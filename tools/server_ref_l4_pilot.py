"""Ref-L4 Validation-Set Small-Scale Pilot on NVIDIA RTX PRO 6000 Blackwell.

Executes Step 1 of the Unseen-Scene Generalization Plan:
  1. Builds a strictly image-deduplicated validation pilot from `ref-l4-val.parquet`
     (never touching `ref-l4-test.parquet`), recording source dataset (`coco` vs `o365`),
     `is_rewrite`, and image SHA-256 for cross-dataset deduplication.
  2. Verifies free-form bounding box output (`bbox_2d`), coordinate conversion across
     three candidate spaces (`qwen25_resized_pixel`, `orig_pixel`, `norm1000`), and
     strict REC scoring (`Acc@IoU0.5`, `Acc@IoU0.75`, `mIoU`, invalid output = 0.0 IoU).
  3. Compares:
     - `Qwen2.5-VL-3B-Instruct` (Base Zero-Shot, Free-BBox Prompt)
     - `Qwen2.5-VL-7B-Instruct` (Base Zero-Shot, Free-BBox Prompt)
     - `Qwen2.5-VL-3B + Same-Scene Fixed-Alias Contract-LoRA` (Legacy Fixed-Alias Prompt vs Free-BBox Prompt)
     - `Qwen2.5-VL-3B / 7B` with the Upgraded Unified Free-BBox Exposure Contract Schema
       (`parse_vlm_contract_json(..., strict_free_bbox=True)`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

from iscas_control.vlm_contract_compiler import (
    coco_xywh_to_xyxy,
    compute_bbox_iou_xyxy,
    parse_vlm_contract_json,
    parse_vlm_free_bbox,
    summarize_rec_evaluation,
)

PILOT_DIR = Path("/root/autodl-tmp/vlm_camera_control/ref_l4_pilot_20261019")
IMG_DIR = PILOT_DIR / "images"
PARQUET_PATH = PILOT_DIR / "ref-l4-val.parquet"
MODEL_PATHS_JSON = Path("/root/autodl-tmp/vlm_camera_control/T_model_paths.json")
SAME_SCENE_LORA_DIR = Path("/root/autodl-tmp/vlm_camera_control/audit_20261018/full/adapter")

FREE_BBOX_SYSTEM_PROMPT = (
    "You are an expert visual grounding model for referring expression comprehension. "
    "Locate the single target object described by the user's referring expression and "
    "output ONLY a compact JSON object: "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "<short target name>"}. '
    "Do not output fixed camera ROI aliases or extra text."
)

UNIFIED_FREE_CONTRACT_SYSTEM_PROMPT = (
    "You are a unified visual grounding and camera exposure contract compiler. "
    "Given the image and the user's referring expression, locate the target region as "
    "a free-form 2D bounding box and compile an exposure contract JSON with keys: "
    '{"bbox_2d": [x1, y1, x2, y2], "target_dn": 300.0, "tolerance_dn": 15.0, '
    '"max_clip_fraction": 0.0, "confirm_frames": 6}. '
    "Output ONLY the JSON object."
)

LEGACY_FIXED_ALIAS_SYSTEM_PROMPT = (
    "You are an expert scientific CMOS (sCMOS) exposure contract compiler for a "
    "12-bit G6504 camera (range 0..4095 DN, empirical baseline offset b≈166.5 DN). "
    "Given the image representation, telemetry summary, and user task instruction, "
    "output ONLY a compact JSON object with keys: "
    '{"roi_code": "<CENTER|LAMP_HOUSING|JUNCTION_BOX|PERFORATION_PLATE|DARK_WALL>", '
    '"grid_cell": "<R0C0..R3C3>", "target_dn": <float>, "tolerance_dn": <float>, '
    '"max_clip_fraction": 0.0, "confirm_frames": 6}. Do not output markdown or extra text.'
)


def extract_deduplicated_val_pilot(target_total: int = 60, max_stream_s: float = 85.0) -> list[dict]:
    """Stream `images.tar.gz` from hf-mirror and extract a strictly image-deduplicated
    pilot sample (`1` row per unique `image_id` and `sha256`) from `ref-l4-val.parquet`.
    """
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET_PATH)
    # Strictly deduplicate by image_id and file_name (1 referring expression per unique image)
    df_dedup = df.drop_duplicates(subset=["image_id"]).drop_duplicates(subset=["file_name"]).copy()
    df_dedup["source_prefix"] = df_dedup["image_id"].str.split("_").str[0]
    row_by_fname = {row["file_name"]: row.to_dict() for _, row in df_dedup.iterrows()}

    manifest_path = PILOT_DIR / "pilot_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(existing) == target_total and all((IMG_DIR / r["file_name"]).exists() for r in existing):
            print(f"Reusing existing deduplicated pilot manifest ({len(existing)} unique images).")
            return existing

    coco_selected: list[dict] = []
    o365_selected: list[dict] = []
    coco_backup: list[dict] = []
    seen_sha256: set[str] = set()

    url = "https://hf-mirror.com/datasets/JierunChen/Ref-L4/resolve/main/images.tar.gz"
    print(f"Streaming validation images from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.perf_counter()

    with urllib.request.urlopen(req, timeout=45) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for member in tf:
                if time.perf_counter() - t0 > max_stream_s and (len(coco_selected) + len(o365_selected) + len(coco_backup)) >= target_total:
                    print(f"Reached stream time limit ({max_stream_s}s); finalizing pilot set.")
                    break
                if not member.isfile():
                    continue
                fname = member.name.split("/")[-1]
                if fname not in row_by_fname:
                    continue
                row = row_by_fname[fname]
                src = row["source_prefix"]

                need_coco = len(coco_selected) < (target_total // 2)
                need_o365 = len(o365_selected) < (target_total - target_total // 2)
                need_backup = len(coco_backup) < (target_total - len(coco_selected))

                if src == "coco" and not (need_coco or need_backup):
                    continue
                if src == "o365" and not need_o365:
                    continue

                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                img_bytes = fobj.read()
                sha = hashlib.sha256(img_bytes).hexdigest()
                if sha in seen_sha256:
                    continue
                seen_sha256.add(sha)

                out_img_path = IMG_DIR / fname
                out_img_path.write_bytes(img_bytes)

                with Image.open(out_img_path) as im:
                    actual_w, actual_h = im.size

                bbox_xywh = [float(x) for x in row["bbox"]]
                gt_xyxy = list(coco_xywh_to_xyxy(bbox_xywh))

                entry = {
                    "id": int(row["id"]),
                    "image_id": str(row["image_id"]),
                    "file_name": fname,
                    "source_prefix": src,
                    "is_rewrite": bool(row["is_rewrite"]),
                    "ori_category_id": str(row["ori_category_id"]),
                    "caption": str(row["caption"]),
                    "width": int(actual_w),
                    "height": int(actual_h),
                    "meta_width": int(row["width"]),
                    "meta_height": int(row["height"]),
                    "bbox_xywh": bbox_xywh,
                    "gt_bbox_xyxy": gt_xyxy,
                    "bbox_area": float(row["bbox_area"]),
                    "image_sha256": sha,
                }
                if src == "coco":
                    if need_coco:
                        coco_selected.append(entry)
                    else:
                        coco_backup.append(entry)
                else:
                    o365_selected.append(entry)

                if len(coco_selected) >= (target_total // 2) and len(o365_selected) >= (target_total - target_total // 2):
                    break

    pilot = coco_selected + o365_selected
    if len(pilot) < target_total:
        needed = target_total - len(pilot)
        pilot.extend(coco_backup[:needed])

    manifest_path.write_text(json.dumps(pilot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Constructed deduplicated Ref-L4 val pilot: total={len(pilot)} "
        f"(coco={sum(1 for x in pilot if x['source_prefix']=='coco')}, "
        f"o365={sum(1 for x in pilot if x['source_prefix']=='o365')})"
    )
    return pilot


@torch.inference_mode()
def run_vlm_grounding_pass(
    model,
    processor,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 64,
) -> tuple[str, float, int, int]:
    """Run one VLM forward pass and return `(raw_output, latency_ms, resized_w, resized_h)`."""
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[img], padding=True, return_tensors="pt").to("cuda:0")

    # Extract exact Qwen2.5-VL resized pixel dimensions from image_grid_thw (grid_h * 14, grid_w * 14)
    thw = inputs["image_grid_thw"][0].detach().cpu().tolist()
    resized_h = int(thw[1] * 14)
    resized_w = int(thw[2] * 14)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
    out_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return out_text, elapsed_ms, resized_w, resized_h


def evaluate_pilot_stage(
    model,
    processor,
    pilot_rows: list[dict],
    stage_name: str,
    prompt_mode: str = "free_bbox",
) -> dict:
    """Evaluate a model configuration across all `pilot_rows` and audit coordinate spaces."""
    records_by_space = {
        "qwen25_resized_pixel": [],
        "orig_pixel": [],
        "norm1000": [],
    }
    latencies_ms = []
    contract_compile_valid = 0

    for row in pilot_rows:
        img_path = IMG_DIR / row["file_name"]
        caption = row["caption"]
        orig_w, orig_h = row["width"], row["height"]
        gt_xyxy = row["gt_bbox_xyxy"]

        if prompt_mode == "free_bbox":
            sys_p = FREE_BBOX_SYSTEM_PROMPT
            usr_p = (
                f'Locate the single target described by the referring expression: "{caption}". '
                'Output ONLY JSON: {"bbox_2d": [x1, y1, x2, y2], "label": "target"}.'
            )
        elif prompt_mode == "unified_free_contract":
            sys_p = UNIFIED_FREE_CONTRACT_SYSTEM_PROMPT
            usr_p = (
                f'Target Instruction: Expose and localize "{caption}". '
                'Output ONLY JSON: {"bbox_2d": [x1, y1, x2, y2], "target_dn": 300.0, '
                '"tolerance_dn": 15.0, "max_clip_fraction": 0.0, "confirm_frames": 6}.'
            )
        elif prompt_mode == "legacy_fixed_alias":
            sys_p = LEGACY_FIXED_ALIAS_SYSTEM_PROMPT
            usr_p = (
                f"Task Instruction: {caption}\n"
                "Identify the target ROI (CENTER / LAMP_HOUSING / JUNCTION_BOX / PERFORATION_PLATE / DARK_WALL) "
                "and target DN."
            )
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

        raw_out, lat_ms, res_w, res_h = run_vlm_grounding_pass(
            model, processor, img_path, sys_p, usr_p, max_new_tokens=64
        )
        latencies_ms.append(lat_ms)

        # Check strict unified contract compilation when applicable
        try:
            _ = parse_vlm_contract_json(
                raw_out,
                strict_free_bbox=True,
                orig_width=orig_w,
                orig_height=orig_h,
                resized_width=res_w,
                resized_height=res_h,
                coord_space="qwen25_resized_pixel",
            )
            contract_compile_valid += 1
        except Exception:
            pass

        for c_space in ("qwen25_resized_pixel", "orig_pixel", "norm1000"):
            parsed = parse_vlm_free_bbox(
                raw_out,
                orig_width=orig_w,
                orig_height=orig_h,
                resized_width=res_w,
                resized_height=res_h,
                default_coord_space=c_space,
            )
            iou = compute_bbox_iou_xyxy(parsed["bbox_xyxy"], gt_xyxy) if parsed["valid"] else 0.0
            records_by_space[c_space].append({
                "id": row["id"],
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "source_prefix": row["source_prefix"],
                "is_rewrite": row["is_rewrite"],
                "caption": caption,
                "orig_size": [orig_w, orig_h],
                "resized_size": [res_w, res_h],
                "gt_bbox_xyxy": [round(v, 2) for v in gt_xyxy],
                "vlm_raw_output": raw_out,
                "latency_ms": round(lat_ms, 2),
                "valid": parsed["valid"],
                "error": parsed["error"],
                "raw_bbox": parsed["raw_bbox"],
                "pred_bbox_xyxy": [round(v, 2) for v in parsed["bbox_xyxy"]] if parsed["bbox_xyxy"] else None,
                "iou": round(iou, 4),
            })

    coord_space_audit = {
        c_space: summarize_rec_evaluation(recs)
        for c_space, recs in records_by_space.items()
    }

    # Primary coordinate space for Qwen2.5-VL is qwen25_resized_pixel
    primary_recs = records_by_space["qwen25_resized_pixel"]
    primary_summary = coord_space_audit["qwen25_resized_pixel"]

    # Breakdown by source_prefix (coco vs o365) and is_rewrite
    by_source = {}
    for src in sorted({r["source_prefix"] for r in primary_recs}):
        sub = [r for r in primary_recs if r["source_prefix"] == src]
        by_source[src] = summarize_rec_evaluation(sub)

    by_rewrite = {}
    for rw_flag in (False, True):
        sub = [r for r in primary_recs if r["is_rewrite"] == rw_flag]
        if sub:
            by_rewrite["rewrite" if rw_flag else "original"] = summarize_rec_evaluation(sub)

    print(
        f"[{stage_name}] ValidBBox={primary_summary['valid_bbox_rate']*100:.1f}% | "
        f"Acc@0.5(resized_px)={primary_summary['acc_at_iou_050']*100:.1f}% | "
        f"Acc@0.75={primary_summary['acc_at_iou_075']*100:.1f}% | "
        f"mIoU={primary_summary['mean_iou']*100:.1f}% | "
        f"Acc@0.5(orig_px)={coord_space_audit['orig_pixel']['acc_at_iou_050']*100:.1f}% | "
        f"Acc@0.5(norm1000)={coord_space_audit['norm1000']['acc_at_iou_050']*100:.1f}%"
    )

    return {
        "stage_name": stage_name,
        "prompt_mode": prompt_mode,
        "num_samples": len(pilot_rows),
        "strict_free_contract_valid_rate": round(contract_compile_valid / len(pilot_rows), 4),
        "median_latency_ms": round(float(np.median(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "primary_summary_qwen25_resized_pixel": primary_summary,
        "coordinate_space_audit": coord_space_audit,
        "breakdown_by_source": by_source,
        "breakdown_by_rewrite": by_rewrite,
        "records": primary_recs,
    }


def main():
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Running Ref-L4 Validation Pilot on {gpu_name} ({vram_gib:.1f} GiB VRAM)")

    model_paths = json.loads(MODEL_PATHS_JSON.read_text(encoding="utf-8"))
    dir_3b = model_paths["3B"]
    dir_7b = model_paths["7B"]

    pilot_rows = extract_deduplicated_val_pilot(target_total=60)

    # 1. Evaluate Qwen2.5-VL-3B-Instruct (Base Zero-Shot: Free-BBox + Unified Free Contract)
    proc_3b = AutoProcessor.from_pretrained(dir_3b, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
    model_3b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        dir_3b, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    res_3b_free = evaluate_pilot_stage(
        model_3b, proc_3b, pilot_rows,
        stage_name="Qwen2.5-VL-3B-Base-FreeBBox",
        prompt_mode="free_bbox",
    )
    res_3b_contract = evaluate_pilot_stage(
        model_3b, proc_3b, pilot_rows,
        stage_name="Qwen2.5-VL-3B-Base-UnifiedFreeContract",
        prompt_mode="unified_free_contract",
    )

    # 2. Evaluate Same-Scene Fixed-Alias Contract-LoRA on Ref-L4 Pilot
    print(f"Loading Same-Scene Fixed-Alias Contract-LoRA from {SAME_SCENE_LORA_DIR} ...")
    lora_model = PeftModel.from_pretrained(model_3b, str(SAME_SCENE_LORA_DIR)).eval()
    res_3b_lora_legacy = evaluate_pilot_stage(
        lora_model, proc_3b, pilot_rows,
        stage_name="Qwen2.5-VL-3B-SameSceneLoRA-LegacyAliasPrompt",
        prompt_mode="legacy_fixed_alias",
    )
    res_3b_lora_free = evaluate_pilot_stage(
        lora_model, proc_3b, pilot_rows,
        stage_name="Qwen2.5-VL-3B-SameSceneLoRA-FreeBBoxPrompt",
        prompt_mode="free_bbox",
    )
    del lora_model
    del model_3b
    torch.cuda.empty_cache()

    # 3. Evaluate Qwen2.5-VL-7B-Instruct (Base Zero-Shot: Free-BBox + Unified Free Contract)
    proc_7b = AutoProcessor.from_pretrained(dir_7b, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
    model_7b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        dir_7b, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    res_7b_free = evaluate_pilot_stage(
        model_7b, proc_7b, pilot_rows,
        stage_name="Qwen2.5-VL-7B-Base-FreeBBox",
        prompt_mode="free_bbox",
    )
    res_7b_contract = evaluate_pilot_stage(
        model_7b, proc_7b, pilot_rows,
        stage_name="Qwen2.5-VL-7B-Base-UnifiedFreeContract",
        prompt_mode="unified_free_contract",
    )
    del model_7b
    torch.cuda.empty_cache()

    out_payload = {
        "benchmark": "Ref-L4 Validation-Set Small-Scale Engineering & Generalization Pilot",
        "split_used": "val_only (ref-l4-val.parquet; ref-l4-test.parquet untouched)",
        "deduplication_policy": "Strict 1 sample per unique image_id, file_name, and image_sha256",
        "license_notice": "Ref-L4 annotations CC BY-NC 4.0; underlying COCO / Objects365 image licenses apply; evaluation only, no image redistribution",
        "pretraining_caveat": "Public COCO / Objects365 images may overlap with base VLM pretraining; evaluated as unseen-from-our-sCMOS-LoRA external REC benchmark",
        "hardware": {
            "gpu_name": gpu_name,
            "vram_gib": round(vram_gib, 2),
            "torch_version": torch.__version__,
        },
        "pilot_sample_counts": {
            "total_unique_images": len(pilot_rows),
            "coco_source_count": sum(1 for r in pilot_rows if r["source_prefix"] == "coco"),
            "o365_source_count": sum(1 for r in pilot_rows if r["source_prefix"] == "o365"),
            "rewrite_true_count": sum(1 for r in pilot_rows if r["is_rewrite"]),
            "rewrite_false_count": sum(1 for r in pilot_rows if not r["is_rewrite"]),
        },
        "stages": {
            "qwen25_vl_3b_base_free_bbox": {k: v for k, v in res_3b_free.items() if k != "records"},
            "qwen25_vl_3b_base_unified_contract": {k: v for k, v in res_3b_contract.items() if k != "records"},
            "qwen25_vl_3b_samescene_lora_legacy_prompt": {k: v for k, v in res_3b_lora_legacy.items() if k != "records"},
            "qwen25_vl_3b_samescene_lora_free_bbox_prompt": {k: v for k, v in res_3b_lora_free.items() if k != "records"},
            "qwen25_vl_7b_base_free_bbox": {k: v for k, v in res_7b_free.items() if k != "records"},
            "qwen25_vl_7b_base_unified_contract": {k: v for k, v in res_7b_contract.items() if k != "records"},
        },
        "sample_records_7b_free": res_7b_free["records"][:20],
        "sample_records_3b_free": res_3b_free["records"][:20],
        "sample_records_3b_samescene_lora": res_3b_lora_free["records"][:20],
    }

    out_file = PILOT_DIR / "T_ref_l4_val_pilot_results.json"
    out_file.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Saved Ref-L4 validation pilot results to {out_file}")


if __name__ == "__main__":
    main()
