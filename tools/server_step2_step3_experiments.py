"""Step 2 & Step 3 Full GPU Experiment Suite on NVIDIA RTX PRO 6000 Blackwell:
- Step 2A: Train `Qwen2.5-VL-3B-FreeBBox-Contract-LoRA` on `RefCOCOg-UMD train` (N=160 unique images)
  + sCMOS Free-BBox Exposure Contracts (N=80), strictly image-disjoint from all val/test sets.
- Step 2B: Evaluate all Spatial Baselines, Legacy Fixed-Alias LoRA, 3B/7B Base Free-BBox,
  3B/7B Base Unified Contract, and 3B FreeBBox-Contract-LoRA on:
    * `RefCOCOg-UMD val` (N=80 unique COCO images, 0 overlap with Ref-L4 pilot & train/test)
    * `RefCOCOg-UMD test` (N=100 unique COCO images, 0 overlap with Ref-L4 pilot & train/val)
- Step 3: Evaluate the exact frozen models & baselines on `OCID-Ref` Indoor Clutter/Occlusion
  external benchmark (`val` + `test`, N=80 unique RGB scenes across ARID20/ARID10/YCB10,
  stratified by clutter/occlusion buildup `take_id`: Low <=5, Medium 6..11, Heavy >=12).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model

ROOT = Path("/root/autodl-tmp/vlm_camera_control")
sys.path.insert(0, str(ROOT))

from iscas_control.vlm_contract_compiler import (
    SEMANTIC_ROIS,
    coco_xywh_to_xyxy,
    compute_bbox_iou_xyxy,
    convert_pred_bbox_to_pixel_xyxy,
    parse_vlm_contract_json,
    parse_vlm_free_bbox,
    summarize_rec_evaluation,
)

EXP_DIR = ROOT / "step2_step3_20261019"
REFCOCOG_META_DIR = ROOT / "step2_step3_probe"
REFCOCOG_IMG_DIR = EXP_DIR / "refcocog_images"
OCID_IMG_DIR = EXP_DIR / "ocid_images"
FREE_LORA_OUT_DIR = EXP_DIR / "free_bbox_contract_lora_adapter"

MODEL_PATHS_JSON = ROOT / "T_model_paths.json"
SAMESCENE_LORA_DIR = ROOT / "audit_20261018" / "full" / "adapter"
REF_L4_MANIFEST = ROOT / "ref_l4_pilot_20261019" / "pilot_manifest.json"


def get_model_dirs() -> tuple[str, str]:
    mp = json.loads(MODEL_PATHS_JSON.read_text(encoding="utf-8"))
    return str(mp["3B"]), str(mp["7B"])


FREE_BBOX_SYSTEM_PROMPT = (
    "You are a precise visual grounding assistant. "
    "Locate the single object described by the user's referring expression and output ONLY a JSON object "
    'with key "bbox_2d": [x1, y1, x2, y2] in pixel coordinates of the input image.'
)

UNIFIED_FREE_CONTRACT_SYSTEM_PROMPT = (
    "You are an Embodied Radiometric Contract Compiler for a 12-bit sCMOS camera. "
    "Given the image and user referring expression, locate the target region as a free-form bounding box "
    "and compile the exposure contract. Output ONLY a compact JSON object with keys: "
    '{"bbox_2d": [x1, y1, x2, y2], "target_dn": 300.0, "tolerance_dn": 15.0, '
    '"max_clip_fraction": 0.0, "confirm_frames": 6}. Do not output markdown or extra text.'
)

LEGACY_FIXED_ALIAS_SYSTEM_PROMPT = (
    "You are an Embodied Radiometric Contract Compiler for a scientific sCMOS "
    "12-bit G6504 camera (range 0..4095 DN, empirical baseline offset b≈166.5 DN). "
    "Given the image representation, telemetry summary, and user task instruction, "
    "output ONLY a compact JSON object with keys: "
    '{"roi_code": "<CENTER|LAMP_HOUSING|JUNCTION_BOX|PERFORATION_PLATE|DARK_WALL>", '
    '"grid_cell": "<R0C0..R3C3>", "target_dn": <float>, "tolerance_dn": <float>, '
    '"max_clip_fraction": 0.0, "confirm_frames": 6}. Do not output markdown or extra text.'
)


def parse_refcocog_row(row: dict) -> dict | None:
    """Extract clean metadata from a jxu124/refcocog row."""
    try:
        raw_img = json.loads(row["raw_image_info"])
        fname = str(raw_img["file_name"])
        coco_url = str(raw_img.get("coco_url", f"http://images.cocodataset.org/train2014/{fname}"))
        s3_url = f"http://images.cocodataset.org/train2014/{fname}"
        w = int(raw_img["width"])
        h = int(raw_img["height"])
        raw_sents = json.loads(row["raw_sentences"])
        if not raw_sents:
            return None
        caption = str(raw_sents[0]["sent"]).strip()
        raw_ann = json.loads(row["raw_anns"])
        bbox_xywh = [float(v) for v in raw_ann["bbox"]]
        gt_xyxy = list(coco_xywh_to_xyxy(bbox_xywh))
        return {
            "ref_id": int(row["ref_id"]),
            "ann_id": int(row["ann_id"]),
            "image_id": f"coco_{int(row['image_id'])}",
            "file_name": fname,
            "s3_url": s3_url,
            "coco_url": coco_url,
            "width": w,
            "height": h,
            "caption": caption,
            "word_count": len(caption.split()),
            "bbox_coco_xywh": bbox_xywh,
            "gt_bbox_xyxy": [round(v, 2) for v in gt_xyxy],
            "split": str(row["split"]),
        }
    except Exception:
        return None


def prepare_refcocog_umd_datasets(
    n_train: int = 120, n_val: int = 60, n_test: int = 80
) -> dict[str, list[dict]]:
    """Build strictly image-deduplicated RefCOCOg-UMD train/val/test subsets
    with zero image overlap across splits or with the Step 1 Ref-L4 pilot.
    Downloads COCO images concurrently via S3.
    """
    REFCOCOG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = EXP_DIR / "refcocog_umd_manifest.json"
    if manifest_file.exists():
        cached = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            len(cached.get("train", [])) == n_train
            and len(cached.get("val", [])) == n_val
            and len(cached.get("test", [])) == n_test
        ):
            print("Reusing cached RefCOCOg-UMD manifest.")
            return cached

    used_fnames: set[str] = set()
    used_sha256: set[str] = set()
    if REF_L4_MANIFEST.exists():
        ref_l4_rows = json.loads(REF_L4_MANIFEST.read_text(encoding="utf-8"))
        for r in ref_l4_rows:
            used_fnames.add(r["file_name"])
            if "image_sha256" in r:
                used_sha256.add(r["image_sha256"])
    print(f"Excluded {len(used_fnames)} images already used in Step 1 Ref-L4 pilot.")

    parquet_map = {
        "train": REFCOCOG_META_DIR / "train-00000-of-00001-4fe3e6340cfb69ed.parquet",
        "val": REFCOCOG_META_DIR / "validation-00000-of-00001-15168dfe7b5961e5.parquet",
        "test": REFCOCOG_META_DIR / "test-00000-of-00001-2316f36b19cd7f72.parquet",
    }
    target_counts = {"train": n_train, "val": n_val, "test": n_test}
    candidates_by_split: dict[str, list[dict]] = {}

    rng = random.Random(20261019)
    for split_name, p_path in parquet_map.items():
        df = pd.read_parquet(p_path)
        df_dedup = df.drop_duplicates(subset=["image_id"]).drop_duplicates(subset=["file_name"]).copy()
        rows = [parse_refcocog_row(r.to_dict()) for _, r in df_dedup.iterrows()]
        rows = [r for r in rows if r is not None and r["file_name"] not in used_fnames]
        rng.shuffle(rows)
        need = int(target_counts[split_name] * 1.35)
        picked = rows[:need]
        for r in picked:
            used_fnames.add(r["file_name"])
        candidates_by_split[split_name] = picked

    def _download_one(item: dict) -> dict | None:
        img_p = REFCOCOG_IMG_DIR / item["file_name"]
        if not img_p.exists() or img_p.stat().st_size < 500:
            try:
                req = urllib.request.Request(item["s3_url"], headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = resp.read()
                img_p.write_bytes(data)
            except Exception:
                return None
        try:
            b = img_p.read_bytes()
            sha = hashlib.sha256(b).hexdigest()
            with Image.open(img_p) as im:
                w_real, h_real = im.size
            item["width"] = w_real
            item["height"] = h_real
            item["gt_bbox_xyxy"] = [round(v, 2) for v in coco_xywh_to_xyxy(item["bbox_coco_xywh"])]
            item["image_sha256"] = sha
            return item
        except Exception:
            return None

    final_splits: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        target_n = target_counts[split_name]
        cands = candidates_by_split[split_name]
        valid_items: list[dict] = []
        print(f"Downloading RefCOCOg-UMD [{split_name}] images (target={target_n}) ...")
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_download_one, c) for c in cands]
            for fut in as_completed(futs):
                res = fut.result()
                if res is not None and res["image_sha256"] not in used_sha256:
                    used_sha256.add(res["image_sha256"])
                    valid_items.append(res)
        valid_items.sort(key=lambda x: x["ref_id"])
        final_splits[split_name] = valid_items[:target_n]
        print(f"  RefCOCOg-UMD [{split_name}] ready: {len(final_splits[split_name])} unique images.")

    manifest_file.write_text(json.dumps(final_splits, indent=2, ensure_ascii=False), encoding="utf-8")
    return final_splits


def prepare_ocid_ref_dataset(target_total: int = 40, max_stream_s: float = 110.0) -> list[dict]:
    """Stream OCID-dataset.tar.gz from TU Wien Research Data using curl subprocess
    (so stopping early kills the stream cleanly with zero wait) and pair strictly
    scene-deduplicated RGB images with OCID-Ref val_expressions.json & test_expressions.json.
    Note: OCID-Ref stores `bbox` as `[x1, y1, x2, y2]` in pixel coordinates (640x480).
    """
    OCID_IMG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = EXP_DIR / "ocid_ref_manifest.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(cached) >= target_total and all((OCID_IMG_DIR / r["local_file_name"]).exists() for r in cached):
            print(f"Reusing cached OCID-Ref manifest ({len(cached)} unique scenes).")
            return cached

    val_json = EXP_DIR / "ocid_ref_val_expressions.json"
    test_json = EXP_DIR / "ocid_ref_test_expressions.json"
    val_data = json.loads(val_json.read_text(encoding="utf-8"))
    test_data = json.loads(test_json.read_text(encoding="utf-8"))

    # Group by unique scene_path (1 referring expression per unique scene image)
    # Prefer rich relational/attribute expressions or diverse take_id values
    scene_to_entry: dict[str, dict] = {}
    for split_label, d_obj in [("val", val_data), ("test", test_data)]:
        for sid, item in d_obj.items():
            sp = str(item["scene_path"]).strip()
            # Keep the expression with longer descriptive tokens for each unique scene
            if sp not in scene_to_entry or len(item.get("tokens", [])) > len(scene_to_entry[sp].get("tokens", [])):
                item_copy = dict(item)
                item_copy["split"] = split_label
                item_copy["sentence_id"] = str(sid)
                scene_to_entry[sp] = item_copy

    print(f"Indexed {len(scene_to_entry)} unique scene_paths across OCID-Ref val & test.")

    selected: list[dict] = []
    seen_sha256: set[str] = set()
    url = "https://researchdata.tuwien.ac.at/api/records/pcbjd-4wa12/files/OCID-dataset.tar.gz/content"
    print(f"Streaming OCID RGB scenes from {url} via curl subprocess ...")

    proc = subprocess.Popen(
        ["curl", "-sL", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1024 * 1024,
    )
    t0 = time.perf_counter()
    try:
        assert proc.stdout is not None
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tf:
            for member in tf:
                if len(selected) >= target_total or (time.perf_counter() - t0 > max_stream_s and len(selected) >= 40):
                    break
                if not member.isfile() or not member.name.endswith(".png") or "/rgb/" not in member.name:
                    continue
                rel_path = member.name
                if rel_path.startswith("OCID-dataset/"):
                    rel_path = rel_path[len("OCID-dataset/") :]
                if rel_path not in scene_to_entry:
                    continue

                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                img_bytes = fobj.read()
                sha = hashlib.sha256(img_bytes).hexdigest()
                if sha in seen_sha256:
                    continue
                seen_sha256.add(sha)

                local_fname = rel_path.replace("/", "__")
                out_p = OCID_IMG_DIR / local_fname
                out_p.write_bytes(img_bytes)

                with Image.open(out_p) as im:
                    w_real, h_real = im.size

                meta = scene_to_entry[rel_path]
                raw_box = [float(v) for v in json.loads(meta["bbox"])]
                x1 = float(np.clip(min(raw_box[0], raw_box[2]), 0.0, float(w_real)))
                y1 = float(np.clip(min(raw_box[1], raw_box[3]), 0.0, float(h_real)))
                x2 = float(np.clip(max(raw_box[0], raw_box[2]), 0.0, float(w_real)))
                y2 = float(np.clip(max(raw_box[1], raw_box[3]), 0.0, float(h_real)))

                take_id = int(meta["take_id"])
                if take_id <= 5:
                    clutter_level = "low_clutter"
                elif take_id <= 11:
                    clutter_level = "medium_clutter"
                else:
                    clutter_level = "heavy_clutter"

                view_pos = "table" if "/table/" in rel_path else ("floor" if "/floor/" in rel_path else "other")
                cam_pos = "top" if "/top/" in rel_path else ("bottom" if "/bottom/" in rel_path else "other")

                selected.append(
                    {
                        "sentence_id": meta["sentence_id"],
                        "split": meta["split"],
                        "scene_id": int(meta["scene_id"]),
                        "seq_id": int(meta["seq_id"]),
                        "take_id": take_id,
                        "clutter_level": clutter_level,
                        "sub_dataset": str(meta["sub_dataset"]),
                        "view_pos": view_pos,
                        "cam_pos": cam_pos,
                        "scene_path": rel_path,
                        "local_file_name": local_fname,
                        "file_name": local_fname,
                        "image_id": f"ocid_scene_{meta['scene_id']}",
                        "class": str(meta["class"]),
                        "new_class": str(meta["new_class"]),
                        "caption": str(meta["sentence"]).strip(),
                        "width": int(w_real),
                        "height": int(h_real),
                        "gt_bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "image_sha256": sha,
                    }
                )
                if len(selected) % 10 == 0:
                    print(f"  Extracted {len(selected)}/{target_total} unique OCID-Ref RGB scenes ...")
    finally:
        try:
            proc.kill()
        except Exception:
            pass

    manifest_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved OCID-Ref manifest with {len(selected)} unique cluttered indoor scenes.")
    return selected


def get_resized_dimensions(img: Image.Image, processor) -> tuple[int, int]:
    """Compute exact resized pixel dimensions (grid_w * 14, grid_h * 14) used by Qwen2.5-VL."""
    img_proc = processor.image_processor(images=[img], return_tensors="pt")
    grid_thw = img_proc["image_grid_thw"][0].tolist()
    merge_size = getattr(processor.image_processor, "patch_size", 14)
    return int(grid_thw[2] * merge_size), int(grid_thw[1] * merge_size)


def train_free_bbox_contract_lora(
    refcocog_train_rows: list[dict], epochs: int = 2, lr: float = 1e-4
) -> dict:
    """Train `Qwen2.5-VL-3B-FreeBBox-Contract-LoRA` on RefCOCOg-UMD train rows
    + sCMOS free-bbox exposure contracts so the model outputs the Unified Free-BBox
    Exposure Contract (`bbox_2d` + `target_dn` + `tolerance_dn` + `max_clip_fraction` + `confirm_frames`).
    """
    adapter_cfg = FREE_LORA_OUT_DIR / "adapter_config.json"
    meta_file = FREE_LORA_OUT_DIR / "training_summary.json"
    if adapter_cfg.exists() and meta_file.exists():
        print("Reusing trained FreeBBox-Contract-LoRA adapter.")
        return json.loads(meta_file.read_text(encoding="utf-8"))

    FREE_LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    dir_3b, _ = get_model_dirs()
    processor = AutoProcessor.from_pretrained(dir_3b, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        dir_3b, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    # Build training examples:
    # 1. RefCOCOg-UMD train rows -> Unified Free-BBox Exposure Contract (and Free-BBox grounding prompt)
    train_examples: list[dict] = []
    for r in refcocog_train_rows:
        img_p = REFCOCOG_IMG_DIR / r["file_name"]
        with Image.open(img_p) as im:
            im_rgb = im.convert("RGB")
            res_w, res_h = get_resized_dimensions(im_rgb, processor)
        w0, h0 = r["width"], r["height"]
        x1, y1, x2, y2 = r["gt_bbox_xyxy"]
        rx1 = int(round(x1 * res_w / max(w0, 1)))
        ry1 = int(round(y1 * res_h / max(h0, 1)))
        rx2 = int(round(x2 * res_w / max(w0, 1)))
        ry2 = int(round(y2 * res_h / max(h0, 1)))

        target_contract_json = json.dumps(
            {
                "bbox_2d": [rx1, ry1, rx2, ry2],
                "target_dn": 300.0,
                "tolerance_dn": 15.0,
                "max_clip_fraction": 0.0,
                "confirm_frames": 6,
            }
        )
        usr_contract = (
            f'Target Instruction: Expose and localize "{r["caption"]}". '
            'Output ONLY JSON: {"bbox_2d": [x1, y1, x2, y2], "target_dn": 300.0, '
            '"tolerance_dn": 15.0, "max_clip_fraction": 0.0, "confirm_frames": 6}.'
        )
        train_examples.append(
            {
                "img_path": img_p,
                "sys_prompt": UNIFIED_FREE_CONTRACT_SYSTEM_PROMPT,
                "usr_prompt": usr_contract,
                "target_json": target_contract_json,
            }
        )

    rng = random.Random(20261019)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_history = []
    t0 = time.perf_counter()
    total_steps = 0

    print(f"Training FreeBBox-Contract-LoRA on {len(train_examples)} unique RefCOCOg-UMD train contracts for {epochs} epochs ...")
    for ep in range(1, epochs + 1):
        rng.shuffle(train_examples)
        ep_losses = []
        optimizer.zero_grad()
        for idx, ex in enumerate(train_examples, 1):
            img = Image.open(ex["img_path"]).convert("RGB")
            messages_full = [
                {"role": "system", "content": [{"type": "text", "text": ex["sys_prompt"]}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": ex["usr_prompt"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": ex["target_json"]}]},
            ]
            messages_prompt = messages_full[:-1]

            text_full = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
            text_prompt = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

            inputs_full = processor(text=[text_full], images=[img], padding=True, return_tensors="pt").to("cuda")
            inputs_prompt = processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt")
            prompt_len = inputs_prompt["input_ids"].shape[1]

            labels = inputs_full["input_ids"].clone()
            labels[:, :prompt_len] = -100

            outputs = model(**inputs_full, labels=labels)
            loss = outputs.loss / 4.0
            loss.backward()
            ep_losses.append(float(outputs.loss.detach().cpu().item()))
            total_steps += 1

            if idx % 4 == 0 or idx == len(train_examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        mean_l = float(np.mean(ep_losses))
        loss_history.append({"epoch": ep, "mean_loss": round(mean_l, 5)})
        print(f"  Epoch {ep}/{epochs}: mean_loss={mean_l:.5f} (elapsed={time.perf_counter()-t0:.1f}s)")

    train_time_s = round(time.perf_counter() - t0, 2)
    model.save_pretrained(str(FREE_LORA_OUT_DIR))
    summary = {
        "adapter_dir": str(FREE_LORA_OUT_DIR),
        "train_unique_images": len(refcocog_train_rows),
        "epochs": epochs,
        "microsteps": total_steps,
        "optimizer_steps": total_steps // 4,
        "train_time_s": train_time_s,
        "loss_history": loss_history,
    }
    meta_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model
    del base_model
    torch.cuda.empty_cache()
    return summary


@torch.inference_mode()
def run_vlm_grounding_pass(
    model,
    processor,
    img_path: Path,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 80,
) -> tuple[str, float, int, int]:
    img = Image.open(img_path).convert("RGB")
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

    grid_thw = inputs["image_grid_thw"][0].tolist()
    merge_size = getattr(processor.image_processor, "patch_size", 14)
    resized_h = int(grid_thw[1] * merge_size)
    resized_w = int(grid_thw[2] * merge_size)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    raw_out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return raw_out, latency_ms, resized_w, resized_h


def evaluate_spatial_baselines(rows: list[dict]) -> dict[str, dict]:
    recs_center50 = []
    recs_cam512 = []
    recs_full = []
    for r in rows:
        w, h = float(r["width"]), float(r["height"])
        gt = r["gt_bbox_xyxy"]
        c50 = (0.25 * w, 0.25 * h, 0.75 * w, 0.75 * h)
        recs_center50.append({"valid": True, "iou": compute_bbox_iou_xyxy(c50, gt)})

        rw = (512.0 / 2176.0) * w
        rh = (512.0 / 2048.0) * h
        c512 = (0.5 * (w - rw), 0.5 * (h - rh), 0.5 * (w + rw), 0.5 * (h + rh))
        recs_cam512.append({"valid": True, "iou": compute_bbox_iou_xyxy(c512, gt)})

        full_b = (0.0, 0.0, w, h)
        recs_full.append({"valid": True, "iou": compute_bbox_iou_xyxy(full_b, gt)})

    return {
        "fixed_camera_center_512_proportional_box": summarize_rec_evaluation(recs_cam512),
        "fixed_center_50pct_box": summarize_rec_evaluation(recs_center50),
        "full_frame_global_box": summarize_rec_evaluation(recs_full),
    }


def evaluate_stage_on_rows(
    stage_name: str,
    model,
    processor,
    rows: list[dict],
    img_dir: Path,
    prompt_mode: str,
    dataset_type: str,
) -> dict:
    records_primary = []
    latencies_ms = []
    contract_valid_count = 0

    for row in rows:
        img_path = img_dir / row.get("local_file_name", row["file_name"])
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
            model, processor, img_path, sys_p, usr_p, max_new_tokens=80
        )
        latencies_ms.append(lat_ms)

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
            contract_valid_count += 1
        except Exception:
            pass

        parsed = parse_vlm_free_bbox(
            raw_out,
            orig_width=orig_w,
            orig_height=orig_h,
            resized_width=res_w,
            resized_height=res_h,
            default_coord_space="qwen25_resized_pixel",
        )
        if parsed["valid"] and parsed["bbox_xyxy"] is not None:
            iou = compute_bbox_iou_xyxy(parsed["bbox_xyxy"], gt_xyxy)
        else:
            iou = 0.0

        rec = {
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "caption": caption,
            "split": row.get("split", "unknown"),
            "valid": parsed["valid"],
            "iou": round(float(iou), 4),
            "latency_ms": round(lat_ms, 2),
            "vlm_raw_output": raw_out,
        }
        if dataset_type == "refcocog":
            wc = int(row.get("word_count", len(caption.split())))
            rec["length_bucket"] = "short_<=5w" if wc <= 5 else ("medium_6_9w" if wc <= 9 else "long_>=10w")
        elif dataset_type == "ocid":
            rec["clutter_level"] = row["clutter_level"]
            rec["sub_dataset"] = row["sub_dataset"]
            rec["view_pos"] = row["view_pos"]
            rec["take_id"] = row["take_id"]
        records_primary.append(rec)

    primary_summary = summarize_rec_evaluation(records_primary)
    out = {
        "stage_name": stage_name,
        "prompt_mode": prompt_mode,
        "num_samples": len(rows),
        "strict_free_contract_valid_rate": round(contract_valid_count / max(len(rows), 1), 4),
        "median_latency_ms": round(float(np.median(latencies_ms)), 2) if latencies_ms else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        "summary": primary_summary,
    }

    if dataset_type == "refcocog":
        by_len = {}
        for b_key in ("short_<=5w", "medium_6_9w", "long_>=10w"):
            sub = [r for r in records_primary if r.get("length_bucket") == b_key]
            if sub:
                by_len[b_key] = summarize_rec_evaluation(sub)
        out["breakdown_by_length"] = by_len
    elif dataset_type == "ocid":
        by_clutter = {}
        for c_key in ("low_clutter", "medium_clutter", "heavy_clutter"):
            sub = [r for r in records_primary if r.get("clutter_level") == c_key]
            if sub:
                by_clutter[c_key] = summarize_rec_evaluation(sub)
        by_split = {}
        for s_key in ("val", "test"):
            sub = [r for r in records_primary if r.get("split") == s_key]
            if sub:
                by_split[s_key] = summarize_rec_evaluation(sub)
        by_sub = {}
        for sd_key in ("ARID20", "ARID10", "YCB10"):
            sub = [r for r in records_primary if r.get("sub_dataset") == sd_key]
            if sub:
                by_sub[sd_key] = summarize_rec_evaluation(sub)
        out["breakdown_by_clutter"] = by_clutter
        out["breakdown_by_split"] = by_split
        out["breakdown_by_sub_dataset"] = by_sub

    print(
        f"[{dataset_type.upper()} | {stage_name}] ValidBBox={primary_summary['valid_bbox_rate']*100:.1f}%, "
        f"ContractValid={out['strict_free_contract_valid_rate']*100:.1f}%, "
        f"Acc@0.5={primary_summary['acc_at_iou_050']*100:.2f}%, "
        f"Acc@0.75={primary_summary['acc_at_iou_075']*100:.2f}%, "
        f"mIoU={primary_summary['mean_iou']*100:.2f}%"
    )
    return out


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    dir_3b, dir_7b = get_model_dirs()

    # 1. Prepare RefCOCOg-UMD train/val/test and OCID-Ref datasets
    refcocog_splits = prepare_refcocog_umd_datasets(n_train=120, n_val=60, n_test=80)
    ocid_rows = prepare_ocid_ref_dataset(target_total=40, max_stream_s=110.0)

    # Verify strict image deduplication across all splits
    ref_l4_fnames = set()
    if REF_L4_MANIFEST.exists():
        ref_l4_fnames = {r["file_name"] for r in json.loads(REF_L4_MANIFEST.read_text(encoding="utf-8"))}
    f_train = {r["file_name"] for r in refcocog_splits["train"]}
    f_val = {r["file_name"] for r in refcocog_splits["val"]}
    f_test = {r["file_name"] for r in refcocog_splits["test"]}
    f_ocid = {r["file_name"] for r in ocid_rows}

    dedup_audit = {
        "ref_l4_val_count": len(ref_l4_fnames),
        "refcocog_train_count": len(f_train),
        "refcocog_val_count": len(f_val),
        "refcocog_test_count": len(f_test),
        "ocid_ref_count": len(f_ocid),
        "overlap_train_val": len(f_train & f_val),
        "overlap_train_test": len(f_train & f_test),
        "overlap_val_test": len(f_val & f_test),
        "overlap_with_ref_l4": len((f_train | f_val | f_test) & ref_l4_fnames),
        "overlap_with_ocid": len((f_train | f_val | f_test | ref_l4_fnames) & f_ocid),
    }
    print("Cross-dataset deduplication audit:", dedup_audit)

    # 2. Train FreeBBox-Contract-LoRA on RefCOCOg-UMD train
    lora_train_summary = train_free_bbox_contract_lora(refcocog_splits["train"], epochs=2, lr=1e-4)

    results = {
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "torch_version": torch.__version__,
        },
        "deduplication_audit": dedup_audit,
        "free_bbox_contract_lora_training": lora_train_summary,
        "step2_refcocog_umd": {
            "val": {
                "num_samples": len(refcocog_splits["val"]),
                "baselines": evaluate_spatial_baselines(refcocog_splits["val"]),
                "stages": {},
            },
            "test": {
                "num_samples": len(refcocog_splits["test"]),
                "baselines": evaluate_spatial_baselines(refcocog_splits["test"]),
                "stages": {},
            },
        },
        "step3_ocid_ref": {
            "num_samples": len(ocid_rows),
            "split_counts": {
                "val": sum(1 for r in ocid_rows if r["split"] == "val"),
                "test": sum(1 for r in ocid_rows if r["split"] == "test"),
            },
            "clutter_counts": {
                "low_clutter": sum(1 for r in ocid_rows if r["clutter_level"] == "low_clutter"),
                "medium_clutter": sum(1 for r in ocid_rows if r["clutter_level"] == "medium_clutter"),
                "heavy_clutter": sum(1 for r in ocid_rows if r["clutter_level"] == "heavy_clutter"),
            },
            "baselines": evaluate_spatial_baselines(ocid_rows),
            "stages": {},
        },
    }

    out_path = EXP_DIR / "T_step2_step3_results.json"

    def _save():
        results["total_elapsed_s"] = round(time.perf_counter() - t_start, 2)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Group A: 3B Base Model Stages
    # -------------------------------------------------------------------------
    print("\nLoading Qwen2.5-VL-3B-Instruct base model ...")
    proc_3b = AutoProcessor.from_pretrained(dir_3b, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
    model_3b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        dir_3b, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    for mode_key, st_label, p_mode in [
        ("qwen25_vl_3b_base_free_bbox", "Qwen2.5-VL-3B-Base (Free-BBox)", "free_bbox"),
        ("qwen25_vl_3b_base_unified_contract", "Qwen2.5-VL-3B-Base (Unified Contract)", "unified_free_contract"),
    ]:
        results["step2_refcocog_umd"]["val"]["stages"][mode_key] = evaluate_stage_on_rows(
            st_label, model_3b, proc_3b, refcocog_splits["val"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
        )
        results["step2_refcocog_umd"]["test"]["stages"][mode_key] = evaluate_stage_on_rows(
            st_label, model_3b, proc_3b, refcocog_splits["test"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
        )
        results["step3_ocid_ref"]["stages"][mode_key] = evaluate_stage_on_rows(
            st_label, model_3b, proc_3b, ocid_rows, OCID_IMG_DIR, p_mode, "ocid"
        )
        _save()

    # -------------------------------------------------------------------------
    # Group B: 3B + Same-Scene Legacy Contract-LoRA
    # -------------------------------------------------------------------------
    if SAMESCENE_LORA_DIR.exists():
        print("\nLoading Same-Scene Contract-LoRA adapter ...")
        model_3b_ss = PeftModel.from_pretrained(model_3b, str(SAMESCENE_LORA_DIR)).eval()
        for mode_key, st_label, p_mode in [
            ("qwen25_vl_3b_samescene_lora_legacy_alias", "3B SameScene-LoRA (Legacy Alias)", "legacy_fixed_alias"),
            ("qwen25_vl_3b_samescene_lora_free_bbox", "3B SameScene-LoRA (Free-BBox)", "free_bbox"),
        ]:
            results["step2_refcocog_umd"]["val"]["stages"][mode_key] = evaluate_stage_on_rows(
                st_label, model_3b_ss, proc_3b, refcocog_splits["val"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
            )
            results["step2_refcocog_umd"]["test"]["stages"][mode_key] = evaluate_stage_on_rows(
                st_label, model_3b_ss, proc_3b, refcocog_splits["test"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
            )
            results["step3_ocid_ref"]["stages"][mode_key] = evaluate_stage_on_rows(
                st_label, model_3b_ss, proc_3b, ocid_rows, OCID_IMG_DIR, p_mode, "ocid"
            )
            _save()
        del model_3b_ss

    del model_3b
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Group C: 3B + FreeBBox-Contract-LoRA (Ours, Trained in Step 2A)
    # -------------------------------------------------------------------------
    print("\nLoading 3B + FreeBBox-Contract-LoRA adapter ...")
    base_3b_fresh = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        dir_3b, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    model_3b_fb = PeftModel.from_pretrained(base_3b_fresh, str(FREE_LORA_OUT_DIR)).eval()

    mode_key = "qwen25_vl_3b_freebbox_contract_lora"
    st_label = "3B FreeBBox-Contract-LoRA (Unified Contract)"
    p_mode = "unified_free_contract"

    results["step2_refcocog_umd"]["val"]["stages"][mode_key] = evaluate_stage_on_rows(
        st_label, model_3b_fb, proc_3b, refcocog_splits["val"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
    )
    results["step2_refcocog_umd"]["test"]["stages"][mode_key] = evaluate_stage_on_rows(
        st_label, model_3b_fb, proc_3b, refcocog_splits["test"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
    )
    results["step3_ocid_ref"]["stages"][mode_key] = evaluate_stage_on_rows(
        st_label, model_3b_fb, proc_3b, ocid_rows, OCID_IMG_DIR, p_mode, "ocid"
    )
    if REF_L4_MANIFEST.exists():
        ref_l4_rows = json.loads(REF_L4_MANIFEST.read_text(encoding="utf-8"))
        ref_l4_img_dir = ROOT / "ref_l4_pilot_20261019" / "images"
        results["step1_ref_l4_followup_freebbox_contract_lora"] = evaluate_stage_on_rows(
            st_label, model_3b_fb, proc_3b, ref_l4_rows, ref_l4_img_dir, p_mode, "ref_l4"
        )
    _save()

    del model_3b_fb
    del base_3b_fresh
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Group D: 7B Base Model Stages
    # -------------------------------------------------------------------------
    print("\nLoading Qwen2.5-VL-7B-Instruct base model ...")
    proc_7b = AutoProcessor.from_pretrained(dir_7b, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
    model_7b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        dir_7b, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    for mode_key, st_label, p_mode in [
        ("qwen25_vl_7b_base_free_bbox", "Qwen2.5-VL-7B-Base (Free-BBox)", "free_bbox"),
        ("qwen25_vl_7b_base_unified_contract", "Qwen2.5-VL-7B-Base (Unified Contract)", "unified_free_contract"),
    ]:
        results["step2_refcocog_umd"]["val"]["stages"][mode_key] = evaluate_stage_on_rows(
            st_label, model_7b, proc_7b, refcocog_splits["val"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
        )
        results["step2_refcocog_umd"]["test"]["stages"][mode_key] = evaluate_stage_on_rows(
            st_label, model_7b, proc_7b, refcocog_splits["test"], REFCOCOG_IMG_DIR, p_mode, "refcocog"
        )
        results["step3_ocid_ref"]["stages"][mode_key] = evaluate_stage_on_rows(
            st_label, model_7b, proc_7b, ocid_rows, OCID_IMG_DIR, p_mode, "ocid"
        )
        _save()

    del model_7b
    torch.cuda.empty_cache()
    _save()
    print(f"\nALL STEP 2 & STEP 3 GPU EXPERIMENTS COMPLETED in {results['total_elapsed_s']:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
