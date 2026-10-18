"""RTX PRO 6000 (98 GB Blackwell) Server Experiment Suite for:
Slow VLM Semantic Contract Compiler + Fast Offset/Secant-Aware Physical Executor.

Executes three core experiments (Experiment 4 removed per user instruction):
  - Experiment 1: Model-Scale (3B vs 7B, Zero-Shot vs Contract-LoRA) x
                  Control Paradigm (P1 Direct VLM Loop vs P2 Compiler+Ratio
                  vs P3 Compiler+SharedOffset vs P4 Compiler+SecantOffset)
  - Experiment 2: sCMOS-ContractBench Multi-Task Evaluation across 4 semantic
                  tasks (T1 Dark Wall, T2 Perforation Localization, T3 Junction
                  Box Label, T4 Center Radiometric) and 4 apertures (f/2.8..f/16)
  - Experiment 3: 12-Bit RAW Visual Representation Ablation (Linear vs MinMax
                  vs Offset-Aware Dual-View Grid) + BF16 Contract-LoRA Training
                  on NVIDIA RTX PRO 6000 Blackwell.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
import math
import numpy as np
import torch
from PIL import Image

from iscas_control.vlm_contract_compiler import (
    SEMANTIC_ROIS,
    SemanticContract,
    parse_vlm_contract_json,
    parse_vlm_direct_lines,
    simulate_dual_layer_episode,
)

ROOT = Path(__file__).resolve().parent
MODEL_CACHE = Path("/root/autodl-tmp/hf_models")


SYSTEM_CONTRACT_PROMPT = (
    "You are an expert scientific CMOS (sCMOS) exposure contract compiler for a "
    "12-bit G6504 camera (range 0..4095 DN, empirical baseline offset b≈166.5 DN). "
    "Given the image representation, telemetry summary, and user task instruction, "
    "output ONLY a compact JSON object with keys: "
    '{"roi_code": "<CENTER|LAMP_HOUSING|JUNCTION_BOX|PERFORATION_PLATE|DARK_WALL>", '
    '"grid_cell": "<R0C0..R3C3>", "target_dn": <float>, "tolerance_dn": <float>, '
    '"max_clip_fraction": 0.0, "confirm_frames": 6}. Do not output markdown or extra text.'
)

SYSTEM_DIRECT_LOOP_PROMPT = (
    "You are directly controlling the integer exposure lines (range 8..512) of a "
    "12-bit sCMOS camera. Given the image, current exposure lines, and user instruction, "
    'output ONLY a JSON object {"next_lines": <int 8..512>} to move brightness toward '
    "the desired task target."
)


def download_qwen_model(model_id: str) -> str:
    """Download Qwen2.5-VL model via ModelScope (fast China mirror on AutoDL)."""
    manifest = ROOT / "T_model_paths.json"
    if manifest.exists():
        size = model_id.split("-")[-2]
        candidate = json.loads(manifest.read_text()).get(size)
        if candidate and Path(candidate, "config.json").is_file():
            return candidate
    from modelscope import snapshot_download
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(model_id, cache_dir=str(MODEL_CACHE))
    return local_dir


def load_vlm(model_dir: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(
        model_dir, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def vlm_generate(model, processor, image_path: Path, system_prompt: str, user_text: str, max_new_tokens: int = 192) -> tuple[str, float]:
    """Measure synchronized generation latency only (not preprocessing/network)."""
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[img], padding=True, return_tensors="pt").to("cuda:0")

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
    vlm_generate.last_metadata = {
        "input_tokens": int(inputs.input_ids.shape[1]),
        "output_tokens": int(gen_ids.numel()),
        "max_new_tokens": max_new_tokens,
        "hit_token_limit": int(gen_ids.numel()) == max_new_tokens,
        "latency_scope": "CUDA-synchronized generation only; excludes image IO and preprocessing",
    }
    return out_text, elapsed_ms


def train_contract_lora(
    model_dir: str,
    train_rows: list[dict],
    output_lora_dir: Path,
    epochs: int = 2,
    lr: float = 2e-4,
) -> tuple:
    """Fine-tune Qwen2.5-VL-3B-Instruct with BF16 LoRA (r=16, alpha=32) on RTX PRO 6000."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_dir, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.config.use_cache = False
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.train()

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr, weight_decay=0.01)
    loss_history = []
    t_start = time.perf_counter()
    step_count = 0
    optimizer_steps = 0
    total_microsteps = epochs * len(train_rows) * 2
    optimizer.zero_grad(set_to_none=True)

    for ep in range(epochs):
        ep_losses = []
        for idx, row in enumerate(train_rows):
            img_path = ROOT / row["images"]["dual_offset_grid"]
            img = Image.open(img_path).convert("RGB")
            ap_key = row["aperture"]
            target_json = row["ground_truth_json"]

            prompt_variants = [
                (
                    f"Task Instruction: {row['instruction']}\n"
                    f"Scene Telemetry: {row['telemetry_summary']}"
                ),
                (
                    f"Task Instruction: {row['instruction']}\n"
                    f"Aperture={ap_key}, CurrentLines={row['start_lines']}. "
                    "Visually locate the target structure in the 4x4 sensor grid (R0C0..R3C3), "
                    "identify its ROI code (CENTER / LAMP_HOUSING / JUNCTION_BOX / PERFORATION_PLATE / DARK_WALL), "
                    "and predict the calibrated target_dn and tolerance_dn."
                ),
            ]

            for user_prompt in prompt_variants:
                step_count += 1
                messages_full = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_CONTRACT_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target_json}]},
                ]
                messages_prompt = messages_full[:-1]

                full_text = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
                prompt_text = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

                inputs_full = processor(text=[full_text], images=[img], return_tensors="pt").to("cuda:0")
                inputs_prompt = processor(text=[prompt_text], images=[img], return_tensors="pt")
                prompt_len = inputs_prompt.input_ids.shape[1]

                labels = inputs_full.input_ids.clone()
                labels[:, :prompt_len] = -100
                if not (labels != -100).any():
                    raise RuntimeError("No supervised assistant tokens after prompt masking")

                outputs = peft_model(**inputs_full, labels=labels)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite LoRA loss at microstep {step_count}")
                # Scale each accumulation group, including a final partial group.
                group_start = ((step_count - 1) // 4) * 4
                group_size = min(4, total_microsteps - group_start)
                (loss / group_size).backward()

                if step_count % 4 == 0 or step_count == total_microsteps:
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_steps += 1
                    print(f"[Contract-LoRA] microstep={step_count}/{total_microsteps} loss={loss.item():.4f}", flush=True)

                ep_losses.append(float(loss.item()))
        mean_ep_loss = float(np.mean(ep_losses))
        loss_history.append({"epoch": ep + 1, "mean_loss": round(mean_ep_loss, 4)})
        print(f"[Contract-LoRA] Epoch {ep + 1}/{epochs} mean_loss={mean_ep_loss:.4f}")

    train_time_s = time.perf_counter() - t_start
    output_lora_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(output_lora_dir))
    processor.save_pretrained(str(output_lora_dir))
    peft_model.config.use_cache = True
    peft_model.eval()
    return peft_model, processor, {
        "epochs": epochs,
        "train_samples": len(train_rows),
        "train_steps": step_count,
        "optimizer_steps": optimizer_steps,
        "train_time_s": round(train_time_s, 2),
        "loss_history": loss_history,
    }


def validate_raw_contract(payload: dict) -> None:
    """Validate model output before the permissive execution parser repairs it."""
    required = {"roi_code", "grid_cell", "target_dn", "tolerance_dn", "max_clip_fraction", "confirm_frames"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Missing contract fields")
    if payload["roi_code"] not in SEMANTIC_ROIS or not re.fullmatch(r"R[0-3]C[0-3]", str(payload["grid_cell"])):
        raise ValueError("Invalid ROI/grid")
    for k, lo, hi in [("target_dn",170,3800),("tolerance_dn",5,100),("max_clip_fraction",0,0.05)]:
        v = payload[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"Invalid {k}")
    if type(payload["confirm_frames"]) is not int or not 2 <= payload["confirm_frames"] <= 10:
        raise ValueError("Invalid confirm_frames")


def evaluate_contract_compilation(
    model,
    processor,
    test_rows: list[dict],
    ledger: dict,
    repr_mode: str = "dual_offset_grid",
    include_telemetry: bool = True,
    model_tag: str = "qwen2.5_vl_3b_lora",
) -> dict:
    """Evaluate VLM contract compilation accuracy + downstream fast-loop execution
    across P2 (raw_ratio), P3 (shared_offset), and P4 (secant_offset).
    """
    records = []
    roi_hits = 0
    roi_code_hits = 0
    grid_cell_hits = 0
    schema_valid = 0
    target_errors = []
    latencies_ms = []

    for row in test_rows:
        ap_key = row["aperture"]
        img_path = ROOT / row["images"][repr_mode]
        if include_telemetry and repr_mode == "dual_offset_grid":
            user_text = (
                f"Task Instruction: {row['instruction']}\n"
                f"Scene Telemetry: {row['telemetry_summary']}"
            )
        else:
            # Visual-only representation test (for Experiment 3 ablation)
            user_text = (
                f"Task Instruction: {row['instruction']}\n"
                f"Aperture={ap_key}, CurrentLines={row['start_lines']}. "
                "Visually locate the target structure in the 4x4 sensor grid (R0C0..R3C3), "
                "identify its ROI code (CENTER / LAMP_HOUSING / JUNCTION_BOX / PERFORATION_PLATE / DARK_WALL), "
                "and predict the calibrated target_dn and tolerance_dn."
            )

        raw_out, lat_ms = vlm_generate(model, processor, img_path, SYSTEM_CONTRACT_PROMPT, user_text, max_new_tokens=192)
        latencies_ms.append(lat_ms)

        gt = row["ground_truth_contract"]
        compiled_grid = ""
        try:
            m_json = re.search(r"\{.*\}", raw_out, flags=re.DOTALL)
            raw_dict = json.loads(m_json.group(0)) if m_json else json.loads(raw_out)
            validate_raw_contract(raw_dict)
            compiled_grid = str(raw_dict.get("grid_cell", "")).upper().strip()
            contract = parse_vlm_contract_json(
                raw_out, version=1, observation_seq=1, task_id=row["task_id"]
            )
            is_valid_json = True
            schema_valid += 1
        except Exception:
            is_valid_json = False
            contract = SemanticContract(
                version=1,
                observation_seq=1,
                task_id=row["task_id"],
                roi_code="CENTER",
                roi_box=SEMANTIC_ROIS["CENTER"],
                target_dn=300.0,
                tolerance_dn=15.0,
            )

        code_ok = (contract.roi_code == gt["roi_code"])
        grid_ok = (compiled_grid == gt["grid_cell"]) or (contract.roi_code == gt["grid_cell"])
        roi_match = bool(code_ok and grid_ok)
        if code_ok:
            roi_code_hits += 1
        if grid_ok:
            grid_cell_hits += 1
        if roi_match:
            roi_hits += 1
        t_err = abs(contract.target_dn - float(gt["target_dn"]))
        target_errors.append(t_err)

        # Evaluate closed-loop convergence against the GROUND-TRUTH task criterion
        # while executing the VLM's compiled contract on the physical lookup curve!
        # Execute the compiled ROI, never silently replace it with GT or grid alias.
        exec_roi_code = contract.roi_code if contract.roi_code in ledger[ap_key]["roi_curves"] else "CENTER"
        exec_curve = {int(k): v for k, v in ledger[ap_key]["roi_curves"][exec_roi_code].items()}
        gt_curve = {int(k): v for k, v in ledger[ap_key]["roi_curves"][gt["roi_code"]].items()}

        policy_outcomes = {}
        for pol, q_mode in (
            ("raw_ratio", "always"),
            ("shared_offset", "always"),
            ("secant_offset", "always"),
            ("shared_offset_selective", "gain_only"),
            ("secant_offset_selective", "gain_only"),
        ):
            base_pol = pol.replace("_selective", "")
            sim = simulate_dual_layer_episode(
                exec_curve,
                contract,
                start_lines=row["start_lines"],
                fast_policy=base_pol,
                quarantine_mode=q_mode,
                vlm_calls=1,
                vlm_latency_ms=lat_ms,
            )
            # Check if terminal exposure actually satisfies the user's true semantic task!
            term_lines = sim["terminal_lines"]
            avail = sorted(gt_curve.keys())
            nearest_l = min(avail, key=lambda x: abs(x - term_lines))
            # Interpolate on true task ROI curve
            idx = int(np.searchsorted(avail, term_lines))
            if term_lines in gt_curve:
                true_mean = gt_curve[term_lines]["mean_dn"]
                true_clip = gt_curve[term_lines]["clip_fraction"]
            elif 0 < idx < len(avail):
                l0, l1 = avail[idx - 1], avail[idx]
                w = (term_lines - l0) / float(l1 - l0)
                true_mean = (1.0 - w) * gt_curve[l0]["mean_dn"] + w * gt_curve[l1]["mean_dn"]
                true_clip = (1.0 - w) * gt_curve[l0]["clip_fraction"] + w * gt_curve[l1]["clip_fraction"]
            else:
                true_mean = gt_curve[nearest_l]["mean_dn"]
                true_clip = gt_curve[nearest_l]["clip_fraction"]

            task_qualified = (
                is_valid_json
                and sim["success"]
                and roi_match
                and abs(true_mean - float(gt["target_dn"])) <= float(gt["tolerance_dn"])
                and true_clip <= float(gt["max_clip_fraction"])
            )
            sim["task_qualified"] = bool(task_qualified)
            sim["true_task_terminal_dn"] = round(float(true_mean), 3)
            sim["true_task_mae_dn"] = round(abs(float(true_mean) - float(gt["target_dn"])), 3)
            policy_outcomes[pol] = sim

        records.append({
            "sample_id": row["sample_id"],
            "aperture": ap_key,
            "task_id": row["task_id"],
            "start_lines": row["start_lines"],
            "vlm_raw_output": raw_out,
            "vlm_latency_ms": round(lat_ms, 2),
            "generation_metadata": getattr(vlm_generate, "last_metadata", {}),
            "schema_valid": is_valid_json,
            "compiled_roi": contract.roi_code,
            "compiled_grid_cell": compiled_grid,
            "gt_roi": gt["roi_code"],
            "gt_grid_cell": gt["grid_cell"],
            "roi_code_match": code_ok,
            "grid_cell_match": grid_ok,
            "roi_match": roi_match,
            "compiled_target_dn": contract.target_dn,
            "gt_target_dn": gt["target_dn"],
            "target_error_dn": round(t_err, 3),
            "policy_outcomes": policy_outcomes,
        })
        print(f"[eval] {model_tag} {len(records)}/{len(test_rows)} schema={is_valid_json} roi={roi_match}", flush=True)

    n = len(test_rows)
    summary_by_policy = {}
    for pol in ("raw_ratio", "shared_offset", "secant_offset", "shared_offset_selective", "secant_offset_selective"):
        sims = [r["policy_outcomes"][pol] for r in records]
        summary_by_policy[pol] = {
            "episodes": n,
            "fast_loop_converged_rate": round(sum(s["success"] for s in sims) / n, 4),
            "semantic_task_qualified_rate": round(sum(s["task_qualified"] for s in sims) / n, 4),
            "semantic_task_qualified_count": int(sum(s["task_qualified"] for s in sims)),
            "mean_online_frames": round(float(np.mean([s["online_frames"] for s in sims])), 3),
            "total_online_frames": int(sum(s["online_frames"] for s in sims)),
            "mean_updates": round(float(np.mean([s["updates"] for s in sims])), 3),
            "mean_vlm_calls": 1.0,
            "median_vlm_latency_ms": round(float(np.median(latencies_ms)), 2),
            "median_total_confirmation_ms": round(float(np.median([s["total_confirmation_ms"] for s in sims])), 2),
            "mean_total_confirmation_ms": round(float(np.mean([s["total_confirmation_ms"] for s in sims])), 2),
            "median_true_task_mae_dn": round(float(np.median([s["true_task_mae_dn"] for s in sims])), 3),
        }

    return {
        "model_tag": model_tag,
        "repr_mode": repr_mode,
        "include_telemetry": include_telemetry,
        "num_samples": n,
        "schema_valid_rate": round(schema_valid / n, 4),
        "roi_code_accuracy": round(roi_code_hits / n, 4),
        "grid_cell_accuracy": round(grid_cell_hits / n, 4),
        "roi_accuracy": round(roi_hits / n, 4),
        "target_dn_mae": round(float(np.mean(target_errors)), 3),
        "median_vlm_latency_ms": round(float(np.median(latencies_ms)), 2),
        "p95_vlm_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "summary_by_policy": summary_by_policy,
        "records": records,
    }


def evaluate_direct_vlm_in_loop(
    model,
    processor,
    test_rows: list[dict],
    ledger: dict,
    model_tag: str = "qwen2.5_vl_7b_direct_loop",
    max_steps: int = 4,
) -> dict:
    """Evaluate Paradigm P1 (Direct VLM-in-the-Loop): VLM is queried at EVERY
    exposure update step to predict `next_lines` directly without a physical
    offset-aware fast controller.
    """
    records = []
    for row in test_rows:
        ap_key = row["aperture"]
        gt = row["ground_truth_contract"]
        gt_curve = {int(k): v for k, v in ledger[ap_key]["roi_curves"][gt["roi_code"]].items()}
        avail_lines = sorted(gt_curve.keys())

        cur_lines = int(row["start_lines"])
        step_lines_seq = []
        step_latencies = []
        step_records = []

        # Query VLM iteratively along the trajectory up to max_steps
        for _ in range(max_steps):
            nearest_img_l = min(
                [int(k) for k in ledger[ap_key]["images"].keys()],
                key=lambda x: abs(x - cur_lines),
            )
            img_path = ROOT / ledger[ap_key]["images"][str(nearest_img_l)]["dual_offset_grid"]
            # Interpolate current ROI mean
            idx = int(np.searchsorted(avail_lines, cur_lines))
            if cur_lines in gt_curve:
                cur_mean = gt_curve[cur_lines]["mean_dn"]
            elif 0 < idx < len(avail_lines):
                l0, l1 = avail_lines[idx - 1], avail_lines[idx]
                w = (cur_lines - l0) / float(l1 - l0)
                cur_mean = (1.0 - w) * gt_curve[l0]["mean_dn"] + w * gt_curve[l1]["mean_dn"]
            else:
                cur_mean = gt_curve[avail_lines[0] if idx == 0 else avail_lines[-1]]["mean_dn"]

            if abs(cur_mean - float(gt["target_dn"])) <= float(gt["tolerance_dn"]):
                break

            prompt = (
                f"Task Instruction: {row['instruction']}\n"
                f"Aperture={ap_key}, CurrentLines={cur_lines}, CurrentROIMean={cur_mean:.1f} DN. "
                'Predict {"next_lines": <int 8..512>}.'
            )
            raw_out, lat_ms = vlm_generate(model, processor, img_path, SYSTEM_DIRECT_LOOP_PROMPT, prompt, max_new_tokens=24)
            step_latencies.append(lat_ms)
            try:
                payload = json.loads(raw_out)
                nxt = payload["next_lines"]
                if type(nxt) is not int or not 8 <= nxt <= 512:
                    raise ValueError("Invalid direct exposure")
                output_valid = True
            except (ValueError, TypeError, KeyError):
                nxt = cur_lines
                output_valid = False
            step_records.append({"current_lines":cur_lines,"image_lines":nearest_img_l,
                "raw_output":raw_out,"schema_valid":output_valid,"next_lines":nxt,
                "generation_ms":lat_ms,"generation_metadata":getattr(vlm_generate,"last_metadata",{})})
            step_lines_seq.append(nxt)
            if nxt == cur_lines:
                break
            cur_lines = nxt

        mean_lat = float(np.mean(step_latencies)) if step_latencies else 0.0
        contract = SemanticContract(
            version=1,
            observation_seq=1,
            task_id=row["task_id"],
            roi_code=gt["roi_code"],
            roi_box=SEMANTIC_ROIS[gt["roi_code"]],
            target_dn=float(gt["target_dn"]),
            tolerance_dn=float(gt["tolerance_dn"]),
        )
        sim = simulate_dual_layer_episode(
            gt_curve,
            contract,
            start_lines=row["start_lines"],
            fast_policy="direct_vlm",
            quarantine_mode="always",
            vlm_latency_ms=mean_lat,
            direct_vlm_step_sequence=step_lines_seq,
        )
        sim["task_qualified"] = bool(sim["success"])
        sim["sample_id"] = row["sample_id"]
        sim["task_id"] = row["task_id"]
        sim["actual_vlm_step_sequence"] = step_lines_seq
        sim["actual_vlm_step_latencies_ms"] = step_latencies
        sim["vlm_step_records"] = step_records
        sim["vlm_calls"] = len(step_latencies)
        sim["vlm_latency_ms"] = round(sum(step_latencies), 2)
        sim["total_confirmation_ms"] = round(sim["acquisition_ms"] + sum(step_latencies), 2)
        sim["p1_limitations"] = "Oracle task ROI telemetry; nearest measured image may differ from interpolated exposure."
        records.append(sim)
        print(f"[direct] {model_tag} {len(records)}/{len(test_rows)} calls={len(step_latencies)} success={sim['success']}", flush=True)

    n = len(records)
    return {
        "model_tag": model_tag,
        "episodes": n,
        "semantic_task_qualified_rate": round(sum(r["task_qualified"] for r in records) / n, 4),
        "semantic_task_qualified_count": int(sum(r["task_qualified"] for r in records)),
        "mean_online_frames": round(float(np.mean([r["online_frames"] for r in records])), 3),
        "total_online_frames": int(sum(r["online_frames"] for r in records)),
        "mean_updates": round(float(np.mean([r["updates"] for r in records])), 3),
        "mean_vlm_calls": round(float(np.mean([r["vlm_calls"] for r in records])), 3),
        "median_total_confirmation_ms": round(float(np.median([r["total_confirmation_ms"] for r in records])), 2),
        "mean_total_confirmation_ms": round(float(np.mean([r["total_confirmation_ms"] for r in records])), 2),
        "records": records,
    }


def evaluate_fixed_center_baseline(test_rows: list[dict], ledger: dict) -> dict:
    """Evaluate traditional Fixed-Center Heuristic AE (always meters CENTER at 300 DN)."""
    records = []
    for row in test_rows:
        ap_key = row["aperture"]
        gt = row["ground_truth_contract"]
        center_curve = {int(k): v for k, v in ledger[ap_key]["roi_curves"]["CENTER"].items()}
        gt_curve = {int(k): v for k, v in ledger[ap_key]["roi_curves"][gt["roi_code"]].items()}
        avail = sorted(gt_curve.keys())

        fixed_contract = SemanticContract(
            version=1,
            observation_seq=1,
            task_id=row["task_id"],
            roi_code="CENTER",
            roi_box=SEMANTIC_ROIS["CENTER"],
            target_dn=300.0,
            tolerance_dn=15.0,
        )
        sim = simulate_dual_layer_episode(
            center_curve,
            fixed_contract,
            start_lines=row["start_lines"],
            fast_policy="shared_offset",
            quarantine_mode="always",
            vlm_calls=0,
            vlm_latency_ms=0.0,
        )
        term_lines = sim["terminal_lines"]
        idx = int(np.searchsorted(avail, term_lines))
        if term_lines in gt_curve:
            true_mean = gt_curve[term_lines]["mean_dn"]
            true_clip = gt_curve[term_lines]["clip_fraction"]
        elif 0 < idx < len(avail):
            l0, l1 = avail[idx - 1], avail[idx]
            w = (term_lines - l0) / float(l1 - l0)
            true_mean = (1.0 - w) * gt_curve[l0]["mean_dn"] + w * gt_curve[l1]["mean_dn"]
            true_clip = (1.0 - w) * gt_curve[l0]["clip_fraction"] + w * gt_curve[l1]["clip_fraction"]
        else:
            nearest_l = min(avail, key=lambda x: abs(x - term_lines))
            true_mean = gt_curve[nearest_l]["mean_dn"]
            true_clip = gt_curve[nearest_l]["clip_fraction"]

        qual = (
            abs(true_mean - float(gt["target_dn"])) <= float(gt["tolerance_dn"])
            and true_clip <= float(gt["max_clip_fraction"])
        )
        sim["task_qualified"] = bool(qual)
        sim["true_task_mae_dn"] = round(abs(float(true_mean) - float(gt["target_dn"])), 3)
        records.append(sim)

    n = len(records)
    return {
        "model_tag": "fixed_center_300dn_baseline",
        "episodes": n,
        "semantic_task_qualified_rate": round(sum(r["task_qualified"] for r in records) / n, 4),
        "semantic_task_qualified_count": int(sum(r["task_qualified"] for r in records)),
        "mean_online_frames": round(float(np.mean([r["online_frames"] for r in records])), 3),
        "total_online_frames": int(sum(r["online_frames"] for r in records)),
        "mean_updates": round(float(np.mean([r["updates"] for r in records])), 3),
        "median_total_confirmation_ms": round(float(np.median([r["total_confirmation_ms"] for r in records])), 2),
        "median_true_task_mae_dn": round(float(np.median([r["true_task_mae_dn"] for r in records])), 3),
    }


def main():
    print("Starting RTX PRO 6000 Dual-Layer VLM + Physical AE Experiment Suite...")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU: {gpu_name} ({vram_gib:.1f} GiB VRAM)")

    ledger = json.loads((ROOT / "sCMOS_physical_ledger.json").read_text(encoding="utf-8"))
    train_rows = [json.loads(line) for line in (ROOT / "train_contracts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    test_rows = [json.loads(line) for line in (ROOT / "test_contracts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    # 1. Fixed-Center Baseline
    fixed_res = evaluate_fixed_center_baseline(test_rows, ledger)
    print("Fixed-Center Baseline qualified rate:", fixed_res["semantic_task_qualified_rate"])

    # 2. Download & Evaluate Qwen2.5-VL-7B-Instruct (Zero-Shot Teacher & Direct Loop)
    dir_7b = download_qwen_model("Qwen/Qwen2.5-VL-7B-Instruct")
    model_7b, proc_7b = load_vlm(dir_7b)

    print("Running Experiment 1A: Qwen2.5-VL-7B Zero-Shot Contract Compiler...")
    res_7b_zs = evaluate_contract_compilation(
        model_7b, proc_7b, test_rows, ledger,
        repr_mode="dual_offset_grid", include_telemetry=True,
        model_tag="Qwen2.5-VL-7B-ZeroShot",
    )
    print("Running Experiment 1B: Qwen2.5-VL-7B Direct VLM-in-the-Loop (P1)...")
    res_7b_direct = evaluate_direct_vlm_in_loop(
        model_7b, proc_7b, test_rows, ledger,
        model_tag="Qwen2.5-VL-7B-DirectLoop",
    )
    del model_7b
    torch.cuda.empty_cache()

    # 3. Download & Evaluate Qwen2.5-VL-3B-Instruct (Zero-Shot + Representation Ablation + BF16 LoRA)
    dir_3b = download_qwen_model("Qwen/Qwen2.5-VL-3B-Instruct")
    model_3b, proc_3b = load_vlm(dir_3b)

    print("Running Experiment 1C: Qwen2.5-VL-3B Zero-Shot Contract Compiler...")
    res_3b_zs = evaluate_contract_compilation(
        model_3b, proc_3b, test_rows, ledger,
        repr_mode="dual_offset_grid", include_telemetry=True,
        model_tag="Qwen2.5-VL-3B-ZeroShot",
    )
    print("Running Experiment 1D: Qwen2.5-VL-3B Direct VLM-in-the-Loop (P1)...")
    res_3b_direct = evaluate_direct_vlm_in_loop(
        model_3b, proc_3b, test_rows, ledger,
        model_tag="Qwen2.5-VL-3B-DirectLoop",
    )

    # Experiment 3A: Visual Representation Ablation (Linear vs MinMax vs Dual-Offset-Grid)
    print("Running Experiment 3A: Visual Representation Ablation on 12-bit RAW...")
    res_repr_linear = evaluate_contract_compilation(
        model_3b, proc_3b, test_rows, ledger,
        repr_mode="linear", include_telemetry=False,
        model_tag="Qwen2.5-VL-3B-Repr-Linear",
    )
    res_repr_minmax = evaluate_contract_compilation(
        model_3b, proc_3b, test_rows, ledger,
        repr_mode="minmax", include_telemetry=False,
        model_tag="Qwen2.5-VL-3B-Repr-MinMax",
    )
    res_repr_dual_visual = evaluate_contract_compilation(
        model_3b, proc_3b, test_rows, ledger,
        repr_mode="dual_offset_grid", include_telemetry=False,
        model_tag="Qwen2.5-VL-3B-Repr-DualOffsetGrid",
    )
    del model_3b
    torch.cuda.empty_cache()

    # Experiment 3B: Fine-Tune Qwen2.5-VL-3B with BF16 Contract-LoRA on RTX PRO 6000
    print("Running Experiment 3B: BF16 Contract-LoRA Fine-Tuning on RTX PRO 6000...")
    lora_out_dir = ROOT / "checkpoints" / "qwen25_vl_3b_contract_lora"
    lora_model, lora_proc, lora_train_stats = train_contract_lora(
        dir_3b, train_rows, lora_out_dir, epochs=2, lr=2e-4
    )

    print("Evaluating Qwen2.5-VL-3B + Contract-LoRA (Dual-Offset-Grid + Telemetry)...")
    res_3b_lora = evaluate_contract_compilation(
        lora_model, lora_proc, test_rows, ledger,
        repr_mode="dual_offset_grid", include_telemetry=True,
        model_tag="Qwen2.5-VL-3B-ContractLoRA",
    )
    print("Evaluating Qwen2.5-VL-3B + Contract-LoRA (Visual-Only Dual-Offset-Grid)...")
    res_3b_lora_visual = evaluate_contract_compilation(
        lora_model, lora_proc, test_rows, ledger,
        repr_mode="dual_offset_grid", include_telemetry=False,
        model_tag="Qwen2.5-VL-3B-ContractLoRA-VisualOnly",
    )

    # Per-task breakdown for Experiment 2 (sCMOS-ContractBench)
    per_task_breakdown = {}
    for task_id in sorted({r["task_id"] for r in test_rows}):
        sub_recs = [r for r in res_3b_lora["records"] if r["task_id"] == task_id]
        n_t = len(sub_recs)
        per_task_breakdown[task_id] = {
            "episodes": n_t,
            "roi_accuracy": round(sum(r["roi_match"] for r in sub_recs) / n_t, 4),
            "target_dn_mae": round(float(np.mean([r["target_error_dn"] for r in sub_recs])), 3),
            "shared_offset_qualified_rate": round(
                sum(r["policy_outcomes"]["shared_offset"]["task_qualified"] for r in sub_recs) / n_t, 4
            ),
            "secant_offset_qualified_rate": round(
                sum(r["policy_outcomes"]["secant_offset"]["task_qualified"] for r in sub_recs) / n_t, 4
            ),
            "secant_offset_mean_frames": round(
                float(np.mean([r["policy_outcomes"]["secant_offset"]["online_frames"] for r in sub_recs])), 3
            ),
            "secant_selective_mean_frames": round(
                float(np.mean([r["policy_outcomes"]["secant_offset_selective"]["online_frames"] for r in sub_recs])), 3
            ),
        }

    final_output = {
        "hardware": {
            "gpu_name": gpu_name,
            "vram_gib": round(vram_gib, 2),
            "torch_version": torch.__version__,
        },
        "benchmark_meta": {
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "apertures": list(ledger.keys()),
        },
        "lora_training": lora_train_stats,
        "experiment_1_paradigm_and_scale": {
            "fixed_center_baseline": fixed_res,
            "qwen25_vl_7b_direct_loop": {k: v for k, v in res_7b_direct.items() if k != "records"},
            "qwen25_vl_3b_direct_loop": {k: v for k, v in res_3b_direct.items() if k != "records"},
            "qwen25_vl_7b_zeroshot_compiler": {k: v for k, v in res_7b_zs.items() if k != "records"},
            "qwen25_vl_3b_zeroshot_compiler": {k: v for k, v in res_3b_zs.items() if k != "records"},
            "qwen25_vl_3b_contract_lora_compiler": {k: v for k, v in res_3b_lora.items() if k != "records"},
        },
        "experiment_2_scmos_contract_bench_by_task": per_task_breakdown,
        "experiment_3_visual_representation_and_lora_ablation": {
            "linear_8bit_visual_only": {k: v for k, v in res_repr_linear.items() if k != "records"},
            "minmax_stretch_visual_only": {k: v for k, v in res_repr_minmax.items() if k != "records"},
            "dual_offset_grid_visual_only_zeroshot": {k: v for k, v in res_repr_dual_visual.items() if k != "records"},
            "dual_offset_grid_visual_only_lora": {k: v for k, v in res_3b_lora_visual.items() if k != "records"},
            "dual_offset_grid_plus_telemetry_zeroshot": {k: v for k, v in res_3b_zs.items() if k != "records"},
            "dual_offset_grid_plus_telemetry_lora": {k: v for k, v in res_3b_lora.items() if k != "records"},
        },
        "sample_records_3b_lora": res_3b_lora["records"][:24],
    }

    out_path = ROOT / "T_rtx6000_results.json"
    out_path.write_text(json.dumps(final_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Saved complete RTX PRO 6000 results to {out_path}")


if __name__ == "__main__":
    main()
