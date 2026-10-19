"""Slow VLM Semantic Contract Compiler + Fast Offset/Secant-Aware Physical Executor.

Implements the dual-layer closed-loop architecture without modifying any frozen
core modules (`hardware.py`, `ae_published.py`, `ae_brightness.py`,
`ae_published_experiments.py`, `types.py`):

- Upper Slow Layer (`VLMContractCompiler`):
  Compiles natural-language task intent + 12-bit RAW visual representation into
  a versioned physical control contract:
      C_v = (version, observation_seq, roi_code, roi_box, target_dn,
             tolerance_dn, max_clip_fraction, confirm_frames)
  Never outputs raw exposure line registers directly in the fast loop.

- Lower Fast Layer (`FastPhysicalClosedLoop`):
  Executes bounded exposure updates at frame rate using `SharedOffsetAE`,
  `SecantOffsetAE` (cold-start online 2-point secant offset), or `RatioAE`,
  with confirmed-output cost accounting:
      T_total = M_vlm * tau_vlm + N_online * tau_frame + U * tau_cmd
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
import numpy as np

from .ae_brightness import BrightnessAE, ResponseModel
from .secant_quarantine import SecantOffsetAE, ShortExposureAction


# Standard 4x4 spatial grid across the 2048x2176 G6504 sCMOS frame
# plus semantic ROI aliases grounded in the physical target layout.
GRID_ROIS: dict[str, tuple[int, int, int, int]] = {}
for _r in range(4):
    for _c in range(4):
        GRID_ROIS[f"R{_r}C{_c}"] = (
            _r * 512,
            (_r + 1) * 512,
            _c * 544,
            (_c + 1) * 544,
        )

SEMANTIC_ROIS: dict[str, tuple[int, int, int, int]] = {
    **GRID_ROIS,
    # Canonical 512x512 center metering ROI used in the 120-episode cohort
    "CENTER": (768, 1280, 832, 1344),
    # Top-left bright lamp housing (rows 0..512, cols 450..994)
    "LAMP_HOUSING": (0, 512, 450, 994),
    # Shadowed junction box with printed model label J208B1/N1-B
    "JUNCTION_BOX": (740, 1180, 1460, 1760),
    # Right-side perforated hole array for sub-pixel localization
    "PERFORATION_PLATE": (420, 920, 1460, 1720),
    # Bottom-left low-reflectance dark wall background
    "DARK_WALL": (1400, 1912, 120, 664),
}


@dataclass(frozen=True)
class SemanticContract:
    """Versioned physical AE contract compiled by the slow VLM layer."""
    version: int
    observation_seq: int
    task_id: str
    roi_code: str
    roi_box: tuple[int, int, int, int]
    target_dn: float
    tolerance_dn: float = 15.0
    max_clip_fraction: float = 0.0
    confirm_frames: int = 6
    rationale: str = ""

    def __post_init__(self):
        if self.version < 1:
            raise ValueError("Contract version must be >= 1")
        if self.observation_seq < 0:
            raise ValueError("Observation sequence must be >= 0")
        r0, r1, c0, c1 = self.roi_box
        if not (0 <= r0 < r1 <= 2048 and 0 <= c0 < c1 <= 2176):
            raise ValueError(f"ROI box {self.roi_box} outside 2048x2176 sensor")
        if not (170.0 <= self.target_dn <= 3800.0):
            raise ValueError(f"target_dn {self.target_dn} outside physical envelope [170, 3800]")
        if not (5.0 <= self.tolerance_dn <= 100.0):
            raise ValueError(f"tolerance_dn {self.tolerance_dn} outside [5, 100]")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["roi_box"] = list(self.roi_box)
        return d


def parse_vlm_contract_json(
    raw_text: str,
    version: int = 1,
    observation_seq: int = 1,
    task_id: str = "custom",
    offset_floor_dn: float = 166.5,
    strict_free_bbox: bool = False,
    orig_width: int = 2176,
    orig_height: int = 2048,
    resized_width: int | None = None,
    resized_height: int | None = None,
    coord_space: str = "qwen25_resized_pixel",
) -> SemanticContract:
    """Parse VLM JSON output into a validated SemanticContract.

    Supports both:
      - Free-form bounding boxes (`bbox_2d` or `bbox` in `[x1, y1, x2, y2]`), mapped
        to sensor `(r0, r1, c0, c1)` via `convert_pred_bbox_to_pixel_xyxy` and
        `xyxy_pixel_to_sensor_roi_box`. When `strict_free_bbox=True`, any output
        lacking a valid 4D free bounding box raises `ValueError` instead of falling
        back to a fixed alias or `'CENTER'`.
      - Legacy fixed ROI aliases (`roi_code` in `SEMANTIC_ROIS`) when `strict_free_bbox=False`.
    """
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    payload: dict = {}
    if match:
        try:
            loaded = json.loads(match.group(0))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}

    has_bbox_key = any(k in payload for k in ("bbox_2d", "bbox", "box")) or bool(
        re.search(r'"(?:bbox_2d|bbox|box)"\s*:\s*\[', raw_text)
    )

    if has_bbox_key:
        b_res = parse_vlm_free_bbox(
            raw_text,
            orig_width=orig_width,
            orig_height=orig_height,
            resized_width=resized_width,
            resized_height=resized_height,
            default_coord_space=coord_space,
        )
        if not b_res["valid"] or b_res["bbox_xyxy"] is None:
            raise ValueError(f"Invalid free bbox in contract: {b_res.get('error')}")
        roi_box = xyxy_pixel_to_sensor_roi_box(
            b_res["bbox_xyxy"], orig_width=orig_width, orig_height=orig_height
        )
        roi_code = str(payload.get("roi_code", "FREE_BBOX")).upper().strip() or "FREE_BBOX"
    elif strict_free_bbox:
        raise ValueError("strict_free_bbox=True requires a 4D free bounding box ('bbox_2d' or 'bbox')")
    else:
        if not payload:
            loaded = json.loads(raw_text)
            if isinstance(loaded, dict):
                payload = loaded
        roi_code = str(payload.get("roi_code", "CENTER")).upper().strip()
        if roi_code in SEMANTIC_ROIS:
            roi_box = SEMANTIC_ROIS[roi_code]
        elif "roi_box" in payload and len(payload["roi_box"]) == 4:
            r0, r1, c0, c1 = [int(v) for v in payload["roi_box"]]
            r0 = max(0, min(r0, 2048 - 64))
            r1 = max(r0 + 64, min(r1, 2048))
            c0 = max(0, min(c0, 2176 - 64))
            c1 = max(c0 + 64, min(c1, 2176))
            roi_box = (r0, r1, c0, c1)
        else:
            roi_code = "CENTER"
            roi_box = SEMANTIC_ROIS["CENTER"]

    def _extract_num(key: str, default_val: float) -> float:
        if key in payload:
            return float(payload[key])
        m_num = re.search(rf'"{key}"\s*:\s*(-?\d+(?:\.\d+)?)', raw_text)
        return float(m_num.group(1)) if m_num else default_val

    raw_target = _extract_num("target_dn", 300.0)
    target_dn = float(np.clip(raw_target, offset_floor_dn + 20.0, 3600.0))
    raw_tol = _extract_num("tolerance_dn", max(15.0, round(0.05 * target_dn, 1)))
    tolerance_dn = float(np.clip(raw_tol, 10.0, 80.0))
    max_clip = float(np.clip(_extract_num("max_clip_fraction", 0.0), 0.0, 0.05))
    confirm = int(np.clip(int(_extract_num("confirm_frames", 6.0)), 2, 10))
    rationale = str(payload.get("rationale", ""))[:200]

    return SemanticContract(
        version=version,
        observation_seq=observation_seq,
        task_id=task_id,
        roi_code=roi_code,
        roi_box=roi_box,
        target_dn=target_dn,
        tolerance_dn=tolerance_dn,
        max_clip_fraction=max_clip,
        confirm_frames=confirm,
        rationale=rationale,
    )


def parse_vlm_direct_lines(raw_text: str, current_lines: int) -> int:
    """Parse direct VLM exposure-line prediction for Paradigm P1 (Direct VLM-in-the-Loop)."""
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if "next_lines" in payload:
                val = int(round(float(payload["next_lines"])))
                return int(np.clip(val, 8, 512))
        except Exception:
            pass
    nums = re.findall(r"\b(\d{1,3})\b", raw_text)
    for n_str in nums:
        val = int(n_str)
        if 8 <= val <= 512:
            return val
    return int(np.clip(current_lines, 8, 512))


def compute_roi_telemetry(image_u16: np.ndarray, roi_box: tuple[int, int, int, int], offset_dn: float = 166.473) -> dict:
    """Compute 12-bit RAW statistics inside a given ROI box."""
    r0, r1, c0, c1 = roi_box
    patch = image_u16[r0:r1, c0:c1].astype(np.float64)
    mean_dn = float(patch.mean())
    std_dn = float(patch.std())
    p99_dn = float(np.percentile(patch, 99.0))
    clip_fraction = float(np.mean(patch >= 4095.0))
    net_signal = max(mean_dn - offset_dn, 0.0)
    snr = float(net_signal / max(std_dn, 1.0))
    # Horizontal & vertical gradient energy for edge/detail localization quality
    gx = np.diff(patch, axis=1)
    gy = np.diff(patch, axis=0)
    grad_rms = float(np.sqrt(0.5 * (np.mean(gx ** 2) + np.mean(gy ** 2))))
    grad_snr = float(grad_rms / max(std_dn, 1.0))
    return {
        "mean_dn": round(mean_dn, 3),
        "std_dn": round(std_dn, 3),
        "p99_dn": round(p99_dn, 3),
        "clip_fraction": round(clip_fraction, 6),
        "net_signal_dn": round(net_signal, 3),
        "snr": round(snr, 3),
        "grad_snr": round(grad_snr, 3),
    }


def render_visual_representation(
    image_u16: np.ndarray,
    mode: str = "dual_offset_grid",
    estimated_offset_dn: float = 166.5,
    downsample: int = 4,
) -> np.ndarray:
    """Render a 12-bit RAW (2048x2176) frame into an 8-bit RGB array (H, W, 3)
    for VLM consumption under three representation modes (Experiment 3):
      - 'linear': standard 8-bit linear full-scale y / 4095
      - 'minmax': auto-stretched min-max view (destroys absolute brightness/clip cues)
      - 'dual_offset_grid': side-by-side [Linear + Saturation Mask | Offset-Subtracted Log + 4x4 Grid]
    """
    img = image_u16[::downsample, ::downsample].astype(np.float64)
    h, w = img.shape

    # 1. Linear full-scale [0, 4095] -> [0, 255]
    lin_norm = np.clip(img / 4095.0, 0.0, 1.0)
    lin_u8 = np.round(lin_norm * 255.0).astype(np.uint8)
    lin_rgb = np.stack([lin_u8, lin_u8, lin_u8], axis=-1)

    if mode == "linear":
        return lin_rgb

    # 2. Min-max auto-stretched
    lo, hi = float(np.percentile(img, 0.5)), float(np.percentile(img, 99.5))
    mm_norm = np.clip((img - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    mm_u8 = np.round(mm_norm * 255.0).astype(np.uint8)
    mm_rgb = np.stack([mm_u8, mm_u8, mm_u8], axis=-1)

    if mode == "minmax":
        return mm_rgb

    if mode != "dual_offset_grid":
        raise ValueError(f"Unknown visual representation mode: {mode}")

    # 3. Offset-Aware Dual-View:
    # Left panel: Linear full-scale with red highlight on saturated pixels (>= 4095 DN)
    left_rgb = lin_rgb.copy()
    sat_mask = img >= 4090.0
    left_rgb[sat_mask] = np.array([255, 40, 40], dtype=np.uint8)

    # Right panel: Offset-subtracted log enhancement log(1 + max(y - b_hat, 0))
    net = np.maximum(img - estimated_offset_dn, 0.0)
    log_val = np.log1p(net)
    log_hi = max(float(np.percentile(log_val, 99.5)), 1.0)
    log_u8 = np.round(np.clip(log_val / log_hi, 0.0, 1.0) * 255.0).astype(np.uint8)
    right_rgb = np.stack([log_u8, log_u8, log_u8], axis=-1)

    # Draw 4x4 grid lines on the right panel so VLM can ground R0C0..R3C3
    for r_idx in range(1, 4):
        ry = (r_idx * h) // 4
        right_rgb[max(0, ry - 1):min(h, ry + 1), :] = np.array([60, 210, 255], dtype=np.uint8)
    for c_idx in range(1, 4):
        cx = (c_idx * w) // 4
        right_rgb[:, max(0, cx - 1):min(w, cx + 1)] = np.array([60, 210, 255], dtype=np.uint8)

    # Concatenate Left | Separator | Right
    sep = np.full((h, 6, 3), 240, dtype=np.uint8)
    return np.concatenate([left_rgb, sep, right_rgb], axis=1)


def simulate_dual_layer_episode(
    roi_curve_lookup: dict[int, dict],
    contract: SemanticContract,
    start_lines: int,
    fast_policy: str = "secant_offset",
    quarantine_mode: str = "gain_only",
    precalibrated_offset_dn: float = 166.473,
    vlm_calls: int = 1,
    vlm_latency_ms: float = 280.0,
    frame_cycle_ms: float = 240.0,
    cmd_overhead_ms: float = 12.0,
    max_updates: int = 10,
    max_online_frames: int = 28,
    direct_vlm_step_sequence: list[int] | None = None,
) -> dict:
    """Run a complete closed-loop episode under either:
      - Fast physical execution ('secant_offset', 'shared_offset', 'raw_ratio')
        guided by a 1-shot VLM compiled SemanticContract, OR
      - Direct VLM-in-the-Loop ('direct_vlm'), where VLM predicts exposure lines
        at every update step (accumulating M_vlm = U + 1 inference latencies).
    """
    avail_lines = sorted(int(k) for k in roi_curve_lookup.keys())

    def query_roi(lines: int) -> dict:
        lines_c = int(np.clip(lines, 8, 512))
        if lines_c in roi_curve_lookup:
            return roi_curve_lookup[lines_c]
        # Interpolate strictly between measured points on the physical curve
        idx = int(np.searchsorted(avail_lines, lines_c))
        if idx == 0:
            return roi_curve_lookup[avail_lines[0]]
        if idx >= len(avail_lines):
            return roi_curve_lookup[avail_lines[-1]]
        l0, l1 = avail_lines[idx - 1], avail_lines[idx]
        w = (lines_c - l0) / float(l1 - l0)
        d0, d1 = roi_curve_lookup[l0], roi_curve_lookup[l1]
        out = {}
        for k in ("mean_dn", "std_dn", "p99_dn", "clip_fraction", "net_signal_dn", "snr", "grad_snr"):
            out[k] = float((1.0 - w) * d0[k] + w * d1[k])
        return out

    ae_cfg = {
        "target_dn": float(contract.target_dn),
        "tolerance_dn": float(contract.tolerance_dn),
        "max_roi_clip_fraction": float(contract.max_clip_fraction),
        "ratio_alpha": 2.0,
        "max_step_factor": 6.0,
        "min_lines": 8,
        "max_lines": 512,
    }
    dummy_model = ResponseModel(
        lines=[8, 512],
        mean_dn=[precalibrated_offset_dn + 16.0, precalibrated_offset_dn + 1024.0],
        offset_dn=float(precalibrated_offset_dn),
    )

    if fast_policy == "shared_offset":
        ctrl = BrightnessAE("offset_ratio", dummy_model, ae_cfg)
    elif fast_policy == "secant_offset":
        ctrl = SecantOffsetAE(ae_cfg)
    elif fast_policy == "raw_ratio":
        ctrl = BrightnessAE("ratio", dummy_model, ae_cfg)
    elif fast_policy == "direct_vlm":
        ctrl = None
    else:
        raise ValueError(f"Unknown fast_policy: {fast_policy}")

    current_lines = int(np.clip(start_lines, 8, 512))
    updates = 0
    consecutive = 0
    online_frames = 0
    trajectory = []
    direct_step_idx = 0

    while online_frames < max_online_frames:
        meas = query_roi(current_lines)
        online_frames += 1
        mean_dn = float(meas["mean_dn"])
        clip_frac = float(meas["clip_fraction"])
        qual = (
            abs(mean_dn - contract.target_dn) <= contract.tolerance_dn
            and clip_frac <= contract.max_clip_fraction
        )
        consecutive = consecutive + 1 if qual else 0
        trajectory.append({
            "online_index": online_frames,
            "role": "evaluation",
            "lines": current_lines,
            "mean_dn": round(mean_dn, 3),
            "clip_fraction": round(clip_frac, 6),
            "snr": round(float(meas["snr"]), 3),
            "grad_snr": round(float(meas["grad_snr"]), 3),
            "qualified": qual,
            "consecutive_qualified": consecutive,
        })
        if consecutive >= contract.confirm_frames:
            break

        # Confirmation frames hold exposure and never invoke a controller.
        if qual:
            continue

        if updates >= max_updates:
            break

        need_frames = 2 if quarantine_mode == "always" else 1
        if online_frames + need_frames > max_online_frames:
            break

        if fast_policy == "direct_vlm":
            if direct_vlm_step_sequence and direct_step_idx < len(direct_vlm_step_sequence):
                next_lines = int(np.clip(direct_vlm_step_sequence[direct_step_idx], 8, 512))
                direct_step_idx += 1
            else:
                # No fabricated ratio-controller continuation after VLM outputs end.
                break
        else:
            next_lines = int(ctrl.choose(current_lines, {"mean_dn": mean_dn, "clip_fraction": clip_frac}))
            if next_lines == current_lines:
                continue

        if next_lines == current_lines:
            continue

        current_lines = next_lines
        updates += 1

        if quarantine_mode == "always":
            q_meas = query_roi(current_lines)
            online_frames += 1
            consecutive = 0
            trajectory.append({
                "online_index": online_frames,
                "role": "transition",
                "lines": current_lines,
                "mean_dn": round(float(q_meas["mean_dn"]), 3),
                "clip_fraction": round(float(q_meas["clip_fraction"]), 6),
                "snr": round(float(q_meas["snr"]), 3),
                "grad_snr": round(float(q_meas["grad_snr"]), 3),
                "qualified": False,
                "consecutive_qualified": 0,
            })

    success = consecutive >= contract.confirm_frames
    actual_vlm_calls = len(direct_vlm_step_sequence or []) if fast_policy == "direct_vlm" else vlm_calls
    vlm_total_ms = actual_vlm_calls * vlm_latency_ms
    acq_total_ms = online_frames * frame_cycle_ms + updates * cmd_overhead_ms
    total_elapsed_ms = vlm_total_ms + acq_total_ms

    eval_tail = [f for f in trajectory if f["role"] == "evaluation"][-contract.confirm_frames:]
    tail_mae = (
        float(np.mean([abs(f["mean_dn"] - contract.target_dn) for f in eval_tail]))
        if success and len(eval_tail) == contract.confirm_frames
        else None
    )

    return {
        "success": bool(success),
        "fast_policy": fast_policy,
        "quarantine_mode": quarantine_mode,
        "start_lines": int(start_lines),
        "terminal_lines": int(current_lines),
        "online_frames": int(online_frames),
        "updates": int(updates),
        "vlm_calls": int(actual_vlm_calls),
        "vlm_latency_ms": round(float(vlm_total_ms), 2),
        "acquisition_ms": round(float(acq_total_ms), 2),
        "total_confirmation_ms": round(float(total_elapsed_ms), 2),
        "tail_mae_dn": round(tail_mae, 3) if tail_mae is not None else None,
        "terminal_snr": trajectory[-1]["snr"] if trajectory else 0.0,
        "terminal_grad_snr": trajectory[-1]["grad_snr"] if trajectory else 0.0,
        "trajectory": trajectory,
    }


# =============================================================================
# Free-Form Bounding Box Grounding, Coordinate Conversion & REC Scoring
# =============================================================================

def coco_xywh_to_xyxy(bbox_xywh: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    """Convert COCO / Ref-L4 [x, y, w, h] pixel bbox to [x1, y1, x2, y2] pixel coordinates."""
    if len(bbox_xywh) != 4:
        raise ValueError(f"Expected 4-element [x, y, w, h], got {bbox_xywh}")
    x, y, w, h = [float(v) for v in bbox_xywh]
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"Non-positive width/height in [x, y, w, h]: {bbox_xywh}")
    return (x, y, x + w, y + h)


def convert_pred_bbox_to_pixel_xyxy(
    raw_bbox: list[float] | tuple[float, ...],
    coord_space: str,
    orig_width: int,
    orig_height: int,
    resized_width: int | None = None,
    resized_height: int | None = None,
) -> tuple[float, float, float, float]:
    """Convert a predicted [x1, y1, x2, y2] bbox from model coordinate space into
    original image pixel coordinates [x1, y1, x2, y2].

    Supported `coord_space` modes:
      - 'qwen25_resized_pixel': coordinates relative to Qwen2.5-VL processor's
        resized input dimensions (`resized_width = grid_w * 14`, `resized_height = grid_h * 14`).
      - 'norm1000': coordinates normalized to [0, 1000] x [0, 1000].
      - 'norm1': coordinates normalized to [0, 1] x [0, 1].
      - 'orig_pixel': already in original image pixel coordinates [0..orig_width, 0..orig_height].
    """
    if len(raw_bbox) != 4:
        raise ValueError(f"Expected 4-element bbox, got {raw_bbox}")
    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    w_orig = float(orig_width)
    h_orig = float(orig_height)

    if coord_space == "qwen25_resized_pixel":
        rw = float(resized_width if resized_width is not None else orig_width)
        rh = float(resized_height if resized_height is not None else orig_height)
        px1 = x1 * (w_orig / max(rw, 1.0))
        py1 = y1 * (h_orig / max(rh, 1.0))
        px2 = x2 * (w_orig / max(rw, 1.0))
        py2 = y2 * (h_orig / max(rh, 1.0))
    elif coord_space == "norm1000":
        px1 = (x1 / 1000.0) * w_orig
        py1 = (y1 / 1000.0) * h_orig
        px2 = (x2 / 1000.0) * w_orig
        py2 = (y2 / 1000.0) * h_orig
    elif coord_space == "norm1":
        px1 = x1 * w_orig
        py1 = y1 * h_orig
        px2 = x2 * w_orig
        py2 = y2 * h_orig
    elif coord_space == "orig_pixel":
        px1, py1, px2, py2 = x1, y1, x2, y2
    else:
        raise ValueError(f"Unknown coord_space: {coord_space}")

    px1 = float(np.clip(px1, 0.0, w_orig))
    py1 = float(np.clip(py1, 0.0, h_orig))
    px2 = float(np.clip(px2, 0.0, w_orig))
    py2 = float(np.clip(py2, 0.0, h_orig))
    if px2 <= px1 or py2 <= py1:
        raise ValueError(f"Degenerate converted bbox [{px1}, {py1}, {px2}, {py2}]")
    return (px1, py1, px2, py2)


def xyxy_pixel_to_sensor_roi_box(
    bbox_xyxy: list[float] | tuple[float, ...],
    orig_width: int,
    orig_height: int,
    sensor_h: int = 2048,
    sensor_w: int = 2176,
    min_side: int = 32,
) -> tuple[int, int, int, int]:
    """Map an arbitrary [x1, y1, x2, y2] image bounding box into integer sensor
    ROI coordinates (r0, r1, c0, c1) for the fast physical exposure loop.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    c0 = int(round(x1 * sensor_w / max(float(orig_width), 1.0)))
    c1 = int(round(x2 * sensor_w / max(float(orig_width), 1.0)))
    r0 = int(round(y1 * sensor_h / max(float(orig_height), 1.0)))
    r1 = int(round(y2 * sensor_h / max(float(orig_height), 1.0)))

    r0 = max(0, min(r0, sensor_h - min_side))
    r1 = max(r0 + min_side, min(r1, sensor_h))
    c0 = max(0, min(c0, sensor_w - min_side))
    c1 = max(c0 + min_side, min(c1, sensor_w))
    return (r0, r1, c0, c1)


def parse_vlm_free_bbox(
    raw_text: str,
    orig_width: int,
    orig_height: int,
    resized_width: int | None = None,
    resized_height: int | None = None,
    default_coord_space: str = "qwen25_resized_pixel",
) -> dict:
    """Strictly parse a free-form bounding box prediction from VLM output.

    Does NOT repair invalid or alias-only outputs into a default 'CENTER' box;
    any invalid output is marked `valid=False` and receives IoU=0.0 in evaluation.
    """
    if not raw_text or not raw_text.strip():
        return {"valid": False, "error": "empty_output", "raw_bbox": None, "bbox_xyxy": None}

    raw_bbox = None
    explicit_space = None

    # 1. Try JSON object or JSON array parsing
    json_candidates = []
    # Check fenced block first
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL)
    json_candidates.extend(fenced)
    obj_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if obj_match:
        json_candidates.append(obj_match.group(0))
    arr_match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if arr_match:
        json_candidates.append(arr_match.group(0))

    for cand in json_candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                for key in ("bbox_2d", "bbox", "box", "roi_box"):
                    if key in parsed and isinstance(parsed[key], (list, tuple)) and len(parsed[key]) == 4:
                        raw_bbox = [float(v) for v in parsed[key]]
                        break
                if "coord_space" in parsed:
                    explicit_space = str(parsed["coord_space"]).strip()
            elif isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(v, (int, float)) for v in parsed):
                raw_bbox = [float(v) for v in parsed]
            if raw_bbox is not None:
                break
        except Exception:
            continue

    # 2. Fallback to Qwen special tag `<box>(x1,y1),(x2,y2)</box>` if present
    if raw_bbox is None:
        m_box = re.search(
            r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)",
            raw_text,
        )
        if m_box:
            raw_bbox = [float(m_box.group(i)) for i in range(1, 5)]

    if raw_bbox is None:
        return {"valid": False, "error": "no_4d_bbox_found", "raw_bbox": None, "bbox_xyxy": None}

    coord_space = explicit_space or default_coord_space
    try:
        bbox_xyxy = convert_pred_bbox_to_pixel_xyxy(
            raw_bbox,
            coord_space=coord_space,
            orig_width=orig_width,
            orig_height=orig_height,
            resized_width=resized_width,
            resized_height=resized_height,
        )
        return {
            "valid": True,
            "error": None,
            "coord_space": coord_space,
            "raw_bbox": raw_bbox,
            "bbox_xyxy": list(bbox_xyxy),
        }
    except Exception as exc:
        return {
            "valid": False,
            "error": str(exc),
            "coord_space": coord_space,
            "raw_bbox": raw_bbox,
            "bbox_xyxy": None,
        }


def compute_bbox_iou_xyxy(
    box_a: list[float] | tuple[float, ...] | None,
    box_b: list[float] | tuple[float, ...] | None,
) -> float:
    """Compute exact continuous 2D Intersection-over-Union (IoU) between two
    [x1, y1, x2, y2] bounding boxes. Returns 0.0 if either box is None or degenerate.
    """
    if box_a is None or box_b is None or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, by2_a = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    area_a = max(0.0, ax2 - ax1) * max(0.0, by2_a - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(by2_a, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def summarize_rec_evaluation(records: list[dict]) -> dict:
    """Summarize Referring Expression Comprehension (REC) metrics across `records`.
    Invalid outputs are included in the denominator with IoU = 0.0.
    """
    n = len(records)
    if n == 0:
        return {"num_samples": 0}
    valid_cnt = sum(1 for r in records if r.get("valid", False))
    ious = [float(r.get("iou", 0.0)) if r.get("valid", False) else 0.0 for r in records]
    acc50 = sum(1 for v in ious if v >= 0.5)
    acc75 = sum(1 for v in ious if v >= 0.75)
    return {
        "num_samples": n,
        "valid_bbox_count": valid_cnt,
        "valid_bbox_rate": round(valid_cnt / n, 4),
        "invalid_output_count": n - valid_cnt,
        "acc_at_iou_050_count": acc50,
        "acc_at_iou_050": round(acc50 / n, 4),
        "acc_at_iou_075_count": acc75,
        "acc_at_iou_075": round(acc75 / n, 4),
        "mean_iou": round(float(np.mean(ious)), 4),
        "median_iou": round(float(np.median(ious)), 4),
    }

