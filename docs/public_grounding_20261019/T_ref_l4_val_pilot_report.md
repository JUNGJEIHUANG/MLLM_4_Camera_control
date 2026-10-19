# Step 1 Pilot Report: `Ref-L4` Validation-Set Free-BBox Grounding, Coordinate Audit & Baseline Comparison

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (`95.0 GiB` VRAM, `PyTorch 2.8.0+cu128`)
- **Dataset Split**: `ref-l4-val.parquet` ONLY (`ref-l4-test.parquet` strictly untouched)
- **Deduplication Policy**: Strictly `1` sample per unique `image_id`, `file_name`, and verified image `SHA-256` (`N = 60` unique validation images: `40` original captions + `20` LLM-rewritten captions)
- **Compliance & Provenance Caveat**:
  - `Ref-L4` annotations are licensed under **CC BY-NC 4.0**, and underlying images remain subject to original COCO / Objects365 terms (used strictly for non-commercial evaluation; no redistribution).
  - Public COCO / Objects365 images may have been seen during base `Qwen2.5-VL` pretraining; therefore, this benchmark tests **external open-vocabulary localization unseen by our sCMOS `Contract-LoRA` fine-tuning**, rather than claiming zero pretraining overlap with the base VLM.

---

## 1. Coordinate Conversion Audit (`qwen25_resized_pixel` vs. `orig_pixel` vs. `norm1000`)

`Ref-L4` stores ground-truth boxes in COCO `[x, y, w, h]` pixel coordinates (`w * h == bbox_area`). We audited three candidate coordinate interpretations for `Qwen2.5-VL` `bbox_2d` outputs via [`convert_pred_bbox_to_pixel_xyxy`](file:///p:/13_VLM_4_Camera_Control/iscas_control/vlm_contract_compiler.py#L458-L510):

| Model Stage | Candidate Coordinate Space | Valid BBox (%) | Acc@IoU>=0.50 | Acc@IoU>=0.75 | Mean IoU (%) | Median IoU (%) |
|---|---|---:|---:|---:|---:|---:|
| `Qwen2.5-VL-3B-Base` (Free-BBox) | **`qwen25_resized_pixel`** (`grid_thw * 14 -> orig`) | **100.00%** (60/60) | **91.67%** (55/60) | 78.33% (47/60) | **83.92%** | **92.37%** |
| `Qwen2.5-VL-3B-Base` (Free-BBox) | `orig_pixel` (unscaled raw pixel) | 100.00% (60/60) | 91.67% (55/60) | 81.67% (49/60) | 83.03% | 89.20% |
| `Qwen2.5-VL-3B-Base` (Free-BBox) | `norm1000` (assuming `[0, 1000]`) | 100.00% (60/60) | 0.00% (0/60) | 0.00% (0/60) | 5.65% | 0.08% |
| `Qwen2.5-VL-7B-Base` (Free-BBox) | **`qwen25_resized_pixel`** (`grid_thw * 14 -> orig`) | **100.00%** (60/60) | **90.00%** (54/60) | **75.00%** (45/60) | **80.44%** | **90.98%** |
| `Qwen2.5-VL-7B-Base` (Free-BBox) | `orig_pixel` (unscaled raw pixel) | 100.00% (60/60) | 90.00% (54/60) | 73.33% (44/60) | 78.82% | 87.43% |
| `Qwen2.5-VL-7B-Base` (Free-BBox) | `norm1000` (assuming `[0, 1000]`) | 100.00% (60/60) | 0.00% (0/60) | 0.00% (0/60) | 5.29% | 0.05% |

---

## 2. Unseen-Scene `Ref-L4` Pilot vs. Spatial & Fixed-Alias Baselines (`N = 60`, Strict `Invalid = 0`)

All methods are scored using [`summarize_rec_evaluation`](file:///p:/13_VLM_4_Camera_Control/iscas_control/vlm_contract_compiler.py#L654-L676) with **zero fallback repair** (any missing/unparseable/alias-only output receives `IoU = 0.0`):

| Category | Method / Stage | Prompt / Interface | Valid Free BBox | Acc@IoU>=0.50 | Acc@IoU>=0.75 | Mean IoU | Median IoU | Median Latency (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Spatial Baseline** | `Fixed Camera-Center 512 Proportional Box` | Fixed Central `[512/2176 × 512/2048]` ROI | 100.00% (60/60) | 0.00% (0/60) | 0.00% (0/60) | 8.39% | 5.88% | 0.0 |
| **Spatial Baseline** | `Fixed Center-50% Box` | Fixed Central `[0.25..0.75 W × 0.25..0.75 H]` | 100.00% (60/60) | 1.67% (1/60) | 0.00% (0/60) | 20.28% | 18.22% | 0.0 |
| **Spatial Baseline** | `Full-Frame Global Box` | Full Image `[0, 0, W, H]` | 100.00% (60/60) | 5.00% (3/60) | 1.67% (1/60) | 19.61% | 14.04% | 0.0 |
| **Legacy Alias Interface** | `3B + Same-Scene Contract-LoRA` | Legacy Fixed-Alias Prompt (`roi_code` in 5 aliases) | **0.00%** (0/60) | **0.00%** (0/60) | **0.00%** (0/60) | **0.00%** | **0.00%** | 1744.1 |
| **Free-BBox Interface** | `3B + Same-Scene Contract-LoRA` | Free-BBox Prompt (`{"bbox_2d": [x1,y1,x2,y2]}`) | **100.00%** (60/60) | **91.67%** (55/60) | **80.00%** (48/60) | **84.36%** | 91.46% | 1007.6 |
| **Free-BBox Interface** | `Qwen2.5-VL-3B-Base` | Free-BBox Prompt (`{"bbox_2d": [x1,y1,x2,y2]}`) | **100.00%** (60/60) | **91.67%** (55/60) | 78.33% (47/60) | 83.92% | **92.37%** | 891.0 |
| **Free-BBox Interface** | `Qwen2.5-VL-7B-Base` | Free-BBox Prompt (`{"bbox_2d": [x1,y1,x2,y2]}`) | **100.00%** (60/60) | 90.00% (54/60) | 75.00% (45/60) | 80.44% | 90.98% | **571.3** |
| **Unified Free Contract** | `Qwen2.5-VL-3B-Base` | Unified Free-BBox Exposure Contract (`bbox_2d + target_dn + ...`) | **100.00%** (60/60) | **90.00%** (54/60) | **75.00%** (45/60) | **81.45%** | **91.81%** | 909.3 |
| **Unified Free Contract** | `Qwen2.5-VL-7B-Base` | Unified Free-BBox Exposure Contract (`bbox_2d + target_dn + ...`) | 98.33% (59/60) | 88.33% (53/60) | 73.33% (44/60) | 79.71% | 91.15% | 941.5 |

---

## 3. Stratification by Caption Complexity (`is_rewrite = False` vs. `True`)

| Stage | Original Captions (`n = 40`) Acc@0.5 | Original Captions (`n = 40`) mIoU | LLM-Rewritten Captions (`n = 20`) Acc@0.5 | LLM-Rewritten Captions (`n = 20`) mIoU |
|---|---:|---:|---:|---:|
| `3B + Same-Scene LoRA` (Free-BBox Prompt) | **95.00%** (38/40) | **86.97%** | 85.00% (17/20) | 79.15% |
| `Qwen2.5-VL-3B-Base` (Free-BBox Prompt) | 92.50% (37/40) | 85.37% | **90.00%** (18/20) | **81.03%** |
| `Qwen2.5-VL-3B-Base` (Unified Contract) | 92.50% (37/40) | 84.16% | 85.00% (17/20) | 76.02% |
| `Qwen2.5-VL-7B-Base` (Free-BBox Prompt) | 90.00% (36/40) | 82.06% | **90.00%** (18/20) | 77.19% |
| `Qwen2.5-VL-7B-Base` (Unified Contract) | 87.50% (35/40) | 80.64% | **90.00%** (18/20) | 77.84% |
