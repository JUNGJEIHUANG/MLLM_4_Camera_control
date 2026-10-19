# Complete Unseen-Scene Generalization Report (Steps 1–3): `Ref-L4`, `RefCOCOg-UMD`, and `OCID-Ref` Indoor Clutter/Occlusion

- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (`95.0 GiB` VRAM, `PyTorch 2.8.0+cu128`)
- **Total GPU Execution Time**: `1,706.3 s` (Step 2 & Step 3 full suite) + Step 1 `Ref-L4` validation pilot
- **Strict Cross-Dataset & Cross-Split Image Deduplication Audit**:
  - `Step 1 (Ref-L4 val pilot)`: `60` unique images (`1` referring expression per unique `image_id`, `file_name`, and `SHA-256`)
  - `Step 2A (RefCOCOg-UMD train)`: `120` unique images (`0` overlap with `Ref-L4`, `RefCOCOg-UMD val/test`, or `OCID-Ref`)
  - `Step 2B (RefCOCOg-UMD val)`: `60` unique images (`0` overlap with any other split)
  - `Step 2C (RefCOCOg-UMD test)`: `80` unique images (`0` overlap with any other split; evaluated once with frozen prompt and weights)
  - `Step 3 (OCID-Ref indoor clutter/occlusion val + test)`: `40` unique RGB indoor scenes across `ARID20` (`n=20`), `ARID10` (`n=14`), and `YCB10` (`n=6`), stratified by object clutter/occlusion buildup `take_id` (`low_clutter <= 5`: `n=14`; `medium_clutter 6..11`: `n=19`; `heavy_clutter >= 12`: `n=7`).
  - **Verified Pairwise Image Overlap**: `0` across all 5 sets (`360` unique images total).

---

## 1. Step 2: `RefCOCOg-UMD` Validation (`N=60`) & Test (`N=80`) vs. Baselines

All methods are scored using [`summarize_rec_evaluation`](file:///p:/13_VLM_4_Camera_Control/iscas_control/vlm_contract_compiler.py#L665-L686) with **strict `Invalid = 0.0` IoU** (no default `CENTER` fallback repair) and [`parse_vlm_contract_json(..., strict_free_bbox=True)`](file:///p:/13_VLM_4_Camera_Control/iscas_control/vlm_contract_compiler.py#L92-L176):

| Split | Category | Method / Stage | Contract Valid (%) | Valid BBox (%) | Acc@IoU>=0.50 | Acc@IoU>=0.75 | Mean IoU | Median IoU | Median Latency (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Val (`N=60`)** | Spatial Baseline | `Camera-Center 512 Proportional Box` | — | 100.0% | 1.67% (1/60) | 0.00% (0/60) | 8.81% | 4.42% | 0.0 |
| **Val (`N=60`)** | Spatial Baseline | `Fixed Center-50% Box` | — | 100.0% | 1.67% (1/60) | 0.00% (0/60) | 19.72% | 18.47% | 0.0 |
| **Val (`N=60`)** | Spatial Baseline | `Full-Frame Global Box` | — | 100.0% | 10.00% (6/60) | 1.67% (1/60) | 21.43% | 14.44% | 0.0 |
| **Val (`N=60`)** | Legacy Alias | `3B + Same-Scene LoRA (Legacy Alias)` | **0.0%** | **0.0%** | **0.00%** (0/60) | **0.00%** (0/60) | **0.00%** | **0.00%** | 1759.0 |
| **Val (`N=60`)** | Free-BBox | `3B + Same-Scene LoRA (Free-BBox)` | 100.0% | 100.0% | 88.33% (53/60) | 75.00% (45/60) | 80.58% | 91.95% | 1188.1 |
| **Val (`N=60`)** | Free-BBox | `Qwen2.5-VL-3B-Base (Free-BBox)` | 100.0% | 100.0% | 86.67% (52/60) | 76.67% (46/60) | 79.90% | 92.17% | 680.8 |
| **Val (`N=60`)** | Free-BBox | `Qwen2.5-VL-7B-Base (Free-BBox)` | 100.0% | 100.0% | **90.00%** (54/60) | **78.33%** (47/60) | 81.86% | 91.50% | **662.6** |
| **Val (`N=60`)** | Unified Contract | `Qwen2.5-VL-3B-Base (Unified Contract)` | 100.0% | 100.0% | 86.67% (52/60) | 75.00% (45/60) | 79.78% | 92.37% | 1095.0 |
| **Val (`N=60`)** | Unified Contract | `Qwen2.5-VL-7B-Base (Unified Contract)` | 100.0% | 100.0% | 88.33% (53/60) | **78.33%** (47/60) | **81.96%** | 91.47% | 1025.9 |
| **Val (`N=60`)** | **Unified Contract (Ours)** | **`3B FreeBBox-Contract-LoRA (Unified Contract)`** | **100.0%** | **100.0%** | **88.33%** (53/60) | **78.33%** (47/60) | **81.62%** | **92.89%** | 1535.8 |
| **Test (`N=80`)** | Spatial Baseline | `Camera-Center 512 Proportional Box` | — | 100.0% | 0.00% (0/80) | 0.00% (0/80) | 8.87% | 5.75% | 0.0 |
| **Test (`N=80`)** | Spatial Baseline | `Fixed Center-50% Box` | — | 100.0% | 5.00% (4/80) | 0.00% (0/80) | 20.81% | 18.40% | 0.0 |
| **Test (`N=80`)** | Spatial Baseline | `Full-Frame Global Box` | — | 100.0% | 5.00% (4/80) | 2.50% (2/80) | 20.01% | 14.11% | 0.0 |
| **Test (`N=80`)** | Legacy Alias | `3B + Same-Scene LoRA (Legacy Alias)` | **0.0%** | **0.0%** | **0.00%** (0/80) | **0.00%** (0/80) | **0.00%** | **0.00%** | 1773.3 |
| **Test (`N=80`)** | Free-BBox | `3B + Same-Scene LoRA (Free-BBox)` | 100.0% | 100.0% | 83.75% (67/80) | 73.75% (59/80) | 78.56% | 93.47% | 1191.2 |
| **Test (`N=80`)** | Free-BBox | `Qwen2.5-VL-3B-Base (Free-BBox)` | 100.0% | 100.0% | 82.50% (66/80) | 73.75% (59/80) | 77.91% | **93.89%** | 668.5 |
| **Test (`N=80`)** | Free-BBox | `Qwen2.5-VL-7B-Base (Free-BBox)` | 100.0% | 100.0% | 86.25% (69/80) | 71.25% (57/80) | 78.46% | 92.57% | **659.7** |
| **Test (`N=80`)** | Unified Contract | `Qwen2.5-VL-3B-Base (Unified Contract)` | 100.0% | 100.0% | 81.25% (65/80) | 75.00% (60/80) | 79.07% | 92.68% | 1099.9 |
| **Test (`N=80`)** | Unified Contract | `Qwen2.5-VL-7B-Base (Unified Contract)` | 100.0% | 100.0% | **91.25%** (73/80) | 78.75% (63/80) | 82.13% | 92.18% | 1032.2 |
| **Test (`N=80`)** | **Unified Contract (Ours)** | **`3B FreeBBox-Contract-LoRA (Unified Contract)`** | **100.0%** | **100.0%** | **87.50%** (70/80) | **82.50%** (66/80) | **82.55%** | **93.89%** | 1530.5 |

---

## 2. Step 3: `OCID-Ref` Indoor Clutter & Occlusion External Evaluation (`N=40` Unique Scenes)

`OCID-Ref` scenes are captured with a physical RGB-D sensor across tabletop and floor environments (`ARID20`, `ARID10`, `YCB10`) with progressive clutter and occlusion (`take_id`), having **zero image overlap with COCO or our sCMOS dataset**:

| Category | Method / Stage | Contract Valid (%) | Overall Acc@0.5 | Overall Acc@0.75 | Overall mIoU | Low Clutter (`take<=5`, `n=14`) Acc@0.5 | Med Clutter (`take 6..11`, `n=19`) Acc@0.5 | Heavy Clutter (`take>=12`, `n=7`) Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Spatial Baseline | `Camera-Center 512 Proportional Box` | — | 0.00% (0/40) | 0.00% (0/40) | 7.13% | 0.00% | 0.00% | 0.00% |
| Spatial Baseline | `Fixed Center-50% Box` | — | 0.00% (0/40) | 0.00% (0/40) | 7.49% | 0.00% | 0.00% | 0.00% |
| Spatial Baseline | `Full-Frame Global Box` | — | 0.00% (0/40) | 0.00% (0/40) | 2.69% | 0.00% | 0.00% | 0.00% |
| Legacy Alias | `3B + Same-Scene LoRA (Legacy Alias)` | **0.0%** | **0.00%** (0/40) | **0.00%** (0/40) | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| Free-BBox | `3B + Same-Scene LoRA (Free-BBox)` | 100.0% | 72.50% (29/40) | 62.50% (25/40) | 63.82% | **92.86%** (13/14) | 57.89% (11/19) | **71.43%** (5/7) |
| Free-BBox | `Qwen2.5-VL-3B-Base (Free-BBox)` | 100.0% | 72.50% (29/40) | **67.50%** (27/40) | **66.77%** | **92.86%** (13/14) | 57.89% (11/19) | **71.43%** (5/7) |
| Free-BBox | `Qwen2.5-VL-7B-Base (Free-BBox)` | 100.0% | **77.50%** (31/40) | 62.50% (25/40) | 66.38% | **92.86%** (13/14) | **73.68%** (14/19) | 57.14% (4/7) |
| Unified Contract | `Qwen2.5-VL-3B-Base (Unified Contract)` | 100.0% | 65.00% (26/40) | 60.00% (24/40) | 62.86% | 85.71% (12/14) | 47.37% (9/19) | **71.43%** (5/7) |
| Unified Contract | `Qwen2.5-VL-7B-Base (Unified Contract)` | 100.0% | **72.50%** (29/40) | 62.50% (25/40) | 64.52% | 85.71% (12/14) | 68.42% (13/19) | 57.14% (4/7) |
| **Unified Contract (Ours)** | **`3B FreeBBox-Contract-LoRA (Unified Contract)`** | **100.0%** | **67.50%** (27/40) | **60.00%** (24/40) | **63.12%** | **85.71%** (12/14) | **52.63%** (10/19) | **71.43%** (5/7) |

---

## 3. Summary Across All Three External Benchmarks (`Ref-L4 Val`, `RefCOCOg-UMD Val/Test`, `OCID-Ref`)

| Benchmark Split | `Cam-Center 512` Baseline Acc@0.5 | `3B Legacy Alias LoRA` Acc@0.5 | `3B Base Unified Contract` Acc@0.5 / mIoU | `7B Base Unified Contract` Acc@0.5 / mIoU | **`Ours: 3B FreeBBox-Contract-LoRA` Acc@0.5 / Acc@0.75 / mIoU** |
|---|---:|---:|---:|---:|---:|
| **Step 1: `Ref-L4 Val` (`N=60`)** | 0.00% | 0.00% | 90.00% / 81.45% | 88.33% / 79.71% | **90.00% / 75.00% / 82.54%** |
| **Step 2: `RefCOCOg-UMD Val` (`N=60`)** | 1.67% | 0.00% | 86.67% / 79.78% | 88.33% / 81.96% | **88.33% / 78.33% / 81.62%** |
| **Step 2: `RefCOCOg-UMD Test` (`N=80`)** | 0.00% | 0.00% | 81.25% / 79.07% | 91.25% / 82.13% | **87.50% / 82.50% / 82.55%** |
| **Step 3: `OCID-Ref Indoor Clutter` (`N=40`)** | 0.00% | 0.00% | 65.00% / 62.86% | 72.50% / 64.52% | **67.50% / 60.00% / 63.12%** |
