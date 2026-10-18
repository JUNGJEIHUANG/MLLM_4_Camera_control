---
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
tags:
  - lora
  - scientific-camera-control
  - same-scene-diagnostic
---

# Qwen2.5-VL-3B camera-contract LoRA — full diagnostic run

**Built with Qwen.** This is a research adapter, not an official Qwen checkpoint.

**Modified-file notice:** this completed model card replaces the automatically
generated PEFT template. The adapter tensor, adapter configuration, tokenizer,
and processor files remain byte-identical to the verified full training output.

## Model and intended use

The adapter compiles camera-task instructions and a rendered scientific image
into a JSON exposure contract with ROI/grid identity, target DN, tolerance,
clipping limit, and confirmation-frame count. It does not directly operate a
camera. The four training tasks use fixed, preconfigured scene ROIs.

Use for non-commercial research/evaluation under the upstream Qwen Research
License. The complete license and attribution accompany this release in
`LICENSE-QWEN-RESEARCH` and `NOTICE` at repository root. It is not Apache/MIT.

## Training provenance

- Base: `Qwen/Qwen2.5-VL-3B-Instruct`; no base-model tensors are mirrored here.
- 180 instruction rows, 2 epochs, two prompt variants per row.
- BF16 LoRA, rank 16, alpha 32, dropout 0.05, learning rate 2e-4; seed 42.
- 720 microsteps and 180 optimizer updates; training-loop time 381.35 seconds.
- Mean epoch losses: 0.0288 and 0.0001. Loss reduction is not generalization evidence.
- Saved adapter was reloaded before evaluation.
- Tensor SHA256: `380450d3df4d95c9d263a318d7ffaa0a6ca6ec1e643063c4f61dae222a885aa0`.

See `source_training_receipt.json` and `base_model_provenance.json`.

## Evidence boundary

**Result label: `same_scene_diagnostic`.** Training/test share all 15 source
images and all 60 physical task/initial-state combinations. Only instruction
paraphrases are held out. Eighteen of 120 test rows are unreachable in the captured
exposure range. The dataset is not distributed in this release.

In that diagnostic, the adapter produced valid contracts for 120/120 rows; the
brightness/clipping proxy passed for 101/120 rows with telemetry and 96/120
with visual input only. These are not independent-scene generalization results,
live camera trials, real localization/OCR/SNR accuracies, or fair method-superiority
claims. P2-P4 obtained the same pass counts. See the repository diagnostic report.

Do not deploy this research adapter for unattended or safety-critical control.
Predictions require validation, reachability checks, device-state checks, and
independent measurements. New scenes and movable ROI layouts are unvalidated.

## Loading

The archived adapter configuration preserves its training-machine base-model
path. Load an official/local base model explicitly, then attach this adapter:

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_id = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter = "checkpoints/qwen25_vl_3b_contract_lora_full"
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_id, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
)
model = PeftModel.from_pretrained(base, adapter).eval()
processor = AutoProcessor.from_pretrained(base_id)
```

For a complete non-actuating inference command, see `tools/infer_contract.py`.
