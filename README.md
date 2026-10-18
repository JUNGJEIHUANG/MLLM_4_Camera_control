# MLLM_4_Camera_control

**Built with Qwen. Research code and a trained LoRA adapter for scientific-camera
exposure-contract compilation.** A slow vision-language model produces a semantic
contract; bounded physical controllers execute exposure updates.

## What is released

- `iscas_control/`: contract parsing, image representations, physical controllers,
  execution guards, and host-side camera interfaces.
- `server_vlm_experiment.py`: the audited BF16 training/evaluation implementation.
- `T_run_verified_suite.py`: staged, resumable diagnostic runner.
- `checkpoints/qwen25_vl_3b_contract_lora_full/`: the full two-epoch 3B LoRA adapter,
  processor/tokenizer files, a model card, and checksums. The tensor file uses Git LFS.
- `tests/`: unit tests and explicitly mocked integration tests.
- `docs/`: the full-run receipt, summary, and limitations report.

**Not released:** Qwen base-model tensors, private RAW images/datasets, camera
vendor drivers, SSH credentials, and workstation caches. See [data availability](data/README.md).

## Important evidence boundary

The included result is labeled **`same_scene_diagnostic`**, not independent-scene
generalization and not a fair superiority claim. Training and test share 15 source
images. Eighteen of 120 test rows are unreachable in the captured exposure range.
Closed-loop evaluation uses measured response-curve interpolation, not live camera
actuation. The P1 baseline and compiler variants also differ in training/information
conditions. Full details: [diagnostic report](docs/full_same_scene_report.md).

The trained adapter plus telemetry reached **101/120 brightness-proxy passes** in
that diagnostic. This does not measure real localization/OCR/SNR accuracy and must
not be described as unseen-scene performance.

## Installation and checkpoint retrieval

```bash
git lfs install
git clone https://github.com/JUNGJEIHUANG/MLLM_4_Camera_control.git
cd MLLM_4_Camera_control
git lfs pull

# Install a compatible CUDA PyTorch build first, then:
python -m pip install -r requirements.txt
```

The recorded environment was Python 3.12.3, PyTorch 2.8.0+cu128, Transformers
5.20.1, PEFT 0.21.2, and Accelerate 1.15.0 on one RTX PRO 6000 Blackwell GPU.
Select your PyTorch build from the [official installation instructions](https://pytorch.org/get-started/locally/).
The archived CUDA inference path uses BF16 on `cuda:0`; CPU inference is not covered.

## Base models: official downloads, not mirrored

- Adapter base: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Zero-shot comparator: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Official ModelScope mirrors: [3B](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct), [7B](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)

The base-model tensor hashes used for the diagnostic are listed in
`checkpoints/qwen25_vl_3b_contract_lora_full/base_model_provenance.json`.
To download with ModelScope and register local paths for the diagnostic runner:

```bash
python tools/download_base_models.py --sizes 3B 7B --cache-dir models/base
```

Base-model licensing differs by release; consult each official model card.
The 3B adapter base uses the **Qwen Research License**, not Apache/MIT.

## Compile a contract without moving camera hardware

Use an official base-model ID, or pass your downloaded 3B model directory:

```bash
python tools/infer_contract.py \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --image /absolute/path/to/your_image.png \
  --instruction "Set an exposure contract for the dark wall region."
```

Optional `--telemetry` supplies scene telemetry. The command prints the raw output,
strict validation result, and model-generation latency. It never actuates a camera.
The adapter configuration retains its original training-machine base-model path
for provenance, so load the base explicitly as this tool does; do not rely on an
automatic loader resolving that archived path. New scenes are not validated.

## Run the archived diagnostics

These commands require the private input bundle described in `data/README.md`:

```bash
python T_run_verified_suite.py --mode smoke --wait-models
python T_run_verified_suite.py --mode full --wait-models
```

The full configuration uses 180 training rows, two epochs/two prompt variants,
720 microsteps, 180 optimizer updates, and 120 test rows per stage. It writes
stage JSON/receipts below `audit_20261018/`. No camera access occurs in this runner.
Its confirmation-time metric combines measured generation latency with assumed
frame/command costs; it is not measured hardware end-to-end latency.

The live host interface is a separate, platform-specific research component. It
depends on locally verified vendor files and device-state checks not shipped here.
Do not treat it as a portable camera driver or run hardware acquisition blindly.

## Tests

```bash
python -m unittest discover -s tests -v
```

Mock-camera tests validate code behavior, not physical-camera performance.

## License and attribution

The adapter, tokenizer/processor assets, and upstream Qwen material are subject to
the [Qwen Research License](LICENSE-QWEN-RESEARCH), for non-commercial research /
evaluation use. See [NOTICE](NOTICE) and the checkpoint model card. No additional
MIT/Apache license has been selected for the original project code; contact the
repository owner about permissions. This repository does not relicense Qwen.
