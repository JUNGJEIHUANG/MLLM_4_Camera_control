# Data availability

Raw camera frames, rendered images, and the original training/test bundle are
not included in this code-and-checkpoint release. No SSH credentials, vendor
camera drivers, workstation caches, or manuscripts are distributed.

The audited experiment entrypoints expect these private inputs at repository root:

```text
train_contracts.jsonl        # 180 instruction rows
test_contracts.jsonl         # 120 instruction rows
sCMOS_physical_ledger.json   # measured response curves
images/                     # referenced visual representations
```

These paths are intentionally ignored by Git. Contact the repository owner for
access. The code/checkpoint release is not a standalone public benchmark: the
complete diagnostic cannot be reproduced without the corresponding input data.

## Evaluation scope

The original training and test sets share all 15 source images and 60 physical
task/initial-condition combinations. The split holds out instruction paraphrases,
not scenes or independent image acquisitions. Eighteen test rows are infeasible
within the captured exposure range. See `docs/full_same_scene_report.md`.

Changing names, augmenting images, or splitting adjacent frames does not make
these data an independently acquired unseen-scene test set.
