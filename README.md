# Anonymous Factor Monitor artifact

This repository contains the anonymous source code, frozen configurations, and
tests for the paper's primary training and evaluation pipeline. Model adapters,
preserved predictions, and dataset metadata are distributed separately because
they are too large for Anonymous GitHub.

## Contents

- `factor_monitor/`: training and monitor-inference implementation.
- `factor_monitor_eval/`: manifest-driven binary, Process, and Intervention scoring.
- `PAPER_ALIGNMENT.md`: binding between paper claims, checkpoints, and result artifacts.
- `DATA_AVAILABILITY.md`: redistribution status and reconstruction policy.

## Install and test

```bash
python -m pip install -e 'factor_monitor[training]'
python -m pip install -e factor_monitor_eval
PYTHONPATH=factor_monitor_eval/src:factor_monitor/src   python -m unittest discover -s factor_monitor_eval/tests -v
```

The active suite has 14 top-level artifacts: 12 standard units across 11
categories and two Intervention bundles expanding to eight reported tasks.
Deception uses `mask_qwen`; SycoBench is retained as `sycobench_judge`.

Full rescoring additionally requires the separately distributed model,
prediction, and reconstructed dataset artifacts. Raw benchmark rows are not
bundled where redistribution permission is absent.
