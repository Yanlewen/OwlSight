# Factor Monitor

Training and scoring code for the paper models. Dataset construction and
historical analysis commands are not part of this public copy. Model weights
and evaluation data are linked from the repository README.

## Configs

- `configs/train_qwen35_4b.yaml`
- `configs/train_qwen35_9b.yaml`
- `configs/train_llama32_3b.yaml`
- `configs/architecture_qwen35_4b.yaml`
- `configs/architecture_qwen35_9b.yaml`
- `configs/architecture_llama32_3b.yaml`
- `configs/taxonomy.yaml`

## Commands

`python -m factor_monitor` supports `train`, `evaluate`, `full-eval`, `score`,
`score-binary-artifact`, `score-intervention-plan`, `score-process`,
`audit-lengths`, `validate-source`, and `contract`.
