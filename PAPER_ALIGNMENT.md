# Paper alignment notes

## Frozen model bindings

- Primary: `qwen35-4b-factor-monitor`, checkpoint 4058, non-targeted active-11 run.
- Scale transfer: `qwen35-9b-factor-monitor`, checkpoint 4058.
- Backbone transfer: `llama32-3b-factor-monitor`, checkpoint 6087.

## Evaluation counts

- New benchmark headline: 32,222 queries (Table 15 excluding both intervention
  units).
- Table 15 total: 36,726 queries.
- Active scoring suite: 14 top-level artifacts and 36,726 queries. The two
  intervention artifacts expand to eight paper-reported tasks.

## Result bindings

- Deception uses the released `mask_qwen` artifact for all three model tables;
  MASK-DeepSeek is excluded from this source tree.
- SycoBench is retained as `sycobench_judge` for all three models, including
  the Qwen3.5-9B and Llama-3.2-3B predictions and summaries.
- The authoritative primary-evaluation outputs are the per-benchmark summaries
  in the separately distributed results artifact under each model's `scored/` directory. The
  `suite_summary.json` files are convenience aggregations and must not be used
  as substitutes for the paper's fine/coarse 11-category macro averages.

## Scope

The combined code, model, results, and dataset-metadata release supports the
primary training and active evaluation pipeline through source code, frozen
configurations, adapter bundles, predictions, summaries, and hashes. Additional
analysis experiments may remain as preserved snapshots or paper tables without
separate end-to-end commands, provided their provenance is documented in the
paper.
