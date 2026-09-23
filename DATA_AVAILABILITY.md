# Data availability

Training rows are not published. Evaluation rows are published at
https://huggingface.co/datasets/270vision/factor-monitor-evaluation.

## Included in full

Task text, chain-of-thought, factors, and labels are included for Process
GPQA, SycoBench, IFBench, HalluLens, ConfAIde, JailbreakBench, OpenAI WMDP,
and the MALT reward-hacking and sandbagging sets. Their supplemental
CoT-transform and GLM / Kimi panels are included on the same terms.

- Process GPQA: CC-BY-4.0 data; OpenAI evaluation code Apache-2.0.
- SycoBench: CC-BY-4.0.
- IFBench: ODC-BY-1.0.
- HalluLens: CC-BY-NC. This subset is not Apache-2.0.
- ConfAIde: MIT.
- JailbreakBench: MIT.
- OpenAI WMDP: evaluation code Apache-2.0; WMDP MIT.
- MALT reward hacking and sandbagging: MIT. Upstream hosting is gated.

OpenAI sycophancy keeps DAFT, Scruples, and GPQA text in the same file as the
AIME rows.

## Task statements replaced by links

`problem_linked/` replaces copyrighted task statements with `[PROBLEM]`.
Official URL, source or task id, statement hash, and slice offsets are in
`problem_links.jsonl`. Chain-of-thought text, factors, and labels stay.
Exact copies of at least 64 characters are replaced. Shorter paraphrases can
remain in the chain of thought.

- Process AIME, including the GLM and Kimi panels: official contest URL and source.
- OpenAI sycophancy: only the AIME rows.
- MASK, SkillEval, and SHADE-Arena, including their supplemental panels: the
  whole task statement. The link is the official dataset plus the task id.

Restore those statements with `scripts/prepare_public_eval_release.py hydrate`
after fetching them from the recorded source.
