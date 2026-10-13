# Small-model alignment assignment

Original PyTorch SFT and DPO implementation for `docs/assignment_manual.pdf`, with a separate TRL/LoRA persona experiment, private Kaggle workflows, blinded judging, and an evidence-derived LaTeX report.

## Correctness checks

```powershell
python -m pytest -q tests --basetemp .pytest_cache/local-run
```

The six functions in `submission_adapter.py` delegate to `alignment/core.py`. SFT uses the mean over scored response tokens. DPO uses summed response log probabilities and a stable batch-mean log-sigmoid objective. Prompt and padding labels are ignored; EOS is appended before truncation. Tests cover exact accumulation weighting, optimizer/RNG resumption, judge blinding, retries, and failure accounting.

## Kaggle workflow

Store the existing modern Kaggle token in local `.env` under `KAGGLE_API`. Credentials never enter the source bundle. All notebooks are private. The upload allowlist includes only Python implementation/tests, JSON configurations, and supplied data. Every new bundle records SHA-256 source hashes.

```powershell
python -X utf8 scripts/kaggle_control.py preflight
python -X utf8 scripts/kaggle_control.py push --stage smoke
python -X utf8 scripts/kaggle_control.py push --stage data
python -X utf8 scripts/kaggle_control.py status --kernel OWNER/small-alignment-data
python -X utf8 scripts/kaggle_control.py push --stage sft --source OWNER/small-alignment-data
python -X utf8 scripts/kaggle_control.py push --stage hh --source OWNER/small-alignment-data --source OWNER/small-alignment-sft
python -X utf8 scripts/kaggle_control.py push --stage uf --source OWNER/small-alignment-data --source OWNER/small-alignment-sft
python -X utf8 scripts/kaggle_control.py push --stage persona --source OWNER/small-alignment-data
```

Wait for each dependency to finish before launching its consumer. Kaggle may limit concurrent GPU sessions. Never cancel unrelated notebooks. `requirements-kaggle.txt` pins the tested APIs while retaining Kaggle's PyTorch/CUDA installation. The smoke run exercised two T4 GPUs with 512-token SFT and DPO batches.

`configs/experiments.json` defines the guide configuration. Fixed subsets use seed 42; single-turn preference pairs are filtered before selecting counts. The saved manifest includes revisions, filtering counts, IDs, and hashes. Training saves best/final checkpoints and optimizer, scheduler, scaler, RNG, epoch, and sample-offset state. A resume requires an identical configuration:

```bash
python -m alignment.train --track sft --data DATA --output OUTPUT --resume CHECKPOINT
```

For DPO add `--sft-checkpoint SFT_BEST`. The LoRA command is `python -m alignment.persona --data DATA --output OUTPUT`. It adapts query/value projections of SmolLM2; the base model with adapters disabled is its frozen reference.

## Generation and judging

Attach data and all four training notebooks when pushing `--stage generate`. Generations are greedy and append immediately to JSONL; settings and input hashes protect resumption. Qwen uses an explicit instruction/response template, while SmolLM2 uses its native chat template. Inspect five samples per model before accepting the full evaluation. Pair files exclude model identities; separate `.identity.json` files hold the mapping.

```powershell
python -X utf8 scripts/benchmark_control.py push
python -X utf8 scripts/benchmark_control.py status
python -X utf8 scripts/benchmark_control.py run
python -X utf8 scripts/benchmark_control.py push --task small-alignment-judging --generations GENERATIONS
python -X utf8 scripts/benchmark_control.py run --task small-alignment-judging
python -X utf8 scripts/benchmark_control.py download --task small-alignment-judging
```

The first three commands validate judge access. Benchmark tasks stay private unless explicitly published. Full judging only reads saved pairs. To resume a task, pass the prior judgment directory with `--previous`. Main comparisons each contain 300 pairs; persona contains 40. Unknowns and failures are not counted as wins. The report distinguishes decisive and tie-adjusted rates and does not claim official length-controlled AlpacaEval scores.

Download each completed notebook's results to a common artifact root:

```powershell
python -X utf8 scripts/kaggle_control.py download --kernel OWNER/small-alignment-sft --output artifacts/experiment
```

This retrieves `results/` and the notebook log without duplicating source files. Add `--pattern` to retrieve only metadata while keeping large checkpoints in saved private notebook outputs.

## LaTeX report

```powershell
python -X utf8 scripts/build_report.py --results artifacts/experiment/results --repository https://github.com/OWNER/REPO --author "Your name"
```

The builder creates vector figures and LaTeX tables from saved metrics and judgments, then compiles with `latexmk`/PDFLaTeX/BibTeX. It rejects missing experiments and PDFs over eight pages. `--draft` explicitly permits incomplete evidence and produces `report/draft_report.pdf`; it never substitutes for a final report. The deliverable is `report/final_report.pdf` plus `main.tex`, bibliography, generated inputs, and figures.

## References and authorship

The math is implemented from the assignment specification and DPO paper, not copied from a public training loop. External API references are cited in `report/references.bib`. AI assistance was used to develop and debug this project; the submitter should review and understand the implementation and follow course disclosure requirements.

The reusable implementation prompt is `docs/implementation_prompt.md`.
