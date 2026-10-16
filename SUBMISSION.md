# Submission — Talha Hassan Ul Haq (Student ID 27100306)

GitHub repository: https://github.com/TalhaHassanUlHaq/small-model-alignment-assignment

This repository is public. Course staff can inspect the source and archived evidence through the link above.

The assignment manual's five submission items are here:

| Item | Location |
| --- | --- |
| Completed GitHub project | Repository URL above |
| Final report PDF | [report/final_report.pdf](report/final_report.pdf) (7 pages, including references) |
| SFT and DPO training plots | [report/figures/training.pdf](report/figures/training.pdf) and [report/figures/preferences.pdf](report/figures/preferences.pdf) |
| DPO-versus-SFT evaluation summary | Report Section 5; machine-readable [HH](report/generated/hh_summary.json), [UltraFeedback](report/generated/uf_summary.json), and [persona](report/generated/persona_summary.json) summaries |
| Root-level adapter | [submission_adapter.py](submission_adapter.py), connected to [alignment/core.py](alignment/core.py) |

The report compares HH-DPO (140 wins, 102 losses, 58 ties; 56.3% tie-adjusted) and UltraFeedback-DPO (112 wins, 128 losses, 60 ties; 47.3%) against the SFT checkpoint on separate fixed 300-prompt sets. Persona LoRA versus its instruction-model baseline yielded 13 wins, 14 losses, and 13 ties (48.8%) on 40 prompts. All 640 primary-judge pairs have a recorded outcome, with zero unresolved failures. These are raw judged pairwise rates, not official AlpacaEval scores.

Recheck the adapter and rebuild the report from the evidence committed to this repository:

```powershell
python -m pytest -q tests --basetemp .pytest_cache/local-run
python -X utf8 scripts/build_report.py --results evidence --repository https://github.com/TalhaHassanUlHaq/small-model-alignment-assignment --author "Talha Hassan Ul Haq (Student ID: 27100306)"
```

The second command requires the Python dependencies and `latexmk`/PDFLaTeX/BibTeX described in [README.md](README.md). Archived configurations, metrics, generations, pair identities, and judgments are in [evidence](evidence); full training checkpoints remain in the private saved Kaggle notebooks listed in the README. AI assistance was used to develop and debug this project; review the implementation and follow the course's disclosure requirements when submitting.

## LMS submission

The assignment manual asks for five items. Paste the public repository URL into the GitHub link field. Upload `report/final_report.pdf` as the final report. If the LMS has separate fields for plots, upload `report/figures/training.pdf` and `report/figures/preferences.pdf`; both are also embedded in the report. For the evaluation-summary field, point to Section 5 of the report and the three linked JSON summaries above. For an adapter/source field, point to the repository's root-level `submission_adapter.py` (or upload that file if the LMS requires a file). If the LMS provides just one file slot, upload the report PDF and put the repository URL in the comments or link field.
