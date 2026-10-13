# Reusable execution prompt

Implement and complete the assignment in `docs/assignment_manual.pdf`. Treat the manual as the source of truth and audit every required artifact. Use the existing six-function adapter contract and leave supplied tests and fixed evaluation data unchanged.

Implement original response-masked SFT and DPO loops in PyTorch. Use TRL/PEFT only for the persona task. Test the numerical objectives independently, including causal shifting, truncation, EOS/pad collisions, accumulation weighting, and frozen reference gradients. Save exact data subsets and dataset/model revisions.

Run the four configured experiments in `configs/experiments.json` on private Kaggle notebooks using free T4 x2 quota. Read the local credential without printing it or uploading it. Run a GPU smoke test first. Preserve best/final checkpoints, per-step metrics, resumable states, and source fingerprints. Monitor jobs and diagnose failures rather than treating submission as success.

Generate and inspect five samples before full evaluations. Compare SFT against independent HH and UltraFeedback DPO checkpoints on their fixed 300-prompt sets. Evaluate persona style/usefulness/safety separately. Judge saved, blinded pairs through Kaggle's supported model interface with five-pair batches, five-second minimum spacing, three attempts, partial-result recovery, and explicit failure accounting.

Create the authorized private GitHub repository and a final LaTeX report of at most eight pages. Derive plots and tables from saved evidence, inspect every PDF page, and include measured outcomes, configuration deviations, representative improvements/regressions, citations, and limitations. Never invent results or promise marks. A report with missing experiments must remain explicitly marked draft.
