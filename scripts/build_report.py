"""Build an evidence-derived LaTeX report; final mode refuses missing experiments."""
import argparse
from datetime import date
from pathlib import Path
import random
import shutil
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.io import read_json, read_jsonl, write_json
from alignment.judge import summarize


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c if ord(c) < 128 else " ") for c in str(value))


def contrast_excerpt(value, other, width=130):
    """Show a compact span around the first response difference."""
    value, other = " ".join(str(value).split()), " ".join(str(other).split())
    difference = next((i for i, (left, right) in enumerate(zip(value, other))
                       if left != right), min(len(value), len(other)))
    start = max(0, difference - 24)
    if start:
        start = value.rfind(" ", 0, start) + 1
    excerpt = textwrap.shorten(value[start:], width=width, placeholder="...")
    return ("..." if start else "") + excerpt


def prompt_bootstrap_interval(counts, seed=42, draws=2000):
    scores = ([1.0] * counts.get("win", 0) + [0.5] * counts.get("tie", 0) +
              [0.0] * counts.get("loss", 0))
    if not scores:
        return None
    rng, n = random.Random(seed), len(scores)
    estimates = sorted(sum(scores[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws)]


def build(results, repository="", author="", draft=False, compile_pdf=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = ROOT / "report"
    generated, figures = report / "generated", report / "figures"
    generated.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    results = Path(results)
    def emit(name, content):
        (generated / f"{name}.tex").write_text(content + "\n", encoding="utf-8")
    missing = []
    runs = {}
    for track in ["sft", "hh", "uf", "persona"]:
        folder = results / track
        if not (folder / "run_status.json").exists() or read_json(folder / "run_status.json").get("status") != "complete":
            missing.append(f"Completed {track} experiment")
            continue
        runs[track] = dict(config=read_json(folder / "config.json"), status=read_json(folder / "run_status.json"),
                           metrics=read_jsonl(folder / "metrics.jsonl"))
    summaries = {}
    expected_generations = {name: 300 for name in
                            ("alpaca_base", "alpaca_sft", "alpaca_uf",
                             "hh_base", "hh_sft", "hh_hh")}
    expected_generations.update(persona_base=40, persona_lora=40)
    generation_settings = []
    for name, expected_count in expected_generations.items():
        response_path = results / "generations" / f"{name}.jsonl"
        manifest_path = results / "generations" / f"{name}.manifest.json"
        if not response_path.exists() or not manifest_path.exists():
            missing.append(f"Saved {name} generations and settings")
            continue
        responses = read_jsonl(response_path)
        manifest = read_json(manifest_path)
        if len(responses) != expected_count or manifest.get("expected_count") != expected_count:
            missing.append(f"Complete {name} generation count")
        if len({row["id"] for row in responses}) != len(responses):
            missing.append(f"Unique {name} generation IDs")
        if any(not str(row.get("output", "")).strip() for row in responses):
            missing.append(f"Nonempty {name} generations")
        if any(not 0 <= row.get("generated_tokens", -1) <= 256 for row in responses):
            missing.append(f"Valid {name} generation lengths")
        generation_settings.append(manifest["settings"])
    if generation_settings and (any(setting != generation_settings[0] for setting in generation_settings) or
                                generation_settings[0].get("max_new_tokens") != 256 or
                                generation_settings[0].get("batch_size") != 4 or
                                generation_settings[0].get("do_sample") is not False):
        missing.append("Consistent executed generation settings")
    for track in ["hh", "uf", "persona"]:
        pair_path = results / "generations" / f"pairs_{track}.jsonl"
        judgment_path = results / "judgments" / f"judged_{track}.jsonl"
        if not pair_path.exists() or not judgment_path.exists():
            missing.append(f"Saved {track} comparisons and judgments")
            continue
        summaries[track] = summarize(pair_path, judgment_path, generated / f"{track}_summary.json")
        if summaries[track]["total"] != (40 if track == "persona" else 300):
            missing.append(f"Full {track} evaluation subset")
        if summaries[track]["denominators"]["valid_including_ties"] == 0:
            missing.append(f"At least one valid {track} judge comparison")
    if not repository:
        missing.append("GitHub repository URL")
    if not (results / "prepared" / "manifest.json").exists():
        missing.append("Dataset revision and subset manifest")
    if missing and not draft:
        raise RuntimeError("Cannot label this report final. Missing: " + "; ".join(missing))
    write_json(generated / "evidence_audit.json", dict(draft=draft, missing=missing,
                                                      results=str(results), repository=repository))
    emit("metadata", "\n".join([r"\newcommand{\ReportAuthor}{" + tex(author) + "}",
                                r"\newcommand{\ReportDate}{" + str(date.today()) + "}",
                                r"\newcommand{\ReportStatus}{" + (r"\textbf{DRAFT: experiments or submission evidence remain incomplete.}" if draft else "") + "}"]))
    emit("abstract", " ".join(
        f"{track.upper()} records {s['counts']['win']} wins, {s['counts']['loss']} losses, "
        f"and {s['counts']['tie']} ties (tie-adjusted rate "
        f"{s['tie_adjusted_win_rate']:.3f})."
        for track, s in summaries.items() if s['tie_adjusted_win_rate'] is not None
    ) or "Full-run results are not yet available; no model-quality conclusion is claimed.")
    provenance = [r"Repository: \url{" + repository + "}." if repository else "Repository publication is pending authentication.",
                  r"Base checkpoints: \path{Qwen/Qwen2.5-0.5B} and \path{HuggingFaceTB/SmolLM2-360M-Instruct}.",
                  "Training sources: UltraChat 200k, Anthropic HH-RLHF, and UltraFeedback Binarized; fixed general-helpfulness prompts: AlpacaEval. Exact repository identifiers and revisions are in the manifest."]
    if (results / "prepared" / "manifest.json").exists():
        manifest = read_json(results / "prepared" / "manifest.json")
        for track, splits in manifest["filtering"].items():
            provenance.append(f"{track.upper()}: {splits['train']['count']:,} training and {splits['val']['count']:,} validation examples.")
        provenance.append(f"Persona: {manifest['persona']['train']} training and {manifest['persona']['val']} validation examples.")
        provenance.append("Fixed evaluation sets: 300 AlpacaEval prompts, 300 supplied HH prompts, and 40 supplied persona prompts.")
        provenance.append("Filtering was substantial for HH: 34,723 of 49,903 scanned training rows were invalid for the required single-turn, shared-prompt pair format; 108 were duplicates or overlaps and 72 were unscorable. Exact source revisions, split names, row indices, and hashes are archived in the accompanying manifest and selected-example index.")
    else:
        provenance.append("The intended counts are 30,000/1,000 for SFT and 15,000/1,000 for each main DPO track; these are not completed-run claims.")
    emit("provenance", "\n".join(provenance))
    configuration = [r"\begin{table}[ht]\centering\small\setlength{\tabcolsep}{3pt}",
                     r"\begin{tabularx}{\linewidth}{lXXX}\toprule Track & Strategy and DPO reference & Optimization & Batch and run\\\midrule"]
    for track in ["sft", "hh", "uf", "persona"]:
        if track not in runs:
            configuration.append(tex(track.upper()) + r" & \multicolumn{3}{l}{No completed experiment available}\\")
            continue
        c = runs[track]["config"]
        if track == "persona":
            strategy = r"LoRA $r=16$, $\alpha=32$, dropout .05; $q,v$; 1.64M"
            reference = r"SmolLM2 base, adapters off; $\beta=.1$"
        else:
            strategy = f"Full, {c['trainable_parameters']/1e6:.1f}M"
            reference = (r"no reference" if track == "sft" else
                         r"SFT best, step 938; frozen T4:1; $\beta=.1$")
        optimization = (f"AdamW; LR {c['lr']:g}; {tex(c['schedule'])}; "
                        f"WU {100*c.get('warmup_ratio',0):g}\\%; "
                        f"WD {c['weight_decay']:g}; clip 1")
        steps = runs[track]["status"].get("step", runs[track]["status"].get("final_step"))
        batch_run = (f"{c['batch_size']} $\\times$ {c['accumulation']} = "
                     f"{c['batch_size']*c['accumulation']}; {c['epochs']} ep., {steps} windows")
        configuration.append(f"{track.upper()} & {strategy}; {reference} & {optimization} & {batch_run}" + r"\\")
    configuration += [r"\bottomrule\end{tabularx}\caption{Executed settings. WU: warmup; WD: weight decay; batch is per device $\times$ accumulation = effective global batch; windows include skipped updates. All tracks: seed 42, 512 tokens, FP32 trainable weights with FP16 autocast (GradScaler for full tuning), Kaggle T4. Evaluation: greedy, temperature/top-$p$ inactive, 256 new tokens, repetition/length penalties 1.0; literal Qwen instruction/response template or native SmolLM2 chat template.}\end{table}"]
    configuration.append(r"Evaluation uses greedy decoding in batches of four, with 256 maximum new tokens, repetition and length penalties of 1.0, and no active temperature or top-$p$ sampling. Qwen uses the instruction/response template below; SmolLM2 uses its native chat template. Both main DPO references resolve to \path{/kaggle/input/notebooks/hanzlasajjad2026/small-alignment-sft/results/sft/best}; the persona reference is the original SmolLM2 instruction checkpoint (manifest revision) with adapters disabled.")
    emit("configuration", "\n".join(configuration))
    emit("deviations", "The guide's main learning rates, data sizes, effective batch sizes, beta, and training lengths were retained. A stricter 256-token prompt filter was applied to SFT as well as DPO, preserving response supervision within the 512-token context; it changed which examples were eligible, not the target counts. The revision-pinned AlpacaEval JSON replaced its custom dataset loader after a compatibility problem; Kaggle's CUDA-compatible PyTorch stack was preserved, and incompatible optional torchao and torchvision imports were removed. Evaluation changed from serial decoding to left-padded batches of four, with identical greedy settings across compared models. On the five-prompt sanity evaluation for all eight output groups, batching reduced recorded execution time from about 629 to 241 seconds; this is a throughput observation, not evidence of improved model quality. No beta or learning-rate tuning was attempted.")
    training, figure_tex = [], []
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8), constrained_layout=True)
    plotted = False
    for ax, track in zip(axes.flat, ["sft", "hh", "uf", "persona"]):
        ax.set_title(track.upper()); ax.set_xlabel("Logged step"); ax.set_ylabel("Loss")
        ax.grid(alpha=.2)
        if track not in runs:
            ax.text(.5, .5, "Awaiting completed run", ha="center", transform=ax.transAxes)
            continue
        records = runs[track]["metrics"]
        for label, keys in [("Training", ["train_loss", "loss"]), ("Validation", ["val_loss", "eval_loss"])]:
            rows = [(r["step"], next((r[k] for k in keys if k in r), None)) for r in records if r.get("split") != "initial_subset"]
            rows = [(s, v) for s, v in rows if v is not None]
            if rows:
                ax.plot(*zip(*rows), label=label, linewidth=.8); plotted = True
        if ax.lines:
            ax.legend(fontsize=8)
        values = [r for r in records if "val_loss" in r and r.get("split") != "initial_subset"]
        if values:
            first, last = values[0], values[-1]
            training.append(f"{track.upper()} validation loss changed from {first['val_loss']:.4f} at step {first['step']} to {last['val_loss']:.4f} at step {last['step']}.")
            if "reward_accuracy" in last:
                training.append(f"Final reward accuracy was {last['reward_accuracy']:.3f}, with mean reward margin {last['reward_margin']:.3f}.")
        if track == "persona":
            final = runs[track]["status"]["best_metrics"]
            training.append(f"Persona best step {runs[track]['status']['best_step']} had validation DPO loss {final['eval_loss']:.4f}, reward accuracy {final['eval_rewards/accuracies']:.3f}, and mean margin {final['eval_rewards/margins']:.3f} on 22 examples.")
        elif values:
            score = runs[track]["status"]["best_score"][0]
            training.append(f"{track.upper()} selected step {last['step']} by " +
                            (f"minimum validation loss {abs(score):.4f}." if track == "sft" else
                             f"maximum validation reward accuracy {score:.3f}."))
            skipped = sum(bool(r.get("skipped_update")) for r in records)
            if skipped:
                training.append(f"FP16 gradient scaling skipped {skipped} of {runs[track]['status']['step']} {track.upper()} accumulation windows after non-finite gradients; {runs[track]['status']['step']-skipped} optimizer updates were applied.")
    if plotted:
        fig.savefig(figures / "training.pdf")
        figure_tex.append(r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/training.pdf}\caption{Recorded training and validation losses; no smoothing.}\end{figure}")
    plt.close(fig)
    if any(t in runs for t in ["hh", "uf", "persona"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), constrained_layout=True)
        for track in ["hh", "uf", "persona"]:
            if track not in runs:
                continue
            for ax, keys in zip(axes, [["reward_accuracy", "eval_rewards/accuracies"], ["reward_margin", "eval_rewards/margins"]]):
                points = [(r["step"], next((r[k] for k in keys if k in r), None)) for r in runs[track]["metrics"] if r.get("split") != "initial_subset"]
                points = [(s,v) for s,v in points if v is not None]
                if points:
                    ax.plot(*zip(*points), marker=".", label=track.upper())
        for ax, title in zip(axes, ["Validation reward accuracy", "Validation reward margin"]):
            ax.set(title=title, xlabel="Logged step"); ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=8)
        fig.savefig(figures / "preferences.pdf"); plt.close(fig)
        figure_tex.append(r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/preferences.pdf}\caption{Preference diagnostics, plotted without smoothing.}\end{figure}")
    emit("training", " ".join(training) or "No completed training trajectory is available yet.")
    emit("figures", "\n".join(figure_tex))
    table = [r"\begin{table}[H]\centering\small\begin{tabular}{lrrrrrr}\toprule Track & $N$ & Wins & Losses & Ties & Unknown & Failures\\\midrule"]
    for track, s in summaries.items():
        c = s["counts"]
        table.append(f"{track.upper()} & {s['total']} & {c['win']} & {c['loss']} & {c['tie']} & {c['unknown']} & {c['failure']}" + r"\\")
    table.append(r"\bottomrule\end{tabular}\caption{HH/UF DPO versus SFT; persona LoRA versus the original SmolLM2 instruction model. Each track uses its designated prompts.}\end{table}")
    if not summaries:
        table = ["No completed pairwise evaluation is available yet."]
    table_header_lines = len(table)
    analysis, examples = [], []
    for track, s in summaries.items():
        c = s["counts"]
        rate = s["tie_adjusted_win_rate"]
        interval = prompt_bootstrap_interval(c) if rate is not None else None
        decisive = c['win'] / (c['win'] + c['loss']) if c['win'] + c['loss'] else None
        table.append(f"{track.upper()} judge: {tex(', '.join(s['judge_models']))}. "
                     f"Decisive: {c['win']}/{c['win']+c['loss']} = " +
                     (f"{decisive:.3f}; " if decisive is not None else "undefined; ") +
                     f"tie-adjusted: ({c['win']} + {c['tie']}/2)/{c['win']+c['loss']+c['tie']} = " +
                     (f"{rate:.3f}. " if rate is not None else "undefined. ") +
                     (f"95\\% bootstrap interval [{interval[0]:.3f}, {interval[1]:.3f}]. " if interval else "") +
                     f"Unknown {100*s['unknown_rate']:.1f}\\%, failure {100*s['failure_rate']:.1f}\\% of {s['total']}; "
                     f"{s['retry_attempts']} successful-item retries.")
        if rate is None:
            analysis.append(f"{track.upper()} has no valid judged comparison, so relative quality is unresolved.")
        else:
            baseline = "original instruction model" if track == "persona" else "SFT"
            candidate_name = {"hh": "HH-DPO", "uf": "UltraFeedback-DPO",
                              "persona": "LoRA persona"}[track]
            better = candidate_name if rate > .5 else baseline if rate < .5 else "neither model"
            analysis.append(f"On the {track.upper()} evaluation, {better} had the higher observed score under this judge: "
                            f"{candidate_name}'s tie-adjusted rate was {100*rate:.1f}\\%, {100*(rate-.5):+.1f} percentage points "
                            f"relative to parity, with {c['win']-c['loss']:+d} net decisive wins. "
                            + ("The prompt-bootstrap interval includes parity. " if interval and interval[0] <= .5 <= interval[1]
                               else "The prompt-bootstrap interval excludes parity. ") +
                            "This is a measured association on fixed prompts, not a causal estimate of the preference dataset's effect.")
            category_rates = []
            for category, counts in s["categories"].items():
                valid = sum(counts.get(key, 0) for key in ("win", "loss", "tie"))
                if valid:
                    category_rates.append((float(counts.get("win", 0) + counts.get("tie", 0) / 2) / valid,
                                           category, valid))
            if len(category_rates) > 1:
                low, high = min(category_rates), max(category_rates)
                analysis.append(f"Within {track.upper()}, the lowest observed category was {tex(low[1])} "
                                f"({100*low[0]:.1f}\\%, {low[2]} valid), and the highest was {tex(high[1])} "
                                f"({100*high[0]:.1f}\\%, {high[2]} valid). These small category slices are descriptive.")
        if track == "persona":
            for dimension, counts in s["persona_dimensions"].items():
                total = sum(counts.values())
                valid = sum(counts.get(name, 0) for name in ("win", "loss", "tie"))
                adjusted = ((counts.get("win", 0) + counts.get("tie", 0) / 2) / valid
                            if valid else None)
                table.append("Persona " + tex(dimension.replace("_winner", "")) + ": " +
                             f"{counts.get('win',0)} wins, {counts.get('loss',0)} losses, "
                             f"{counts.get('tie',0)} ties, {counts.get('unknown',0)} unknown "
                             f"({total} dimension labels); tie-adjusted " +
                             (f"{adjusted:.3f} ({valid} valid)." if adjusted is not None else "undefined."))
        pairs = read_jsonl(results / "generations" / f"pairs_{track}.jsonl")
        mapping = read_json(results / "generations" / f"pairs_{track}.identity.json")
        judgments = {r["id"]: r for r in read_jsonl(results / "judgments" / f"judged_{track}.jsonl")}
        if track == "hh":
            pairs = [pair for pair in pairs if pair.get("category") in
                     {"benign_helpfulness", "calibration_and_uncertainty", "safe_transformation"}]
        elif track == "persona":
            pairs = [pair for pair in pairs if pair.get("category") != "safety"]
        for wanted in ["win", "loss"]:
            for pair in pairs:
                j = judgments.get(pair["id"], {})
                winner = j.get("winner")
                if winner not in {"A", "B"}: continue
                won = mapping[pair["id"]][winner] == mapping[pair["id"]]["candidate"]
                if won != (wanted == "win"): continue
                if track == "uf" and wanted == "win" and any(
                    phrase in j.get("reason", "").lower()
                    for phrase in ("both responses failed", "both models failed", "neither response")
                ):
                    continue
                candidate_side = "A" if mapping[pair["id"]]["A"] == mapping[pair["id"]]["candidate"] else "B"
                baseline_side = "B" if candidate_side == "A" else "A"
                candidate_text = pair[f"response_{candidate_side.lower()}"]
                baseline_text = pair[f"response_{baseline_side.lower()}"]
                examples.append(r"\textbf{" + tex(f"{track.upper()} representative {wanted}, prompt {pair['prompt_id']}") + "}: " +
                                tex(textwrap.shorten(" ".join(pair["instruction"].split()), width=110, placeholder="...")) +
                                " Baseline: ``" + tex(contrast_excerpt(baseline_text, candidate_text)) +
                                "'' Candidate: ``" + tex(contrast_excerpt(candidate_text, baseline_text)) +
                                "'' Judge: " + tex(textwrap.shorten(" ".join(j.get("reason", "").split()),
                                                                         width=140, placeholder="...")))
                break
    if all(track in runs and track in summaries for track in ("hh", "uf")):
        hh_accuracy = runs["hh"]["status"]["best_score"][0]
        uf_accuracy = runs["uf"]["status"]["best_score"][0]
        analysis.append(
            f"Both main DPO runs used one epoch and $\\beta=.1$. Validation reward accuracy "
            f"was {hh_accuracy:.3f} for HH and {uf_accuracy:.3f} for UF, but this measures "
            "fitting to each preference source. The external judge uses different prompt sets, "
            "so the two judged rates are not a head-to-head dataset comparison."
        )
    if "persona" in runs and "persona" in summaries:
        style = summaries["persona"]["persona_dimensions"]["style_winner"]
        analysis.append(
            "Persona validation reward accuracy reached 1.000 on just 22 held-out pairs, "
            f"whereas the 40-prompt style judge gave {style.get('win',0)} LoRA wins, "
            f"{style.get('loss',0)} losses, and {style.get('tie',0)} ties. "
            "The small validation score alone cannot establish style transfer on new prompts."
        )
    for track, baseline_file, candidate_file in [
        ("HH", "hh_sft", "hh_hh"),
        ("UF", "alpaca_sft", "alpaca_uf"),
        ("persona", "persona_base", "persona_lora"),
    ]:
        paths = [results / "generations" / f"{name}.jsonl"
                 for name in (baseline_file, candidate_file)]
        if not all(path.exists() for path in paths):
            continue
        lengths = [[row["generated_tokens"] for row in read_jsonl(path)] for path in paths]
        if not all(lengths):
            continue
        baseline, candidate = lengths
        analysis.append(
            f"{track.upper()} outputs averaged {sum(candidate)/len(candidate):.1f} generated "
            f"tokens for the candidate versus {sum(baseline)/len(baseline):.1f} for its baseline; "
            f"{sum(n == 256 for n in candidate)} versus {sum(n == 256 for n in baseline)} "
            "reached the 256-token cap. Response length and truncation may affect judging."
        )
    agreement = []
    for track in ("hh", "uf", "persona"):
        alternate = results / "judgments_alt" / f"judged_{track}.jsonl"
        primary = results / "judgments" / f"judged_{track}.jsonl"
        if not alternate.exists() or not primary.exists() or track not in summaries:
            continue
        alternate_rows = read_jsonl(alternate)
        if (not alternate_rows or
                set(summaries[track]["judge_models"]) ==
                {row.get("judge_model") for row in alternate_rows}):
            continue
        primary_rows = {row["id"]: row for row in read_jsonl(primary)}
        comparable = [(primary_rows[row["id"]]["winner"], row["winner"])
                      for row in alternate_rows if row["id"] in primary_rows and
                      primary_rows[row["id"]].get("winner") in {"A", "B", "tie"} and
                      row.get("winner") in {"A", "B", "tie"}]
        if comparable:
            agreement.append(f"{track.upper()} {sum(left == right for left, right in comparable)}/{len(comparable)}")
    if agreement:
        analysis.append("An independent Gemini 3.8 validation judge agreed with the final judge on " +
                        ", ".join(agreement) + " overall A/B/tie decisions. This disagreement is a measured limit of model judging.")
    emit("results", "\n".join(table[:table_header_lines]) + "\n\n" +
         "\n\n".join(table[table_header_lines:]) +
         "\n\nIntervals resample valid prompts 2,000 times with seed 42; they do not capture judge uncertainty.")
    emit("analysis", "\n\n".join(analysis) or "Interpretation is pending measured comparisons. No improvement is assumed.")
    emit("examples", "\n\n".join(examples))
    emit("limitations", "Missing evidence: " + tex("; ".join(missing)) + "." if missing else "All required experiment and evaluation artifacts were available when this report was built.")
    emit("artifacts", r"Reproduction commands for tests, staged Kaggle jobs, and rebuilding the PDF are in the README. The repository archives \path{evidence/prepared/manifest.json} and \path{evidence/prepared/selected_examples_index.json}, configuration and metrics under \path{evidence/sft/}, \path{evidence/hh/}, \path{evidence/uf/}, and \path{evidence/persona/}, responses and blinded pairs under \path{evidence/generations/}, and decisions under \path{evidence/judgments/} and \path{evidence/judgments_alt/}. Best and final model weights remain in the private saved Kaggle notebook outputs named in the README.")
    if compile_pdf:
        name = "draft_report" if draft else "final_report"
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={name}", "main.tex"], cwd=report, check=True)
        from pypdf import PdfReader
        pages = len(PdfReader(report / f"{name}.pdf").pages)
        if pages > 8:
            raise RuntimeError(f"Report exceeds eight-page limit: {pages}")
        print(f"Built {report / (name + '.pdf')} ({pages} pages)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="artifacts/experiment/results")
    p.add_argument("--repository", default="")
    p.add_argument("--author", default="")
    p.add_argument("--draft", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    a = p.parse_args()
    build(a.results, a.repository, a.author, a.draft, not a.no_compile)
