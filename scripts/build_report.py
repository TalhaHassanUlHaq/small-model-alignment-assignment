"""Build an evidence-derived LaTeX report; final mode refuses missing experiments."""
import argparse
from datetime import date
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.io import read_json, read_jsonl, write_json
from alignment.judge import summarize


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c if ord(c) < 128 else " ") for c in str(value))


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
                  r"Base models: \texttt{Qwen2.5-0.5B} and \texttt{SmolLM2-360M-Instruct} (HuggingFaceTB)."]
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
    configuration = [r"\begin{table}[ht]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
                     r"\begin{tabularx}{\linewidth}{lXXXX}\toprule Track & Trainable parameters & Optimization & Batch and run & DPO reference\\\midrule"]
    for track in ["sft", "hh", "uf", "persona"]:
        if track not in runs:
            configuration.append(tex(track.upper()) + r" & \multicolumn{4}{l}{No completed experiment available}\\")
            continue
        c = runs[track]["config"]
        if track == "persona":
            strategy = r"LoRA $r=16$, $\alpha=32$, dropout .05; $q,v$ projections; 1.64M"
            reference = r"SmolLM2-360M-Instruct, adapters disabled; $\beta=.1$"
        else:
            strategy = f"Full; {c['trainable_parameters']/1e6:.1f}M"
            reference = (r"---" if track == "sft" else
                         r"SFT best (step 938), frozen on T4:1; $\beta=.1$")
        optimization = (f"AdamW; {c['lr']:g}; {tex(c['schedule'])}; "
                        f"warmup {100*c.get('warmup_ratio',0):g}\\%; "
                        f"WD {c['weight_decay']:g}; clip 1")
        steps = runs[track]["status"].get("step", runs[track]["status"].get("final_step"))
        batch_run = (f"{c['batch_size']} $\\times$ {c['accumulation']} = "
                     f"{c['batch_size']*c['accumulation']}; {c['epochs']} ep., {steps} logged steps")
        configuration.append(f"{track.upper()} & {strategy} & {optimization} & {batch_run} & {reference}" + r"\\")
    configuration += [r"\bottomrule\end{tabularx}\caption{Executed settings; batch is per device $\times$ accumulation = effective global batch. Logged steps count accumulation windows, including skipped updates. All tracks: seed 42, 512 tokens, FP32 trainable weights with FP16 autocast (GradScaler for full tuning), Kaggle T4 hardware.}\end{table}"]
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
    table = [r"\begin{table}[ht]\centering\small\begin{tabular}{lrrrrrr}\toprule Track & $N$ & Wins & Losses & Ties & Unknown & Failures\\\midrule"]
    for track, s in summaries.items():
        c = s["counts"]
        table.append(f"{track.upper()} & {s['total']} & {c['win']} & {c['loss']} & {c['tie']} & {c['unknown']} & {c['failure']}" + r"\\")
    table.append(r"\bottomrule\end{tabular}\caption{Candidate versus baseline, reported separately for each evaluation target.}\end{table}")
    if not summaries:
        table = ["No completed pairwise evaluation is available yet."]
    analysis, examples = [], []
    for track, s in summaries.items():
        c = s["counts"]
        rate = s["tie_adjusted_win_rate"]
        decisive = c['win'] / (c['win'] + c['loss']) if c['win'] + c['loss'] else None
        table.append(f"{track.upper()} judge: {tex(', '.join(s['judge_models']))}. "
                     f"Decisive: {c['win']}/{c['win']+c['loss']} = " +
                     (f"{decisive:.3f}; " if decisive is not None else "undefined; ") +
                     f"tie-adjusted: ({c['win']} + {c['tie']}/2)/{c['win']+c['loss']+c['tie']} = " +
                     (f"{rate:.3f}. " if rate is not None else "undefined. ") +
                     f"Unknown {100*s['unknown_rate']:.1f}\\%, failure {100*s['failure_rate']:.1f}\\% of {s['total']}; "
                     f"{s['retry_attempts']} successful-item retries.")
        if rate is None:
            analysis.append(f"{track.upper()} has no valid judged comparison, so relative quality is unresolved.")
        else:
            baseline = "original instruction model" if track == "persona" else "SFT"
            better = "candidate" if rate > .5 else baseline if rate < .5 else "neither model"
            analysis.append(f"On the {track.upper()} evaluation, {better} performed better under this judge: "
                            f"the candidate's tie-adjusted rate was {100*rate:.1f}\\%, {100*(rate-.5):+.1f} percentage points "
                            f"relative to parity, with {c['win']-c['loss']:+d} net decisive wins. "
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
        for wanted in ["win", "loss"]:
            for pair in pairs:
                j = judgments.get(pair["id"], {})
                winner = j.get("winner")
                if winner not in {"A", "B"}: continue
                won = mapping[pair["id"]][winner] == mapping[pair["id"]]["candidate"]
                if won != (wanted == "win"): continue
                candidate_side = "A" if mapping[pair["id"]]["A"] == mapping[pair["id"]]["candidate"] else "B"
                baseline_side = "B" if candidate_side == "A" else "A"
                candidate_text = pair[f"response_{candidate_side.lower()}"]
                baseline_text = pair[f"response_{baseline_side.lower()}"]
                examples.append(r"\textbf{" + tex(f"{track.upper()} representative {wanted}, prompt {pair['prompt_id']}") + "}: " +
                                tex(pair["instruction"][:110]) + " Baseline: ``" + tex(baseline_text[:110]) +
                                "'' Candidate: ``" + tex(candidate_text[:110]) + "'' Judge: " +
                                tex(j.get("reason", "")[:140]) + ".")
                break
    emit("results", "\n".join(table))
    emit("analysis", "\n".join(analysis) or "Interpretation is pending measured comparisons. No improvement is assumed.")
    emit("examples", "\n".join(examples))
    emit("limitations", "Missing evidence: " + tex("; ".join(missing)) + "." if missing else "All required experiment and evaluation artifacts were available when this report was built.")
    emit("artifacts", r"Reproduction commands for tests, staged Kaggle jobs, and rebuilding the PDF are in the README. The repository archives \path{evidence/prepared/manifest.json} and \path{evidence/prepared/selected_examples_index.json}, configuration and metrics under \path{evidence/sft/}, \path{evidence/hh/}, \path{evidence/uf/}, and \path{evidence/persona/}, responses and blinded pairs under \path{evidence/generations/}, and decisions under \path{evidence/judgments/}. Best and final model weights remain in the private saved Kaggle notebook outputs named in the README.")
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
