"""Private, resumable Kaggle benchmark judging tasks from saved pairs."""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile
from kaggle_control import api_client, ROOT
from alignment.io import append_jsonl, read_jsonl, write_json
from alignment.judge import parse_response, summarize


def allow_kaggle_validating_state():
    """Handle a server state not yet present in the current Kaggle SDK enum."""
    from kagglesdk.kaggle_object import EnumSerializer
    from kagglesdk.benchmarks.types.benchmark_enums import BenchmarkTaskVersionCreationState
    original = EnumSerializer._from_str

    def parse(cls, value):
        if (cls is BenchmarkTaskVersionCreationState and
                value == "BENCHMARK_TASK_VERSION_CREATION_STATE_VALIDATING"):
            return cls.BENCHMARK_TASK_VERSION_CREATION_STATE_QUEUED
        return original(cls, value)

    EnumSerializer._from_str = staticmethod(parse)


def prepare(task, generations=None, previous=None):
    folder = ROOT / ".kaggle-build" / task
    folder.mkdir(parents=True, exist_ok=True)
    if generations is None:
        source = '''import kaggle_benchmarks as kbench
@kbench.task(name="small-alignment-judge-smoke")
def check_judge(llm):
    response = llm.prompt('Return exactly the text READY.')
    print('JUDGE_PREFLIGHT_RESPONSE', response)
    kbench.assertions.assert_in('READY', response)
check_judge.run(kbench.llm)
'''
    else:
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename in ["__init__.py", "io.py", "judge.py"]:
                archive.writestr(f"alignment/{filename}", (ROOT / "alignment" / filename).read_bytes())
            for path in Path(generations).glob("pairs_*.jsonl"):
                archive.writestr("pairs/" + path.name, path.read_bytes())
            for path in Path(generations).glob("pairs_*.identity.json"):
                archive.writestr("pairs/" + path.name, path.read_bytes())
            if previous:
                for path in Path(previous).glob("judged_*"):
                    if path.suffix in {".json", ".jsonl"}:
                        archive.writestr("judgments/" + path.name, path.read_bytes())
        encoded = base64.b64encode(payload.getvalue()).decode()
        source = f'''import kaggle_benchmarks as kbench
import base64, io, zipfile, pathlib, sys
root = pathlib.Path('/kaggle/working/alignment_judge')
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as z:
    z.extractall(root)
sys.path.insert(0, str(root))
from alignment.judge import run_judge, summarize
@kbench.task(name={task!r})
def evaluate_saved_pairs(llm):
    def call(prompt):
        with kbench.chats.new('independent_pair_batch'):
            return llm.prompt(prompt)
    for track in ['hh', 'uf', 'persona']:
        pairs = root / 'pairs' / f'pairs_{{track}}.jsonl'
        output = root / 'judgments' / f'judged_{{track}}.jsonl'
        run_judge(pairs, output, call, 'benchmark_selected_model')
        summary = summarize(pairs, output, output.with_suffix('.summary.json'))
        print('SUMMARY', track, summary)
evaluate_saved_pairs.run(kbench.llm)
'''
    path = folder / "task.py"
    path.write_text(source, encoding="utf-8")
    return path


def recover_judgments(download, generations, output, task, previous=None):
    """Recover saved judge answers from Benchmark run conversations.

    The Benchmark download endpoint exports conversations and assertions, but
    does not export arbitrary files the task wrote under /kaggle/working.
    Every prompt and response is included in its run JSON, so those records
    are the durable evidence for reconstruction and resumption.
    """
    download, generations, output = map(Path, (download, generations, output))
    candidates = list(download.rglob(f"{task}*.run.json"))
    if not candidates:
        raise FileNotFoundError(f"No saved run JSON for {task} under {download}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in candidates]
    runs = [r for r in runs if r.get("modelVersion", {}).get("slug") == "google/gemini-3.6-flash"]
    if not runs:
        raise ValueError("No Gemini 3.6 Flash benchmark run was downloaded")
    run = max(runs, key=lambda r: r.get("endTime") or r.get("startTime") or "")
    model = run["modelVersion"]["slug"]
    expected = {}
    for track in ("hh", "uf", "persona"):
        for pair in read_jsonl(generations / f"pairs_{track}.jsonl"):
            if pair["id"] in expected:
                raise ValueError(f"Duplicate pair ID across tracks: {pair['id']}")
            expected[pair["id"]] = track
    recovered, requests_seen, invalid = {}, {}, []
    if previous:
        for track in ("hh", "uf", "persona"):
            for row in read_jsonl(Path(previous) / f"judged_{track}.jsonl"):
                if row["id"] not in expected or expected[row["id"]] != track:
                    raise ValueError("Previous judgments contain an unexpected pair ID")
                recovered[row["id"]] = row
    for conversation in run.get("conversations", []):
        for request in conversation.get("requests", []):
            contents = request.get("contents", [])
            prompts = [part.get("text", "") for content in contents
                       if content.get("role") == "CONTENT_ROLE_USER" for part in content.get("parts", [])]
            answers = [part.get("text", "") for content in contents
                       if content.get("role") == "CONTENT_ROLE_ASSISTANT" for part in content.get("parts", [])]
            if not prompts or "DATA:\n" not in prompts[-1]:
                continue
            visible = json.loads(prompts[-1].split("DATA:\n", 1)[1])
            ids = [item["id"] for item in visible]
            if any(pair_id not in expected for pair_id in ids):
                raise ValueError("Benchmark response contains an unexpected pair ID")
            for pair_id in ids:
                requests_seen[pair_id] = requests_seen.get(pair_id, 0) + 1
            if not answers:
                invalid.append(dict(request=request.get("id"), ids=ids, error="missing_response"))
                continue
            try:
                parsed = parse_response(answers[-1])
            except (ValueError, KeyError, TypeError) as error:
                invalid.append(dict(request=request.get("id"), ids=ids, error=type(error).__name__))
                continue
            for pair_id in ids:
                result = parsed.get(pair_id, {})
                required = ["winner"]
                if expected[pair_id] == "persona":
                    required += ["style_winner", "usefulness_winner", "safety_winner"]
                if not all(result.get(key) in {"A", "B", "tie", "unknown"} for key in required):
                    invalid.append(dict(request=request.get("id"), ids=[pair_id], error="invalid_judgment"))
                    continue
                recovered[pair_id] = dict(result, judge_model=model, attempts=requests_seen[pair_id])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "recovery_audit.json", dict(task=task, model=model, run_state=run.get("state"),
                                                     expected=len(expected), recovered=len(recovered),
                                                     requests_seen=requests_seen, invalid=invalid))
    summaries = {}
    for track in ("hh", "uf", "persona"):
        pairs = generations / f"pairs_{track}.jsonl"
        judgments = output / f"judged_{track}.jsonl"
        judgments.write_text("", encoding="utf-8")
        for pair in read_jsonl(pairs):
            if pair["id"] in recovered:
                append_jsonl(judgments, recovered[pair["id"]])
        summaries[track] = summarize(pairs, judgments, output / f"judged_{track}.summary.json")
    return summaries


def main():
    allow_kaggle_validating_state()
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["push", "run", "status", "download", "logs", "recover"])
    p.add_argument("--task", default="small-alignment-judge-smoke")
    p.add_argument("--generations")
    p.add_argument("--previous")
    p.add_argument("--output", default="artifacts/benchmark")
    p.add_argument("--download", default="artifacts/benchmark")
    args = p.parse_args()
    if args.action == "recover":
        if not args.generations:
            p.error("recover requires --generations")
        print(recover_judgments(args.download, args.generations, args.output, args.task, args.previous))
        return
    api = api_client()
    if args.action == "push":
        api.benchmarks_tasks_push_cli(args.task, str(prepare(args.task, args.generations, args.previous)))
    elif args.action == "run":
        api.benchmarks_tasks_run_cli(args.task, model=["gemini-3.6-flash"])
    elif args.action == "status":
        api.benchmarks_tasks_status_cli(args.task)
    elif args.action == "download":
        api.benchmarks_tasks_download_cli(args.task, output=args.output, force=True)
    elif args.action == "logs":
        api.benchmarks_tasks_log_cli(args.task)


if __name__ == "__main__":
    main()
