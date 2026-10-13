"""Private, resumable Kaggle benchmark judging tasks from saved pairs."""
import argparse
import base64
import io
from pathlib import Path
import zipfile
from kaggle_control import api_client, ROOT


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
    judge = kbench.llms['google/gemini-3.6-flash']
    def call(prompt):
        with kbench.chats.new('independent_pair_batch'):
            return judge.prompt(prompt)
    for track in ['hh', 'uf', 'persona']:
        pairs = root / 'pairs' / f'pairs_{{track}}.jsonl'
        output = root / 'judgments' / f'judged_{{track}}.jsonl'
        run_judge(pairs, output, call, 'google/gemini-3.6-flash')
        summary = summarize(pairs, output, output.with_suffix('.summary.json'))
        print('SUMMARY', track, summary)
evaluate_saved_pairs.run(kbench.llms['google/gemini-3.6-flash'])
'''
    path = folder / "task.py"
    path.write_text(source, encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["push", "run", "status", "download", "logs"])
    p.add_argument("--task", default="small-alignment-judge-smoke")
    p.add_argument("--generations")
    p.add_argument("--previous")
    p.add_argument("--output", default="artifacts/benchmark")
    args = p.parse_args()
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
