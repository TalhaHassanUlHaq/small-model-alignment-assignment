"""Copy compact, auditable experiment evidence into the private repository."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.io import write_json


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    required = [source / "prepared" / name for name in
                ("manifest.json", "eval_alpaca.json", "eval_hh.json", "eval_persona.json")]
    for track in ("sft", "hh", "uf", "persona"):
        required.extend(source / track / name for name in
                        ("config.json", "metrics.jsonl", "run_status.json",
                         "best/complete.json", "final/complete.json"))
    for track in ("hh", "uf", "persona"):
        required.extend((source / "generations" / f"pairs_{track}.jsonl",
                         source / "generations" / f"pairs_{track}.identity.json",
                         source / "judgments" / f"judged_{track}.jsonl",
                         source / "judgments" / f"judged_{track}.summary.json"))
    required.extend(source.glob("generations/*.jsonl"))
    required.extend(source.glob("generations/*.manifest.json"))
    required.extend(source.glob("judgments/*.jsonl"))
    required.extend(source.glob("judgments/*.json"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Evidence is incomplete: " + ", ".join(missing))
    index = {}
    for path in sorted(set(required)):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        payload = target.read_bytes()
        index[relative.as_posix()] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    write_json(destination / "index.json", index)
    print(f"Archived {len(index)} evidence files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=ROOT / "artifacts" / "experiment" / "results")
    parser.add_argument("--destination", default=ROOT / "evidence")
    args = parser.parse_args()
    archive(args.source, args.destination)
