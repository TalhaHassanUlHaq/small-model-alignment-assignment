"""Copy compact, auditable experiment evidence into the private repository."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.io import read_json, write_json


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    required = [source / "prepared" / name for name in
                ("manifest.json", "eval_alpaca.json", "eval_hh.json", "eval_persona.json",
                 "persona_train.json", "persona_val.json")]
    selection_paths = [source / "prepared" / f"{track}_{split}.json"
                       for track in ("sft", "hh", "uf") for split in ("train", "val")]
    for track in ("sft", "hh", "uf", "persona"):
        required.extend(source / track / name for name in
                        ("config.json", "metrics.jsonl", "run_status.json"))
        checkpoint_metadata = (("best/adapter_config.json", "final/adapter_config.json")
                               if track == "persona" else
                               ("best/complete.json", "final/complete.json"))
        required.extend(source / track / name for name in checkpoint_metadata)
    for track in ("hh", "uf", "persona"):
        required.extend((source / "generations" / f"pairs_{track}.jsonl",
                         source / "generations" / f"pairs_{track}.identity.json",
                         source / "judgments" / f"judged_{track}.jsonl",
                         source / "judgments" / f"judged_{track}.summary.json"))
    required.extend(source.glob("generations/*.jsonl"))
    required.extend(source.glob("generations/*.manifest.json"))
    required.extend(source.glob("judgments/*.jsonl"))
    required.extend(source.glob("judgments/*.json"))
    required.extend(source.glob("judgments_alt/*.jsonl"))
    required.extend(source.glob("judgments_alt/*.json"))
    missing = [str(path) for path in required + selection_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Evidence is incomplete: " + ", ".join(missing))
    index = {}
    for path in sorted(set(required)):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        target.write_bytes(payload)
        index[relative.as_posix()] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    selected = {path.stem: [dict(id=row["id"], source_index=row["source_index"])
                            for row in read_json(path)] for path in selection_paths}
    selected_path = destination / "prepared" / "selected_examples_index.json"
    write_json(selected_path, selected)
    payload = selected_path.read_bytes().replace(b"\r\n", b"\n")
    selected_path.write_bytes(payload)
    index[selected_path.relative_to(destination).as_posix()] = dict(
        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    index_path = destination / "index.json"
    write_json(index_path, index)
    index_path.write_bytes(index_path.read_bytes().replace(b"\r\n", b"\n"))
    print(f"Archived {len(index)} evidence files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=ROOT / "artifacts" / "experiment" / "results")
    parser.add_argument("--destination", default=ROOT / "evidence")
    args = parser.parse_args()
    archive(args.source, args.destination)
