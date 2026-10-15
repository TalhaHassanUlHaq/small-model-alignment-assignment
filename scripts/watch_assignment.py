"""Record independent Kaggle run status every 30 minutes without changing jobs."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.io import append_jsonl
from kaggle_control import api_client


def snapshot(api, owner):
    controller_file = ROOT / "artifacts" / "controller.json"
    try:
        controller = json.loads(controller_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        controller = {}
    stage = controller.get("stage")
    row = {"checked_at": time.time(), "owner": owner, "controller_stage": stage,
           "controller_status": controller.get("status")}
    if stage:
        kernel = f"{owner}/small-alignment-{stage.replace('_', '-')}"
        try:
            response = api.kernels_status(kernel)
            row["kaggle_status"] = str(response.status).split(".")[-1].upper()
        except Exception as error:
            row["kaggle_error_type"] = type(error).__name__
    try:
        quota = api.quota_view().gpu_quota
        row["gpu_hours_remaining"] = round(max(0, (quota.total_time_allowed - quota.time_used).total_seconds() / 3600), 2)
    except Exception as error:
        row["quota_error_type"] = type(error).__name__
    for track in ("sft", "hh", "uf", "persona"):
        marker = ROOT / "artifacts" / "experiment" / "results" / track / "run_status.json"
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            continue
        if data.get("status") == "complete":
            row[f"{track}_complete"] = True
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="hanzlasajjad2026")
    parser.add_argument("--interval", type=int, default=1800)
    args = parser.parse_args()
    log = ROOT / "artifacts" / "assignment-30min-checks.jsonl"
    while True:
        try:
            row = snapshot(api_client(), args.owner)
        except Exception as error:
            row = {"checked_at": time.time(), "owner": args.owner,
                   "monitor_error_type": type(error).__name__}
        append_jsonl(log, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if (ROOT / "report" / "final_report.pdf").exists():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
