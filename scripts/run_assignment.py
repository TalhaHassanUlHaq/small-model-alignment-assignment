"""Sequential job controller. Stops on real failures; never cancels unrelated jobs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from requests.exceptions import RequestException

from kaggle_control import api_client, prepare_job, ROOT
from alignment.io import read_json, write_json


def status_name(response):
    return str(response.status).split(".")[-1].upper()


def record_controller_status(path, value):
    """Keep monitoring when Windows briefly locks this advisory status file."""
    try:
        write_json(path, value)
    except PermissionError:
        print("Controller status file is temporarily locked; continuing Kaggle work", flush=True)


def retry_transient(operation, label, controller=None):
    """Retry only transport failures without treating a Kaggle job as failed."""
    attempts = 0
    while True:
        try:
            return operation()
        except RequestException as error:
            attempts += 1
            delay = min(60, 5 * 2 ** min(attempts - 1, 4))
            if controller is not None:
                record_controller_status(controller, dict(status="WAITING_FOR_NETWORK", operation=label,
                                                          retry=attempts, updated_at=time.time()))
            print(f"Temporary network failure during {label}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def owned_kernel_refs(api, controller=None):
    """Return the current user's kernel references.

    Kaggle returns a misleading ``kernels.get`` permission error for a private
    notebook slug that has not been created.  Looking up the account's own
    notebook list first lets the controller distinguish that normal case from
    a real status-query failure on a notebook that does exist.
    """
    kernels = retry_transient(lambda: api.kernels_list(mine=True, page_size=100), "kernel list", controller)
    return {str(kernel.ref).lower() for kernel in kernels}


def run(repository, owner="talhahassanulhaq", start="sft", stop_after=None):
    api = api_client()
    artifacts = ROOT / "artifacts" / "experiment"
    controller = ROOT / "artifacts" / "controller.json"
    stages = ["sft", "hh", "uf", "persona", "generation_sanity", "generate"]
    if start not in stages:
        raise ValueError(start)
    known_kernels = owned_kernel_refs(api, controller)
    for stage in stages[stages.index(start):]:
        kernel = f"{owner}/small-alignment-{stage.replace('_','-')}"
        sources = [f"{owner}/small-alignment-data"]
        if stage in {"hh", "uf"}:
            sources += [f"{owner}/small-alignment-sft"]
        elif stage in {"generation_sanity", "generate"}:
            sources += [f"{owner}/small-alignment-{track}" for track in ["sft", "hh", "uf", "persona"]]
        if kernel.lower() not in known_kernels:
            status = "NOT_FOUND"
        else:
            try:
                status = status_name(retry_transient(lambda: api.kernels_status(kernel),
                                                     f"status for {stage}", controller))
            except Exception:
                # This kernel was listed as owned, so an access failure here
                # is real and must stop the controller for diagnosis.
                raise
        if status == "ERROR":
            api.kernels_output(kernel, str(artifacts), file_pattern="^$", force=True, page_size=100)
            raise RuntimeError(f"Existing {stage} job failed; diagnose it before resubmitting")
        if status == "NOT_FOUND":
            quota = retry_transient(lambda: api.quota_view().gpu_quota, "GPU quota check", controller)
            remaining = (quota.total_time_allowed - quota.time_used).total_seconds() / 3600
            required = {"sft": 4, "hh": 4, "uf": 4, "persona": 1, "generation_sanity": .5, "generate": 2}[stage]
            if remaining < required:
                raise RuntimeError(f"Free quota too low for {stage}: {remaining:.2f} hours remain")
            folder = prepare_job(stage, owner, sources)
            while True:
                response = retry_transient(lambda: api.kernels_push(str(folder), timeout="32400", acc="NvidiaTeslaT4"),
                                           f"start {stage}", controller)
                if not response.error:
                    print("Started", stage, response.url, flush=True)
                    known_kernels.add(kernel.lower())
                    break
                if "Maximum batch GPU session count" not in response.error:
                    raise RuntimeError(response.error)
                print("Waiting for a free Kaggle GPU session; no other jobs will be cancelled", flush=True)
                time.sleep(60)
            status = "RUNNING"
        while status not in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            record_controller_status(controller, dict(stage=stage, status=status, kernel=kernel,
                                                      updated_at=time.time()))
            print(stage, status, flush=True)
            time.sleep(60)
            status = status_name(retry_transient(lambda: api.kernels_status(kernel),
                                                 f"status for {stage}", controller))
        if status != "COMPLETE":
            api.kernels_output(kernel, str(artifacts), file_pattern="^$", force=True, page_size=100)
            raise RuntimeError(f"{stage} ended with {status}")
        # Preserve large weights remotely; download all evidence needed for analysis/report.
        pattern = r"^results/.*\.(json|jsonl)$"
        api.kernels_output(kernel, str(artifacts), file_pattern=pattern, force=True, page_size=100)
        record_controller_status(controller, dict(stage=stage, status="COMPLETE", kernel=kernel,
                                                  updated_at=time.time()))
        print("Completed and retrieved", stage, flush=True)
        if stage == "generation_sanity":
            # A manual sample inspection is explicitly required by the assignment.
            print("Inspect artifacts/experiment/results/sanity_generations before full generation.", flush=True)
            return
        if stage == stop_after:
            return
    print("Generations ready. Run the private judging task, retrieve its judgments, then build the report.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="https://github.com/TalhaHassanUlHaq/small-model-alignment-assignment")
    parser.add_argument("--owner", default="talhahassanulhaq")
    parser.add_argument("--start", default="sft")
    parser.add_argument("--stop-after")
    args = parser.parse_args()
    try:
        run(args.repository, args.owner, args.start, args.stop_after)
    except Exception as error:
        record_controller_status(ROOT / "artifacts" / "controller.json",
                                 dict(status="BLOCKED", error_type=type(error).__name__,
                                      message=str(error) if isinstance(error, (RuntimeError, ValueError))
                                      else "External API request failed"))
        raise
