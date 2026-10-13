"""Private Kaggle jobs. Credentials are read locally and never bundled."""
import argparse
import base64
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment.io import write_json


def api_client():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                if name.strip() in {"KAGGLE_API", "KAGGLE_API_TOKEN"}:
                    os.environ["KAGGLE_API_TOKEN"] = value.strip().strip('"').strip("'")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def source_bundle():
    # Explicit allowlist: never walk credentials, outputs, or arbitrary local files.
    paths = [ROOT / p for p in ["pyproject.toml", "requirements-kaggle.txt", "submission_adapter.py"]]
    for directory, suffixes in [("alignment", {".py"}), ("tests", {".py"}), ("configs", {".json"}),
                                ("data/persona_lora", {".json"}), ("eval_sets", {".json"})]:
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in suffixes)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for path in sorted(paths):
            name, contents = path.relative_to(ROOT).as_posix(), path.read_bytes()
            archive.writestr(name, contents)
            hashes[name] = hashlib.sha256(contents).hexdigest()
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(hashes, indent=2))
    return base64.b64encode(buffer.getvalue()).decode()


def prepare_job(stage, owner, sources=()):
    folder = ROOT / ".kaggle-build" / stage
    folder.mkdir(parents=True, exist_ok=True)
    bootstrap = f'''import base64, io, zipfile, pathlib, os, subprocess, sys
root = pathlib.Path('/kaggle/working/project')
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({source_bundle()!r}))) as z:
    z.extractall(root)
os.chdir(root)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao', 'torchvision'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-kaggle.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
subprocess.run([sys.executable, '-m', 'alignment.stage', '--stage', {stage!r}], check=True)
'''
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")),
                    cells=[dict(cell_type="code", id="run-stage", metadata={}, execution_count=None, outputs=[], source=bootstrap)])
    write_json(folder / "notebook.ipynb", notebook)
    title = "Small Alignment " + stage.replace("_", " ").title()
    slug = title.lower().replace(" ", "-")
    write_json(folder / "kernel-metadata.json", dict(id=f"{owner}/{slug}", title=title, code_file="notebook.ipynb",
               language="python", kernel_type="notebook", is_private=True, enable_gpu=stage not in {"data", "judge"},
               enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=list(sources)))
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "push", "status", "download", "prepare", "logs"])
    parser.add_argument("--stage", default="smoke")
    parser.add_argument("--owner")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--kernel")
    parser.add_argument("--output", default="artifacts/kaggle")
    parser.add_argument("--pattern", default="^results/")
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare_job(args.stage, args.owner or "talhahassanulhaq", args.source))
        return
    api = api_client()
    owner = args.owner or api.get_config_value("username")
    if args.action == "preflight":
        api.kernels_list(mine=True, page_size=1)
        print("Authenticated account:", owner)
        api.quota_view_cli()
        api.benchmarks_tasks_models_cli()
    elif args.action == "push":
        folder = prepare_job(args.stage, owner, args.source)
        response = api.kernels_push(str(folder), timeout="32400",
                                    acc="NvidiaTeslaT4" if args.stage not in {"data", "judge"} else None)
        if getattr(response, "error", None):
            print("Kaggle rejected the job:", response.error, file=sys.stderr)
            sys.exit(2)
        print("Kernel:", getattr(response, "url", None))
        print("Response:", str(response))
    elif args.action == "status":
        print(api.kernels_status(args.kernel))
    elif args.action == "download":
        api.kernels_output(args.kernel, path=args.output, file_pattern=args.pattern, force=True, page_size=100)
    elif args.action == "logs":
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
        from alignment.io import append_jsonl
        events, errors = [], []
        log_path = Path(args.output) / (args.kernel.split("/")[-1] + ".live.jsonl")
        def consume():
            try:
                with api.build_kaggle_client() as client:
                    request = ApiGetKernelSessionLogsStreamRequest()
                    request.user_name, request.kernel_slug = args.kernel.split("/")
                    request.wait_for_logs_url_seconds = 10
                    response = client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
                    if "text/event-stream" in response.headers.get("Content-Type", ""):
                        for line in response.iter_lines():
                            line = line.decode("utf-8") if isinstance(line, bytes) else line
                            if line.startswith("data:"):
                                try: entry = json.loads(line[5:])
                                except ValueError: continue
                                events.append(entry); append_jsonl(log_path, entry)
                    else:
                        entries = response.json()
                        if isinstance(entries, list):
                            for entry in entries:
                                events.append(entry); append_jsonl(log_path, entry)
            except Exception as error:
                errors.append(type(error).__name__)
        worker = threading.Thread(target=consume, daemon=True)
        worker.start(); worker.join(min(45, args.seconds))
        lines = [e.get("data", "") for e in events if isinstance(e, dict)]
        selected = [s for s in lines if any(k in s for k in ["train_loss", "Error", "Traceback", "SANITY", "Generated"])]
        print("".join((selected or lines)[-8:])[-5000:])
        print("Log events:", len(events), "Errors:", errors)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not render request headers or credential-bearing exception messages.
        print("Kaggle operation failed:", type(error).__name__, "status:", getattr(error, "status", None), file=sys.stderr)
        sys.exit(1)
