"""From-scratch full-parameter SFT and DPO with exact accumulation weighting."""
import argparse
import contextlib
import importlib.metadata
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from alignment.core import build_sft_batch, build_dpo_batch, response_scores, dpo_scores
from alignment.io import read_json, write_json, append_jsonl, prompt_text, digest


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else contextlib.nullcontext()


def batch_rows(tokenizer, rows, track, max_length):
    prompts = [prompt_text(r["instruction"]) for r in rows]
    if track == "sft":
        return build_sft_batch(tokenizer, prompts, [r["response"] for r in rows], max_length)
    return build_dpo_batch(tokenizer, prompts, [r["chosen"] for r in rows], [r["rejected"] for r in rows], max_length)


def evaluate(model, reference, tokenizer, rows, config, track):
    model.eval()
    numerator, denominator, margin_sum, correct = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), config["batch_size"]):
            selected = rows[start:start + config["batch_size"]]
            batch = batch_rows(tokenizer, selected, track, config["max_length"])
            with autocast(next(model.parameters()).device):
                if track == "sft":
                    logps, counts = response_scores(model, batch)
                    numerator += -logps.sum().item()
                    denominator += counts.sum().item()
                else:
                    loss, margins = dpo_scores(model, reference, batch, config["beta"])
                    numerator += loss.item() * len(selected)
                    denominator += len(selected)
                    correct += (margins > 0).sum().item()
                    margin_sum += margins.sum().item()
    model.train()
    result = dict(val_loss=numerator / max(denominator, 1))
    if track != "sft":
        result.update(reward_accuracy=correct / len(rows), reward_margin=margin_sum / len(rows))
    return result


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, scaler, state, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "complete.json").unlink(missing_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    temporary = path / "training_state.tmp"
    torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(), rng=rng_state(), state=state,
                    config_digest=digest(config)), temporary)
    temporary.replace(path / "training_state.pt")
    write_json(path / "run_config.json", config)
    write_json(path / "complete.json", state)


def train(track, data, output, sft_checkpoint=None, resume=None, max_steps=None,
          config_path="configs/experiments.json", allow_cpu=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = read_json(config_path)
    config = {k: v for k, v in root.items() if not isinstance(v, dict)} | root[track]
    data = Path(data)
    manifest = read_json(data / "manifest.json")
    config.update(track=track, reference=str(sft_checkpoint) if sft_checkpoint else None,
                  data_manifest_digest=digest(manifest), max_steps_override=max_steps,
                  precision="fp32 weights; fp16 autocast and GradScaler", optimizer="AdamW",
                  schedule="cosine" if track == "sft" else "constant",
                  sft_reduction="mean over scored response tokens")
    seed_all(config["seed"])
    if not allow_cpu and torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs required for the assignment")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ref_device = "cuda:1" if torch.cuda.device_count() >= 2 else device
    if track != "sft" and not sft_checkpoint:
        raise ValueError("DPO requires the completed SFT best checkpoint")
    if sft_checkpoint and not (Path(sft_checkpoint) / "complete.json").exists():
        raise ValueError("SFT checkpoint is incomplete")
    initial = sft_checkpoint or config["model"]
    revision = manifest["model_revision"] if track == "sft" and not resume else None
    model = AutoModelForCausalLM.from_pretrained(resume or initial, revision=revision,
                                                torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    tokenizer = AutoTokenizer.from_pretrained(resume or initial, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = None
    if track != "sft":
        reference = AutoModelForCausalLM.from_pretrained(initial, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                                                        attn_implementation="sdpa").to(ref_device)
        reference.requires_grad_(False)
        reference.eval()
        reference.config.use_cache = False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    rows = read_json(data / f"{track}_train.json")
    validation = read_json(data / f"{track}_val.json")
    if len(rows) != config["train_size"] or len(validation) != config["val_size"]:
        raise ValueError("Data sizes differ from the experiment configuration")
    config.update(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  versions={n: importlib.metadata.version(n) for n in ["torch", "transformers", "datasets"]},
                  devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    write_json(out / "data_manifest.json", manifest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    effective = config["batch_size"] * config["accumulation"]
    steps = math.ceil(len(rows) / effective) * config["epochs"]
    warmup = max(1, math.ceil(steps * config.get("warmup_ratio", 0)))
    def lr_factor(step):
        if track != "sft":
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    state = dict(step=0, epoch=0, offset=0, best_score=None, elapsed_seconds=0.0)
    if resume:
        if not (Path(resume) / "complete.json").exists():
            raise ValueError("Resume checkpoint incomplete")
        saved = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        if saved["config_digest"] != digest(config):
            raise ValueError("Resume configuration differs from saved configuration")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        restore_rng(saved["rng"])
    start_time = time.monotonic()
    previous_elapsed = state["elapsed_seconds"]
    model.train()
    if not resume:
        initial_metrics = evaluate(model, reference, tokenizer, validation[:min(32, len(validation))], config, track)
        append_jsonl(out / "metrics.jsonl", dict(step=0, split="initial_subset", **initial_metrics))
    for epoch in range(state["epoch"], config["epochs"]):
        order = list(range(len(rows)))
        random.Random(config["seed"] + epoch).shuffle(order)
        offset = state["offset"] if state["epoch"] == epoch else 0
        for start in range(offset, len(rows), effective):
            window = [rows[i] for i in order[start:start + effective]]
            batches = [batch_rows(tokenizer, window[i:i + config["batch_size"]], track, config["max_length"])
                       for i in range(0, len(window), config["batch_size"])]
            tokens = sum(b["labels"][:, 1:].ne(-100).sum().item() for b in batches) if track == "sft" else 0
            optimizer.zero_grad(set_to_none=True)
            window_loss = 0.0
            for batch in batches:
                with autocast(device):
                    if track == "sft":
                        logps, _ = response_scores(model, batch)
                        loss = -logps.sum() / max(tokens, 1)
                    else:
                        raw, _ = dpo_scores(model, reference, batch, config["beta"])
                        loss = raw * batch["chosen"]["input_ids"].shape[0] / len(window)
                window_loss += loss.detach().item()
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if not skipped:
                scheduler.step()
            state.update(step=state["step"] + 1, epoch=epoch, offset=start + len(window),
                         elapsed_seconds=previous_elapsed + time.monotonic() - start_time)
            record = dict(step=state["step"], epoch=epoch, train_loss=window_loss,
                          lr=optimizer.param_groups[0]["lr"], gradient_norm=float(grad_norm),
                          skipped_update=skipped, elapsed_seconds=state["elapsed_seconds"])
            finish = state["offset"] >= len(rows) or (max_steps and state["step"] >= max_steps)
            if state["step"] % config["eval_every"] == 0 or finish:
                record.update(evaluate(model, reference, tokenizer, validation, config, track))
                score = [-record["val_loss"]] if track == "sft" else [record["reward_accuracy"], -record["val_loss"]]
                if state["best_score"] is None or score > state["best_score"]:
                    state["best_score"] = score
                    save_checkpoint(out / "best", model, tokenizer, optimizer, scheduler, scaler, state, config)
                # Latest resumable checkpoint is overwritten, bounding disk consumption.
                save_checkpoint(out / "final", model, tokenizer, optimizer, scheduler, scaler, state, config)
            append_jsonl(out / "metrics.jsonl", record)
            print(record, flush=True)
            if max_steps and state["step"] >= max_steps:
                write_json(out / "run_status.json", dict(status="smoke_complete", **state))
                return
        state.update(epoch=epoch + 1, offset=0)
    write_json(out / "run_status.json", dict(status="complete", **state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["sft", "hh", "uf"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--config", default="configs/experiments.json")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    train(args.track, args.data, args.output, args.sft_checkpoint, args.resume,
          args.max_steps, args.config, args.allow_cpu)
