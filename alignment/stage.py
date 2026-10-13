"""Kaggle stage entrypoint; upstream artifacts are explicit notebook inputs."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys

from alignment.io import read_json, write_json


def upstream(relative):
    candidates = list(Path("/kaggle/input").glob(f"**/{relative}"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one upstream {relative}; found {len(candidates)}")
    return candidates[0]


def smoke(out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from alignment.core import build_sft_batch, response_scores, build_dpo_batch, dpo_scores
    from alignment.train import autocast
    if torch.cuda.device_count() != 2 or not all("T4" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Expected exactly two T4 GPUs")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32).to("cuda:0")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16).to("cuda:1")
    ref.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    prompts = ["### Instruction:\nExplain photosynthesis.\n\n### Response:\n"] * 2
    # Exercise the full 512-token memory envelope, not only tiny prompts.
    responses = ["Plants convert light into chemical energy. " * 100] * 2
    batch = build_sft_batch(tokenizer, prompts, responses, 512)
    with autocast("cuda:0"):
        scores, counts = response_scores(model, batch)
        loss = -scores.sum() / counts.sum()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    sft_loss = float(loss.detach())
    pair = build_dpo_batch(tokenizer, prompts[:1], responses[:1], ["Plants get energy from rocks. " * 100], 512)
    with autocast("cuda:0"):
        loss, margin = dpo_scores(model, ref, pair, 0.1)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    scaler.step(optimizer); scaler.update()
    write_json(out / "smoke.json", dict(status="complete", sft_loss=sft_loss, dpo_loss=float(loss.detach()),
               gpus=[torch.cuda.get_device_name(i) for i in range(2)],
               peak_memory_gb=[torch.cuda.max_memory_allocated(i)/1e9 for i in range(2)],
               versions={n: importlib.metadata.version(n) for n in ["torch", "transformers", "datasets", "trl", "peft"]}))


def main(stage):
    out = Path("/kaggle/working/results")
    out.mkdir(parents=True, exist_ok=True)
    if stage == "smoke":
        smoke(out)
    elif stage == "data":
        from alignment.data import prepare
        prepare(out / "prepared")
    elif stage in {"sft", "hh", "uf"}:
        from alignment.train import train
        data = upstream("prepared/manifest.json").parent
        sft = upstream("sft/best/complete.json").parent if stage != "sft" else None
        train(stage, data, out / stage, sft_checkpoint=sft)
    elif stage == "persona":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from alignment.persona import train_persona
        train_persona(upstream("prepared/manifest.json").parent, out / "persona")
    elif stage in {"generate", "generation_sanity"}:
        from alignment.generate import generate_all
        generate_all(upstream("prepared/manifest.json").parent,
                     out / ("sanity_generations" if stage == "generation_sanity" else "generations"),
                     upstream, sanity_only=stage == "generation_sanity")
    else:
        raise ValueError(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    main(args.stage)
