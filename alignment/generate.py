"""Deterministic saved generations and identity-blinded comparisons."""
import gc
import random
from pathlib import Path

from alignment.io import read_json, write_json, append_jsonl, read_jsonl, prompt_text, digest


def generate(model_path, prompts, output, generator, config, revision=None, persona=False, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, torch_dtype=torch.float16).to("cuda:0")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    output = Path(output)
    signature = digest(dict(model=str(model_path), revision=revision, adapter=str(adapter), config=config,
                            prompts=prompts, generator=generator))
    manifest = output.with_suffix(".manifest.json")
    if manifest.exists() and read_json(manifest)["signature"] != signature:
        raise ValueError("Generation settings changed; use a new output path")
    write_json(manifest, dict(signature=signature, model=str(model_path), revision=revision,
                             adapter=str(adapter), settings=config, expected_count=len(prompts)))
    completed = {r["id"] for r in read_jsonl(output)}
    batch_size = config.get("batch_size", 1)
    if batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            pending = [(i, row) for i, row in enumerate(prompts[start:start+batch_size], start)
                       if row["id"] not in completed]
            if not pending:
                continue
            encoded = []
            for _, row in pending:
                if persona:
                    formatted = tokenizer.apply_chat_template(
                        [dict(role="user", content=row["instruction"])],
                        tokenize=False, add_generation_prompt=True)
                else:
                    formatted = prompt_text(row["instruction"])
                # Keep the response prefix when the evaluation prompt is long.
                encoded.append(tokenizer.encode(formatted, add_special_tokens=False)[-256:])
            width = max(map(len, encoded))
            token_ids = [[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in encoded]
            masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded]
            ids = torch.tensor(token_ids, device="cuda:0")
            attention = torch.tensor(masks, device="cuda:0")
            result = model.generate(input_ids=ids, attention_mask=attention,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    do_sample=False, max_new_tokens=config["max_new_tokens"],
                                    repetition_penalty=config["repetition_penalty"])
            for (i, row), sequence in zip(pending, result):
                new_ids = sequence[width:].tolist()
                response = tokenizer.decode(new_ids, skip_special_tokens=True)
                if not response.strip():
                    print("WARNING: empty generation", row["id"], flush=True)
                eos = tokenizer.eos_token_id
                generated_tokens = new_ids.index(eos) + 1 if eos in new_ids else len(new_ids)
                append_jsonl(output, dict(row, output=response, generator=generator,
                                          generated_tokens=generated_tokens))
                if i < 5:
                    print("SANITY", generator, row["id"], repr(response), flush=True)
            if (start + len(pending)) % 25 == 0 or start + batch_size >= len(prompts):
                print("Generated", generator, min(start+batch_size, len(prompts)), "/", len(prompts), flush=True)
    rows = read_jsonl(output)
    if len(rows) != len(prompts) or len({r["id"] for r in rows}) != len(prompts):
        raise ValueError("Generation count mismatch")
    write_json(output.with_suffix(".json"), rows)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def pair_outputs(baseline_path, candidate_path, output, seed=42):
    baseline = read_jsonl(baseline_path)
    candidate = {r["id"]: r for r in read_jsonl(candidate_path)}
    if set(candidate) != {r["id"] for r in baseline}:
        raise ValueError("Compared models must have exactly matching prompt IDs")
    rng = random.Random(seed)
    pairs, mapping = [], {}
    for base in baseline:
        other = candidate[base["id"]]
        if base["instruction"] != other["instruction"]:
            raise ValueError("Mismatched instructions")
        swap = rng.choice([False, True])
        a, b = (other, base) if swap else (base, other)
        pair_id = digest([base["id"], base["generator"], other["generator"], base["output"], other["output"]])[:24]
        pair = dict(id=pair_id, prompt_id=base["id"], instruction=base["instruction"],
                    response_a=a["output"], response_b=b["output"], dataset=base["dataset"],
                    category=base.get("category", "general"))
        pairs.append(pair)
        mapping[pair_id] = dict(A=a["generator"], B=b["generator"], baseline=base["generator"], candidate=other["generator"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    for pair in pairs:
        append_jsonl(output, pair)
    write_json(output.with_suffix(".identity.json"), mapping)


def generate_all(data, output, locate, sanity_only=False):
    data, out = Path(data), Path(output)
    cfg = read_json("configs/experiments.json")
    manifest = read_json(data / "manifest.json")
    sft = locate("sft/best/complete.json").parent
    for evaluation, track in [("alpaca", "uf"), ("hh", "hh")]:
        prompts = read_json(data / f"eval_{evaluation}.json")
        if sanity_only:
            prompts = prompts[:5]
        for name, model, revision in [("base", cfg["sft"]["model"], manifest["model_revision"]),
                                     ("sft", sft, None), (track, locate(f"{track}/best/complete.json").parent, None)]:
            generate(model, prompts, out / f"{evaluation}_{name}.jsonl", name, cfg["generation"], revision)
        pair_outputs(out / f"{evaluation}_sft.jsonl", out / f"{evaluation}_{track}.jsonl", out / f"pairs_{track}.jsonl")
    persona_path = locate("persona/best/adapter_config.json").parent
    prompts = read_json(data / "eval_persona.json")
    if sanity_only:
        prompts = prompts[:5]
    for name, adapter in [("persona_base", None), ("persona_lora", persona_path)]:
        generate(cfg["persona"]["model"], prompts, out / f"{name}.jsonl", name, cfg["generation"],
                 manifest["persona_model_revision"], persona=True, adapter=adapter)
    pair_outputs(out / "persona_base.jsonl", out / "persona_lora.jsonl", out / "pairs_persona.jsonl")
