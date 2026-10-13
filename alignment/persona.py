"""The only track using TRL's trainer; base model remains the frozen reference."""
import argparse
from pathlib import Path

from alignment.io import read_json, write_json, append_jsonl


def train_persona(data, output, config_path="configs/experiments.json", max_steps=-1):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer
    from alignment.train import seed_all
    root = read_json(config_path)
    cfg = root["persona"]
    seed_all(root["seed"])
    manifest = read_json(Path(data) / "manifest.json")
    revision = manifest["persona_model_revision"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=revision, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    def dataset(split):
        rows = read_json(Path(data) / f"persona_{split}.json")
        return Dataset.from_list([dict(prompt=[dict(role="user", content=r["prompt"])],
                                       chosen=[dict(role="assistant", content=r["chosen"])],
                                       rejected=[dict(role="assistant", content=r["rejected"])]) for r in rows])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    class Logger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                append_jsonl(out / "metrics.jsonl", dict(step=state.global_step, **logs))
    arguments = DPOConfig(output_dir=str(out / "trainer"), learning_rate=cfg["lr"],
                          per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=2,
                          gradient_accumulation_steps=cfg["accumulation"], num_train_epochs=cfg["epochs"],
                          max_steps=max_steps, beta=cfg["beta"], max_length=root["max_length"],
                          max_prompt_length=root["max_prompt_length"], fp16=torch.cuda.is_available(),
                          gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                          eval_strategy="epoch", save_strategy="epoch", save_total_limit=3,
                          load_best_model_at_end=False, logging_steps=1, report_to="none",
                          lr_scheduler_type="constant", warmup_ratio=0, weight_decay=0,
                          max_grad_norm=1, seed=root["seed"], data_seed=root["seed"],
                          remove_unused_columns=False, disable_tqdm=True)
    lora = LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                      target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, ref_model=None, args=arguments, processing_class=tokenizer,
                         train_dataset=dataset("train"), eval_dataset=dataset("val"), peft_config=lora,
                         callbacks=[Logger()])
    write_json(out / "config.json", dict(cfg, seed=root["seed"], max_length=root["max_length"],
               max_prompt_length=root["max_prompt_length"], reference=cfg["model"], revision=revision,
               precision="FP32 adapter weights, FP16 autocast", optimizer="AdamW", schedule="constant",
               gradient_clip=1, trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
    trainer.train()
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(out / "final")
    evaluations = [r for r in trainer.state.log_history if "eval_rewards/accuracies" in r]
    if not evaluations:
        metrics = trainer.evaluate()
        evaluations = [dict(step=trainer.state.global_step, **metrics)]
    best = max(evaluations, key=lambda r: (r["eval_rewards/accuracies"], -r["eval_loss"]))
    # Retain all three epoch checkpoints until best selection.
    checkpoint = out / "trainer" / f"checkpoint-{best['step']}"
    if checkpoint.exists():
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=revision)
        best_model = PeftModel.from_pretrained(base, checkpoint)
        best_model.save_pretrained(out / "best")
    else:
        if best["step"] != trainer.state.global_step:
            raise RuntimeError("Best persona checkpoint was not retained")
        trainer.save_model(str(out / "best"))
    tokenizer.save_pretrained(out / "best")
    write_json(out / "run_status.json", dict(status="complete" if max_steps < 0 else "smoke_complete",
               best_step=best["step"], final_step=trainer.state.global_step, best_metrics=best))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=-1)
    a = p.parse_args()
    train_persona(a.data, a.output, max_steps=a.max_steps)
