"""Numerical core. No Transformers/TRL dependency; usable by the offline grader."""
import torch
import torch.nn.functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if max_length < 1 or len(prompts) != len(responses):
        raise ValueError("Positive max_length and matching input lengths required")
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = pad if pad is not None else (eos if eos is not None else 0)
    rows, labels = [], []
    for prompt, response in zip(prompts, responses):
        p = list(tokenizer.encode(prompt, add_special_tokens=False))
        r = list(tokenizer.encode(response, add_special_tokens=False))
        if eos is not None:
            r.append(eos)
        rows.append((p + r)[:max_length])
        labels.append(([-100] * len(p) + r)[:max_length])
    width = max((len(row) for row in rows), default=0)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    target = torch.full_like(ids, -100)
    for i, (row, lab) in enumerate(zip(rows, labels)):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        target[i, :len(row)] = torch.tensor(lab, dtype=torch.long)
        mask[i, :len(row)] = 1
    return dict(input_ids=ids, attention_mask=mask, labels=target)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def response_scores(model, batch):
    """Return summed causal log-probabilities and scored-token counts per row."""
    device = next(model.parameters()).device
    batch = to_device(batch, device)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    target = batch["labels"][:, 1:]
    mask = target.ne(-100)
    # FP32 cross entropy avoids half-precision log-softmax underflow and retains gradients.
    if logits.shape[1] < 2:
        return logits.sum(dim=(1, 2)) * 0, mask.sum(dim=1)
    nll = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), target,
                          reduction="none", ignore_index=-100)
    return -nll.sum(dim=1), mask.sum(dim=1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return response_scores(model, build_sft_batch(tokenizer, prompts, responses, max_length))[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    logps, counts = response_scores(model, build_sft_batch(tokenizer, prompts, responses, max_length))
    return -logps.sum() / counts.sum().clamp_min(1)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {"chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
            "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length)}


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                               reference_chosen_logps, reference_rejected_logps, beta):
    gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = (reference_chosen_logps - reference_rejected_logps).detach().to(gap.device)
    return -F.logsigmoid(beta * (gap - reference_gap)).mean()


def dpo_scores(policy, reference, batch, beta):
    pc, _ = response_scores(policy, batch["chosen"])
    pr, _ = response_scores(policy, batch["rejected"])
    # Preserve caller state while making reference scores deterministic.
    was_training = reference.training
    reference.eval()
    try:
        with torch.no_grad():
            rc, _ = response_scores(reference, batch["chosen"])
            rr, _ = response_scores(reference, batch["rejected"])
    finally:
        reference.train(was_training)
    loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    margin = beta * ((pc - pr) - (rc - rr).to(pc.device))
    return loss, margin


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return dpo_scores(policy_model, reference_model, batch, beta)[0]
