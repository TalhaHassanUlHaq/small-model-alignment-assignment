import math
from types import SimpleNamespace

import torch
from torch import nn

from alignment import core
from tests.toy_lm import ToyTokenizer, ToyCausalLM


class ContextLM(nn.Module):
    """Different logits at each position catches accidental unshifted scoring."""
    def __init__(self, vocab):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(vocab * vocab, dtype=torch.float32).reshape(vocab, vocab).sin())

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.weight[input_ids])


def test_logprobs_match_independent_token_loop():
    tok = ToyTokenizer()
    model = ContextLM(tok.vocab_size)
    prompts, responses = ["a: ", "abc: "], ["zy", "x"]
    actual = core.compute_response_logprobs(model, tok, prompts, responses, 20)
    expected = []
    for p, r in zip(prompts, responses):
        prefix = tok.encode(p)
        continuation = tok.encode(r) + [tok.eos_token_id]
        full = prefix + continuation
        score = 0.0
        for position in range(len(prefix), len(full)):
            logits = model.weight[full[position - 1]].double()
            score += logits[full[position]] - torch.logsumexp(logits, dim=0)
        expected.append(score)
    torch.testing.assert_close(actual.double(), torch.stack(expected), rtol=1e-6, atol=1e-6)


def test_eos_pad_collision_and_truncation():
    tok = ToyTokenizer()
    tok.pad_token_id = tok.eos_token_id
    batch = core.build_sft_batch(tok, ["ab", "a"], ["xyz", "x"], 4)
    assert batch["labels"][0].tolist() == [-100, -100, tok.vocab["x"], tok.vocab["y"]]
    assert batch["labels"][1].tolist() == [-100, tok.vocab["x"], 1, -100]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0]


def test_no_eos_or_pad_and_all_masked_loss():
    tok = ToyTokenizer()
    tok.eos_token_id = None
    tok.pad_token_id = None
    model = ToyCausalLM(tok.vocab_size)
    batch = core.build_sft_batch(tok, ["abcd", "ab"], ["x", ""], 2)
    assert batch["labels"].eq(-100).all()
    loss = core.compute_sft_loss(model, tok, ["abcd"], ["x"], 2)
    assert loss.item() == 0
    loss.backward()
    assert model.logit_bias.grad.eq(0).all()


def test_row_order_and_empty_response_eos():
    tok = ToyTokenizer()
    model = ContextLM(tok.vocab_size)
    p, r = ["a", "b", "c"], ["", "xyz", "q"]
    scores = core.compute_response_logprobs(model, tok, p, r, 20)
    for i in range(3):
        torch.testing.assert_close(scores[i], core.compute_response_logprobs(model, tok, p[i:i+1], r[i:i+1], 20)[0])


def test_dpo_extremes_and_sign_and_reference_detached():
    pc = torch.tensor([10000., -10000., 2.], requires_grad=True)
    pr = torch.zeros(3, requires_grad=True)
    rc = torch.zeros(3, requires_grad=True)
    rr = torch.zeros(3, requires_grad=True)
    loss = core.compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.1)
    expected = torch.logaddexp(torch.zeros(3), -0.1 * pc).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(loss)
    assert pc.grad[-1] < 0 and pr.grad[-1] > 0
    assert rc.grad is None and rr.grad is None


def test_accumulation_equals_full_token_mean_with_unequal_lengths():
    tok = ToyTokenizer()
    full, accumulated = ContextLM(tok.vocab_size), ContextLM(tok.vocab_size)
    p, r = ["abc", "d", "ef"], ["x", "yyyy", "zz"]
    core.compute_sft_loss(full, tok, p, r, 20).backward()
    batch = core.build_sft_batch(tok, p, r, 20)
    denominator = batch["labels"][:, 1:].ne(-100).sum()
    for i in range(3):
        scores = core.compute_response_logprobs(accumulated, tok, p[i:i+1], r[i:i+1], 20)
        (-scores.sum() / denominator).backward()
    torch.testing.assert_close(full.weight.grad, accumulated.weight.grad)


def test_reference_unchanged_and_identical_models_log_two():
    tok = ToyTokenizer()
    policy, reference = ContextLM(tok.vocab_size), ContextLM(tok.vocab_size)
    before = reference.weight.detach().clone()
    loss = core.compute_dpo_loss(policy, reference, tok, ["a"], ["z"], ["x"], .1, 10)
    assert abs(loss.item() - math.log(2)) < 1e-6
    loss.backward()
    torch.optim.SGD(policy.parameters(), lr=.1).step()
    torch.testing.assert_close(reference.weight, before)
    assert reference.weight.grad is None
