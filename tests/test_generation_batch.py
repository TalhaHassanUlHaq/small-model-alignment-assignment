"""Batched decoder-only generation must keep response tokens off the prompt."""
import sys
from types import SimpleNamespace

import torch

from alignment.generate import generate
from alignment.io import read_jsonl


def test_left_padded_batch_extracts_only_new_tokens(tmp_path, monkeypatch):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99
        padding_side = "right"

        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 50 + 1 for c in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join("X" for token in ids if token not in {self.pad_token_id, self.eos_token_id})

    class Model:
        config = SimpleNamespace(use_cache=False)

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, **kwargs):
            assert input_ids.shape[0] == 2
            assert attention_mask[0, 0] == 0  # Shorter prompt is padded on the left.
            assert attention_mask[1, 0] == 1
            continuation = torch.tensor([[7, 99], [8, 99]])
            return torch.cat([input_ids, continuation], dim=1)

    tokenizer = Tokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model())))
    original_tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda data, **kwargs: original_tensor(data))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    prompts = [dict(id="a", instruction="a"), dict(id="b", instruction="a longer prompt")]
    output = tmp_path / "generated.jsonl"
    generate("fake-model", prompts, output, "model", dict(batch_size=2, max_new_tokens=2,
                                                           repetition_penalty=1.0))
    rows = read_jsonl(output)
    assert [row["output"] for row in rows] == ["X", "X"]
    assert [row["generated_tokens"] for row in rows] == [2, 2]
    assert tokenizer.padding_side == "left"
