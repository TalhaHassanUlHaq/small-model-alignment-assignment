import random
from pathlib import Path

import numpy as np
import torch

from alignment.train import save_checkpoint, restore_rng, seed_all
from tests.toy_lm import ToyCausalLM


class SerializableModel(ToyCausalLM):
    def save_pretrained(self, path, safe_serialization=True):
        torch.save(self.state_dict(), Path(path) / "weights.pt")


class SerializableTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text('{}')


def update(model, optimizer, scheduler):
    optimizer.zero_grad()
    loss = (model.logit_bias - torch.rand_like(model.logit_bias)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_checkpoint_resume_matches_uninterrupted_update(tmp_path):
    seed_all(42)
    model = SerializableModel(8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    update(model, optimizer, scheduler)
    save_checkpoint(tmp_path / 'checkpoint', model, SerializableTokenizer(), optimizer, scheduler, scaler,
                    dict(step=1, epoch=0, offset=2), dict(lr=.01))
    update(model, optimizer, scheduler)
    expected = model.logit_bias.detach().clone()
    continued_rng = (random.random(), float(np.random.random()), torch.rand(1))
    restored = SerializableModel(8)
    restored.load_state_dict(torch.load(tmp_path / 'checkpoint' / 'weights.pt', weights_only=True))
    opt = torch.optim.AdamW(restored.parameters(), lr=.01)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 1, .9)
    saved = torch.load(tmp_path / 'checkpoint' / 'training_state.pt', weights_only=False)
    opt.load_state_dict(saved['optimizer'])
    schedule.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'])
    update(restored, opt, schedule)
    torch.testing.assert_close(restored.logit_bias, expected, atol=0, rtol=0)
    assert continued_rng[0] == random.random()
    assert continued_rng[1] == float(np.random.random())
    torch.testing.assert_close(continued_rng[2], torch.rand(1), atol=0, rtol=0)
