import json
from pathlib import Path

import pytest

from alignment.io import append_jsonl, read_jsonl, read_json
from alignment.generate import pair_outputs
from alignment.judge import run_judge, summarize, judge_prompt


def prepare(tmp_path, count=3):
    for generator in ["sft-secret", "dpo-secret"]:
        for i in range(count):
            append_jsonl(tmp_path / f"{generator}.jsonl", dict(id=str(i), instruction=f"Question {i}",
                         output=f"Answer {i}", generator=generator, dataset="alpaca_eval"))
    pairs = tmp_path / "pairs.jsonl"
    pair_outputs(tmp_path / "sft-secret.jsonl", tmp_path / "dpo-secret.jsonl", pairs)
    return pairs


def test_blinding_and_resume(tmp_path):
    pairs = prepare(tmp_path)
    assert "sft-secret" not in judge_prompt(read_jsonl(pairs))
    assert "dpo-secret" not in judge_prompt(read_jsonl(pairs))
    calls = []
    def call(prompt):
        calls.append(prompt)
        data = json.loads(prompt.split("DATA:\n")[1])
        return json.dumps([dict(id=r["id"], winner="A", reason="test") for r in data])
    output = tmp_path / "judged.jsonl"
    waits = []
    run_judge(pairs, output, call, "mock", batch_size=2, sleep=waits.append, clock=lambda: 0)
    assert waits == [5]
    assert len(calls) == 2
    run_judge(pairs, output, call, "mock", sleep=lambda _: None)
    assert len(calls) == 2
    summary = summarize(pairs, output, tmp_path / "summary.json")
    assert summary["counts"]["win"] + summary["counts"]["loss"] == 3
    assert summary["failure_rate"] == 0


def test_invalid_and_failed_judgments_are_not_wins(tmp_path):
    pairs = prepare(tmp_path)
    output = tmp_path / "judged.jsonl"
    calls = []
    def call(prompt):
        calls.append(prompt)
        return "invalid JSON"
    run_judge(pairs, output, call, "mock", sleep=lambda _: None)
    assert len(calls) == 3
    assert len(read_jsonl(output.with_suffix(".failures.jsonl"))) == 3
    summary = summarize(pairs, output, tmp_path / "summary.json")
    assert summary["failure_rate"] == 1
    assert summary["decisive_win_rate"] is None


def test_partial_batch_retries_only_missing_pairs(tmp_path):
    pairs = prepare(tmp_path)
    output = tmp_path / "judged.jsonl"
    seen = []
    def call(prompt):
        rows = json.loads(prompt.split("DATA:\n")[1])
        seen.append(len(rows))
        return json.dumps([dict(id=rows[0]["id"], winner="tie", reason="equal")])
    run_judge(pairs, output, call, "mock", sleep=lambda _: None)
    assert seen == [3, 2, 1]
    assert len(read_jsonl(output)) == 3


def test_resume_rejects_changed_judge(tmp_path):
    pairs = prepare(tmp_path)
    out = tmp_path / "judged.jsonl"
    run_judge(pairs, out, lambda _: "[]", "mock", sleep=lambda _: None)
    with pytest.raises(ValueError, match="changed"):
        run_judge(pairs, out, lambda _: "[]", "different")
