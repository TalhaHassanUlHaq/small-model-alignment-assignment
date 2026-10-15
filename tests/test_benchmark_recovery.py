"""The Kaggle Benchmark download preserves conversations, not task files."""
import json
import sys
from pathlib import Path

from alignment.io import append_jsonl, read_jsonl, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_control import recover_judgments


def test_recover_saved_conversations_and_retry_counts(tmp_path):
    generations = tmp_path / "generations"
    generations.mkdir()
    rows = {}
    for track in ("hh", "uf", "persona"):
        pair = dict(id=f"pair-{track}", instruction="Question", response_a="A", response_b="B",
                    category="general", dataset=track)
        rows[track] = pair
        append_jsonl(generations / f"pairs_{track}.jsonl", pair)
        write_json(generations / f"pairs_{track}.identity.json",
                   {pair["id"]: dict(A="candidate", B="baseline", candidate="candidate", baseline="baseline")})
    requests = []
    for track, pair in rows.items():
        prompt = "DATA:\n" + json.dumps([pair])
        response = dict(id=pair["id"], winner="A", reason="better")
        if track == "persona":
            response.update(style_winner="A", usefulness_winner="tie", safety_winner="tie")
        if track == "hh":
            requests.append(dict(contents=[dict(role="CONTENT_ROLE_USER", parts=[dict(text=prompt)])]))
        requests.append(dict(contents=[dict(role="CONTENT_ROLE_USER", parts=[dict(text=prompt)]),
                                       dict(role="CONTENT_ROLE_ASSISTANT", parts=[dict(text=json.dumps([response]))])]))
    download = tmp_path / "download"
    download.mkdir()
    write_json(download / "judge.run.json", dict(modelVersion=dict(slug="google/gemini-3.6-flash"),
                                                   state="BENCHMARK_TASK_RUN_STATE_COMPLETED",
                                                   endTime="2026-10-15T12:00:00Z",
                                                   conversations=[dict(requests=requests)]))
    output = tmp_path / "judgments"
    summaries = recover_judgments(download, generations, output, "judge")
    assert all(summaries[track]["counts"]["win"] == 1 for track in rows)
    assert read_jsonl(output / "judged_hh.jsonl")[0]["attempts"] == 2
