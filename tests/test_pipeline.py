"""Offline tests: the whole pipeline runs against a fake transcript and a fake model.

    python -m tests.test_pipeline

No network, no API key and no Ollama needed, which is also how you would test
this in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exporters import to_anki_csv, to_json, to_markdown  # noqa: E402
from src.llm import LLM, extract_json  # noqa: E402
from src.pipeline import run  # noqa: E402
from src.transcript import (  # noqa: E402
    Snippet,
    chunk_snippets,
    extract_video_id,
    format_timestamp,
    parse_subtitle_file,
)


class FakeLLM(LLM):
    """Stands in for a real model: returns canned JSON shaped like the prompts ask for."""

    def __init__(self) -> None:
        self.model, self.temperature, self.timeout, self.base_url = "fake-model", 0.0, 1, "http://localhost:11434"
        self.calls = 0

    def json(self, system: str, user: str, retries: int = 2):  # type: ignore[override]
        self.calls += 1
        if "one excerpt" in user:
            return {
                "topic": "Gradient descent basics",
                "notes": ["Loss measures error", "Gradients point uphill", "Learning rate controls step size"],
                "key_points": [{"point": "The learning rate must be tuned", "offset_seconds": 12}],
                "terms": ["gradient", "learning rate"],
            }
        if "section notes" in user and "revision questions" not in user:
            return {
                "title": "Introduction to gradient descent",
                "tldr": "The lecture explains how gradient descent minimises a loss function.",
                "summary_bullets": ["Loss quantifies error", "Gradients give the direction of steepest ascent"],
                "chapters": [
                    {"title": "What is loss", "start_seconds": 0, "summary": "Defines the loss function."},
                    {"title": "Choosing a learning rate", "start_seconds": 60, "summary": "Too large diverges."},
                ],
                "key_terms": [{"term": "learning rate", "definition": "The size of each update step."}],
                "study_checklist": ["Explain why a large learning rate can diverge"],
            }
        return {
            "questions": [
                {
                    "type": "mcq",
                    "question": "What does the learning rate control?",
                    "options": ["Step size", "Batch size", "Layer count", "Loss type"],
                    "answer_index": 0,
                    "explanation": "It scales each parameter update.",
                    "timestamp_seconds": 60,
                },
                {
                    "type": "short",
                    "question": "Why can a large learning rate cause divergence?",
                    "answer": "Steps overshoot the minimum, so the loss grows instead of shrinking.",
                    "timestamp_seconds": 60,
                },
            ]
        }


def fake_snippets(n: int = 400) -> list[Snippet]:
    lines = [
        "Today we look at how gradient descent actually works in practice.",
        "The loss function measures how wrong the model currently is.",
        "The gradient tells us the direction of steepest increase in that loss.",
        "So we step in the opposite direction, scaled by the learning rate.",
    ]
    return [Snippet(text=lines[i % len(lines)], start=float(i * 4), duration=4.0) for i in range(n)]


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise AssertionError(label)


def main() -> int:
    print("\nURL parsing")
    for url in [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ]:
        check(url, extract_video_id(url) == "dQw4w9WgXcQ")

    print("\nTimestamp formatting")
    check("125s -> 2:05", format_timestamp(125) == "2:05")
    check("3725s -> 1:02:05", format_timestamp(3725) == "1:02:05")

    print("\nChunking")
    snippets = fake_snippets()
    chunks = chunk_snippets(snippets, max_chars=1500)  # same defaults the pipeline uses
    check(f"produced {len(chunks)} chunks", len(chunks) > 1)
    check("chunks are time-ordered", all(a.start <= b.start for a, b in zip(chunks, chunks[1:])))
    check("chunks respect the size limit", all(len(c.text) < 2200 for c in chunks))
    check("first chunk starts at 0", chunks[0].start == 0.0)

    print("\nSubtitle parsing")
    srt = (
        "1\n00:00:01,000 --> 00:00:04,500\nWelcome to week three.\n\n"
        "2\n00:00:04,500 --> 00:00:08,000\n[Music]\n\n"
        "3\n00:01:00,000 --> 00:01:03,000\n<i>Gradient</i> descent again.\n"
    )
    parsed = parse_subtitle_file(srt)
    check("sound-effect lines are dropped", len(parsed) == 2)
    check("timecodes become seconds", parsed[1].start == 60.0)
    check("styling tags are stripped", parsed[1].text == "Gradient descent again.")

    print("\nJSON repair")
    check("fenced json", extract_json('```json\n{"a": 1}\n```') == {"a": 1})
    check("json with prose around it", extract_json('Sure! {"a": 1} hope that helps') == {"a": 1})

    print("\nFull pipeline (fake model, no network)")
    llm = FakeLLM()
    result = run(
        "https://youtu.be/dQw4w9WgXcQ",
        llm,
        n_questions=4,
        use_cache=False,
        snippets=snippets,
        max_chars=1500,
    )
    check("sections were produced", len(result.sections) == len(chunks))
    check("summary has a title", bool(result.summary["title"]))
    check("chapters are sorted", [c["seconds"] for c in result.summary["chapters"]] == sorted(c["seconds"] for c in result.summary["chapters"]))
    check("quiz has questions", len(result.quiz) == 2)
    check("mcq answer index is in range", result.quiz[0]["answer_index"] < len(result.quiz[0]["options"]))
    check("key point timestamps are absolute", all(kp["seconds"] >= s["start"] for s in result.sections for kp in s["key_points"]))
    check("one model call per chunk plus two", llm.calls == len(chunks) + 2)

    print("\nExporters")
    markdown = to_markdown(result)
    check("markdown has a heading", markdown.startswith("# "))
    check("markdown links back to the video", "youtube.com/watch?v=dQw4w9WgXcQ&t=" in markdown)
    check("markdown includes the quiz", "**Q1.**" in markdown)
    csv_text = to_anki_csv(result)
    check("csv has one row per question", len(csv_text.strip().splitlines()) >= 2)
    check("json round-trips", json.loads(to_json(result))["video_id"] == "dQw4w9WgXcQ")

    print("\nAll checks passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
