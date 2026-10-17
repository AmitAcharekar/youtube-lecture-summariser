"""Turn a Result into files a student can actually use: markdown notes and Anki cards."""

from __future__ import annotations

import csv
import io
import json

from .pipeline import Result
from .transcript import format_timestamp, youtube_link


def to_markdown(result: Result, include_sections: bool = True) -> str:
    summary = result.summary
    out: list[str] = []
    out.append(f"# {summary.get('title', 'Lecture summary')}")
    out.append("")
    out.append(f"Source: {result.url}")
    out.append(
        f"Length: {result.stats.get('duration_human', '?')} · "
        f"{result.stats.get('words', 0):,} words · "
        f"Model: {result.meta.get('model', '?')} (local, Ollama)"
    )
    out.append("")

    if summary.get("tldr"):
        out.append("## TL;DR")
        out.append("")
        out.append(summary["tldr"])
        out.append("")

    if summary.get("bullets"):
        out.append("## Key takeaways")
        out.append("")
        out.extend(f"- {b}" for b in summary["bullets"])
        out.append("")

    if summary.get("chapters"):
        out.append("## Chapters")
        out.append("")
        for chapter in summary["chapters"]:
            stamp = format_timestamp(chapter["seconds"])
            link = youtube_link(result.video_id, chapter["seconds"])
            out.append(f"- [`{stamp}`]({link}) **{chapter['title']}** — {chapter.get('summary', '')}")
        out.append("")

    if summary.get("key_terms"):
        out.append("## Key terms")
        out.append("")
        for term in summary["key_terms"]:
            out.append(f"- **{term['term']}** — {term['definition']}")
        out.append("")

    if summary.get("checklist"):
        out.append("## After this lecture you should be able to")
        out.append("")
        out.extend(f"- [ ] {item}" for item in summary["checklist"])
        out.append("")

    if result.quiz:
        out.append("## Revision questions")
        out.append("")
        for i, question in enumerate(result.quiz, start=1):
            stamp = format_timestamp(question["seconds"])
            link = youtube_link(result.video_id, question["seconds"])
            out.append(f"**Q{i}.** {question['question']}  ([`{stamp}`]({link}))")
            out.append("")
            if question["type"] == "mcq":
                for j, option in enumerate(question["options"]):
                    out.append(f"   {chr(65 + j)}. {option}")
                out.append("")
                answer = chr(65 + question["answer_index"])
                out.append(f"   <details><summary>Answer</summary>")
                out.append("")
                out.append(f"   **{answer}.** {question.get('explanation', '')}")
                out.append("")
                out.append("   </details>")
            else:
                out.append("   <details><summary>Answer</summary>")
                out.append("")
                out.append(f"   {question.get('answer', '')}")
                out.append("")
                out.append("   </details>")
            out.append("")

    if include_sections and result.sections:
        out.append("## Section notes")
        out.append("")
        for section in result.sections:
            stamp = format_timestamp(section["start"])
            link = youtube_link(result.video_id, section["start"])
            out.append(f"### [`{stamp}`]({link}) {section['topic']}")
            out.append("")
            out.extend(f"- {note}" for note in section["notes"])
            out.append("")

    return "\n".join(out)


def to_anki_csv(result: Result) -> str:
    """Two-column CSV that imports straight into Anki or Quizlet."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for question in result.quiz:
        if question["type"] == "mcq":
            options = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(question["options"]))
            front = f"{question['question']}\n{options}"
            back = f"{chr(65 + question['answer_index'])}. {question['options'][question['answer_index']]}"
            if question.get("explanation"):
                back += f"\n\n{question['explanation']}"
        else:
            front = question["question"]
            back = question.get("answer", "")
        back += f"\n\n[{format_timestamp(question['seconds'])}] {youtube_link(result.video_id, question['seconds'])}"
        writer.writerow([front, back])
    return buffer.getvalue()


def to_json(result: Result) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
