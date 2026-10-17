"""The summarisation pipeline: transcript -> chunk notes -> summary, chapters, quiz.

Long lectures do not fit in one prompt, so this uses the classic map-reduce
pattern: summarise each chunk independently (map), then summarise the summaries
(reduce). Timestamps are carried through every stage so the final output can
link back to the exact moment in the video.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .llm import LLM, LLMError
from .transcript import (
    Chunk,
    Snippet,
    chunk_snippets,
    fetch_transcript,
    format_timestamp,
    plain_text,
    transcript_stats,
)

CACHE_DIR = Path(".cache")
ProgressFn = Callable[[float, str], None]

# --------------------------------------------------------------------- prompts

MAP_SYSTEM = (
    "You are a precise note-taker for university lectures. "
    "You only use information present in the transcript excerpt you are given. "
    "If the excerpt is filler or an aside, say so instead of inventing content. "
    "Always reply with valid JSON and nothing else."
)

MAP_USER = """Below is one excerpt from a lecture transcript, covering {start} to {end}.

Return JSON with exactly this shape:
{{
  "topic": "<short title for this excerpt, max 8 words>",
  "notes": ["<concise bullet, one idea each>", "..."],
  "key_points": [
    {{"point": "<an important claim, definition or result>", "offset_seconds": <integer seconds from the START of this excerpt where it is discussed>}}
  ],
  "terms": ["<technical term defined or used>", "..."]
}}

Rules:
- 3 to 6 notes, 1 to 3 key_points, 0 to 5 terms.
- offset_seconds must be between 0 and {span} seconds.
- Use the speaker's own terminology. Do not add outside knowledge.

TRANSCRIPT EXCERPT:
\"\"\"
{text}
\"\"\""""

REDUCE_SYSTEM = (
    "You are an expert at turning lecture notes into a study guide. "
    "You work only from the section notes provided. "
    "Always reply with valid JSON and nothing else."
)

REDUCE_USER = """These are ordered section notes from a single {duration} lecture.

Return JSON with exactly this shape:
{{
  "title": "<descriptive title for the lecture, max 12 words>",
  "tldr": "<2 to 3 sentence summary of the whole lecture>",
  "summary_bullets": ["<the main things a student should take away>", "..."],
  "chapters": [
    {{"title": "<chapter title>", "start_seconds": <integer>, "summary": "<1 sentence>"}}
  ],
  "key_terms": [{{"term": "<term>", "definition": "<short definition in the lecture's sense>"}}],
  "study_checklist": ["<what a student should be able to do after this lecture>", "..."]
}}

Rules:
- 5 to 9 summary_bullets, 3 to 8 chapters, up to 8 key_terms, 3 to 6 checklist items.
- chapters must be in ascending time order and use the start_seconds values given below.
- Do not invent material that is not in the notes.

SECTION NOTES:
{notes}"""

QUIZ_SYSTEM = (
    "You write fair exam questions that test understanding, not trivia. "
    "Every question must be answerable from the lecture notes supplied. "
    "Always reply with valid JSON and nothing else."
)

QUIZ_USER = """Write {n} revision questions for this lecture, at {difficulty} difficulty.

Return JSON with exactly this shape:
{{
  "questions": [
    {{
      "type": "mcq",
      "question": "<question text>",
      "options": ["<A>", "<B>", "<C>", "<D>"],
      "answer_index": <0-3>,
      "explanation": "<why that answer is right, 1-2 sentences>",
      "timestamp_seconds": <integer, where in the lecture this is covered>
    }},
    {{
      "type": "short",
      "question": "<question text>",
      "answer": "<model answer, 1-3 sentences>",
      "timestamp_seconds": <integer>
    }}
  ]
}}

Rules:
- Roughly two thirds mcq, one third short.
- Distractors must be plausible and of similar length to the correct option.
- timestamp_seconds must come from the chapter times given below.
- Do not ask about anything absent from the notes.

LECTURE TITLE: {title}

CHAPTERS:
{chapters}

SECTION NOTES:
{notes}"""


# ---------------------------------------------------------------------- result


@dataclass
class Result:
    video_id: str
    url: str
    stats: dict
    sections: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    quiz: list[dict] = field(default_factory=list)
    transcript_text: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "stats": self.stats,
            "sections": self.sections,
            "summary": self.summary,
            "quiz": self.quiz,
            "meta": self.meta,
        }


# ----------------------------------------------------------------------- cache


def _cache_key(video_id: str, model: str, n_questions: int, difficulty: str) -> str:
    raw = f"{video_id}|{model}|{n_questions}|{difficulty}|v2"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_cached(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def save_cached(key: str, payload: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -------------------------------------------------------------------- pipeline


def summarise_chunk(llm: LLM, chunk: Chunk) -> dict:
    """Map step: notes for one excerpt, with offsets converted to absolute time."""
    span = max(1, int(chunk.end - chunk.start))
    user = MAP_USER.format(
        start=format_timestamp(chunk.start),
        end=format_timestamp(chunk.end),
        span=span,
        text=chunk.text,
    )
    data = llm.json(MAP_SYSTEM, user)
    if not isinstance(data, dict):
        raise LLMError("Chunk summary was not a JSON object.")

    key_points = []
    for item in data.get("key_points") or []:
        if not isinstance(item, dict):
            continue
        offset = item.get("offset_seconds", 0)
        try:
            offset = max(0, min(int(float(offset)), span))
        except (TypeError, ValueError):
            offset = 0
        key_points.append({"point": str(item.get("point", "")).strip(), "seconds": chunk.start + offset})

    return {
        "index": chunk.index,
        "topic": str(data.get("topic", f"Section {chunk.index + 1}")).strip(),
        "notes": [str(n).strip() for n in (data.get("notes") or []) if str(n).strip()],
        "key_points": [kp for kp in key_points if kp["point"]],
        "terms": [str(t).strip() for t in (data.get("terms") or []) if str(t).strip()],
        "start": chunk.start,
        "end": chunk.end,
    }


def _notes_block(sections: list[dict]) -> str:
    lines = []
    for section in sections:
        lines.append(f"## [{format_timestamp(section['start'])}] {section['topic']} (start_seconds={int(section['start'])})")
        lines.extend(f"- {note}" for note in section["notes"])
        for kp in section["key_points"]:
            lines.append(f"- KEY ({format_timestamp(kp['seconds'])}): {kp['point']}")
        if section["terms"]:
            lines.append(f"- TERMS: {', '.join(section['terms'])}")
        lines.append("")
    return "\n".join(lines)


def reduce_sections(llm: LLM, sections: list[dict], duration_human: str) -> dict:
    """Reduce step: one study guide from all the section notes."""
    data = llm.json(REDUCE_SYSTEM, REDUCE_USER.format(duration=duration_human, notes=_notes_block(sections)))
    if not isinstance(data, dict):
        raise LLMError("Summary was not a JSON object.")

    chapters = []
    for item in data.get("chapters") or []:
        if not isinstance(item, dict):
            continue
        try:
            start = int(float(item.get("start_seconds", 0)))
        except (TypeError, ValueError):
            start = 0
        chapters.append(
            {
                "title": str(item.get("title", "")).strip() or "Untitled",
                "seconds": max(0, start),
                "summary": str(item.get("summary", "")).strip(),
            }
        )
    chapters.sort(key=lambda c: c["seconds"])

    return {
        "title": str(data.get("title", "Lecture summary")).strip(),
        "tldr": str(data.get("tldr", "")).strip(),
        "bullets": [str(b).strip() for b in (data.get("summary_bullets") or []) if str(b).strip()],
        "chapters": chapters,
        "key_terms": [
            {"term": str(t.get("term", "")).strip(), "definition": str(t.get("definition", "")).strip()}
            for t in (data.get("key_terms") or [])
            if isinstance(t, dict) and str(t.get("term", "")).strip()
        ],
        "checklist": [str(c).strip() for c in (data.get("study_checklist") or []) if str(c).strip()],
    }


def generate_quiz(llm: LLM, summary: dict, sections: list[dict], n: int, difficulty: str) -> list[dict]:
    chapters_block = "\n".join(
        f"- [{format_timestamp(c['seconds'])}] {c['title']} (start_seconds={c['seconds']})"
        for c in summary.get("chapters", [])
    )
    data = llm.json(
        QUIZ_SYSTEM,
        QUIZ_USER.format(
            n=n,
            difficulty=difficulty,
            title=summary.get("title", ""),
            chapters=chapters_block or "- (none)",
            notes=_notes_block(sections),
        ),
    )
    raw = data.get("questions") if isinstance(data, dict) else data
    questions: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        try:
            seconds = max(0, int(float(item.get("timestamp_seconds", 0))))
        except (TypeError, ValueError):
            seconds = 0
        if str(item.get("type", "")).lower() == "mcq":
            options = [str(o).strip() for o in (item.get("options") or []) if str(o).strip()]
            if len(options) < 2:
                continue
            try:
                answer_index = int(item.get("answer_index", 0))
            except (TypeError, ValueError):
                answer_index = 0
            questions.append(
                {
                    "type": "mcq",
                    "question": str(item["question"]).strip(),
                    "options": options,
                    "answer_index": max(0, min(answer_index, len(options) - 1)),
                    "explanation": str(item.get("explanation", "")).strip(),
                    "seconds": seconds,
                }
            )
        else:
            questions.append(
                {
                    "type": "short",
                    "question": str(item["question"]).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "seconds": seconds,
                }
            )
    return questions


def run(
    url: str,
    llm: LLM,
    n_questions: int = 8,
    difficulty: str = "moderate",
    languages: tuple[str, ...] = ("en", "en-US", "en-GB"),
    max_chars: int = 6000,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
    snippets: list[Snippet] | None = None,
) -> Result:
    """End-to-end: URL in, study guide out.

    `snippets` can be supplied directly, which is what the tests use to run the
    pipeline without touching the network.
    """
    from .transcript import extract_video_id

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    video_id = extract_video_id(url)
    key = _cache_key(video_id, llm.model or "", n_questions, difficulty)

    if use_cache:
        cached = load_cached(key)
        if cached:
            report(1.0, "Loaded from cache")
            result = Result(**{k: v for k, v in cached.items() if k != "transcript_text"})
            result.transcript_text = cached.get("transcript_text", "")
            return result

    started = time.time()
    report(0.05, "Fetching transcript...")
    if snippets is None:
        snippets = fetch_transcript(video_id, languages=languages)
    stats = transcript_stats(snippets)

    chunks = chunk_snippets(snippets, max_chars=max_chars)
    report(0.15, f"Transcript: {stats['words']:,} words in {len(chunks)} sections")

    sections: list[dict] = []
    for i, chunk in enumerate(chunks):
        report(0.15 + 0.55 * (i / max(1, len(chunks))), f"Summarising section {i + 1} of {len(chunks)}...")
        sections.append(summarise_chunk(llm, chunk))

    report(0.75, "Writing the overall summary...")
    summary = reduce_sections(llm, sections, stats["duration_human"])

    report(0.9, f"Writing {n_questions} quiz questions...")
    quiz = generate_quiz(llm, summary, sections, n_questions, difficulty)

    result = Result(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        stats=stats,
        sections=sections,
        summary=summary,
        quiz=quiz,
        transcript_text=plain_text(snippets),
        meta={
            "model": llm.model,
            "chunks": len(chunks),
            "llm_calls": len(chunks) + 2,
            "seconds_taken": round(time.time() - started, 1),
            "difficulty": difficulty,
        },
    )
    if use_cache:
        payload = result.to_dict()
        payload["transcript_text"] = result.transcript_text
        save_cached(key, payload)
    report(1.0, f"Done in {result.meta['seconds_taken']}s")
    return result
