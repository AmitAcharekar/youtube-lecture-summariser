"""Fetch and prepare YouTube transcripts.

Handles both the old (<1.0) and new (>=1.0) youtube-transcript-api interfaces,
because which one you get depends on when you ran `pip install`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

_ID_PATTERNS = [
    r"(?:v=|\/v\/)([0-9A-Za-z_-]{11})",
    r"youtu\.be\/([0-9A-Za-z_-]{11})",
    r"\/embed\/([0-9A-Za-z_-]{11})",
    r"\/shorts\/([0-9A-Za-z_-]{11})",
    r"\/live\/([0-9A-Za-z_-]{11})",
]


class TranscriptError(RuntimeError):
    """Raised when no usable transcript can be retrieved."""


def extract_video_id(url_or_id: str) -> str:
    """Pull the 11-character video id out of any common YouTube URL form."""
    value = (url_or_id or "").strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", value):
        return value
    for pattern in _ID_PATTERNS:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    raise TranscriptError(f"Could not find a YouTube video id in: {url_or_id!r}")


@dataclass
class Snippet:
    """One caption line."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Chunk:
    """A block of transcript text small enough to fit in one LLM call."""

    index: int
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return asdict(self)


def format_timestamp(seconds: float) -> str:
    """3725.4 -> '1:02:05'  |  125.0 -> '2:05'"""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def youtube_link(video_id: str, seconds: float) -> str:
    """Deep link that opens the video at a given moment."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


def _normalise(raw: Iterable) -> list[Snippet]:
    """Accept dicts (old API) or FetchedTranscriptSnippet objects (new API)."""
    snippets: list[Snippet] = []
    for item in raw:
        if isinstance(item, dict):
            text, start, duration = item.get("text", ""), item.get("start", 0.0), item.get("duration", 0.0)
        else:
            text = getattr(item, "text", "")
            start = getattr(item, "start", 0.0)
            duration = getattr(item, "duration", 0.0)
        text = re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()
        # Caption files mark sound effects like [Music] or [Applause] - drop them.
        if not text or re.fullmatch(r"\[.*?\]", text):
            continue
        snippets.append(Snippet(text=text, start=float(start), duration=float(duration)))
    return snippets


def fetch_transcript(video_id: str, languages: Sequence[str] = ("en", "en-US", "en-GB")) -> list[Snippet]:
    """Download captions for a video, falling back to any available language.

    Raises TranscriptError with a readable message when captions are unavailable
    (private video, no captions uploaded, or YouTube blocking the request).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise TranscriptError("youtube-transcript-api is not installed. Run: pip install -r requirements.txt") from exc

    errors: list[str] = []

    # --- new interface (v1.0+): instance methods, returns an object ---
    try:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=list(languages))
        except Exception:
            # Requested language missing: take whatever the video has.
            listing = api.list(video_id)
            transcript = next(iter(listing))
            fetched = transcript.fetch()
        snippets = _normalise(getattr(fetched, "snippets", fetched))
        if snippets:
            return snippets
    except Exception as exc:  # noqa: BLE001 - fall through to the legacy path
        errors.append(f"new API: {exc}")

    # --- legacy interface (<1.0): class methods, returns list[dict] ---
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
        snippets = _normalise(raw)
        if snippets:
            return snippets
    except Exception as exc:  # noqa: BLE001
        errors.append(f"legacy API: {exc}")

    raise TranscriptError(
        "No transcript could be fetched for this video. It may have captions disabled, "
        "be private or age-restricted, or YouTube may be rate-limiting this IP.\n"
        + "\n".join(errors)
    )


def chunk_snippets(snippets: list[Snippet], max_chars: int = 6000, overlap_chars: int = 300) -> list[Chunk]:
    """Group caption lines into overlapping, timestamped chunks.

    Overlap keeps a sentence that straddles a boundary from losing its context
    in the chunk summary.
    """
    if not snippets:
        return []

    chunks: list[Chunk] = []
    buffer: list[Snippet] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if not buffer:
            return
        chunks.append(
            Chunk(
                index=len(chunks),
                text=" ".join(s.text for s in buffer),
                start=buffer[0].start,
                end=buffer[-1].end,
            )
        )
        # Carry the tail of this chunk into the next one.
        tail: list[Snippet] = []
        tail_chars = 0
        for snippet in reversed(buffer):
            if tail_chars >= overlap_chars:
                break
            tail.insert(0, snippet)
            tail_chars += len(snippet.text) + 1
        buffer = tail
        size = tail_chars

    for snippet in snippets:
        buffer.append(snippet)
        size += len(snippet.text) + 1
        if size >= max_chars:
            flush()
    # Final flush, but ignore a leftover that is only the carried-over overlap.
    if buffer and (not chunks or len(" ".join(s.text for s in buffer)) > overlap_chars):
        tail_only = buffer
        buffer = tail_only
        chunks.append(
            Chunk(
                index=len(chunks),
                text=" ".join(s.text for s in buffer),
                start=buffer[0].start,
                end=buffer[-1].end,
            )
        )
    return chunks


_TIMECODE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def parse_subtitle_file(content: str) -> list[Snippet]:
    """Parse an .srt or .vtt subtitle file into snippets.

    Useful when YouTube refuses the caption request, which happens often from
    cloud IP addresses. The user can download captions manually and feed them in.
    """
    snippets: list[Snippet] = []
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    for block in blocks:
        match = _TIMECODE.search(block)
        if not match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        lines = [ln for ln in block.split("\n") if not _TIMECODE.search(ln) and not ln.strip().isdigit()]
        text = re.sub(r"<[^>]+>", " ", " ".join(lines))  # strip VTT styling tags
        text = re.sub(r"\s+", " ", text).strip()
        if text and not re.fullmatch(r"\[.*?\]", text) and not text.upper().startswith("WEBVTT"):
            snippets.append(Snippet(text=text, start=start, duration=max(0.0, end - start)))
    if not snippets:
        raise TranscriptError("No subtitle blocks found. Expected an .srt or .vtt file.")
    return snippets


def transcript_stats(snippets: list[Snippet]) -> dict:
    words = sum(len(s.text.split()) for s in snippets)
    duration = snippets[-1].end if snippets else 0.0
    return {
        "lines": len(snippets),
        "words": words,
        "duration_seconds": duration,
        "duration_human": format_timestamp(duration),
    }


def plain_text(snippets: list[Snippet], with_timestamps: bool = True) -> str:
    if with_timestamps:
        return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in snippets)
    return " ".join(s.text for s in snippets)
