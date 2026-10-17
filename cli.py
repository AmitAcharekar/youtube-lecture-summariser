"""Command line version, handy for batch processing a whole unit's lectures.

    python cli.py "https://www.youtube.com/watch?v=..." --out notes.md
    python cli.py VIDEO_ID --model llama3.2:3b --questions 10 --difficulty hard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.exporters import to_anki_csv, to_json, to_markdown
from src.llm import DEFAULT_MODEL, LLM, LLMError
from src.pipeline import run
from src.transcript import TranscriptError


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise a YouTube lecture into notes and quiz questions.")
    parser.add_argument("url", help="YouTube URL or 11-character video id")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--difficulty", default="moderate", choices=["easy", "moderate", "hard"])
    parser.add_argument("--chunk-chars", type=int, default=6000)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default=None, help="Write markdown notes here (default: <video_id>_notes.md)")
    parser.add_argument("--cards", action="store_true", help="Also write an Anki CSV")
    parser.add_argument("--json", action="store_true", help="Also write the raw JSON")
    args = parser.parse_args()

    def progress(fraction: float, message: str) -> None:
        print(f"[{int(fraction * 100):3d}%] {message}", flush=True)

    llm = LLM(model=args.model)
    ok, message = llm.health_check()
    if not ok:
        print(f"Error: {message}", file=sys.stderr)
        return 1

    try:
        result = run(
            args.url,
            llm,
            n_questions=args.questions,
            difficulty=args.difficulty,
            max_chars=args.chunk_chars,
            use_cache=not args.no_cache,
            progress=progress,
        )
    except (TranscriptError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out or f"{result.video_id}_notes.md")
    out_path.write_text(to_markdown(result), encoding="utf-8")
    print(f"\nNotes written to {out_path}")

    if args.cards:
        cards_path = out_path.with_suffix(".cards.csv")
        cards_path.write_text(to_anki_csv(result), encoding="utf-8")
        print(f"Flashcards written to {cards_path}")

    if args.json:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(to_json(result), encoding="utf-8")
        print(f"Raw data written to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
