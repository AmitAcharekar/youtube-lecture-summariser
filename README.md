# 🎓 YouTube Lecture Summariser

Turn any lecture video into a study guide: a summary, chapter timestamps that link back
to the video, and revision questions you can answer in the browser or export as flashcards.

Built because reviewing a two-hour recorded lecture before an exam is slow, and scrubbing
through the video to find the one thing you half-remember is slower.

> **Demo:** _add your deployed link here_
> **Screenshot / GIF:** _add `docs/demo.gif` here_

---

## What it does

| | |
|---|---|
| **Input** | A YouTube URL (or an `.srt` / `.vtt` file if captions are blocked) |
| **Output** | TL;DR, key takeaways, chapters with clickable timestamps, key terms, a study checklist, and a quiz |
| **Exports** | Markdown notes, Anki/Quizlet CSV flashcards, raw JSON |
| **Model** | Any local model served by [Ollama](https://ollama.com) — no API key, no cost, nothing leaves your machine |

Every chapter and every quiz question carries a timestamp, so each one is a link that
opens the video at the exact moment the point is made.

---

## Quick start

**1. Install Ollama and pull a model** (one-off, ~5GB) from [ollama.com/download](https://ollama.com/download):

```bash
ollama pull llama3.1:8b
ollama list                             # confirms it is there
```

Keep the Ollama app running while you use this.

**2. Set up the project:**

```bash
git clone <your-repo-url>
cd youtube-lecture-summariser
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then paste a lecture URL and press **Summarise**.

> Use `python -m pip` and `python -m streamlit` rather than the bare commands. On machines
> with Anaconda or several Pythons installed, a bare `streamlit` often belongs to a different
> interpreter that cannot see this project's packages.

### Choosing a model

The sidebar lists every model you have pulled. On a laptop:

| Model | Speed | Notes |
|---|---|---|
| `llama3.2:3b` | fastest | good notes, weaker multiple-choice distractors |
| `llama3.1:8b` | balanced | the default |
| `gemma2:9b` / `qwen2.5:7b` | slower | slightly better structure |

Expect roughly a minute per section on an M-series MacBook Air. Results are cached, so the
second run on the same video is instant.

### Command line

```bash
python cli.py "https://www.youtube.com/watch?v=VIDEO_ID" --out week3.md --cards
python cli.py VIDEO_ID --model llama3.2:3b --questions 12 --difficulty hard
```

---

## How it works

A three-hour lecture is roughly 30,000 words, which will not fit in one prompt for a small
local model. The pipeline uses **map-reduce summarisation**:

```
YouTube URL
     |
     v
[1] Fetch captions ......... youtube-transcript-api (with .srt/.vtt fallback)
     |
     v
[2] Chunk ................... ~6,000 chars per chunk, 300 chars overlap,
     |                        timestamps preserved on every chunk
     v
[3] MAP ..................... one LLM call per chunk -> topic, notes,
     |                        key points with absolute timestamps
     v
[4] REDUCE .................. one LLM call over all section notes ->
     |                        title, TL;DR, chapters, key terms, checklist
     v
[5] QUIZ .................... one LLM call -> MCQs + short answers,
     |                        each tied to a chapter timestamp
     v
Study guide  ->  Markdown / Anki CSV / JSON
```

Total model calls: `number_of_chunks + 2`.

### Design decisions worth knowing

- **Overlapping chunks.** A 300-character overlap stops an idea that straddles a chunk
  boundary from being lost by both chunk summaries.
- **Timestamps survive every stage.** The map step asks for an *offset within the chunk*,
  which is then converted to an absolute time. Models are bad at absolute arithmetic over
  long transcripts but fine with small local offsets.
- **JSON everywhere, defensively.** Every call uses Ollama's native JSON mode, and
  `extract_json()` repairs fenced or prose-wrapped responses. A failed parse is retried with a
  firmer instruction. Small local models need this far more than large hosted ones.
- **Local only, by design.** Lecture recordings and your notes stay on your own machine, which
  also means no per-token cost while developing and testing.
- **Disk cache.** Results are cached by video + model + settings, so re-running a demo is
  instant instead of a two-minute wait in front of an interviewer.
- **Validation after generation.** Answer indices are clamped to the option list, chapters
  are sorted by time, and malformed entries are dropped rather than crashing the app.

---

## Project layout

```
.
├── app.py                  Streamlit UI
├── cli.py                  Command line version (batch a whole unit)
├── src/
│   ├── transcript.py       URL parsing, caption fetching, chunking, .srt/.vtt parsing
│   ├── llm.py              Ollama client, JSON mode, retries and JSON repair
│   ├── pipeline.py         Map-reduce summarisation, quiz generation, caching
│   └── exporters.py        Markdown, Anki CSV, JSON
└── tests/
    └── test_pipeline.py    Offline tests with a fake model and fake transcript
```

## Tests

```bash
python -m tests.test_pipeline
```

Runs the entire pipeline with a stub model and a synthetic transcript, so it needs no
network, no Ollama and no GPU. It covers URL parsing, chunk boundaries, subtitle parsing,
JSON repair, timestamp arithmetic and all three exporters — 23 checks in total.

---

## Limitations

- **Ollama must be running.** The app is useless without it; there is no cloud fallback,
  which is deliberate.
- **Captions required.** Videos with captions disabled will not work. Adding Whisper
  transcription would fix this, at the cost of a much slower first run.
- **YouTube blocks cloud IPs.** Caption requests from Streamlit Cloud, Render and similar
  hosts are frequently rate-limited. The `.srt` / `.vtt` upload is the workaround; a proxy
  or a self-hosted deployment is the other.
- **Summary quality tracks model size.** `llama3.1:8b` is good for notes and acceptable for
  quizzes; multiple-choice distractors are noticeably weaker than a larger model would write.
- **Speed is bounded by your laptop.** One model call per section, run locally, so a long
  lecture takes minutes rather than seconds. Raising the section size trades detail for speed.
- **No speaker separation.** A panel discussion or Q&A is summarised as one voice.
- **English-focused prompts.** The transcript fetch handles other languages, but the prompts
  are written in English.

## Possible extensions

- Whisper fallback for videos without captions
- Playlist mode: summarise an entire unit and build a combined revision guide
- Semantic search across every lecture you have processed (embeddings + a vector store)
- Spaced repetition scheduling instead of a flat flashcard export
