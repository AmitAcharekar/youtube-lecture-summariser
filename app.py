"""Streamlit front end for the YouTube Lecture Summariser.

Runs entirely on your machine against a local Ollama model.

    python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.exporters import to_anki_csv, to_json, to_markdown
from src.llm import DEFAULT_MODEL, LLM, SUGGESTED_MODELS, LLMError
from src.pipeline import run
from src.transcript import TranscriptError, format_timestamp, parse_subtitle_file, youtube_link

st.set_page_config(page_title="Lecture Summariser", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")

    available = LLM().list_models()
    if available:
        default_index = available.index(DEFAULT_MODEL) if DEFAULT_MODEL in available else 0
        model = st.selectbox("Ollama model", available, index=default_index)
    else:
        st.warning("No models found. Is the Ollama app running?")
        model = st.text_input("Ollama model", value=DEFAULT_MODEL)
        st.caption("Pull one with: `ollama pull " + DEFAULT_MODEL + "`")

    with st.expander("Models worth trying"):
        st.caption("\n".join(f"- `{m}`" for m in SUGGESTED_MODELS))
        st.caption("Smaller models are much faster on a laptop but write weaker quiz questions.")

    n_questions = st.slider("Quiz questions", 3, 15, 8)
    difficulty = st.select_slider("Difficulty", ["easy", "moderate", "hard"], value="moderate")
    chunk_chars = st.slider(
        "Section size (characters)", 2000, 12000, 6000, step=1000,
        help="Larger sections mean fewer model calls, so it finishes faster but with less detail.",
    )
    use_cache = st.checkbox("Use cached results", value=True)

    st.divider()
    if st.button("Check Ollama connection"):
        ok, message = LLM(model=model).health_check()
        (st.success if ok else st.error)(message)

# ------------------------------------------------------------------- header

st.title("🎓 YouTube Lecture Summariser")
st.caption("Paste a lecture link to get a summary, chapter timestamps and revision questions. Runs locally with Ollama.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

manual_snippets = None
with st.expander("Captions blocked? Upload a subtitle file instead (.srt / .vtt)"):
    st.caption(
        "YouTube rate-limits caption requests from cloud servers. If the fetch fails, "
        "download the captions (for example with yt-dlp) and upload the file here."
    )
    uploaded = st.file_uploader("Subtitle file", type=["srt", "vtt"])
    if uploaded is not None:
        try:
            manual_snippets = parse_subtitle_file(uploaded.getvalue().decode("utf-8", errors="ignore"))
            st.success(f"Loaded {len(manual_snippets)} caption lines from {uploaded.name}")
        except TranscriptError as exc:
            st.error(str(exc))

go = st.button("Summarise", type="primary", disabled=not url)

if go:
    progress_bar = st.progress(0.0)
    status = st.empty()

    def on_progress(fraction: float, message: str) -> None:
        progress_bar.progress(min(1.0, fraction))
        status.write(message)

    try:
        st.session_state["result"] = run(
            url,
            LLM(model=model),
            n_questions=n_questions,
            difficulty=difficulty,
            max_chars=chunk_chars,
            use_cache=use_cache,
            progress=on_progress,
            snippets=manual_snippets,
        )
        progress_bar.empty()
        status.empty()
    except TranscriptError as exc:
        progress_bar.empty()
        status.empty()
        st.error(f"Transcript problem: {exc}")
    except LLMError as exc:
        progress_bar.empty()
        status.empty()
        st.error(f"Model problem: {exc}")

# ------------------------------------------------------------------- results

result = st.session_state.get("result")

if result:
    summary = result.summary
    st.subheader(summary.get("title", "Lecture summary"))

    columns = st.columns(4)
    columns[0].metric("Length", result.stats.get("duration_human", "?"))
    columns[1].metric("Words", f"{result.stats.get('words', 0):,}")
    columns[2].metric("Sections", result.meta.get("chunks", 0))
    columns[3].metric("Time taken", f"{result.meta.get('seconds_taken', 0)}s")

    summary_tab, chapters_tab, quiz_tab, transcript_tab = st.tabs(
        ["Summary", "Chapters", "Quiz", "Transcript"]
    )

    with summary_tab:
        if summary.get("tldr"):
            st.info(summary["tldr"])
        if summary.get("bullets"):
            st.markdown("#### Key takeaways")
            for bullet in summary["bullets"]:
                st.markdown(f"- {bullet}")
        left, right = st.columns(2)
        with left:
            if summary.get("key_terms"):
                st.markdown("#### Key terms")
                for term in summary["key_terms"]:
                    st.markdown(f"**{term['term']}** — {term['definition']}")
        with right:
            if summary.get("checklist"):
                st.markdown("#### You should be able to")
                for item in summary["checklist"]:
                    st.checkbox(item, key=f"check_{item[:40]}")

    with chapters_tab:
        for chapter in summary.get("chapters", []):
            stamp = format_timestamp(chapter["seconds"])
            link = youtube_link(result.video_id, chapter["seconds"])
            st.markdown(f"##### [{stamp}]({link}) · {chapter['title']}")
            st.caption(chapter.get("summary", ""))

    with quiz_tab:
        if not result.quiz:
            st.warning("No questions were generated. Try a different model or a longer video.")
        for i, question in enumerate(result.quiz, start=1):
            with st.container(border=True):
                st.markdown(f"**Q{i}. {question['question']}**")
                if question["type"] == "mcq":
                    choice = st.radio(
                        "Choose one",
                        question["options"],
                        key=f"q{i}",
                        index=None,
                        label_visibility="collapsed",
                    )
                    if choice is not None:
                        correct = question["options"][question["answer_index"]]
                        if choice == correct:
                            st.success("Correct.")
                        else:
                            st.error(f"Not quite — the answer is: {correct}")
                        if question.get("explanation"):
                            st.caption(question["explanation"])
                else:
                    with st.expander("Show model answer"):
                        st.write(question.get("answer", ""))
                stamp = format_timestamp(question["seconds"])
                st.caption(f"Covered at [{stamp}]({youtube_link(result.video_id, question['seconds'])})")

    with transcript_tab:
        st.text_area("Full transcript", result.transcript_text, height=400)

    st.divider()
    st.markdown("#### Download")
    downloads = st.columns(3)
    downloads[0].download_button(
        "Study notes (.md)", to_markdown(result), file_name=f"{result.video_id}_notes.md", mime="text/markdown"
    )
    downloads[1].download_button(
        "Flashcards (.csv)", to_anki_csv(result), file_name=f"{result.video_id}_cards.csv", mime="text/csv"
    )
    downloads[2].download_button(
        "Raw data (.json)", to_json(result), file_name=f"{result.video_id}.json", mime="application/json"
    )
else:
    st.markdown(
        """
        **How it works**

        1. Captions are downloaded for the video.
        2. The transcript is split into overlapping, timestamped sections.
        3. Each section is summarised separately (the *map* step).
        4. Those section notes are summarised into one study guide (the *reduce* step).
        5. A final pass writes quiz questions, each linked to the moment it is covered.

        Everything runs locally through Ollama, so nothing leaves your machine.
        """
    )
