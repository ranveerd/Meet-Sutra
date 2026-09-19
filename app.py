"""
Minutes - Streamlit UI for the AI Meeting Assistant.

Run from the project root (next to main.py, utils/ and core/):
    streamlit run app.py

Design note: the look is a stenographer's pad - pale green paper, ink-green
type, a red margin line and a highlighter. The one animated moment is the
pen-trace under the headline, drawn once on load.
"""

import html
import math
import os
import re
import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

APP_NAME = "Minutes"

st.set_page_config(
    page_title=f"{APP_NAME} - AI meeting assistant",
    page_icon="🎙️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..800&family=Literata:ital,opsz,wght@0,7..72,400..700;1,7..72,400..700&display=swap');

:root {
  --paper: #E3EBD8;
  --paper-light: #EEF3E6;
  --paper-deep: #D2DDC5;
  --ink: #16251D;
  --muted: #55665B;
  --red: #C63D2F;
  --blue: #274B9F;
  --marker: rgba(243, 221, 91, .78);
  --rule: rgba(22, 37, 29, .16);
  --display: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif;
  --text: 'Literata', Georgia, 'Times New Roman', serif;
}

.stApp { background: var(--paper); color: var(--ink); font-family: var(--display); color-scheme: light; }

/* ---------- theme colours (so no .streamlit/config.toml is needed) ---------- */
.stApp :where(p, li, label, h1, h2, h3, h4) { color: var(--ink); }
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background: var(--ink); border-color: var(--ink);
}
.stButton button[kind="primary"] :is(p, span), .stButton button[data-testid="stBaseButton-primary"] :is(p, span) { color: var(--paper); }
.stButton button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) { background: transparent; border: 1px solid var(--ink); }
.stCheckbox label:has(input:checked) > span:first-child { background-color: var(--ink) !important; border-color: var(--ink) !important; }
.stRadio label:has(input:checked) > div:first-child { background-color: var(--ink) !important; border-color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] :where(span, small, p, div) { color: var(--ink); }
[data-testid="stFileUploaderDropzone"] button { background: var(--paper); color: var(--ink); border: 1px solid var(--ink); }
[data-testid="stChatInput"], [data-testid="stChatInput"] textarea { background: var(--paper-light); color: var(--ink); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 2.6rem; padding-bottom: 5rem; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--ink); border-right: none; }
section[data-testid="stSidebar"] :is(p, span, label, div, li) { color: var(--paper); }
.brand { font-family: var(--display); font-weight: 800; font-size: 1.7rem; letter-spacing: -.02em; line-height: 1; }
.brand small { display: block; margin-top: .35rem; font-weight: 400; font-size: .86rem; letter-spacing: 0; opacity: .75; }
.side-note { font-family: var(--text); font-size: .9rem; line-height: 1.6; opacity: .88; margin-top: 1.4rem; }
.side-rule { height: 1px; background: rgba(227, 235, 216, .22); margin: 1.4rem 0 1.1rem; }
section[data-testid="stSidebar"] .side-title { font-weight: 700; font-size: .95rem; margin-bottom: .5rem; }
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stDownloadButton button {
  width: 100%; background: transparent; border: 1px solid rgba(227, 235, 216, .55); border-radius: 6px;
}
section[data-testid="stSidebar"] .stButton button:hover,
section[data-testid="stSidebar"] .stDownloadButton button:hover { border-color: #F3DD5B; }
section[data-testid="stSidebar"] :is(button[kind="primary"], button[data-testid="stBaseButton-primary"]) {
  background: #F3DD5B; border-color: #F3DD5B;
}
section[data-testid="stSidebar"] :is(button[kind="primary"], button[data-testid="stBaseButton-primary"]) :is(p, span) { color: var(--ink); }

/* ---------- hero ---------- */
.hero-title {
  font-family: 'Manrope', var(--display); font-weight: 700;
  font-size: clamp(1.9rem, 3.2vw, 2.7rem); line-height: 1.15; letter-spacing: -.02em;
  margin: 0 0 1rem; max-width: 22ch;
}
.hero-copy { font-family: var(--text); font-size: 1.08rem; line-height: 1.7; color: var(--muted); max-width: 60ch; margin: 0; }
.trace { display: block; width: 100%; height: 56px; margin: 1.6rem 0 2rem; color: var(--ink); overflow: visible; }
.trace path { stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw 2.6s cubic-bezier(.45, .05, .25, 1) forwards; }
@keyframes draw { to { stroke-dashoffset: 0; } }
@media (prefers-reduced-motion: reduce) { .trace path { animation: none; stroke-dashoffset: 0; } }

/* ---------- form ---------- */
.form-title { font-weight: 700; font-size: 1.15rem; margin: 0 0 .4rem; }
.stTextInput input, .stTextArea textarea {
  background: var(--paper-light) !important; border: 1.5px solid var(--ink) !important; border-radius: 6px !important;
  color: var(--ink) !important; font-family: var(--display) !important;
}
.stTextInput input:focus { box-shadow: 0 0 0 3px var(--marker) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--paper-light); border: 1.5px dashed var(--ink); border-radius: 6px; }
.stRadio label p { font-family: var(--display); font-size: .98rem; }
.stButton button, .stDownloadButton button { font-family: var(--display); font-weight: 600; border-radius: 6px; padding: .55rem 1.1rem; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible, .stTextInput input:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted); font-family: var(--text); }

/* ---------- what comes back ---------- */
.gets { border-left: 1.5px solid var(--red); padding-left: 1.4rem; }
.gets-row { padding: .7rem 0; border-bottom: 1px solid var(--rule); }
.gets-row:last-child { border-bottom: none; }
.gets-row b { font-family: var(--display); font-weight: 700; display: block; font-size: 1.02rem; }
.gets-row span { font-family: var(--text); font-size: .95rem; color: var(--muted); }

/* ---------- result header ---------- */
.brief-title { font-family: 'Manrope', var(--display); font-weight: 700; font-size: clamp(1.6rem, 2.8vw, 2.2rem); line-height: 1.2; letter-spacing: -.02em; margin: 0 0 .7rem; max-width: 30ch; }
.brief-meta { font-family: var(--text); color: var(--muted); font-size: .95rem; margin: 0 0 1rem; }
.brief-strip { font-family: var(--text); font-size: 1.08rem; line-height: 1.75; max-width: 66ch; margin: 0 0 2rem; }
.brief-strip b { font-weight: 700; background: linear-gradient(transparent 58%, var(--marker) 58%); }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1.5px solid var(--ink); }
.stTabs button[data-baseweb="tab"] { padding: .7rem 0; background: transparent; color: var(--muted); }
.stTabs button[data-baseweb="tab"] p { font-family: var(--display); font-weight: 600; font-size: 1rem; }
.stTabs button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--ink); height: 4px; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-testid="stMarkdownContainer"] :is(p, li) { font-family: var(--text); font-size: 1.05rem; line-height: 1.75; max-width: 70ch; }
.stTabs button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p { font-family: var(--display); max-width: none; line-height: 1.3; }
.tab-lead { font-family: var(--text); color: var(--muted); margin: 1.1rem 0 .6rem; }

/* ---------- action items (tickable) ---------- */
.stCheckbox { border-bottom: 1px solid var(--rule); padding: .6rem 0; max-width: 72ch; }
.stCheckbox label { align-items: flex-start; }

/* ---------- decisions and questions ---------- */
.note { display: grid; grid-template-columns: 2rem 1fr; gap: .5rem; padding: .75rem 0; border-bottom: 1px solid var(--rule); max-width: 72ch; }
.note .glyph { font-family: var(--display); font-weight: 800; font-size: 1.3rem; line-height: 1.4; }
.note .body { font-family: var(--text); font-size: 1.04rem; line-height: 1.7; }
.note.decision .glyph { color: var(--blue); }
.note.question .glyph { color: var(--red); }

/* ---------- transcript as a ruled notebook page ---------- */
.notebook {
  max-height: 560px; overflow-y: auto; padding: 0 1.4rem 0 4.4rem; margin-top: .8rem;
  border: 1px solid var(--rule); background-color: var(--paper-light);
  background-image:
    linear-gradient(to right, transparent 3.3rem, var(--red) 3.3rem, var(--red) calc(3.3rem + 1.5px), transparent calc(3.3rem + 1.5px)),
    repeating-linear-gradient(to bottom, transparent 0, transparent calc(2rem - 1px), var(--rule) calc(2rem - 1px), var(--rule) 2rem);
  background-attachment: local, local;
  font-family: var(--text); font-size: 1rem; line-height: 2rem;
}
.notebook p { margin: 0 0 2rem; }
.notebook mark { background: var(--marker); color: var(--ink); padding: 0 .1em; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background: var(--paper-light); border: 1px solid var(--rule); border-radius: 6px; }
[data-testid="stChatInput"] textarea { font-family: var(--display); }
"""


def inject_css() -> None:
    st.markdown("<style>" + re.sub(r"\n\s*\n", "\n", CSS) + "</style>", unsafe_allow_html=True)


def _h(markup: str) -> None:
    """Render an HTML snippet. Newlines are collapsed so Markdown can't mangle it."""
    st.markdown(re.sub(r"\n\s*", "", markup.strip()), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.*)")


def parse_items(value) -> list[str]:
    """Turn LLM output (markdown bullets, numbered list, or a Python list) into clean items."""
    if value is None:
        return []
    lines = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value).splitlines()
    items: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _BULLET.match(line)
        if match:
            items.append(match.group(1).strip())
        elif line.endswith(":"):
            continue  # intro line such as "Here are the action items:"
        elif items:
            items[-1] += " " + line  # wrapped continuation of the previous bullet
        else:
            items.append(line)
    return items


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {v}" for v in value)
    return str(value)


def inline_md(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "meeting-brief"


def short_source(source: str) -> str:
    cleaned = re.sub(r"^https?://(www\.)?", "", source)
    return cleaned if len(cleaned) <= 60 else cleaned[:57] + "..."


def trace_svg(width: int = 1000, height: int = 56, n: int = 180) -> str:
    """A pen-trace of a voice: quiet, a burst of speech, quiet again."""
    points = []
    for i in range(n + 1):
        x = i / n * width
        envelope = math.sin(math.pi * i / n) ** 1.4
        wobble = 0.6 * math.sin(i * 0.9) + 0.4 * math.sin(i * 2.3 + 1)
        y = height / 2 + envelope * height * 0.44 * wobble
        points.append(f"{x:.1f},{y:.1f}")
    d = "M" + " L".join(points)
    return (
        f'<svg class="trace" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">'
        f'<path d="{d}" pathLength="1" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>'
    )


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------

def build_txt(r: dict) -> str:
    def block(head: str, body: str) -> str:
        return f"{head}\n{'-' * len(head)}\n{body.strip() or 'None found.'}\n"

    return "\n".join(
        [
            r["title"],
            "=" * len(r["title"]),
            f"Source: {r['source_label']}",
            f"Generated: {r['generated']}",
            "",
            block("Summary", as_text(r["summary"])),
            block("Action items", as_text(r["action_items"])),
            block("Key decisions", as_text(r["key_decisions"])),
            block("Open questions", as_text(r["open_questions"])),
            block("Transcript", r["transcript"]),
        ]
    )


_PDF_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2022": "-", "\u2026": "...",
    "\u00a0": " ", "\u2192": "->",
}


def pdf_safe(text) -> str:
    """The built-in PDF fonts only cover Latin-1, so map or replace anything outside it."""
    text = str(text)
    for old, new in _PDF_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(r: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()
    nxt = {"new_x": "LMARGIN", "new_y": "NEXT"}

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(22, 37, 29)
    pdf.multi_cell(0, 9, pdf_safe(r["title"]), **nxt)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(95, 110, 100)
    pdf.multi_cell(0, 5, pdf_safe(f"Source: {r['source_label']}   Generated: {r['generated']}"), **nxt)
    pdf.ln(2)

    def section(head: str, body: str) -> None:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(22, 37, 29)
        pdf.cell(0, 8, head, **nxt)
        pdf.set_draw_color(198, 61, 47)
        pdf.set_line_width(0.6)
        y = pdf.get_y()
        pdf.line(18, y, 44, y)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(40, 52, 46)
        pdf.multi_cell(0, 5.8, pdf_safe(body).strip() or "None found.", **nxt)

    section("Summary", as_text(r["summary"]))
    section("Action items", as_text(r["action_items"]))
    section("Key decisions", as_text(r["key_decisions"]))
    section("Open questions", as_text(r["open_questions"]))
    section("Transcript", r["transcript"])
    return bytes(pdf.output())


# --------------------------------------------------------------------------
# Pipeline (same steps as run_pipeline, with progress shown on screen)
# --------------------------------------------------------------------------

def run_with_progress(source: str, language: str, source_label: str):
    started = time.time()
    box = st.status("Working through the recording", expanded=True)
    try:
        box.write("Downloading and splitting the audio")
        chunks = process_input(source)

        box.write("Transcribing with Whisper on this computer. This is the slowest step.")
        transcript = transcribe_all(chunks, language=language)
        if not transcript or not transcript.strip():
            raise ValueError("No speech was found in this recording.")

        box.write("Writing the title and summary")
        title = generate_title(transcript)
        summary = summarize(transcript)

        box.write("Pulling out action items, decisions and open questions")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        box.write("Indexing the transcript so you can chat with it")
        rag_chain = build_rag_chain(transcript)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the person
        box.update(label="Could not finish", state="error", expanded=True)
        st.error(
            f"{exc}\n\nThings to check: FFmpeg is installed, the link is public, "
            "and MISTRAL_API_KEY is set in your .env file."
        )
        return None

    elapsed = int(time.time() - started)
    box.update(label=f"Done in {elapsed // 60}m {elapsed % 60}s", state="complete", expanded=False)

    result = {
        "rid": int(time.time() * 1000),
        "title": str(title).strip().strip('"'),
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "source_label": source_label,
        "language": language,
        "generated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }
    result["txt"] = build_txt(result)
    try:
        result["pdf"] = build_pdf(result)
    except Exception:  # noqa: BLE001 - PDF is optional; TXT still works
        result["pdf"] = None
    return result


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------

def render_sidebar() -> None:
    result = st.session_state.result
    with st.sidebar:
        _h(f'<div class="brand">{APP_NAME}<small>AI meeting assistant</small></div>')

        if result is None:
            _h(
                '<div class="side-note">Audio is transcribed on this computer. '
                "Only transcript text, never the audio, is sent to the Mistral API "
                "to write the brief and answer your questions.</div>"
            )
            return

        _h('<div class="side-rule"></div><div class="side-title">Take it with you</div>')
        slug = slugify(result["title"])
        if result.get("pdf"):
            st.download_button(
                "Download PDF", data=result["pdf"], file_name=f"{slug}.pdf",
                mime="application/pdf", type="primary", key="dl_pdf",
            )
        st.download_button(
            "Download TXT", data=result["txt"], file_name=f"{slug}.txt",
            mime="text/plain", key="dl_txt",
        )
        _h('<div class="side-rule"></div>')
        if st.button("Start a new meeting", key="reset"):
            st.session_state.result = None
            st.session_state.chat = []
            st.rerun()


def render_landing() -> None:
    _h(
        '<div class="hero-title">Hand over the recording. Get back the decisions.</div>'
        '<p class="hero-copy">Paste a YouTube link or upload a meeting recording. Whisper transcribes it '
        "on your machine, then you get a summary, action items, decisions, open questions "
        "and a chat that answers from the transcript.</p>"
        + trace_svg()
    )

    form, gets = st.columns([1.25, 1], gap="large")

    with form:
        _h('<div class="form-title">Add a recording</div>')
        mode = st.radio(
            "Source", ["YouTube link", "Upload a file"],
            horizontal=True, label_visibility="collapsed", key="mode",
        )
        url, upload = "", None
        if mode == "YouTube link":
            url = st.text_input(
                "YouTube link", placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed", key="url",
            ).strip()
        else:
            upload = st.file_uploader(
                "Recording", type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "mp4", "mkv", "webm", "mov"],
                label_visibility="collapsed", key="upload",
            )

        language = st.radio(
            "Spoken language", ["english", "hinglish"],
            format_func=str.title, horizontal=True, key="language",
        )
        st.caption("Choose Hinglish for a Hindi and English mix. It is translated to English.")
        go = st.button("Generate brief", type="primary", key="go")

        if go:
            source, label = None, ""
            if mode == "YouTube link":
                if not url:
                    st.warning("Paste a YouTube link first.")
                else:
                    source, label = url, short_source(url)
            elif upload is None:
                st.warning("Choose a recording to upload first.")
            else:
                folder = tempfile.mkdtemp(prefix="minutes_")
                source = os.path.join(folder, os.path.basename(upload.name))
                with open(source, "wb") as fh:
                    fh.write(upload.getbuffer())
                label = upload.name

            if source:
                result = run_with_progress(source, language, label)
                if result:
                    st.session_state.result = result
                    st.session_state.chat = []
                    st.rerun()

    with gets:
        _h(
            '<div class="gets">'
            '<div class="gets-row"><b>Summary</b><span>The meeting in a few short paragraphs.</span></div>'
            '<div class="gets-row"><b>Action items</b><span>A list you can tick off as work gets done.</span></div>'
            '<div class="gets-row"><b>Decisions</b><span>What the group agreed on.</span></div>'
            '<div class="gets-row"><b>Open questions</b><span>What is still unresolved.</span></div>'
            '<div class="gets-row"><b>Transcript</b><span>Full text with keyword search.</span></div>'
            '<div class="gets-row"><b>Chat</b><span>Ask anything and get answers from the transcript.</span></div>'
            "</div>"
        )


def render_notes(value, kind: str, glyph: str, empty: str) -> None:
    items = parse_items(value)
    if not items:
        st.markdown(f'<p class="tab-lead">{empty}</p>', unsafe_allow_html=True)
        return
    _h("".join(
        f'<div class="note {kind}"><span class="glyph">{glyph}</span>'
        f'<span class="body">{inline_md(item)}</span></div>'
        for item in items
    ))


def render_transcript(text: str) -> None:
    query = st.text_input(
        "Search the transcript", placeholder="Search the transcript",
        label_visibility="collapsed", key="tx_search",
    ).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    paragraphs = [" ".join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]

    matches = 0
    out = []
    for para in paragraphs:
        safe = html.escape(para)
        if query:
            pattern = re.compile(re.escape(html.escape(query)), re.I)
            matches += len(pattern.findall(safe))
            safe = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
        out.append(f"<p>{safe}</p>")

    if query:
        st.caption(f"{plural(matches, 'match')} for \"{query}\"" if matches else f"No matches for \"{query}\".")
    _h(f'<div class="notebook">{"".join(out)}</div>')


def render_chat(result: dict) -> None:
    history = st.session_state.chat
    suggestions = [
        "What were the main topics?",
        "Who is responsible for what?",
        "What was decided, and why?",
    ]

    clicked = None
    if not history:
        st.markdown(
            '<p class="tab-lead">Ask about anything said in the recording. '
            "Answers come from the transcript only.</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(suggestions))
        for col, text in zip(cols, suggestions):
            if col.button(text, key=f"chip_{text}"):
                clicked = text

    log = st.container()
    typed = st.chat_input("Ask about this meeting")
    question = typed or clicked

    if question:
        history.append({"role": "user", "content": question})
        try:
            with st.spinner("Reading the transcript"):
                answer = ask_question(result["rag_chain"], question)
        except Exception as exc:  # noqa: BLE001
            answer = f"I could not get an answer: {exc}. Check that MISTRAL_API_KEY is set in your .env file."
        history.append({"role": "assistant", "content": str(answer)})

    with log:
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


def render_result() -> None:
    r = st.session_state.result

    actions = parse_items(r["action_items"])
    decisions = parse_items(r["key_decisions"])
    questions = parse_items(r["open_questions"])
    words = len(r["transcript"].split())
    minutes = max(1, round(words / 200))

    language = "Hinglish, translated to English" if r["language"] == "hinglish" else "English"
    _h(
        f'<div class="brief-title">{html.escape(r["title"])}</div>'
        f'<p class="brief-meta">From {html.escape(r["source_label"])}, generated {r["generated"]}. Spoken in {language}.</p>'
        f'<p class="brief-strip">The transcript runs to about <b>{words:,} words</b>, a <b>{minutes}-minute</b> read. '
        f"It produced <b>{plural(len(actions), 'action item')}</b>, "
        f"<b>{plural(len(decisions), 'decision')}</b> and "
        f"<b>{plural(len(questions), 'open question')}</b>.</p>"
    )

    tabs = st.tabs([
        "Summary",
        f"Action items ({len(actions)})",
        f"Decisions ({len(decisions)})",
        f"Open questions ({len(questions)})",
        "Transcript",
        "Chat",
    ])

    with tabs[0]:
        st.markdown(as_text(r["summary"]))

    with tabs[1]:
        if not actions:
            st.markdown('<p class="tab-lead">No action items were found in this recording.</p>', unsafe_allow_html=True)
        else:
            keys = [f"act_{r['rid']}_{i}" for i in range(len(actions))]
            done = sum(bool(st.session_state.get(k)) for k in keys)
            st.markdown(f'<p class="tab-lead">{done} of {len(actions)} done</p>', unsafe_allow_html=True)
            for key, item in zip(keys, actions):
                st.checkbox(item, key=key)

    with tabs[2]:
        render_notes(r["key_decisions"], "decision", "\u2713", "No decisions were found in this recording.")

    with tabs[3]:
        render_notes(r["open_questions"], "question", "?", "No open questions were found in this recording.")

    with tabs[4]:
        render_transcript(r["transcript"])

    with tabs[5]:
        render_chat(r)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("chat", [])

    inject_css()
    render_sidebar()
    if st.session_state.result is None:
        render_landing()
    else:
        render_result()


main()
