"""A friendly, deployment-ready Gradio interface for finding names in text."""

from __future__ import annotations

import html
import logging
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import gradio as gr


APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = APP_DIR / "checkpoints" / "transformer_ner_model"
LOCAL_TOKENIZER_DIR = APP_DIR / "checkpoints" / "transformer_ner_tokenizer"

# On cloud hosts, set HF_TOKEN and optionally HF_MODEL_ID to use hosted inference.
# If the project checkpoint is available, it remains the preferred model.
REMOTE_MODEL_ID = os.getenv("HF_MODEL_ID", "dslim/distilbert-NER").strip()
BACKEND_PREFERENCE = os.getenv("NER_BACKEND", "auto").strip().lower()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

MAX_INPUT_CHARS = 3_000
CHUNK_SIZE = 900
MAX_MODEL_TOKENS = 384
LIVE_MIN_CHARS = 3

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("name-finder")


@dataclass(frozen=True)
class EntitySpan:
    """One model prediction aligned to the original passage."""

    text: str
    label: str
    start: int
    end: int


ENTITY_META = {
    "PER": {
        "name": "Person",
        "plural": "People",
        "class_name": "person",
    },
    "ORG": {
        "name": "Organization",
        "plural": "Organizations",
        "class_name": "organization",
    },
    "LOC": {
        "name": "Place",
        "plural": "Places",
        "class_name": "place",
    },
    "MISC": {
        "name": "Other name",
        "plural": "Other names",
        "class_name": "other",
    },
}

SAMPLES = {
    "Global news": (
        "Elon Musk met with officials from the United Nations in New York "
        "during the climate summit."
    ),
    "Business": (
        "Apple opened a new office in Cairo while Tim Cook visited Egypt."
    ),
    "Sports": (
        "Serena Williams attended Wimbledon in London with Nike representatives."
    ),
}


THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="green",
    neutral_hue="zinc",
    radius_size=gr.themes.sizes.radius_sm,
    font=["Aptos", "Segoe UI", "Arial", "sans-serif"],
).set(
    body_background_fill="#0c0f13",
    body_background_fill_dark="#0c0f13",
    body_text_color="#f2f6f8",
    body_text_color_dark="#f2f6f8",
    block_background_fill="#12171d",
    block_background_fill_dark="#12171d",
    block_border_color="#2b3742",
    block_border_color_dark="#2b3742",
    input_background_fill="#10151a",
    input_background_fill_dark="#10151a",
    input_border_color="#34434f",
    input_border_color_dark="#34434f",
    button_primary_background_fill="#356b4d",
    button_primary_background_fill_dark="#356b4d",
    button_primary_text_color="#f4faf6",
    button_primary_text_color_dark="#f4faf6",
)


CSS = r"""
:root {
    --ink: #f2f6f8;
    --muted: #9aa8b2;
    --quiet: #71808b;
    --page: #0c0f13;
    --surface: #12171d;
    --surface-raised: #181f27;
    --surface-deep: #0f1419;
    --line: #2b3742;
    --line-strong: #41515f;
    --accent: #78b68e;
    --action: #356b4d;
    --action-hover: #407a59;
    --person: #72d2ff;
    --person-bg: #102c3a;
    --organization: #ff8fa3;
    --organization-bg: #381b25;
    --place: #9fe870;
    --place-bg: #1d3020;
    --other: #ffc96b;
    --other-bg: #392b13;
}

* {
    box-sizing: border-box;
}

html,
body,
.gradio-container {
    min-height: 100%;
    background: var(--page) !important;
    color: var(--ink) !important;
}

body {
    margin: 0;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100vh !important;
    padding: 0 !important;
}

.gradio-container .main {
    min-height: 100vh;
    padding: 0 !important;
}

#app-root {
    min-height: 100vh;
    gap: 0 !important;
}

#app-header {
    border-bottom: 1px solid var(--line);
    background: var(--surface);
    padding: 20px clamp(22px, 3vw, 52px);
}

.brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-mark {
    position: relative;
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
    border: 1px solid var(--line-strong);
    border-radius: 6px;
    background: var(--surface-deep);
}

.brand-mark span {
    position: absolute;
    left: 8px;
    right: 8px;
    height: 3px;
    border-radius: 2px;
}

.brand-mark span:nth-child(1) { top: 8px; background: var(--person); }
.brand-mark span:nth-child(2) { top: 14px; background: var(--organization); }
.brand-mark span:nth-child(3) { top: 20px; background: var(--place); }
.brand-mark span:nth-child(4) { top: 26px; background: var(--other); }

.brand-name {
    margin: 0;
    color: var(--ink);
    font-size: clamp(1.25rem, 2vw, 1.6rem);
    font-weight: 750;
    line-height: 1.05;
    letter-spacing: 0;
}

.brand-subtitle {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.3;
}

.legend {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 8px 18px;
}

.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: var(--muted);
    font-size: 0.78rem;
    white-space: nowrap;
}

.legend-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    flex: 0 0 8px;
    border-radius: 50%;
}

.dot-person { background: var(--person); }
.dot-organization { background: var(--organization); }
.dot-place { background: var(--place); }
.dot-other { background: var(--other); }

#workspace {
    min-height: calc(100vh - 80px);
    margin: 0 !important;
    gap: 0 !important;
    align-items: stretch;
}

#input-pane,
#result-pane {
    min-width: 0;
    padding: clamp(24px, 3vw, 48px);
}

#input-pane {
    background: var(--surface-deep);
    border-right: 1px solid var(--line);
}

#result-pane {
    background: var(--page);
}

.pane-heading {
    margin-bottom: 20px;
}

.pane-kicker {
    display: block;
    margin-bottom: 8px;
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0;
}

.pane-title {
    margin: 0;
    color: var(--ink);
    font-size: clamp(1.35rem, 2vw, 1.85rem);
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: 0;
}

.pane-copy {
    max-width: 58ch;
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.55;
}

#source-text {
    margin-top: 2px;
}

.field-label {
    margin: 0 0 8px;
    color: #cbd5da;
    font-size: 0.84rem;
    font-weight: 650;
    line-height: 1.3;
}

#source-text textarea {
    min-height: clamp(260px, 39vh, 440px) !important;
    resize: vertical !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 6px !important;
    background: #0b1014 !important;
    color: var(--ink) !important;
    font-size: 1.04rem !important;
    line-height: 1.7 !important;
    padding: 18px !important;
    box-shadow: none !important;
}

#source-text textarea::placeholder {
    color: #6f7d87 !important;
}

#source-text textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(120, 182, 142, 0.16) !important;
}

.action-row {
    margin-top: 10px;
    gap: 10px !important;
}

#find-button,
#clear-button,
.sample-button {
    min-height: 44px;
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
    transition: background-color 140ms ease, border-color 140ms ease, transform 140ms ease;
}

#find-button {
    border: 1px solid var(--accent) !important;
    background: var(--action) !important;
    color: #f4faf6 !important;
}

#find-button:hover {
    border-color: #8bc9a1 !important;
    background: var(--action-hover) !important;
    transform: translateY(-1px);
}

#clear-button,
.sample-button {
    border: 1px solid var(--line-strong) !important;
    background: var(--surface-raised) !important;
    color: var(--ink) !important;
}

#clear-button:hover,
.sample-button:hover {
    border-color: #667783 !important;
    background: #202933 !important;
}

#find-button:focus-visible,
#clear-button:focus-visible,
.sample-button:focus-visible {
    outline: 3px solid rgba(120, 182, 142, 0.3) !important;
    outline-offset: 2px;
}

.sample-heading {
    margin: 24px 0 10px;
    padding-top: 20px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0;
}

.sample-row {
    gap: 8px !important;
}

.sample-button {
    min-width: 120px !important;
    font-size: 0.84rem !important;
}

#result-html {
    display: block;
    min-height: clamp(440px, 68vh, 740px);
}

.result-shell {
    min-height: clamp(440px, 68vh, 740px);
    border: 1px solid var(--line);
    border-radius: 7px;
    overflow: hidden;
    background: var(--surface);
}

.result-status {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    min-height: 54px;
    padding: 13px 18px;
    border-bottom: 1px solid var(--line);
    background: var(--surface-raised);
}

.result-status strong {
    color: var(--ink);
    font-size: 0.94rem;
    font-weight: 700;
}

.result-status span {
    color: var(--muted);
    font-size: 0.78rem;
    text-align: right;
}

.document-output {
    min-height: 230px;
    padding: clamp(20px, 2.5vw, 34px);
    border-bottom: 1px solid var(--line);
    background: #0f1419;
}

.document-copy {
    margin: 0;
    color: #e8eef2;
    font-size: clamp(1.05rem, 1.3vw, 1.22rem);
    line-height: 2.05;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.entity-mark {
    display: inline;
    margin: 0 0.06em;
    padding: 0.16em 0.3em 0.2em;
    border-radius: 4px;
    color: var(--ink);
    font-weight: 650;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
}

.entity-mark .entity-type {
    display: inline-block;
    margin-left: 0.4em;
    font-size: 0.58em;
    font-weight: 800;
    line-height: 1;
    text-transform: uppercase;
    white-space: nowrap;
    letter-spacing: 0;
    vertical-align: 0.08em;
}

.entity-person {
    border-bottom: 2px solid var(--person);
    background: var(--person-bg);
}

.entity-person .entity-type { color: var(--person); }

.entity-organization {
    border-bottom: 2px solid var(--organization);
    background: var(--organization-bg);
}

.entity-organization .entity-type { color: var(--organization); }

.entity-place {
    border-bottom: 2px solid var(--place);
    background: var(--place-bg);
}

.entity-place .entity-type { color: var(--place); }

.entity-other {
    border-bottom: 2px solid var(--other);
    background: var(--other-bg);
}

.entity-other .entity-type { color: var(--other); }

.entity-list {
    padding: 20px clamp(20px, 2.5vw, 34px) 24px;
}

.entity-list-title {
    margin: 0 0 12px;
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0;
}

.entity-rows {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0 24px;
}

.entity-row {
    display: grid;
    grid-template-columns: 9px minmax(0, 1fr) auto;
    align-items: center;
    gap: 10px;
    min-height: 48px;
    border-top: 1px solid var(--line);
}

.entity-row:nth-child(1),
.entity-row:nth-child(2) {
    border-top-color: transparent;
}

.entity-swatch {
    width: 7px;
    height: 28px;
    border-radius: 3px;
}

.swatch-person { background: var(--person); }
.swatch-organization { background: var(--organization); }
.swatch-place { background: var(--place); }
.swatch-other { background: var(--other); }

.entity-name {
    min-width: 0;
    color: var(--ink);
    font-size: 0.91rem;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.entity-kind {
    color: var(--muted);
    font-size: 0.75rem;
    white-space: nowrap;
}

.empty-result,
.error-result {
    display: flex;
    min-height: clamp(440px, 68vh, 740px);
    align-items: center;
    justify-content: center;
    padding: 32px;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: var(--surface);
    text-align: center;
}

.empty-inner,
.error-inner {
    max-width: 390px;
}

.empty-glyph {
    display: grid;
    width: 56px;
    height: 56px;
    margin: 0 auto 18px;
    place-items: center;
    border: 1px solid var(--line-strong);
    border-radius: 7px;
    background: var(--surface-deep);
    color: var(--accent);
    font-family: Consolas, monospace;
    font-size: 1.25rem;
    font-weight: 700;
}

.empty-result h3,
.error-result h3 {
    margin: 0 0 8px;
    color: var(--ink);
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0;
}

.empty-result p,
.error-result p {
    margin: 0;
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.55;
}

.error-result {
    border-color: #70414b;
    background: #191317;
}

footer {
    display: none !important;
}

@media (max-width: 980px) {
    .brand-row {
        align-items: flex-start;
        flex-direction: column;
        gap: 14px;
    }

    .legend {
        justify-content: flex-start;
    }

    #workspace {
        flex-direction: column;
    }

    #input-pane {
        border-right: 0;
        border-bottom: 1px solid var(--line);
    }

    #source-text textarea {
        min-height: 230px !important;
    }

    #result-html,
    .result-shell,
    .empty-result,
    .error-result {
        min-height: 420px;
    }
}

@media (max-width: 620px) {
    #app-header,
    #input-pane,
    #result-pane {
        padding: 20px 16px;
    }

    .legend {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        width: 100%;
    }

    .action-row,
    .sample-row {
        flex-direction: column;
    }

    #find-button,
    #clear-button,
    .sample-button {
        width: 100%;
    }

    .entity-rows {
        grid-template-columns: 1fr;
    }

    .entity-row:nth-child(2) {
        border-top-color: var(--line);
    }

    .result-status {
        align-items: flex-start;
        flex-direction: column;
        gap: 4px;
    }

    .result-status span {
        text-align: left;
    }
}

@media (prefers-reduced-motion: reduce) {
    #find-button,
    #clear-button,
    .sample-button {
        transition: none;
    }
}
"""


HEADER_HTML = """
<header class="brand-row">
    <div class="brand-lockup">
        <div class="brand-mark" aria-hidden="true">
            <span></span><span></span><span></span><span></span>
        </div>
        <div>
            <h1 class="brand-name">Name Finder</h1>
            <p class="brand-subtitle">People, places, organizations, and other named details</p>
        </div>
    </div>
    <div class="legend" aria-label="Highlight colors">
        <span class="legend-item"><i class="legend-dot dot-person"></i>Person</span>
        <span class="legend-item"><i class="legend-dot dot-organization"></i>Organization</span>
        <span class="legend-item"><i class="legend-dot dot-place"></i>Place</span>
        <span class="legend-item"><i class="legend-dot dot-other"></i>Other name</span>
    </div>
</header>
"""

INPUT_HEADING_HTML = """
<div class="pane-heading">
    <span class="pane-kicker">Source text</span>
    <h2 class="pane-title">What should we check?</h2>
    <p class="pane-copy">Add a sentence or short passage, up to 3,000 characters.</p>
</div>
"""

RESULT_HEADING_HTML = """
<div class="pane-heading">
    <span class="pane-kicker">Result</span>
    <h2 class="pane-title">Highlighted names</h2>
    <p class="pane-copy">Each name is marked by type for quick scanning.</p>
</div>
"""

EMPTY_RESULT_HTML = """
<section class="empty-result" aria-live="polite">
    <div class="empty-inner">
        <div class="empty-glyph" aria-hidden="true">Aa</div>
        <h3>Your highlighted text will appear here</h3>
        <p>Add a passage or choose a sample to begin.</p>
    </div>
</section>
"""


def _read_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize_label(item: Any) -> str | None:
    raw_label = str(
        _read_value(item, "entity_group")
        or _read_value(item, "entity")
        or ""
    ).upper()
    label = re.sub(r"^[BI]-", "", raw_label)
    if label in ENTITY_META:
        return label
    return None


def _iter_text_chunks(text: str) -> Iterable[tuple[str, int]]:
    """Split long passages at natural boundaries while preserving offsets."""

    start = 0
    while start < len(text):
        hard_end = min(start + CHUNK_SIZE, len(text))
        end = hard_end

        if hard_end < len(text):
            search_start = start + int(CHUNK_SIZE * 0.6)
            window = text[search_start:hard_end]
            boundary = -1
            for match in re.finditer(r"(?:[.!?]\s+|\n+|\s+)", window):
                boundary = match.end()
            if boundary > 0:
                end = search_start + boundary

        if end <= start:
            end = hard_end

        yield text[start:end], start
        start = end


class EntityRecognizer:
    """Lazy model adapter with local and hosted inference paths."""

    def __init__(self) -> None:
        self._runner: Any = None
        self._mode: str | None = None
        self._lock = threading.RLock()

    def _preferred_mode(self) -> str:
        if BACKEND_PREFERENCE in {"local", "api"}:
            return BACKEND_PREFERENCE
        if LOCAL_MODEL_DIR.is_dir():
            return "local"
        if HF_TOKEN:
            return "api"
        return "local"

    def _load_api_runner(self) -> Any:
        from huggingface_hub import InferenceClient

        LOGGER.info("Using hosted inference")
        return InferenceClient(
            model=REMOTE_MODEL_ID,
            token=HF_TOKEN,
            provider="hf-inference",
            timeout=45,
        )

    def _load_local_runner(self) -> Any:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_source = (
            str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else REMOTE_MODEL_ID
        )
        tokenizer_source = (
            str(LOCAL_TOKENIZER_DIR)
            if LOCAL_TOKENIZER_DIR.is_dir()
            else model_source
        )

        LOGGER.info("Loading the local name-finding model")
        if not torch.cuda.is_available():
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        tokenizer.model_max_length = min(
            int(getattr(tokenizer, "model_max_length", MAX_MODEL_TOKENS)),
            MAX_MODEL_TOKENS,
        )
        model = AutoModelForTokenClassification.from_pretrained(model_source)
        model.eval()

        return pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )

    def _ensure_runner(self) -> None:
        if self._runner is not None:
            return

        self._mode = self._preferred_mode()
        self._runner = (
            self._load_api_runner()
            if self._mode == "api"
            else self._load_local_runner()
        )

    def _run_chunk(self, chunk: str) -> list[Any]:
        self._ensure_runner()
        if self._mode == "api":
            return list(
                self._runner.token_classification(
                    chunk,
                    aggregation_strategy="simple",
                )
            )

        return list(self._runner(chunk))

    def find(self, text: str) -> list[EntitySpan]:
        spans: list[EntitySpan] = []

        with self._lock:
            for chunk, chunk_offset in _iter_text_chunks(text):
                try:
                    predictions = self._run_chunk(chunk)
                except Exception:
                    if self._mode != "api" or BACKEND_PREFERENCE == "api":
                        raise

                    LOGGER.warning(
                        "Hosted inference was unavailable; using local inference",
                        exc_info=True,
                    )
                    self._mode = "local"
                    self._runner = self._load_local_runner()
                    predictions = self._run_chunk(chunk)

                for item in predictions:
                    label = _normalize_label(item)
                    start = _read_value(item, "start")
                    end = _read_value(item, "end")
                    if label is None or not isinstance(start, int) or not isinstance(end, int):
                        continue

                    absolute_start = chunk_offset + start
                    absolute_end = chunk_offset + end
                    if absolute_start < absolute_end <= len(text):
                        spans.append(
                            EntitySpan(
                                text=text[absolute_start:absolute_end],
                                label=label,
                                start=absolute_start,
                                end=absolute_end,
                            )
                        )

        return _remove_overlaps(spans)


RECOGNIZER = EntityRecognizer()


def _remove_overlaps(spans: list[EntitySpan]) -> list[EntitySpan]:
    clean: list[EntitySpan] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start))):
        if clean and span.start < clean[-1].end:
            continue
        if clean and span.start == clean[-1].end and span.label == clean[-1].label:
            previous = clean[-1]
            clean[-1] = EntitySpan(
                text=previous.text + span.text,
                label=previous.label,
                start=previous.start,
                end=span.end,
            )
            continue
        clean.append(span)
    return clean


def _highlighted_passage(text: str, spans: list[EntitySpan]) -> str:
    pieces: list[str] = []
    cursor = 0

    for span in spans:
        pieces.append(html.escape(text[cursor:span.start]))
        meta = ENTITY_META[span.label]
        pieces.append(
            f'<mark class="entity-mark entity-{meta["class_name"]}">'
            f'{html.escape(span.text)}'
            f'<span class="entity-type">{meta["name"]}</span>'
            "</mark>"
        )
        cursor = span.end

    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _entity_rows(spans: list[EntitySpan]) -> str:
    unique_entities: list[EntitySpan] = []
    seen: set[tuple[str, str]] = set()

    for span in spans:
        key = (span.text.casefold(), span.label)
        if key not in seen:
            unique_entities.append(span)
            seen.add(key)

    rows = []
    for span in unique_entities:
        meta = ENTITY_META[span.label]
        rows.append(
            '<div class="entity-row">'
            f'<span class="entity-swatch swatch-{meta["class_name"]}"></span>'
            f'<span class="entity-name">{html.escape(span.text)}</span>'
            f'<span class="entity-kind">{meta["name"]}</span>'
            "</div>"
        )
    return "".join(rows)


def _count_summary(spans: list[EntitySpan]) -> str:
    counts = Counter(span.label for span in spans)
    parts = []
    for label in ("PER", "ORG", "LOC", "MISC"):
        count = counts.get(label, 0)
        if count:
            label_text = ENTITY_META[label]["name" if count == 1 else "plural"]
            parts.append(f"{count} {label_text.lower()}")
    return " &middot; ".join(parts)


def _render_result(text: str, spans: list[EntitySpan]) -> str:
    if not spans:
        return """
        <section class="result-shell" aria-live="polite">
            <div class="result-status">
                <strong>No names found</strong>
                <span>Try a passage with a person, place, or organization.</span>
            </div>
            <div class="document-output">
                <p class="document-copy">{}</p>
            </div>
            <div class="entity-list">
                <p class="entity-list-title">Names found</p>
                <p class="pane-copy">There are no highlighted names in this passage.</p>
            </div>
        </section>
        """.format(html.escape(text))

    total = len(spans)
    total_label = "name" if total == 1 else "names"
    return f"""
    <section class="result-shell" aria-live="polite">
        <div class="result-status">
            <strong>{total} {total_label} found</strong>
            <span>{_count_summary(spans)}</span>
        </div>
        <div class="document-output">
            <p class="document-copy">{_highlighted_passage(text, spans)}</p>
        </div>
        <div class="entity-list">
            <p class="entity-list-title">Names found</p>
            <div class="entity-rows">{_entity_rows(spans)}</div>
        </div>
    </section>
    """


def _render_error(title: str, message: str) -> str:
    return f"""
    <section class="error-result" aria-live="assertive">
        <div class="error-inner">
            <div class="empty-glyph" aria-hidden="true">!</div>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(message)}</p>
        </div>
    </section>
    """


def find_names(text: str) -> str:
    clean_text = (text or "").replace("\r\n", "\n").strip()
    if not clean_text:
        return _render_error(
            "Add some text first",
            "Enter a sentence or choose one of the samples, then find the names.",
        )
    if len(clean_text) > MAX_INPUT_CHARS:
        return _render_error(
            "This passage is a little long",
            f"Shorten it to {MAX_INPUT_CHARS:,} characters and try again.",
        )

    try:
        spans = RECOGNIZER.find(clean_text)
    except Exception:
        LOGGER.exception("Name finding failed")
        return _render_error(
            "We could not check this text",
            "The service may be busy. Wait a moment, then try again.",
        )

    return _render_result(clean_text, spans)


def find_names_live(text: str) -> str:
    """Keep the result in sync without showing errors for unfinished input."""

    clean_text = (text or "").strip()
    if len(clean_text) < LIVE_MIN_CHARS:
        return EMPTY_RESULT_HTML
    return find_names(text)


def clear_workspace() -> tuple[str, str]:
    return "", EMPTY_RESULT_HTML


def use_sample(sample_name: str) -> tuple[str, str]:
    sample_text = SAMPLES[sample_name]
    return sample_text, find_names(sample_text)


with gr.Blocks(
    title="Name Finder",
    fill_width=True,
    fill_height=True,
    analytics_enabled=False,
) as demo:
    with gr.Column(elem_id="app-root"):
        gr.HTML(HEADER_HTML, elem_id="app-header", padding=False)

        with gr.Row(elem_id="workspace", equal_height=True):
            with gr.Column(scale=5, min_width=360, elem_id="input-pane"):
                gr.HTML(INPUT_HEADING_HTML, padding=False)
                gr.HTML('<p class="field-label">Text to check</p>', padding=False)
                source_text = gr.Textbox(
                    label="Text to check",
                    show_label=False,
                    container=False,
                    placeholder="For example: Marie Curie studied at the University of Paris...",
                    lines=12,
                    max_lines=18,
                    max_length=MAX_INPUT_CHARS,
                    autofocus=True,
                    elem_id="source-text",
                )

                with gr.Row(elem_classes=["action-row"]):
                    find_button = gr.Button(
                        "Find names",
                        variant="primary",
                        elem_id="find-button",
                        scale=3,
                    )
                    clear_button = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_id="clear-button",
                        scale=1,
                    )

                gr.HTML('<p class="sample-heading">Try a sample</p>', padding=False)
                with gr.Row(elem_classes=["sample-row"]):
                    news_sample = gr.Button(
                        "Global news",
                        size="sm",
                        elem_classes=["sample-button"],
                    )
                    business_sample = gr.Button(
                        "Business",
                        size="sm",
                        elem_classes=["sample-button"],
                    )
                    sports_sample = gr.Button(
                        "Sports",
                        size="sm",
                        elem_classes=["sample-button"],
                    )

            with gr.Column(scale=7, min_width=420, elem_id="result-pane"):
                gr.HTML(RESULT_HEADING_HTML, padding=False)
                result_html = gr.HTML(
                    value=EMPTY_RESULT_HTML,
                    elem_id="result-html",
                    padding=False,
                )

    find_event = find_button.click(
        fn=find_names,
        inputs=source_text,
        outputs=result_html,
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id="name-finding",
    )
    live_event = source_text.input(
        fn=find_names_live,
        inputs=source_text,
        outputs=result_html,
        show_progress="hidden",
        trigger_mode="always_last",
        concurrency_limit=1,
        concurrency_id="name-finding",
        api_visibility="private",
    )
    submit_event = source_text.submit(
        fn=find_names,
        inputs=source_text,
        outputs=result_html,
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id="name-finding",
    )
    clear_button.click(
        fn=clear_workspace,
        inputs=None,
        outputs=[source_text, result_html],
        show_progress="hidden",
        cancels=[find_event, live_event, submit_event],
    )
    news_sample.click(
        fn=lambda: use_sample("Global news"),
        inputs=None,
        outputs=[source_text, result_html],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id="name-finding",
    )
    business_sample.click(
        fn=lambda: use_sample("Business"),
        inputs=None,
        outputs=[source_text, result_html],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id="name-finding",
    )
    sports_sample.click(
        fn=lambda: use_sample("Sports"),
        inputs=None,
        outputs=[source_text, result_html],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id="name-finding",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
        theme=THEME,
        css=CSS,
    )
