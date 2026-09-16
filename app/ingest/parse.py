from __future__ import annotations

import re
from itertools import groupby
from typing import Any

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.utils import extract_words
from pydantic import BaseModel

_HEADING_PATTERN = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z][A-Za-z ]{2,40})\s*$")

_KNOWN_SECTIONS = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "model architecture",
    "methodology",
    "method",
    "approach",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "acknowledgments",
    "references",
)

_COLUMN_BAND_FRACTION = (0.3, 0.7)
_COLUMN_SCAN_RESOLUTION = 2.0
_MIN_GUTTER_WIDTH_POINTS = 4.0
_LINE_GROUPING_TOLERANCE = 3.0


class PaperSection(BaseModel):
    paper_id: str
    section: str
    text: str


def _find_column_boundary(words: list[dict[str, Any]], *, page_width: float) -> float | None:
    """Procura o maior vão sem texto na faixa central da página (30%-70% da
    largura). Retorna None se a página não parecer ter duas colunas de fato
    (ex: título, página de referências em coluna única)."""
    if not words:
        return None

    band_start = page_width * _COLUMN_BAND_FRACTION[0]
    band_end = page_width * _COLUMN_BAND_FRACTION[1]
    step_count = int((band_end - band_start) / _COLUMN_SCAN_RESOLUTION) + 1
    positions = [band_start + i * _COLUMN_SCAN_RESOLUTION for i in range(step_count)]

    occupied: set[int] = set()
    for word in words:
        x0, x1 = word["x0"], word["x1"]
        if x1 < band_start or x0 > band_end:
            continue
        start_idx = max(0, int((x0 - band_start) / _COLUMN_SCAN_RESOLUTION))
        end_idx = min(len(positions) - 1, int((x1 - band_start) / _COLUMN_SCAN_RESOLUTION))
        occupied.update(range(start_idx, end_idx + 1))

    free_indices = [i for i in range(len(positions)) if i not in occupied]
    if not free_indices:
        return None

    best_run: list[int] = []
    current_run: list[int] = []
    for index in free_indices:
        if current_run and index != current_run[-1] + 1:
            best_run = current_run if len(current_run) > len(best_run) else best_run
            current_run = []
        current_run.append(index)
    best_run = current_run if len(current_run) > len(best_run) else best_run

    if len(best_run) * _COLUMN_SCAN_RESOLUTION < _MIN_GUTTER_WIDTH_POINTS:
        return None

    return positions[best_run[len(best_run) // 2]]


def _group_into_lines(words: list[dict[str, Any]]) -> list[str]:
    """Reconstrói linhas a partir de palavras soltas, agrupando por posição
    vertical (top) e ordenando por posição horizontal dentro da linha."""
    if not words:
        return []

    def bucket(word: dict[str, Any]) -> int:
        return round(word["top"] / _LINE_GROUPING_TOLERANCE)

    words_sorted = sorted(words, key=lambda w: (bucket(w), w["x0"]))
    return [" ".join(w["text"] for w in group) for _, group in groupby(words_sorted, key=bucket)]


def _extract_page_text(page: Page) -> str:
    """Extrai o texto de uma página respeitando a ordem de leitura real,
    incluindo o layout de duas colunas comum em papers do arXiv. Em vez de
    cortar a página por uma linha fixa (que corta palavras ao meio quando
    uma delas cai sobre o ponto de corte), agrupa as palavras pela posição
    e só separa em colunas quando encontra um vão real entre elas."""
    upright_chars = [char for char in page.chars if char.get("upright", True)]
    words = extract_words(upright_chars, x_tolerance=1, keep_blank_chars=False)
    boundary = _find_column_boundary(words, page_width=page.width)

    if boundary is None:
        return "\n".join(_group_into_lines(words))

    left_words = [word for word in words if (word["x0"] + word["x1"]) / 2 < boundary]
    right_words = [word for word in words if (word["x0"] + word["x1"]) / 2 >= boundary]
    return "\n".join(_group_into_lines(left_words) + _group_into_lines(right_words))


def _match_known_section(line: str) -> str | None:
    match = _HEADING_PATTERN.match(line.strip())
    if match is None:
        return None
    title = match.group(1).strip().lower()
    for known in _KNOWN_SECTIONS:
        if title == known or title.startswith(known):
            return known
    return None


def extract_sections(pdf_path: str, *, paper_id: str) -> list[PaperSection]:
    """Extrai texto por seção via pdfplumber. Heurística baseada em nomes de
    seção conhecidos (abstract, introduction, ..., references); seções que
    aparecem mais de uma vez (ex: por causa de um sumário) são mescladas."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(_extract_page_text(page) for page in pdf.pages)

    section_order: list[str] = ["abstract"]
    section_lines: dict[str, list[str]] = {"abstract": []}
    current_section = "abstract"

    for line in full_text.splitlines():
        heading = _match_known_section(line)
        if heading is not None:
            current_section = heading
            if current_section not in section_lines:
                section_lines[current_section] = []
                section_order.append(current_section)
            continue
        section_lines[current_section].append(line)

    sections = [
        PaperSection(paper_id=paper_id, section=name, text="\n".join(section_lines[name]).strip())
        for name in section_order
    ]
    return [section for section in sections if section.text]
