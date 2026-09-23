from __future__ import annotations

import re
from pathlib import Path

from citadel.schema import Chunk

_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def split_title(text: str, fallback: str) -> tuple[str, str]:
    match = _TITLE.search(text)
    if not match:
        return fallback, text.strip()
    title = match.group(1).strip()
    body = (text[: match.start()] + text[match.end() :]).strip()
    return title, body


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    max_chars: int = 280,
    overlap: int = 40,
) -> list[Chunk]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip() or title]
    pieces: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            start = 0
            step = max(max_chars - overlap, 1)
            while start < len(paragraph):
                pieces.append(paragraph[start : start + max_chars])
                if start + max_chars >= len(paragraph):
                    break
                start += step
            continue
        candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            pieces.append(buffer)
            buffer = paragraph
    if buffer:
        pieces.append(buffer)
    return [
        Chunk(id=f"{doc_id}-{index}", doc_id=doc_id, title=title, text=piece)
        for index, piece in enumerate(pieces)
    ]


def load_kb(directory: Path | None = None) -> list[Chunk]:
    root = directory or Path(__file__).resolve().parent / "data" / "kb"
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        title, body = split_title(path.read_text(encoding="utf-8"), path.stem)
        chunks.extend(chunk_document(path.stem, title, body))
    return chunks
