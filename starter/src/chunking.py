"""Fixed-size and paragraph-aware chunking strategies."""
import hashlib
import re

from .models import Chunk, DocumentPage

_BLOCK_START = re.compile(
    r"^(?:[•·]\s*|\d+\.\s+|Bước\s+\d+\b|Cách\s+\d+\b|Phương thức\b)",
    re.IGNORECASE,
)


def _make_chunk(
    page: DocumentPage,
    index: int,
    text: str,
    char_start: int | None,
    char_end: int | None,
) -> Chunk:
    digest = hashlib.sha256(
        f"{page.document_id}:{page.page}:{index}:{char_start}:{char_end}:{text}".encode("utf-8")
    ).hexdigest()[:24]
    metadata = dict(page.metadata)
    metadata.update({"chunk_index": index, "char_count": len(text)})
    if char_start is not None and char_end is not None:
        metadata.update({"char_start": char_start, "char_end": char_end})
    return Chunk(
        chunk_id=digest,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    text = page.text
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(_make_chunk(page, len(chunks), chunk_text, start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _split_long_block(block: str, max_size: int) -> list[str]:
    pieces: list[str] = []
    remaining = block
    while len(remaining) > max_size:
        split_at = remaining.rfind(" ", 0, max_size + 1)
        if split_at <= 0:
            split_at = max_size
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _logical_blocks(page: DocumentPage) -> list[str]:
    blocks: list[str] = []
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            blocks.append(" ".join(current_lines))
            current_lines = []

    for raw_line in page.text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()

        if not stripped:
            flush()
        elif not current_lines or _BLOCK_START.match(stripped):
            flush()
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    flush()
    return blocks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    if max_size <= 0:
        raise ValueError("Maximum chunk size must be greater than zero.")

    blocks = _logical_blocks(page)
    units = [part for block in blocks for part in _split_long_block(block, max_size)]

    chunks: list[Chunk] = []
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        if current_parts:
            chunks.append(
                _make_chunk(
                    page,
                    len(chunks),
                    "\n".join(current_parts),
                    None,
                    None,
                )
            )
            current_parts = []

    for part in units:
        candidate = "\n".join((*current_parts, part))
        if current_parts and len(candidate) > max_size:
            flush()
        current_parts.append(part)
    flush()
    return chunks
