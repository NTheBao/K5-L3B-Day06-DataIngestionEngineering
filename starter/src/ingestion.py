"""Read PDF pages while preserving their source, page, and section metadata."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

_SECTIONS = (
    ("1. Đăng ký/Đăng nhập", re.compile(r"1\.\s*Đăng ký/Đăng nhập", re.IGNORECASE)),
    ("2. Xác thực tài khoản", re.compile(r"2\.\s*Xác thực tài khoản", re.IGNORECASE)),
    ("3. Liên kết với Agribank", re.compile(r"3\.\s*Liên kết với Agribank", re.IGNORECASE)),
    ("4. Huỷ liên kết ngân hàng", re.compile(r"4\.\s*Huỷ liên kết ngân hàng", re.IGNORECASE)),
    ("5. Nạp tiền vào ví ZaloPay", re.compile(r"5\.\s*Nạp tiền vào ví ZaloPay", re.IGNORECASE)),
    ("6. Rút tiền về Agribank", re.compile(r"6\.\s*Rút tiền về Agribank", re.IGNORECASE)),
    ("7. Thanh toán qua ví ZaloPay", re.compile(r"7\.\s*Thanh toán qua ví ZaloPay", re.IGNORECASE)),
)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    current_section: str | None = None

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        normalized_for_matching = " ".join(text.split())
        matches = [
            section
            for section, pattern in _SECTIONS
            if pattern.search(normalized_for_matching)
        ]

        if len(matches) == 1:
            current_section = matches[0]
        elif len(matches) > 1:
            current_section = None

        metadata: dict[str, str] = {
            "extraction_status": "ok" if text else "no_text",
        }
        if current_section is not None:
            metadata["section"] = current_section

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=metadata,
            )
        )

    return pages


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in normalized.split("\n"):
        cleaned = re.sub(r"[^\S\n]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
        elif lines and lines[-1] != "":
            lines.append("")

    return "\n".join(lines).strip()
