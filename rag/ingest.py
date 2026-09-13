from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

from .models import Chunk, RagError


@dataclass(frozen=True)
class UploadLimits:
    files: int = 3
    bytes_per_file: int = 2 * 1024 * 1024
    chunks: int = 50
    pages: int = 30


LOCAL_LIMITS = UploadLimits(100, 10 * 1024 * 1024, 5000, 1000)


def split_text(text: str, size: int = 800, overlap: int = 160) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            sentence = max(text.rfind(". ", start + size // 2, end),
                           text.rfind("? ", start + size // 2, end))
            boundary = sentence + 1 if sentence >= 0 else text.rfind(" ", start + 1, end)
            if boundary > start:
                end = boundary
        pieces.append(text[start:end].strip())
        if end == len(text):
            break
        new_start = max(start + 1, end - overlap)
        space = text.find(" ", new_start, end)
        start = space + 1 if space >= 0 else end
    return pieces


def ingest(files: list[tuple[str, bytes]], limits: UploadLimits = UploadLimits()) -> tuple[list[Chunk], list[str]]:
    if not files or len(files) > limits.files:
        raise RagError(f"Choose between 1 and {limits.files} files.")
    chunks, warnings, seen = [], [], set()
    for original_name, content in files:
        name = re.split(r"[/\\]", original_name)[-1][:160]
        if not content or len(content) > limits.bytes_per_file:
            raise RagError("A file is empty or exceeds the allowed upload size.")
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen:
            warnings.append("Duplicate file content was indexed once.")
            continue
        seen.add(digest)
        suffix = name.rsplit(".", 1)[-1].lower()
        if suffix == "pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content), strict=False)
                if reader.is_encrypted:
                    raise RagError("Encrypted PDFs are not supported. Supply an unencrypted text PDF.")
                if len(reader.pages) > limits.pages:
                    raise RagError(f"A PDF exceeds the {limits.pages}-page limit.")
                pages = []
                for number, page in enumerate(reader.pages, 1):
                    # Bound unusually large decompressed page streams before text extraction.
                    stream = page.get_contents()
                    if stream and len(stream.get_data()) > 8 * 1024 * 1024:
                        raise RagError("A PDF page is too complex for this small demo.")
                    text = page.extract_text() or ""
                    if not text.strip():
                        warnings.append(f"Page {number} has no extractable text (blank or scanned).")
                    pages.append((number, text))
                if not any(text.strip() for _, text in pages):
                    raise RagError("This PDF has no extractable text. Scanned PDFs need OCR, which this demo does not include.")
            except RagError:
                raise
            except Exception:
                raise RagError("The PDF could not be read. Supply a valid text PDF.") from None
        elif suffix in {"txt", "md"}:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise RagError("Text files must use UTF-8 encoding.") from None
            if "\x00" in text:
                raise RagError("Binary content is not supported as a text file.")
            pages = [(None, text)]
        else:
            raise RagError("Supported files are text PDFs, .txt, and .md.")
        for page, text in pages:
            for index, part in enumerate(split_text(text)):
                chunk_id = f"{digest[:16]}-{page or 0}-{index}"
                chunks.append(Chunk(chunk_id, digest, name, page, part))
                if len(chunks) > limits.chunks:
                    raise RagError(f"These documents exceed {limits.chunks} chunks. Use fewer or shorter files.")
    if not chunks:
        raise RagError("No readable text was found.")
    return chunks, warnings

