import io

import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, DictionaryObject

from rag.config import DIMENSIONS
from rag.ingest import ingest, UploadLimits, split_text
from rag.models import RagError
from rag.store import Store, Index, build_index


def pdf_bytes(text=None, encrypted=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("password")
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def test_pdf_text_and_provenance():
    chunks, _ = ingest([("policy.pdf", pdf_bytes("Refunds within 30 days."))])
    assert chunks[0].page == 1
    assert "30 days" in chunks[0].text
    assert len(chunks[0].document_hash) == 64


@pytest.mark.parametrize("data", [b"not a pdf", pdf_bytes(), pdf_bytes(encrypted=True)])
def test_reject_unreadable_scanned_and_encrypted_pdfs(data):
    with pytest.raises(RagError):
        ingest([("policy.pdf", data)])


@pytest.mark.parametrize("files,limits", [([("x.exe", b"x")], UploadLimits()),
    ([("a.txt", b"a"), ("b.txt", b"b")], UploadLimits(files=1)),
    ([("a.txt", b"abcd")], UploadLimits(bytes_per_file=3)),
    ([("a.txt", b"words " * 500)], UploadLimits(chunks=1)),
    ([("a.txt", b"\xff")], UploadLimits()), ([("a.txt", b" ")], UploadLimits())])
def test_file_limits(files, limits):
    with pytest.raises(RagError):
        ingest(files, limits)


def test_duplicate_contents_and_path_sanitization():
    chunks, warnings = ingest([("../../a.txt", b"Policy"), ("b.txt", b"Policy")])
    assert len(chunks) == 1 and chunks[0].filename == "a.txt"
    assert warnings


def test_chunking_preserves_tail_and_progress():
    text = " ".join(f"word{i}" for i in range(1000))
    parts = split_text(text)
    assert all(len(p) <= 800 for p in parts)
    assert parts[-1].endswith("word999")
    assert set(text.split()) == set(" ".join(parts).split())


def test_index_roundtrip_and_retrieval(tmp_path):
    chunks, _ = ingest([("a.txt", b"Annual leave"), ("b.txt", b"Refund policy")])
    vectors = np.eye(2, DIMENSIONS)
    index = Index(chunks, vectors)
    store = Store(tmp_path / "research.sqlite")
    store.save_index(index)
    assert store.load_index().retrieve(vectors[1])[0].chunk.filename == "b.txt"
    store.close()


def test_session_storage_isolation():
    a, b = Store(), Store()
    a.put("visitor", {"text": "Only A"})
    assert b.get("visitor") is None
    a.close()
    b.close()


def test_index_rejects_model_mismatch_and_invalid_vectors():
    chunks, _ = ingest([("a.txt", b"Policy")])
    index = Index(chunks, np.ones((1, DIMENSIONS)))
    serialized = index.to_dict()
    serialized["config"] = dict(serialized["config"], model="different-model")
    with pytest.raises(RagError):
        Index.from_dict(serialized)
    with pytest.raises(RagError):
        Index(chunks, np.zeros((1, DIMENSIONS)))

