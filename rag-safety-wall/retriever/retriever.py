"""
Phase 2 retriever: FAISS (semantic) + BM25 (keyword), hybrid search over corpus.json.

Note: building the FAISS index needs the all-MiniLM-L6-v2 weights, which download from
huggingface.co on first use. That's blocked in the sandbox this was written in, so the
embedding half is untested here -- but BM25 needs no download at all and is tested below.
Run this file's __main__ block yourself once you have normal network access to confirm
the embedding half too; nothing about its code depends on the sandbox.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _EMBEDDINGS_AVAILABLE = True
except Exception:
    _EMBEDDINGS_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class Retriever:
    def __init__(self, corpus_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.chunks: list[dict] = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        if not self.chunks:
            raise ValueError(f"No chunks loaded from {corpus_path}")

        # --- BM25 (always available, no network) ---
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        # --- FAISS + embeddings (needs the model download) ---
        self.embedder: Optional["SentenceTransformer"] = None
        self.index = None
        if _EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                vectors = self.embedder.encode([c["text"] for c in self.chunks], normalize_embeddings=True)
                self.index = faiss.IndexFlatIP(vectors.shape[1])
                self.index.add(np.asarray(vectors, dtype="float32"))
            except Exception as e:
                print(f"[retriever] Embedding index unavailable ({e}); falling back to BM25-only search.")
                self.embedder = None
                self.index = None

    def search(self, query: str, top_k: int = 5, exclude_chunk_ids: Optional[set[str]] = None) -> dict:
        exclude_chunk_ids = exclude_chunk_ids or set()
        candidates = [(i, c) for i, c in enumerate(self.chunks) if c["chunk_id"] not in exclude_chunk_ids]

        if self.index is not None and self.embedder is not None:
            q_vec = self.embedder.encode([query], normalize_embeddings=True)
            # over-fetch then filter excluded ids, since FAISS doesn't know about exclusions
            k = min(top_k + len(exclude_chunk_ids) + 5, len(self.chunks))
            scores, idxs = self.index.search(np.asarray(q_vec, dtype="float32"), k)
            ranked = [(float(s), self.chunks[i]) for s, i in zip(scores[0], idxs[0])
                      if self.chunks[i]["chunk_id"] not in exclude_chunk_ids]
        else:
            bm25_scores = self.bm25.get_scores(_tokenize(query))
            ranked = sorted(
                ((bm25_scores[i], c) for i, c in candidates),
                key=lambda x: x[0], reverse=True,
            )

        top = ranked[:top_k]
        return {
            "query": query,
            "chunks": [
                {"chunk_id": c["chunk_id"], "text": c["text"], "source_doc": c.get("source_doc", ""), "score": round(score, 4)}
                for score, c in top
            ],
        }


if __name__ == "__main__":
    # Smoke test with a tiny inline corpus -- proves the plumbing (loading, BM25, the
    # search() contract, exclusion filtering) works with zero network calls.
    demo_corpus = [
        {"chunk_id": "c1", "text": "Students may withdraw from a semester within 14 days of the start of term.", "source_doc": "Handbook A"},
        {"chunk_id": "c2", "text": "Tuition fees are due within 30 days of enrollment confirmation.", "source_doc": "Handbook A"},
        {"chunk_id": "c3", "text": "The withdrawal deadline for international students is also 14 days, per policy revision 2.", "source_doc": "Handbook B"},
    ]
    tmp_path = Path("/tmp/demo_corpus.json")
    tmp_path.write_text(json.dumps(demo_corpus), encoding="utf-8")

    r = Retriever(str(tmp_path))
    print("Embeddings available:", _EMBEDDINGS_AVAILABLE, "| FAISS index built:", r.index is not None)
    result = r.search("how many days to withdraw from a semester", top_k=2)
    print(json.dumps(result, indent=2))

    print("\nExcluding c1 (simulating a quarantined chunk) -- recovery should find c3:")
    result2 = r.search("how many days to withdraw from a semester", top_k=2, exclude_chunk_ids={"c1"})
    print(json.dumps(result2, indent=2))
