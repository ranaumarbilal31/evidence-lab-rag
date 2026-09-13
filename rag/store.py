from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

import numpy as np

from .config import EMBED_MODEL, DIMENSIONS, EMBED_VERSION
from .models import Chunk, Hit, RagError

INDEX_CONFIG = {"model": EMBED_MODEL, "dimensions": DIMENSIONS, "version": EMBED_VERSION}


class Store:
    """One owner only: a private visitor session or a local research process."""

    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS corpus (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")

    def get(self, key):
        row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(value)))

    def save_index(self, index):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO corpus VALUES (1,?)", (json.dumps(index.to_dict()),))

    def load_index(self):
        row = self.db.execute("SELECT value FROM corpus WHERE id=1").fetchone()
        return Index.from_dict(json.loads(row[0])) if row else None

    def close(self):
        self.db.close()


class Index:
    def __init__(self, chunks, vectors):
        self.chunks = chunks
        array = np.asarray(vectors, dtype=np.float32)
        if (array.shape != (len(chunks), DIMENSIONS) or not len(chunks)
                or not np.isfinite(array).all() or len({c.id for c in chunks}) != len(chunks)):
            raise RagError("Invalid or incompatible document index.")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms == 0).any():
            raise RagError("The embedding API returned an empty vector.")
        self.vectors = array / norms

    def retrieve(self, vector, k=8):
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (DIMENSIONS,) or not np.isfinite(query).all() or not np.linalg.norm(query):
            raise RagError("The query embedding is invalid.")
        scores = self.vectors @ (query / np.linalg.norm(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]

    def to_dict(self):
        return {"config": INDEX_CONFIG, "chunks": [asdict(c) for c in self.chunks], "vectors": self.vectors.tolist()}

    @classmethod
    def from_dict(cls, data):
        if data.get("config") != INDEX_CONFIG:
            raise RagError("The embedding model or settings changed. Rebuild the index explicitly.")
        return cls([Chunk(**item) for item in data["chunks"]], data["vectors"])


def build_index(chunks, client, progress=None):
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append(client.embed(chunk.text))
        if progress:
            progress((i + 1) / len(chunks))
    return Index(chunks, vectors)
