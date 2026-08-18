from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from src.knowledge import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "is",
    "are",
    "my",
    "i",
    "it",
    "on",
    "at",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _tfidf_matrix(docs: list[str]) -> tuple[list[str], np.ndarray]:
    tokenized = [tokenize(d) for d in docs]
    vocab = sorted({tok for doc in tokenized for tok in doc})
    index = {t: i for i, t in enumerate(vocab)}
    n = len(docs)
    df = Counter()
    for doc in tokenized:
        df.update(set(doc))
    idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1.0 for t in vocab], dtype=np.float32)
    matrix = np.zeros((n, len(vocab)), dtype=np.float32)
    for row, doc in enumerate(tokenized):
        counts = Counter(doc)
        length = max(len(doc), 1)
        for tok, c in counts.items():
            matrix[row, index[tok]] = (c / length) * idf[index[tok]]
    return vocab, matrix


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class EmbeddingIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks = chunks
        self.matrix = _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def search(self, query_vector: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[int(i)], float(scores[i])) for i in order]


class TfidfIndex:
    """Local keyword retriever so RAG still works without an embedding API."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        docs = [c.document for c in chunks]
        self.vocab, matrix = _tfidf_matrix(docs)
        self.matrix = _l2_normalize(matrix)
        self._index = {t: i for i, t in enumerate(self.vocab)}

    def encode_query(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        vec = np.zeros((len(self.vocab),), dtype=np.float32)
        counts = Counter(tokens)
        length = max(len(tokens), 1)
        for tok, c in counts.items():
            if tok in self._index:
                vec[self._index[tok]] = c / length
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        return vec

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.encode_query(query)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        results = []
        for i in order:
            score = float(scores[i])
            if score <= 0:
                continue
            results.append((self.chunks[int(i)], score))
        return results
