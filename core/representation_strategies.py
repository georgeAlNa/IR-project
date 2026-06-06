from __future__ import annotations

import math
from collections import Counter
from hashlib import blake2b
from typing import Any, Literal, Protocol

try:  # pragma: no cover - optional dependency
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover - optional dependency fallback
    Word2Vec = None

from core.inverted_index import InvertedIndex


RepresentationType = Literal["tfidf", "bm25", "embeddings"]


class RepresentationStrategy(Protocol):
    representation_type: RepresentationType

    def fit(self, corpus: list[list[str]], index: InvertedIndex) -> None:
        """Fit the strategy on the indexed corpus."""

    def represent_query(self, query_tokens: list[str], index: InvertedIndex) -> Any:
        """Convert a query into a representation-specific payload."""


class TFIDFRepresentationStrategy:
    representation_type: RepresentationType = "tfidf"

    def fit(self, corpus: list[list[str]], index: InvertedIndex) -> None:
        self._document_count = index.document_count

    def represent_query(self, query_tokens: list[str], index: InvertedIndex) -> dict[str, float]:
        if not query_tokens:
            return {}

        query_term_frequencies = Counter(query_tokens)
        vector: dict[str, float] = {}

        for term in sorted(query_term_frequencies):
            document_frequency = index.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue
            idf = math.log((index.document_count + 1) / (document_frequency + 1)) + 1.0
            vector[term] = query_term_frequencies[term] * idf

        return vector


class BM25RepresentationStrategy:
    representation_type: RepresentationType = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def fit(self, corpus: list[list[str]], index: InvertedIndex) -> None:
        self._document_count = index.document_count

    def represent_query(self, query_tokens: list[str], index: InvertedIndex) -> dict[str, float]:
        if not query_tokens:
            return {}

        query_term_frequencies = Counter(query_tokens)
        query_length = len(query_tokens)
        average_document_length = max(index.average_document_length, 1.0)
        vector: dict[str, float] = {}

        for term in sorted(query_term_frequencies):
            document_frequency = index.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue

            idf = math.log(1 + (index.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = query_term_frequencies[term] + self.k1 * (
                1 - self.b + self.b * (query_length / average_document_length)
            )
            vector[term] = idf * ((query_term_frequencies[term] * (self.k1 + 1)) / denominator)

        return vector


class EmbeddingRepresentationStrategy:
    representation_type: RepresentationType = "embeddings"

    def __init__(self, vector_size: int = 64) -> None:
        self.vector_size = vector_size
        self._model: Any = None
        self._use_gensim = Word2Vec is not None

    def fit(self, corpus: list[list[str]], index: InvertedIndex) -> None:
        if not corpus:
            self._model = None
            return

        if self._use_gensim:
            self._model = Word2Vec(
                sentences=corpus,
                vector_size=self.vector_size,
                window=5,
                min_count=1,
                workers=1,
                sg=1,
                seed=42,
                epochs=30,
            )
        else:
            self._model = {token for sentence in corpus for token in sentence}

    def _hash_vector(self, token: str) -> list[float]:
        digest = blake2b(token.encode("utf-8"), digest_size=self.vector_size).digest()
        return [((byte / 255.0) * 2.0) - 1.0 for byte in digest[: self.vector_size]]

    def represent_document(self, document_tokens: list[str]) -> list[float]:
        if self._model is None or not document_tokens:
            return [0.0] * self.vector_size

        if self._use_gensim:
            valid_vectors = [self._model.wv[token] for token in document_tokens if token in self._model.wv]
            if not valid_vectors:
                return [0.0] * self.vector_size

            aggregated = [0.0] * self.vector_size
            for vector in valid_vectors:
                for position, value in enumerate(vector):
                    aggregated[position] += float(value)

            vector_count = float(len(valid_vectors))
            return [value / vector_count for value in aggregated]

        aggregated = [0.0] * self.vector_size
        valid_tokens = [token for token in document_tokens if token in self._model]
        if not valid_tokens:
            return [0.0] * self.vector_size

        for token in valid_tokens:
            token_vector = self._hash_vector(token)
            for position, value in enumerate(token_vector):
                aggregated[position] += value

        token_count = float(len(valid_tokens))
        return [value / token_count for value in aggregated]

    def represent_query(self, query_tokens: list[str], index: InvertedIndex) -> list[float]:
        return self.represent_document(query_tokens)

