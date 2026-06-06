from __future__ import annotations

import threading
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.data_loader import DEFAULT_DOCUMENT_LIMIT, load_documents
from core.inverted_index import IndexedDocumentEntity, InvertedIndex, InvertedIndexBuilder, normalize_text
from core.preprocessing_pipeline import PreprocessingPipeline
from core.representation_strategies import (
    BM25RepresentationStrategy,
    EmbeddingRepresentationStrategy,
    RepresentationStrategy,
    RepresentationType,
    TFIDFRepresentationStrategy,
)
from core.state import set_indexed_dataset
from schemas.matching_ranking_schema import RankedDocument, SearchQueryRepresentation


@dataclass(frozen=True)
class IndexingResult:
    document_count: int
    vocabulary_size: int
    average_document_length: float
    active_representation: RepresentationType


class IndexingService:
    def __init__(
        self,
        builder: InvertedIndexBuilder | None = None,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
    ) -> None:
        self._builder = builder or InvertedIndexBuilder()
        self._preprocessing_pipeline = preprocessing_pipeline or PreprocessingPipeline.default()
        self._lock = threading.RLock()
        self._documents: list[IndexedDocumentEntity] = []
        self._index: InvertedIndex | None = None
        self._strategies: dict[RepresentationType, RepresentationStrategy] = {}

    def index_documents(
        self,
        documents: list[tuple[str, str]],
        representation_type: RepresentationType = "tfidf",
        k1: float = 1.5,
        b: float = 0.75,
        vector_size: int = 64,
        dataset_name: str | None = None,
    ) -> IndexingResult:
        with self._lock:
            if not documents:
                raise RuntimeError("No documents were provided for indexing.")

            indexed_documents = [
                IndexedDocumentEntity(
                    document_id=document_id,
                    processed_text=processed_text,
                    tokens=tuple(normalize_text(processed_text)),
                )
                for document_id, processed_text in documents
            ]

            self._documents = indexed_documents
            self._index = self._builder.build(indexed_documents)

            corpus = [list(document.tokens) for document in indexed_documents]
            self._strategies = {
                "tfidf": TFIDFRepresentationStrategy(),
                "bm25": BM25RepresentationStrategy(k1=k1, b=b),
                "embeddings": EmbeddingRepresentationStrategy(vector_size=vector_size),
            }

            for strategy in self._strategies.values():
                strategy.fit(corpus, self._index)

            if dataset_name:
                set_indexed_dataset(
                    dataset_name=dataset_name,
                    documents=self._build_ranked_documents(indexed_documents, self._index, k1, b),
                )

            return IndexingResult(
                document_count=self._index.document_count,
                vocabulary_size=len(self._index.vocabulary),
                average_document_length=self._index.average_document_length,
                active_representation=representation_type,
            )

    def index_ir_dataset(
        self,
        dataset_name: str,
        representation_type: RepresentationType = "tfidf",
        k1: float = 1.5,
        b: float = 0.75,
        vector_size: int = 64,
        max_documents: int = DEFAULT_DOCUMENT_LIMIT,
    ) -> IndexingResult:
        raw_documents = load_documents(dataset_name=dataset_name, limit=max_documents)

        processed_documents: list[tuple[str, str]] = []
        for document in raw_documents:
            processed_text = self._preprocessing_pipeline.process(document.text)
            if processed_text.strip():
                processed_documents.append((document.document_id, processed_text))

        if not processed_documents:
            raise RuntimeError(f"No non-empty documents available after preprocessing for dataset '{dataset_name}'.")

        return self.index_documents(
            documents=processed_documents,
            representation_type=representation_type,
            k1=k1,
            b=b,
            vector_size=vector_size,
            dataset_name=dataset_name,
        )

    def _build_tfidf_document_vector(self, tokens: tuple[str, ...], index: InvertedIndex) -> dict[str, float]:
        term_frequencies = Counter(tokens)
        vector: dict[str, float] = {}
        for term in sorted(term_frequencies):
            document_frequency = index.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue
            idf = math.log((index.document_count + 1) / (document_frequency + 1)) + 1.0
            vector[term] = term_frequencies[term] * idf
        return vector

    def _build_bm25_document_vector(
        self,
        tokens: tuple[str, ...],
        index: InvertedIndex,
        k1: float,
        b: float,
    ) -> dict[str, float]:
        term_frequencies = Counter(tokens)
        document_length = len(tokens)
        average_document_length = max(index.average_document_length, 1.0)
        vector: dict[str, float] = {}

        for term in sorted(term_frequencies):
            document_frequency = index.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue

            idf = math.log(1 + (index.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequencies[term] + k1 * (
                1 - b + b * (document_length / average_document_length)
            )
            vector[term] = idf * ((term_frequencies[term] * (k1 + 1)) / denominator)

        return vector

    def _build_ranked_documents(
        self,
        indexed_documents: list[IndexedDocumentEntity],
        index: InvertedIndex,
        k1: float,
        b: float,
    ) -> list[RankedDocument]:
        embedding_strategy = self._strategies.get("embeddings")
        ranked_documents: list[RankedDocument] = []

        for document in indexed_documents:
            embedding_vector: list[float] | None = None
            if isinstance(embedding_strategy, EmbeddingRepresentationStrategy):
                embedding_vector = embedding_strategy.represent_document(list(document.tokens))

            ranked_documents.append(
                RankedDocument(
                    document_id=document.document_id,
                    tfidf_vector=self._build_tfidf_document_vector(document.tokens, index),
                    bm25_vector=self._build_bm25_document_vector(document.tokens, index, k1, b),
                    embedding_vector=embedding_vector,
                )
            )

        return ranked_documents

    def represent_query(
        self,
        query: str,
        representation_type: RepresentationType = "tfidf",
        k1: float | None = None,
        b: float | None = None,
    ) -> tuple[list[str], RepresentationType, Any]:
        with self._lock:
            if self._index is None:
                raise RuntimeError("Index is empty. Call /index before /represent.")

            query_tokens = normalize_text(query)
            if representation_type == "bm25":
                strategy = BM25RepresentationStrategy(
                    k1=k1 if k1 is not None else 1.5,
                    b=b if b is not None else 0.75,
                )
                strategy.fit([], self._index)
                vector = strategy.represent_query(query_tokens, self._index)
                return query_tokens, representation_type, vector

            strategy = self._strategies.get(representation_type)
            if strategy is None:
                raise RuntimeError(f"Representation strategy '{representation_type}' is not available.")

            vector = strategy.represent_query(query_tokens, self._index)
            return query_tokens, representation_type, vector

    def build_search_query_representation(
        self,
        query: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> SearchQueryRepresentation:
        processed_query = self._preprocessing_pipeline.process(query)
        _, _, tfidf_vector = self.represent_query(processed_query, "tfidf")
        _, _, bm25_vector = self.represent_query(processed_query, "bm25", k1=k1, b=b)
        _, _, embedding_vector = self.represent_query(processed_query, "embeddings")
        return SearchQueryRepresentation(
            tfidf_vector=tfidf_vector,
            bm25_vector=bm25_vector,
            embedding_vector=embedding_vector,
        )


def build_indexing_service() -> IndexingService:
    return IndexingService()

