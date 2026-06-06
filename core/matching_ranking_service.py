from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from schemas.matching_ranking_schema import RankedDocument, SearchQueryRepresentation, SearchRepresentationType


RRF_CONSTANT = 60.0


@dataclass(frozen=True)
class RankedDocumentScore:
    document_id: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    ranked_document_ids: list[str]


class MatchingRankingService:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def cosine_similarity_dense(self, query_vector: list[float], document_vector: list[float]) -> float:
        if not query_vector or not document_vector:
            return 0.0

        dot_product = sum(query_value * document_value for query_value, document_value in zip(query_vector, document_vector))
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        document_norm = math.sqrt(sum(value * value for value in document_vector))

        if query_norm == 0.0 or document_norm == 0.0:
            return 0.0

        return dot_product / (query_norm * document_norm)

    def cosine_similarity_sparse(self, query_vector: dict[str, float], document_vector: dict[str, float]) -> float:
        if not query_vector or not document_vector:
            return 0.0

        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        document_norm = math.sqrt(sum(value * value for value in document_vector.values()))
        if query_norm == 0.0 or document_norm == 0.0:
            return 0.0

        shared_terms = set(query_vector).intersection(document_vector)
        dot_product = sum(query_vector[term] * document_vector[term] for term in shared_terms)
        return dot_product / (query_norm * document_norm)

    def bm25_similarity(self, query_vector: dict[str, float], document_vector: dict[str, float]) -> float:
        if not query_vector or not document_vector:
            return 0.0

        shared_terms = set(query_vector).intersection(document_vector)
        return sum(query_vector[term] * document_vector[term] for term in shared_terms)

    def _score_document(
        self,
        representation_type: SearchRepresentationType,
        query_representation: SearchQueryRepresentation,
        document: RankedDocument,
    ) -> float:
        if representation_type == "tfidf":
            return self.cosine_similarity_sparse(query_representation.tfidf_vector or {}, document.tfidf_vector or {})

        if representation_type == "bm25":
            return self.bm25_similarity(query_representation.bm25_vector or {}, document.bm25_vector or {})

        if representation_type == "embeddings":
            return self.cosine_similarity_dense(query_representation.embedding_vector or [], document.embedding_vector or [])

        return 0.0

    def _score_bm25(self, query_representation: SearchQueryRepresentation, document: RankedDocument) -> float:
        return self.bm25_similarity(query_representation.bm25_vector or {}, document.bm25_vector or {})

    def _score_embeddings(self, query_representation: SearchQueryRepresentation, document: RankedDocument) -> float:
        return self.cosine_similarity_dense(query_representation.embedding_vector or [], document.embedding_vector or [])

    def _rank_documents(self, scored_documents: list[RankedDocumentScore]) -> list[RankedDocumentScore]:
        return sorted(scored_documents, key=lambda item: (-item.score, item.document_id))

    def _build_rank_map(self, scored_documents: list[RankedDocumentScore]) -> dict[str, int]:
        return {item.document_id: position + 1 for position, item in enumerate(self._rank_documents(scored_documents))}

    def _apply_rrf(self, rank_maps: list[dict[str, int]]) -> list[RankedDocumentScore]:
        fused_scores: dict[str, float] = {}
        for rank_map in rank_maps:
            for document_id, rank in rank_map.items():
                fused_scores[document_id] = fused_scores.get(document_id, 0.0) + 1.0 / (RRF_CONSTANT + float(rank))

        return [
            RankedDocumentScore(document_id=document_id, score=score)
            for document_id, score in sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
        ]

    def search_documents(
        self,
        representation_type: SearchRepresentationType,
        query_representation: SearchQueryRepresentation,
        dataset: list[RankedDocument],
        top_k: int = 10,
        candidate_k: int = 20,
    ) -> SearchResult:
        with self._lock:
            if representation_type in ("tfidf", "bm25", "embeddings"):
                scored_documents = [
                    RankedDocumentScore(
                        document_id=document.document_id,
                        score=self._score_document(representation_type, query_representation, document),
                    )
                    for document in dataset
                ]
                ranked_documents = self._rank_documents(scored_documents)
                return SearchResult(ranked_document_ids=[item.document_id for item in ranked_documents[:top_k]])

            bm25_scores = [
                RankedDocumentScore(document_id=document.document_id, score=self._score_bm25(query_representation, document))
                for document in dataset
            ]
            embedding_scores = [
                RankedDocumentScore(document_id=document.document_id, score=self._score_embeddings(query_representation, document))
                for document in dataset
            ]

            if representation_type == "hybrid_parallel":
                bm25_rank_map = self._build_rank_map(bm25_scores)
                embedding_rank_map = self._build_rank_map(embedding_scores)
                ranked_documents = self._apply_rrf([bm25_rank_map, embedding_rank_map])
                return SearchResult(ranked_document_ids=[item.document_id for item in ranked_documents[:top_k]])

            if representation_type == "hybrid_serial":
                bm25_ranked_documents = self._rank_documents(bm25_scores)
                candidate_documents = bm25_ranked_documents[: min(candidate_k, len(bm25_ranked_documents))]
                candidate_lookup = {document.document_id: document for document in dataset}
                reranked_documents = [
                    RankedDocumentScore(
                        document_id=item.document_id,
                        score=self._score_embeddings(query_representation, candidate_lookup[item.document_id]),
                    )
                    for item in candidate_documents
                ]
                fallback_bm25_scores = {item.document_id: item.score for item in candidate_documents}
                ranked_documents = sorted(
                    reranked_documents,
                    key=lambda item: (-item.score, -fallback_bm25_scores.get(item.document_id, 0.0), item.document_id),
                )
                return SearchResult(ranked_document_ids=[item.document_id for item in ranked_documents[:top_k]])

            raise RuntimeError(f"Representation strategy '{representation_type}' is not supported.")


def build_matching_ranking_service() -> MatchingRankingService:
    return MatchingRankingService()