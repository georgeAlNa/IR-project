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
        self._dataset_cache = {}

    def _get_dataset_cache(self, dataset: list[RankedDocument]):
        dataset_id = id(dataset)
        if dataset_id not in self._dataset_cache:
            import numpy as np
            from collections import defaultdict
            
            emb_matrix = np.zeros((len(dataset), 64), dtype=np.float32)
            doc_ids = []
            bm25_postings = defaultdict(list)
            tfidf_postings = defaultdict(list)

            for i, doc in enumerate(dataset):
                doc_ids.append(str(doc.document_id))
                
                if doc.embedding_vector:
                    vec = doc.embedding_vector
                    length = min(len(vec), 64)
                    emb_matrix[i, :length] = vec[:length]
                
                if doc.bm25_vector:
                    for term in doc.bm25_vector.keys():
                        bm25_postings[term].append(doc)
                
                if doc.tfidf_vector:
                    for term in doc.tfidf_vector.keys():
                        tfidf_postings[term].append(doc)

            norms = np.linalg.norm(emb_matrix, axis=1)
            norms[norms == 0] = 1e-9
            
            self._dataset_cache[dataset_id] = {
                "emb_matrix": emb_matrix,
                "norms": norms,
                "doc_ids": doc_ids,
                "bm25_postings": bm25_postings,
                "tfidf_postings": tfidf_postings,
            }
        return self._dataset_cache[dataset_id]

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
            cache = self._get_dataset_cache(dataset)
            
            if representation_type == "embeddings":
                import numpy as np
                q_vec = query_representation.embedding_vector or []
                q_emb = np.zeros(64, dtype=np.float32)
                q_len = min(len(q_vec), 64)
                if q_len > 0:
                    q_emb[:q_len] = q_vec[:q_len]
                q_norm = np.linalg.norm(q_emb)
                
                if q_norm == 0:
                    return SearchResult(ranked_document_ids=[])
                    
                scores = np.dot(cache["emb_matrix"], q_emb) / (cache["norms"] * q_norm)
                top_indices = np.argsort(scores)[-top_k:][::-1]
                ranked_ids = [cache["doc_ids"][i] for i in top_indices if scores[i] > 0]
                return SearchResult(ranked_document_ids=ranked_ids)
                
            elif representation_type == "tfidf":
                q_vec = query_representation.tfidf_vector or {}
                if not q_vec:
                    return SearchResult(ranked_document_ids=[])
                
                candidates_set = set()
                for term in q_vec.keys():
                    candidates_set.update(id(d) for d in cache["tfidf_postings"].get(term, []))
                    
                candidates = [d for d in dataset if id(d) in candidates_set]
                scored_documents = [
                    RankedDocumentScore(
                        document_id=str(doc.document_id),
                        score=self.cosine_similarity_sparse(q_vec, doc.tfidf_vector or {})
                    )
                    for doc in candidates
                ]
                ranked_documents = self._rank_documents(scored_documents)
                return SearchResult(ranked_document_ids=[item.document_id for item in ranked_documents[:top_k]])
                
            elif representation_type == "bm25":
                q_vec = query_representation.bm25_vector or {}
                if not q_vec:
                    return SearchResult(ranked_document_ids=[])
                
                candidates_set = set()
                for term in q_vec.keys():
                    candidates_set.update(id(d) for d in cache["bm25_postings"].get(term, []))
                    
                candidates = [d for d in dataset if id(d) in candidates_set]
                scored_documents = [
                    RankedDocumentScore(
                        document_id=str(doc.document_id),
                        score=self.bm25_similarity(q_vec, doc.bm25_vector or {})
                    )
                    for doc in candidates
                ]
                ranked_documents = self._rank_documents(scored_documents)
                return SearchResult(ranked_document_ids=[item.document_id for item in ranked_documents[:top_k]])
                
            elif representation_type == "hybrid_parallel":
                q_bm25 = query_representation.bm25_vector or {}
                bm25_scores = []
                if q_bm25:
                    c_set = set()
                    for term in q_bm25.keys():
                        c_set.update(id(d) for d in cache["bm25_postings"].get(term, []))
                    candidates = [d for d in dataset if id(d) in c_set]
                    bm25_scores = [
                        RankedDocumentScore(str(doc.document_id), self.bm25_similarity(q_bm25, doc.bm25_vector or {}))
                        for doc in candidates
                    ]
                
                import numpy as np
                q_vec = query_representation.embedding_vector or []
                q_emb = np.zeros(64, dtype=np.float32)
                q_len = min(len(q_vec), 64)
                if q_len > 0:
                    q_emb[:q_len] = q_vec[:q_len]
                q_norm = np.linalg.norm(q_emb)
                
                embedding_scores = []
                if q_norm > 0:
                    scores = np.dot(cache["emb_matrix"], q_emb) / (cache["norms"] * q_norm)
                    top_indices = np.argsort(scores)[-max(candidate_k, 2000):][::-1]
                    embedding_scores = [
                        RankedDocumentScore(cache["doc_ids"][i], float(scores[i]))
                        for i in top_indices if scores[i] > 0
                    ]
                
                bm25_rank_map = self._build_rank_map(bm25_scores)
                embedding_rank_map = self._build_rank_map(embedding_scores)
                ranked_documents = self._apply_rrf([bm25_rank_map, embedding_rank_map])
                return SearchResult(ranked_document_ids=[item.document_id for item in ranked_documents[:top_k]])
                
            elif representation_type == "hybrid_serial":
                q_bm25 = query_representation.bm25_vector or {}
                if not q_bm25:
                    return SearchResult(ranked_document_ids=[])
                c_set = set()
                for term in q_bm25.keys():
                    c_set.update(id(d) for d in cache["bm25_postings"].get(term, []))
                candidates = [d for d in dataset if id(d) in c_set]
                bm25_scores = [
                    RankedDocumentScore(str(doc.document_id), self.bm25_similarity(q_bm25, doc.bm25_vector or {}))
                    for doc in candidates
                ]
                
                bm25_ranked_documents = self._rank_documents(bm25_scores)
                candidate_docs = bm25_ranked_documents[: min(candidate_k, len(bm25_ranked_documents))]
                
                candidate_lookup = {doc.document_id: doc for doc in candidates}
                reranked_documents = [
                    RankedDocumentScore(
                        document_id=item.document_id,
                        score=self._score_embeddings(query_representation, candidate_lookup[item.document_id]),
                    )
                    for item in candidate_docs
                ]
                fallback_bm25_scores = {item.document_id: item.score for item in candidate_docs}
                ranked_documents = sorted(
                    reranked_documents,
                    key=lambda item: (-item.score, -fallback_bm25_scores.get(item.document_id, 0.0), item.document_id),
                )
                return SearchResult(ranked_document_ids=[item.document_id for item in ranked_documents[:top_k]])
                
            raise RuntimeError(f"Representation strategy '{representation_type}' is not supported.")


def build_matching_ranking_service() -> MatchingRankingService:
    return MatchingRankingService()