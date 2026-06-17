"""
api/matching_ranking.py
========================
/search endpoint — updated to serve requests from pre-loaded offline indexes.

Search flow:
  1. If dataset_name is provided → load its OfflineDatasetBundle from the
     in-memory store (zero disk I/O, sub-millisecond).
  2. If payload.dataset is provided (small custom datasets) → use the legacy
     dynamic path via core.state (backward-compatible).
  3. representation_type == "bert" uses the FAISS index for dense retrieval
     via the pre-loaded SentenceTransformer model.
  4. All other representation_types (tfidf, bm25, embeddings, hybrid_*)
     delegate to the existing MatchingRankingService without any changes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from core.matching_ranking_service import MatchingRankingService, build_matching_ranking_service
from core.offline_store import get_bundle
from core.state import get_indexed_documents
from schemas.matching_ranking_schema import (
    RankedDocument,
    SearchQueryRepresentation,
    SearchRequest,
    SearchResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["Matching and Ranking"])

_SERVICE = build_matching_ranking_service()


def get_matching_ranking_service() -> MatchingRankingService:
    return _SERVICE


# ─────────────────────────────────────────────────────────────────────────────
# BERT + FAISS helper
# ─────────────────────────────────────────────────────────────────────────────
def _bert_faiss_search(
    query_text: str,
    bundle: Any,
    top_k: int,
) -> list[str]:
    """
    Encode *query_text* with the bundle's SentenceTransformer model,
    search the pre-built FAISS index, and return the top-k document IDs.
    """
    if bundle.bert_model is None or bundle.faiss_index is None or bundle.faiss_doc_ids is None:
        raise RuntimeError(
            f"BERT+FAISS index is not available for dataset '{bundle.dataset_name}'. "
            "Run scripts/build_offline_indexes.py with BERT enabled first."
        )
    import numpy as np

    query_vector = bundle.bert_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, bundle.faiss_index.ntotal)
    _, indices = bundle.faiss_index.search(query_vector, k)

    return [bundle.faiss_doc_ids[i] for i in indices[0] if 0 <= i < len(bundle.faiss_doc_ids)]


# ─────────────────────────────────────────────────────────────────────────────
# /search endpoint
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/search", response_model=SearchResponse)
def search_documents(
    payload: SearchRequest,
    query_text: str | None = Query(
        default=None,
        alias="query_text",
        description="Raw query string required only for representation_type='bert'.",
    ),
    service: MatchingRankingService = Depends(get_matching_ranking_service),
) -> SearchResponse:
    try:
        # ── BERT + FAISS path ────────────────────────────────────────────────
        if payload.representation_type == "bert":
            if not payload.dataset_name:
                raise RuntimeError(
                    "Dataset_Name is required in the request body for BERT search."
                )
            if not query_text:
                raise RuntimeError(
                    "query_text query-parameter is required for representation_type='bert'."
                )
            bundle = get_bundle(payload.dataset_name)
            ranked_ids = _bert_faiss_search(query_text, bundle, top_k=payload.top_k)
            doc_lookup = {doc.document_id: doc for doc in bundle.ranked_documents}
            ranked_documents = [doc_lookup[did] for did in ranked_ids if did in doc_lookup]
            return SearchResponse(
                ranked_document_ids=ranked_ids,
                ranked_documents=ranked_documents,
            )

        # ── Offline store path (pre-indexed large datasets) ──────────────────
        if payload.dataset_name:
            bundle = get_bundle(payload.dataset_name)
            dataset = bundle.ranked_documents
        # ── Legacy dynamic path (small custom datasets) ──────────────────────
        elif payload.dataset:
            dataset = payload.dataset
        else:
            # Fall back to core.state (set by the /index dynamic endpoint)
            raise RuntimeError(
                "Provide Dataset in the request body or Dataset_Name for a pre-indexed dataset."
            )

        result = service.search_documents(
            representation_type=payload.representation_type,
            query_representation=payload.query_representation,
            dataset=dataset,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_lookup = {doc.document_id: doc for doc in dataset}
    ranked_documents = [
        document_lookup[did]
        for did in result.ranked_document_ids
        if did in document_lookup
    ]

    return SearchResponse(
        ranked_document_ids=result.ranked_document_ids,
        ranked_documents=ranked_documents,
    )
