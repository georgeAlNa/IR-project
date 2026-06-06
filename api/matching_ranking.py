from fastapi import APIRouter, Depends, HTTPException

from core.state import get_indexed_documents
from core.matching_ranking_service import MatchingRankingService, build_matching_ranking_service
from schemas.matching_ranking_schema import SearchRequest, SearchResponse

router = APIRouter(tags=["Matching and Ranking"])

_SERVICE = build_matching_ranking_service()


def get_matching_ranking_service() -> MatchingRankingService:
    return _SERVICE


@router.post("/search", response_model=SearchResponse)
def search_documents(
    payload: SearchRequest,
    service: MatchingRankingService = Depends(get_matching_ranking_service),
) -> SearchResponse:
    try:
        dataset = payload.dataset
        if payload.dataset_name:
            dataset = get_indexed_documents(payload.dataset_name)
        if not dataset:
            raise RuntimeError("Provide Dataset in the request body or Dataset_Name for a pre-indexed dataset.")

        result = service.search_documents(
            representation_type=payload.representation_type,
            query_representation=payload.query_representation,
            dataset=dataset,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(ranked_document_ids=result.ranked_document_ids)
