from fastapi import APIRouter, Depends

from core.query_refinement import QueryRefinementService, build_query_refinement_service
from schemas.query_refinement_schema import RefineRequest, RefineResponse

router = APIRouter(tags=["Query Refinement"])

_SERVICE = build_query_refinement_service()


def get_query_refinement_service() -> QueryRefinementService:
    return _SERVICE


@router.post("/refine", response_model=RefineResponse)
def refine_query(
    payload: RefineRequest,
    service: QueryRefinementService = Depends(get_query_refinement_service),
) -> RefineResponse:
    result = service.refine(payload.query)
    return RefineResponse(
        original_query=result.original_query,
        corrected_query=result.corrected_query,
        expanded_query=result.expanded_query,
        expanded_terms=result.expanded_terms,
    )
