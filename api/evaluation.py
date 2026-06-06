from fastapi import APIRouter, Depends, HTTPException, Query

from core.data_loader import load_queries_and_qrels
from core.evaluation_service import EvaluationService, build_evaluation_service
from core.matching_ranking_service import MatchingRankingService
from core.state import get_indexed_documents
from api.indexing import get_indexing_service
from schemas.evaluation_schema import EvaluateRequest, EvaluateResponse, QrelQuery, SystemRun

router = APIRouter(tags=["Evaluation"])

_SERVICE = build_evaluation_service()


def get_evaluation_service() -> EvaluationService:
    return _SERVICE


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_models(
    payload: EvaluateRequest,
    dataset_name: str | None = Query(default=None),
    max_queries: int | None = Query(default=None, ge=1),
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluateResponse:
    try:
        qrels: list[QrelQuery] | None = payload.qrels
        system_results = payload.system_results
        if dataset_name:
            query_lookup, qrels = load_queries_and_qrels(dataset_name=dataset_name, max_queries=max_queries)
            if not system_results:
                indexed_documents = get_indexed_documents(dataset_name)
                indexing_service = get_indexing_service()
                ranking_service = MatchingRankingService()
                model_representations = {
                    "TF-IDF": "tfidf",
                    "BM25": "bm25",
                    "Embedding": "embeddings",
                    "Hybrid Parallel": "hybrid_parallel",
                }
                system_results = []

                for qrel in qrels:
                    query_text = query_lookup.get(qrel.query_id)
                    if not query_text:
                        continue
                    query_representation = indexing_service.build_search_query_representation(query_text)
                    for model_name, representation_type in model_representations.items():
                        ranking_result = ranking_service.search_documents(
                            representation_type=representation_type,
                            query_representation=query_representation,
                            dataset=indexed_documents,
                            top_k=payload.cutoff,
                            candidate_k=max(payload.cutoff, 50),
                        )
                        system_results.append(
                            SystemRun(
                                model_name=model_name,
                                query_id=qrel.query_id,
                                ranked_document_ids=ranking_result.ranked_document_ids,
                            )
                        )

        if not qrels:
            raise RuntimeError("No qrels available. Provide payload Qrels or pass dataset_name.")
        if not system_results:
            raise RuntimeError("No system results available. Provide System_Results or pass dataset_name for server-side evaluation.")

        result = service.evaluate(
            EvaluateRequest(
                qrels=qrels,
                system_results=system_results,
                cutoff=payload.cutoff,
            )
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EvaluateResponse(metrics_by_model=result.metrics_by_model)
