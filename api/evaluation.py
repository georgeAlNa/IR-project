from fastapi import APIRouter, Depends, HTTPException, Query
import json
from pathlib import Path

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
    force_recalculate: bool = Query(default=False),
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluateResponse:
    try:
        cache_file = None
        if dataset_name:
            safe_name = dataset_name.replace("/", "_")
            cache_file = Path(f"offline_indexes/evaluation_cache_{safe_name}.json")
            if not force_recalculate and cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    return EvaluateResponse(**json.load(f))

        qrels: list[QrelQuery] | None = payload.qrels
        system_results = payload.system_results
        if dataset_name:
            # Load ALL queries first so we can filter down to those actually in our subset
            query_lookup, qrels = load_queries_and_qrels(dataset_name=dataset_name, max_queries=None)
            if not system_results:
                from core.offline_store import get_bundle, available_datasets
                from schemas.matching_ranking_schema import SearchQueryRepresentation
                from core.inverted_index import normalize_text

                is_offline = dataset_name in available_datasets()
                if is_offline:
                    bundle = get_bundle(dataset_name)
                    indexed_documents = bundle.ranked_documents
                else:
                    indexed_documents = get_indexed_documents(dataset_name)

                ranking_service = MatchingRankingService()
                model_representations = {
                    "TF-IDF": "tfidf",
                    "BM25": "bm25",
                    "Embedding": "embeddings",
                    "Hybrid Parallel": "hybrid_parallel",
                }
                system_results = []

                # Filter Qrels to only include queries where the target document actually exists in our subset
                valid_doc_ids = {str(doc.document_id) for doc in indexed_documents}
                filtered_qrels = []
                for qrel in qrels:
                    if any(str(rel_doc.document_id) in valid_doc_ids for rel_doc in qrel.relevant_documents):
                        filtered_qrels.append(qrel)
                
                # Evaluate on ALL queries for comprehensive evaluation
                qrels = filtered_qrels

                for qrel in qrels:
                    query_text = query_lookup.get(qrel.query_id)
                    if not query_text:
                        continue
                        
                    if is_offline:
                        query_tokens = normalize_text(query_text)
                        tfidf_vec = bundle.strategies.get("tfidf").represent_query(query_tokens, bundle.inverted_index) if bundle.strategies.get("tfidf") else {}
                        bm25_vec = bundle.strategies.get("bm25").represent_query(query_tokens, bundle.inverted_index) if bundle.strategies.get("bm25") else {}
                        emb_vec = bundle.strategies.get("embeddings").represent_query(query_tokens, bundle.inverted_index) if bundle.strategies.get("embeddings") else []
                        query_representation = SearchQueryRepresentation(
                            tfidf_vector=tfidf_vec, bm25_vector=bm25_vec, embedding_vector=emb_vec
                        )
                    else:
                        indexing_service = get_indexing_service()
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
        response = EvaluateResponse(
            total_queries_evaluated=len(qrels) if qrels else 0,
            metrics_by_model=result.metrics_by_model
        )
        
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response.model_dump(by_alias=True), f, indent=2)
                
        return response
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
