from fastapi import APIRouter, Depends, HTTPException, Query

from core.data_loader import DEFAULT_DOCUMENT_LIMIT
from core.indexing_service import IndexingService, build_indexing_service
from core.offline_store import get_bundle
from schemas.indexing_schema import IndexRequest, IndexResponse, RepresentRequest, RepresentResponse

router = APIRouter(tags=["Indexing"])

_SERVICE = build_indexing_service()


def get_indexing_service() -> IndexingService:
    return _SERVICE


@router.post("/index", response_model=IndexResponse)
def index_documents(
    payload: IndexRequest,
    dataset_name: str | None = Query(default=None),
    max_documents: int = Query(default=DEFAULT_DOCUMENT_LIMIT, ge=1),
    service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    try:
        if dataset_name:
            from core.offline_store import available_datasets, get_bundle
            if dataset_name in available_datasets():
                bundle = get_bundle(dataset_name)
                return IndexResponse(
                    document_count=bundle.inverted_index.document_count,
                    vocabulary_size=len(bundle.inverted_index.vocabulary),
                    average_document_length=bundle.inverted_index.average_document_length,
                    active_representation=payload.representation_type,
                )
            
            result = service.index_ir_dataset(
                dataset_name=dataset_name,
                representation_type=payload.representation_type,
                k1=payload.k1,
                b=payload.b,
                vector_size=payload.vector_size,
                max_documents=max_documents,
            )
        else:
            if not payload.documents:
                raise RuntimeError("Provide Documents in the payload or pass dataset_name as query parameter.")
            result = service.index_documents(
                documents=[
                    (document.document_id, document.processed_text, document.original_text)
                    for document in payload.documents
                ],
                representation_type=payload.representation_type,
                k1=payload.k1,
                b=payload.b,
                vector_size=payload.vector_size,
            )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndexResponse(
        document_count=result.document_count,
        vocabulary_size=result.vocabulary_size,
        average_document_length=result.average_document_length,
        active_representation=result.active_representation,
    )


@router.post("/represent", response_model=RepresentResponse)
def represent_query(
    payload: RepresentRequest,
    dataset_name: str | None = Query(
        default=None,
        description="When provided, use the pre-loaded offline strategies for this dataset.",
    ),
    service: IndexingService = Depends(get_indexing_service),
) -> RepresentResponse:
    try:
        # ── Offline path: use pre-fitted strategies + index from disk ─────────
        if dataset_name:
            bundle = get_bundle(dataset_name)
            strategy = bundle.strategies.get(payload.representation_type)
            if strategy is None:
                raise RuntimeError(
                    f"Representation strategy '{payload.representation_type}' is not "
                    f"available for dataset '{dataset_name}'."
                )
            from core.inverted_index import normalize_text
            query_tokens = normalize_text(payload.query)
            vector = strategy.represent_query(query_tokens, bundle.inverted_index)
            return RepresentResponse(
                representation_type=payload.representation_type,
                query_tokens=query_tokens,
                vector=vector,
            )

        # ── Dynamic path: use the in-process IndexingService (legacy) ─────────
        query_tokens, representation_type, vector = service.represent_query(
            query=payload.query,
            representation_type=payload.representation_type,
            k1=payload.k1,
            b=payload.b,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RepresentResponse(
        representation_type=representation_type,
        query_tokens=query_tokens,
        vector=vector,
    )
