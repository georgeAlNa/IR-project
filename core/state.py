from __future__ import annotations

import threading
from dataclasses import dataclass

from schemas.matching_ranking_schema import RankedDocument, SearchQueryRepresentation


@dataclass(frozen=True)
class IndexedDatasetState:
    dataset_name: str
    documents: list[RankedDocument]
    query_vectors: dict[str, SearchQueryRepresentation]


_LOCK = threading.RLock()
_INDEXED_DATASETS: dict[str, IndexedDatasetState] = {}


def set_indexed_dataset(
    dataset_name: str,
    documents: list[RankedDocument],
    query_vectors: dict[str, SearchQueryRepresentation] | None = None,
) -> None:
    with _LOCK:
        _INDEXED_DATASETS[dataset_name] = IndexedDatasetState(
            dataset_name=dataset_name,
            documents=documents,
            query_vectors=query_vectors or {},
        )


def get_indexed_documents(dataset_name: str) -> list[RankedDocument]:
    with _LOCK:
        state = _INDEXED_DATASETS.get(dataset_name)
        if state is None:
            raise RuntimeError(f"Dataset '{dataset_name}' is not indexed. Call /index first.")
        return state.documents


def set_query_vector(dataset_name: str, query_id: str, query_representation: SearchQueryRepresentation) -> None:
    with _LOCK:
        state = _INDEXED_DATASETS.get(dataset_name)
        if state is None:
            raise RuntimeError(f"Dataset '{dataset_name}' is not indexed. Call /index first.")
        updated_query_vectors = dict(state.query_vectors)
        updated_query_vectors[query_id] = query_representation
        _INDEXED_DATASETS[dataset_name] = IndexedDatasetState(
            dataset_name=state.dataset_name,
            documents=state.documents,
            query_vectors=updated_query_vectors,
        )


def get_query_vector(dataset_name: str, query_id: str) -> SearchQueryRepresentation | None:
    with _LOCK:
        state = _INDEXED_DATASETS.get(dataset_name)
        if state is None:
            raise RuntimeError(f"Dataset '{dataset_name}' is not indexed. Call /index first.")
        return state.query_vectors.get(query_id)
