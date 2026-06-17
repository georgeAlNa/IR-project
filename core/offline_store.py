"""
core/offline_store.py
======================
Thread-safe, singleton store for all pre-loaded offline indexes.

Loaded ONCE at server startup via the FastAPI lifespan and then read by
every request handler without any re-loading cost.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.inverted_index import InvertedIndex
    from core.representation_strategies import RepresentationStrategy, RepresentationType
    from schemas.matching_ranking_schema import RankedDocument


@dataclass
class OfflineDatasetBundle:
    """Everything needed to serve search requests for one dataset."""

    dataset_name: str

    # Traditional IR
    inverted_index: Any           # InvertedIndex
    strategies: dict[str, Any]   # dict[RepresentationType, RepresentationStrategy]
    ranked_documents: list[Any]   # list[RankedDocument]

    # Dense BERT + FAISS (optional — None when not built)
    faiss_index: Any | None = None       # faiss.Index
    faiss_doc_ids: list[str] | None = None
    bert_model: Any | None = None        # SentenceTransformer


_LOCK = threading.RLock()
_BUNDLES: dict[str, OfflineDatasetBundle] = {}


def register_bundle(bundle: OfflineDatasetBundle) -> None:
    with _LOCK:
        _BUNDLES[bundle.dataset_name] = bundle


def get_bundle(dataset_name: str) -> OfflineDatasetBundle:
    with _LOCK:
        bundle = _BUNDLES.get(dataset_name)
    if bundle is None:
        available = list(_BUNDLES.keys())
        raise RuntimeError(
            f"Dataset '{dataset_name}' is not loaded in the offline store. "
            f"Available: {available}. "
            "Run scripts/build_offline_indexes.py first."
        )
    return bundle


def available_datasets() -> list[str]:
    with _LOCK:
        return list(_BUNDLES.keys())
