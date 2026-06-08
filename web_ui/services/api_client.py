from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import requests


_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_DEFAULT_VECTOR_SIZE = 64


@dataclass(frozen=True)
class DatasetDocument:
    document_id: str
    text: str


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    query_id: str
    benchmark_query: str
    documents: tuple[DatasetDocument, ...]
    qrels: dict[str, int]
    ir_dataset_name: str | None = None
    default_document_limit: int = 250_000


def get_dataset_catalog() -> dict[str, DatasetDefinition]:
    return {
        "News Retrieval Demo": DatasetDefinition(
            name="News Retrieval Demo",
            query_id="Q1",
            benchmark_query="brown fox search",
            documents=(
                DatasetDocument("D1", "Brown fox jumps over the lazy dog in the park"),
                DatasetDocument("D2", "A quick brown fox outruns the hounds"),
                DatasetDocument("D3", "City news covers local sports and weather"),
                DatasetDocument("D4", "Dogs and foxes appear in wildlife reports"),
            ),
            qrels={"D1": 3, "D2": 2, "D4": 1},
        ),
        "IR Research Demo": DatasetDefinition(
            name="IR Research Demo",
            query_id="Q2",
            benchmark_query="fastapi microservice evaluation",
            documents=(
                DatasetDocument("D1", "FastAPI microservices expose search and evaluation endpoints"),
                DatasetDocument("D2", "Streamlit creates a simple web interface for retrieval demos"),
                DatasetDocument("D3", "BM25 and embeddings can be combined in hybrid ranking"),
                DatasetDocument("D4", "This document discusses preprocessing and query refinement"),
            ),
            qrels={"D1": 3, "D3": 2, "D4": 1},
        ),
        "MS MARCO Passage": DatasetDefinition(
            name="MS MARCO Passage",
            query_id="",
            benchmark_query="what is information retrieval",
            documents=(),
            qrels={},
            ir_dataset_name="msmarco-passage",
        ),
        "BEIR Quora": DatasetDefinition(
            name="BEIR Quora",
            query_id="",
            benchmark_query="how can i improve search results",
            documents=(),
            qrels={},
            ir_dataset_name="beir/quora",
        ),
    }


def normalize_text(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    response = requests.request(method=method, url=url, json=payload, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data
    return {"data": data}


def refine_query(base_url: str, query: str) -> dict[str, Any]:
    return _request_json("POST", f"{base_url.rstrip('/')}/refine", {"Query": query})


def preprocess_text(base_url: str, text: str) -> dict[str, Any]:
    return _request_json("POST", f"{base_url.rstrip('/')}/preprocess", {"Text": text})


def index_documents(
    base_url: str,
    documents: list[dict[str, str]],
    representation_type: str,
    k1: float,
    b: float,
    vector_size: int,
    dataset_name: str | None = None,
    max_documents: int | None = None,
) -> dict[str, Any]:
    payload = {
        "Documents": documents,
        "Representation_Type": representation_type,
        "K1": k1,
        "B": b,
        "Vector_Size": vector_size,
    }
    params: dict[str, Any] = {}
    if dataset_name:
        params["dataset_name"] = dataset_name
    if max_documents is not None:
        params["max_documents"] = max_documents
    return _request_json("POST", f"{base_url.rstrip('/')}/index", payload, params=params or None, timeout=300)


def represent_query(base_url: str, query: str, representation_type: str, k1: float | None = None, b: float | None = None, vector_size: int = _DEFAULT_VECTOR_SIZE) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Query": query,
        "Representation_Type": representation_type,
    }
    if representation_type == "bm25":
        payload["K1"] = k1 if k1 is not None else 1.5
        payload["B"] = b if b is not None else 0.75
    if representation_type == "embeddings":
        payload["Vector_Size"] = vector_size
    return _request_json("POST", f"{base_url.rstrip('/')}/represent", payload)


def search_documents(
    base_url: str,
    representation_type: str,
    query_representation: dict[str, Any],
    dataset: list[dict[str, Any]] | None,
    top_k: int,
    candidate_k: int,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    payload = {
        "Representation_Type": representation_type,
        "Query_Representation": query_representation,
        "Top_K": top_k,
        "Candidate_K": candidate_k,
    }
    if dataset_name:
        payload["Dataset_Name"] = dataset_name
    else:
        payload["Dataset"] = dataset or []
    return _request_json("POST", f"{base_url.rstrip('/')}/search", payload, timeout=120)


def evaluate_models(
    base_url: str,
    qrels: list[dict[str, Any]],
    system_results: list[dict[str, Any]],
    cutoff: int = 10,
    dataset_name: str | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    payload = {
        "Qrels": qrels,
        "System_Results": system_results,
        "Cutoff": cutoff,
    }
    params: dict[str, Any] = {}
    if dataset_name:
        params["dataset_name"] = dataset_name
    if max_queries is not None:
        params["max_queries"] = max_queries
    return _request_json("POST", f"{base_url.rstrip('/')}/evaluate", payload, params=params or None, timeout=120)


def _hash_vector(token: str, vector_size: int) -> list[float]:
    digest = blake2b(token.encode("utf-8"), digest_size=vector_size).digest()
    return [((byte / 255.0) * 2.0) - 1.0 for byte in digest[:vector_size]]


def _aggregate_embeddings(tokens: list[str], vector_size: int) -> list[float]:
    if not tokens:
        return [0.0] * vector_size

    aggregated = [0.0] * vector_size
    for token in tokens:
        token_vector = _hash_vector(token, vector_size)
        for position, value in enumerate(token_vector):
            aggregated[position] += value

    token_count = float(len(tokens))
    return [value / token_count for value in aggregated]


def _build_corpus_statistics(tokenized_documents: list[list[str]]) -> tuple[dict[str, int], float]:
    document_frequencies: dict[str, int] = {}
    total_length = 0
    for tokens in tokenized_documents:
        total_length += len(tokens)
        for term in set(tokens):
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    average_length = total_length / len(tokenized_documents) if tokenized_documents else 0.0
    return document_frequencies, average_length


def _build_tfidf_vector(tokens: list[str], document_frequencies: dict[str, int], document_count: int) -> dict[str, float]:
    if not tokens:
        return {}

    term_frequencies: dict[str, int] = {}
    for token in tokens:
        term_frequencies[token] = term_frequencies.get(token, 0) + 1

    vector: dict[str, float] = {}
    for term in sorted(term_frequencies):
        document_frequency = document_frequencies.get(term, 0)
        if document_frequency == 0:
            continue
        idf = math.log((document_count + 1) / (document_frequency + 1)) + 1.0
        vector[term] = term_frequencies[term] * idf

    return vector


def _build_bm25_vector(tokens: list[str], document_frequencies: dict[str, int], document_count: int, average_length: float, k1: float, b: float) -> dict[str, float]:
    if not tokens:
        return {}

    term_frequencies: dict[str, int] = {}
    for token in tokens:
        term_frequencies[token] = term_frequencies.get(token, 0) + 1

    query_length = len(tokens)
    average_document_length = max(average_length, 1.0)
    vector: dict[str, float] = {}

    for term in sorted(term_frequencies):
        document_frequency = document_frequencies.get(term, 0)
        if document_frequency == 0:
            continue

        idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        denominator = term_frequencies[term] + k1 * (1 - b + b * (query_length / average_document_length))
        vector[term] = idf * ((term_frequencies[term] * (k1 + 1)) / denominator)

    return vector


def prepare_dataset_vectors(documents: list[dict[str, Any]], k1: float, b: float, vector_size: int) -> list[dict[str, Any]]:
    tokenized_documents = [normalize_text(document["Processed_Text"]) for document in documents]
    document_frequencies, average_length = _build_corpus_statistics(tokenized_documents)
    document_count = len(tokenized_documents)

    prepared_documents: list[dict[str, Any]] = []
    for document, tokens in zip(documents, tokenized_documents, strict=False):
        prepared_documents.append(
            {
                "Document_Id": document["Document_Id"],
                "Original_Text": document.get("Original_Text"),
                "TFIDF_Vector": _build_tfidf_vector(tokens, document_frequencies, document_count),
                "BM25_Vector": _build_bm25_vector(tokens, document_frequencies, document_count, average_length, k1, b),
                "Embedding_Vector": _aggregate_embeddings(tokens, vector_size),
                "Processed_Text": document["Processed_Text"],
            }
        )

    return prepared_documents


def build_query_vectors(base_url: str, query: str, k1: float, b: float, vector_size: int) -> dict[str, Any]:
    tfidf = represent_query(base_url, query, "tfidf")
    bm25 = represent_query(base_url, query, "bm25", k1=k1, b=b)
    embeddings = represent_query(base_url, query, "embeddings", vector_size=vector_size)
    return {
        "TFIDF_Vector": tfidf.get("Vector", {}),
        "BM25_Vector": bm25.get("Vector", {}),
        "Embedding_Vector": embeddings.get("Vector", []),
    }


def build_search_payload(representation_type: str, query_vectors: dict[str, Any]) -> dict[str, Any]:
    if representation_type == "tfidf":
        return {"TFIDF_Vector": query_vectors.get("TFIDF_Vector", {})}
    if representation_type == "bm25":
        return {"BM25_Vector": query_vectors.get("BM25_Vector", {})}
    if representation_type == "embeddings":
        return {"Embedding_Vector": query_vectors.get("Embedding_Vector", [])}
    return {
        "BM25_Vector": query_vectors.get("BM25_Vector", {}),
        "Embedding_Vector": query_vectors.get("Embedding_Vector", []),
    }


def build_ranked_documents(dataset: list[dict[str, Any]], ranked_ids: list[str]) -> list[dict[str, Any]]:
    lookup = {document["Document_Id"]: document for document in dataset}
    return [lookup[document_id] for document_id in ranked_ids if document_id in lookup]


def build_evaluation_payload(query_id: str, qrels: dict[str, int], model_runs: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qrels_payload = [
        {
            "Query_Id": query_id,
            "Relevant_Documents": [
                {
                    "Document_Id": document_id,
                    "Relevance": relevance,
                }
                for document_id, relevance in qrels.items()
            ],
        }
    ]

    system_results: list[dict[str, Any]] = []
    for model_name, ranked_ids in model_runs.items():
        system_results.append(
            {
                "Model_Name": model_name,
                "Query_Id": query_id,
                "Ranked_Document_Ids": ranked_ids,
            }
        )

    return qrels_payload, system_results
