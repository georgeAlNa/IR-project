from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice

import ir_datasets

from schemas.evaluation_schema import QrelDocument, QrelQuery


DEFAULT_DOCUMENT_LIMIT = None
SUPPORTED_DATASETS = ("beir/quora/test",)


@dataclass(frozen=True)
class DatasetConfig:
    docs_dataset_name: str
    evaluation_dataset_name: str


@dataclass(frozen=True)
class LoadedDocument:
    document_id: str
    text: str


DATASET_CONFIGS = {
    "beir/quora/test": DatasetConfig(
        docs_dataset_name="beir/quora/test",
        evaluation_dataset_name="beir/quora/test",
    ),
}


def _patch_ir_datasets_tsv_encoding() -> None:
    from ir_datasets.formats import tsv

    if getattr(tsv.FileLineIter, "_ir_project_utf8_patched", False):
        return

    def __next__(self):
        if self.stop is not None and self.start >= self.stop:
            self.ctxt.close()
            raise StopIteration
        if self.stream is None:
            if isinstance(self.dlc, list):
                self.stream = io.TextIOWrapper(
                    self.ctxt.enter_context(self.dlc[self.stream_idx].stream()),
                    encoding="utf-8",
                    errors="replace",
                )
            else:
                self.stream = io.TextIOWrapper(
                    self.ctxt.enter_context(self.dlc.stream()),
                    encoding="utf-8",
                    errors="replace",
                )
        line = ""
        while self.pos < self.start:
            line = self.stream.readline()
            if line != "\n":
                self.pos += 1
        if line == "":
            if isinstance(self.dlc, list):
                self.stream_idx += 1
                if self.stream_idx < len(self.dlc):
                    self.stream = io.TextIOWrapper(
                        self.ctxt.enter_context(self.dlc[self.stream_idx].stream()),
                        encoding="utf-8",
                        errors="replace",
                    )
                    line = self.stream.readline()
                else:
                    raise StopIteration()
            else:
                raise StopIteration()
        self.start += self.step
        return line

    tsv.FileLineIter.__next__ = __next__
    tsv.FileLineIter._ir_project_utf8_patched = True


def validate_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip()
    if normalized not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}.")
    return normalized


def _dataset_config(dataset_name: str) -> DatasetConfig:
    return DATASET_CONFIGS[validate_dataset_name(dataset_name)]


def _extract_document_text(document: object) -> str:
    text = getattr(document, "text", None)
    title = getattr(document, "title", None)
    body = getattr(document, "body", None)

    chunks: list[str] = []
    if isinstance(title, str) and title.strip():
        chunks.append(title.strip())
    if isinstance(text, str) and text.strip():
        chunks.append(text.strip())
    if isinstance(body, str) and body.strip():
        chunks.append(body.strip())

    return " ".join(chunks)


def load_documents(dataset_name: str, limit: int | None = DEFAULT_DOCUMENT_LIMIT) -> list[LoadedDocument]:
    _patch_ir_datasets_tsv_encoding()

    config = _dataset_config(dataset_name)
    if limit is not None and limit <= 0:
        raise ValueError("Document limit must be greater than 0.")

    dataset = ir_datasets.load(config.docs_dataset_name)
    loaded_documents: list[LoadedDocument] = []

    for document in islice(dataset.docs_iter(), limit):
        document_id = str(getattr(document, "doc_id", "")).strip()
        document_text = _extract_document_text(document)
        if not document_id or not document_text:
            continue
        loaded_documents.append(LoadedDocument(document_id=document_id, text=document_text))

    if not loaded_documents:
        raise RuntimeError(f"No documents were loaded from dataset '{config.docs_dataset_name}'.")

    return loaded_documents


def load_queries_and_qrels(dataset_name: str, max_queries: int | None = None) -> tuple[dict[str, str], list[QrelQuery]]:
    _patch_ir_datasets_tsv_encoding()

    config = _dataset_config(dataset_name)
    dataset = ir_datasets.load(config.evaluation_dataset_name)

    query_lookup: dict[str, str] = {}
    for query in dataset.queries_iter():
        query_id = str(getattr(query, "query_id", "")).strip()
        query_text = str(getattr(query, "text", "")).strip()
        if not query_id or not query_text:
            continue
        query_lookup[query_id] = query_text

    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be greater than 0 when provided.")

    grouped_qrels: dict[str, list[QrelDocument]] = defaultdict(list)
    for qrel in dataset.qrels_iter():
        query_id = str(getattr(qrel, "query_id", "")).strip()
        if not query_id:
            continue
        if max_queries is not None and query_id not in grouped_qrels and len(grouped_qrels) >= max_queries:
            continue

        document_id = str(getattr(qrel, "doc_id", "")).strip()
        relevance = int(getattr(qrel, "relevance", 0))
        if not document_id:
            continue

        grouped_qrels[query_id].append(QrelDocument(document_id=document_id, relevance=relevance))

    qrels: list[QrelQuery] = []
    for query_id, relevant_documents in grouped_qrels.items():
        positive_rels = [document for document in relevant_documents if document.relevance > 0]
        if not positive_rels:
            continue
        qrels.append(QrelQuery(query_id=query_id, relevant_documents=positive_rels))

    if not qrels:
        raise RuntimeError(f"No qrels were loaded from dataset '{config.evaluation_dataset_name}'.")

    return query_lookup, qrels


def load_first_query(dataset_name: str) -> tuple[str, str]:
    query_lookup, _ = load_queries_and_qrels(dataset_name=dataset_name, max_queries=1)
    first_query_id = next(iter(query_lookup), None)
    if first_query_id is None:
        raise RuntimeError(f"No queries were loaded from dataset '{dataset_name}'.")
    return first_query_id, query_lookup[first_query_id]
