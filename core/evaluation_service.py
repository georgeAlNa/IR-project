from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from schemas.evaluation_schema import EvaluateRequest, EvaluateResponse, MetricSummary, QrelQuery, SystemRun


@dataclass(frozen=True)
class PerQueryMetrics:
    average_precision: float
    recall: float
    precision_at_10: float
    ndcg: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics_by_model: dict[str, MetricSummary]


class EvaluationService:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def _relevant_documents(self, qrel_query: QrelQuery) -> dict[str, int]:
        return {
            document.document_id: document.relevance
            for document in qrel_query.relevant_documents
            if document.relevance > 0
        }

    def _average_precision(self, ranked_document_ids: list[str], relevant_documents: dict[str, int]) -> float:
        if not relevant_documents:
            return 0.0

        hits = 0
        precision_sum = 0.0
        seen: set[str] = set()

        for position, document_id in enumerate(ranked_document_ids, start=1):
            if document_id in seen:
                continue
            seen.add(document_id)
            if document_id in relevant_documents:
                hits += 1
                precision_sum += hits / position

        return precision_sum / len(relevant_documents)

    def _recall(self, ranked_document_ids: list[str], relevant_documents: dict[str, int]) -> float:
        if not relevant_documents:
            return 0.0

        retrieved_relevant = len(set(ranked_document_ids).intersection(relevant_documents))
        return retrieved_relevant / len(relevant_documents)

    def _precision_at_k(self, ranked_document_ids: list[str], relevant_documents: dict[str, int], cutoff: int) -> float:
        if cutoff <= 0:
            return 0.0

        top_documents = ranked_document_ids[:cutoff]
        retrieved_relevant = len(set(top_documents).intersection(relevant_documents))
        return retrieved_relevant / cutoff

    def _ndcg(self, ranked_document_ids: list[str], relevant_documents: dict[str, int], cutoff: int) -> float:
        if not relevant_documents:
            return 0.0

        def gain(relevance: int) -> float:
            return (2.0**relevance) - 1.0

        dcg = 0.0
        for position, document_id in enumerate(ranked_document_ids[:cutoff], start=1):
            relevance = relevant_documents.get(document_id, 0)
            if relevance <= 0:
                continue
            dcg += gain(relevance) / math.log2(position + 1)

        ideal_relevances = sorted(relevant_documents.values(), reverse=True)[:cutoff]
        idcg = 0.0
        for position, relevance in enumerate(ideal_relevances, start=1):
            idcg += gain(relevance) / math.log2(position + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def _score_query(self, ranked_document_ids: list[str], relevant_documents: dict[str, int], cutoff: int) -> PerQueryMetrics:
        return PerQueryMetrics(
            average_precision=self._average_precision(ranked_document_ids, relevant_documents),
            recall=self._recall(ranked_document_ids, relevant_documents),
            precision_at_10=self._precision_at_k(ranked_document_ids, relevant_documents, 10),
            ndcg=self._ndcg(ranked_document_ids, relevant_documents, cutoff),
        )

    def evaluate(self, payload: EvaluateRequest) -> EvaluationResult:
        with self._lock:
            if not payload.qrels:
                raise RuntimeError("Qrels are required for evaluation.")

            qrels_by_query = {qrel.query_id: self._relevant_documents(qrel) for qrel in payload.qrels}
            runs_by_model: dict[str, dict[str, list[str]]] = {}

            for system_run in payload.system_results:
                runs_by_model.setdefault(system_run.model_name, {})[system_run.query_id] = system_run.ranked_document_ids

            metrics_by_model: dict[str, MetricSummary] = {}
            for model_name, query_runs in runs_by_model.items():
                per_query_metrics: list[PerQueryMetrics] = []
                for query_id, relevant_documents in qrels_by_query.items():
                    ranked_document_ids = query_runs.get(query_id, [])
                    per_query_metrics.append(self._score_query(ranked_document_ids, relevant_documents, payload.cutoff))

                query_count = float(len(per_query_metrics)) if per_query_metrics else 1.0
                metrics_by_model[model_name] = MetricSummary(
                    map_score=sum(item.average_precision for item in per_query_metrics) / query_count,
                    recall=sum(item.recall for item in per_query_metrics) / query_count,
                    precision_at_10=sum(item.precision_at_10 for item in per_query_metrics) / query_count,
                    ndcg=sum(item.ndcg for item in per_query_metrics) / query_count,
                )

            return EvaluationResult(metrics_by_model=metrics_by_model)


def build_evaluation_service() -> EvaluationService:
    return EvaluationService()