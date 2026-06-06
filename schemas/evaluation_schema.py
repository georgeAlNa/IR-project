from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QrelDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    document_id: str = Field(..., alias="Document_Id", min_length=1)
    relevance: int = Field(..., alias="Relevance", ge=0)


class QrelQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    query_id: str = Field(..., alias="Query_Id", min_length=1)
    relevant_documents: list[QrelDocument] = Field(..., alias="Relevant_Documents", min_length=1)


class SystemRun(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    model_name: str = Field(..., alias="Model_Name", min_length=1)
    query_id: str = Field(..., alias="Query_Id", min_length=1)
    ranked_document_ids: list[str] = Field(..., alias="Ranked_Document_Ids", min_length=1)


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    qrels: list[QrelQuery] | None = Field(default=None, alias="Qrels")
    system_results: list[SystemRun] = Field(default_factory=list, alias="System_Results")
    cutoff: int = Field(10, alias="Cutoff", ge=1)


class MetricSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    map_score: float = Field(..., alias="MAP")
    recall: float = Field(..., alias="Recall")
    precision_at_10: float = Field(..., alias="Precision@10")
    ndcg: float = Field(..., alias="nDCG")


class EvaluateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    metrics_by_model: dict[str, MetricSummary] = Field(..., alias="Metrics_By_Model")
