from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SearchRepresentationType = Literal["tfidf", "bm25", "embeddings", "hybrid_parallel", "hybrid_serial", "bert"]


class RankedDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    document_id: str = Field(..., alias="Document_Id", min_length=1)
    processed_text: str | None = Field(default=None, alias="Processed_Text")
    original_text: str | None = Field(default=None, alias="Original_Text")
    tfidf_vector: dict[str, float] | None = Field(default=None, alias="TFIDF_Vector")
    bm25_vector: dict[str, float] | None = Field(default=None, alias="BM25_Vector")
    embedding_vector: list[float] | None = Field(default=None, alias="Embedding_Vector")


class SearchQueryRepresentation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    tfidf_vector: dict[str, float] | None = Field(default=None, alias="TFIDF_Vector")
    bm25_vector: dict[str, float] | None = Field(default=None, alias="BM25_Vector")
    embedding_vector: list[float] | None = Field(default=None, alias="Embedding_Vector")


class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    representation_type: SearchRepresentationType = Field(..., alias="Representation_Type")
    query_representation: SearchQueryRepresentation = Field(..., alias="Query_Representation")
    dataset: list[RankedDocument] | None = Field(default=None, alias="Dataset")
    dataset_name: str | None = Field(default=None, alias="Dataset_Name", min_length=1)
    top_k: int = Field(10, alias="Top_K", ge=1)
    candidate_k: int = Field(20, alias="Candidate_K", ge=1)
    query_text: str | None = Field(default=None, alias="query_text")


class SearchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ranked_document_ids: list[str] = Field(..., alias="Ranked_Document_Ids")
    ranked_documents: list[RankedDocument] = Field(default_factory=list, alias="Ranked_Documents")
