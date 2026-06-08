from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RepresentationType = Literal["tfidf", "bm25", "embeddings"]


class IndexedDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    document_id: str = Field(..., alias="Document_Id", min_length=1)
    processed_text: str = Field(..., alias="Processed_Text", min_length=1)
    original_text: str | None = Field(default=None, alias="Original_Text")


class IndexRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    documents: list[IndexedDocument] = Field(default_factory=list, alias="Documents")
    representation_type: RepresentationType = Field("tfidf", alias="Representation_Type")
    k1: float = Field(1.5, alias="K1", ge=0.0)
    b: float = Field(0.75, alias="B", ge=0.0, le=1.0)
    vector_size: int = Field(64, alias="Vector_Size", ge=8)


class IndexResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_count: int = Field(..., alias="Document_Count")
    vocabulary_size: int = Field(..., alias="Vocabulary_Size")
    average_document_length: float = Field(..., alias="Average_Document_Length")
    active_representation: RepresentationType = Field(..., alias="Active_Representation")


class RepresentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    query: str = Field(..., alias="Query", min_length=1)
    representation_type: RepresentationType = Field("tfidf", alias="Representation_Type")
    k1: float | None = Field(default=None, alias="K1", ge=0.0)
    b: float | None = Field(default=None, alias="B", ge=0.0, le=1.0)


class RepresentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    representation_type: RepresentationType = Field(..., alias="Representation_Type")
    query_tokens: list[str] = Field(..., alias="Query_Tokens")
    vector: list[float] | dict[str, float] = Field(..., alias="Vector")

