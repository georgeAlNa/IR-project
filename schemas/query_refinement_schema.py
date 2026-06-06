from pydantic import BaseModel, ConfigDict, Field


class RefineRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    query: str = Field(..., alias="Query", min_length=1)


class RefineResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    original_query: str = Field(..., alias="Original_Query")
    corrected_query: str = Field(..., alias="Corrected_Query")
    expanded_query: str = Field(..., alias="Expanded_Query")
    expanded_terms: list[str] = Field(..., alias="Expanded_Terms")
