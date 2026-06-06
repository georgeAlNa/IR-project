from pydantic import BaseModel, ConfigDict, Field


class PreprocessRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    text: str = Field(..., alias="Text", min_length=1)


class PreprocessResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    processed_text: str = Field(..., alias="Processed_Text")
