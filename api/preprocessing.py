from fastapi import APIRouter, Depends

from core.preprocessing_pipeline import PreprocessingPipeline
from schemas.preprocessing_schema import PreprocessRequest, PreprocessResponse

router = APIRouter(tags=["Preprocessing"])

_PIPELINE = PreprocessingPipeline.default()


def get_preprocessing_pipeline() -> PreprocessingPipeline:
    return _PIPELINE


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(
    payload: PreprocessRequest,
    pipeline: PreprocessingPipeline = Depends(get_preprocessing_pipeline),
) -> PreprocessResponse:
    processed_text = pipeline.process(payload.text)
    return PreprocessResponse(processed_text=processed_text)
