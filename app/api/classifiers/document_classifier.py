from fastapi import APIRouter, Request, UploadFile

from app.schemas.DocumentClassifierResult import DocumentClassifierResult
from app.core.config import settings
from app.utils.globals import g
from app.utils.logger import logger

router = APIRouter()


@router.get("/classifier_model_details/", response_model=dict)
async def get_model_details() -> dict:
    logger.info("Request received: Get model details")
    return {
        "model_name": settings.MODEL_NAME,
        "tokenizer_name": settings.TOKENIZER_NAME,
    }


@router.post("/classify_file/", response_model=DocumentClassifierResult)
async def classify_file(
    request: Request, pdf_file: UploadFile
) -> DocumentClassifierResult:
    logger.info(f"Request received: Classify file '{pdf_file.filename}'")
    document_classifier = g.lmv3_classifier
    result = await document_classifier.predict_from_uploadfile(file=pdf_file)

    logger.info(f"Classification result, predicted class: {result.predicted_class}")
    return result
