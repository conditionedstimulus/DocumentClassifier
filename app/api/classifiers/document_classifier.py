# from typing import Any
from fastapi import APIRouter, Request, UploadFile  # , HTTPException
# from app.schemas import DocumentClassifierResult
from app.core.config import settings
from app.utils.globals import g

router = APIRouter()


@router.get("/classifier_model_details/", response_model=dict)
async def get_model_details() -> dict:
    return {
        "model_name": settings.MODEL_NAME}


@router.post("/classify/")
async def classify_text(
        request: Request,
        pdf_file: UploadFile):

    document_classifier = g.lilt_classifier
    result = await document_classifier.predict_from_uploadfile(file=pdf_file)

    return result
