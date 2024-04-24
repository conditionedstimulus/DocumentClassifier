from app.schemas.DocumentType import DocumentType
from pydantic import BaseModel
from typing import Union


class DocumentClassifierResult(BaseModel):
    predicted_class: Union[str, None] = None
    probabilities: Union[dict[DocumentType, float], None] = None
