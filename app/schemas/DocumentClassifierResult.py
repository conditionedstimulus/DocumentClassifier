from typing import Union

from pydantic import BaseModel

from app.schemas.DocumentType import DocumentType


class DocumentClassifierResult(BaseModel):
    predicted_class: Union[str, None] = None
    probabilities: Union[dict[DocumentType, float], None] = None
