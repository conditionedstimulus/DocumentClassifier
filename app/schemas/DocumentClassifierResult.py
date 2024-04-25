from typing import Union

from pydantic import BaseModel, field_validator

from app.schemas.DocumentType import DocumentType


class DocumentClassifierResult(BaseModel):
    predicted_class: Union[DocumentType, None] = None
    probabilities: Union[dict[DocumentType, float], None] = None

    @field_validator("probabilities")
    def round_and_sort_probabilities(cls, value):
        if value is not None:
            # round
            rounded_probabilities = {
                doc_type: round(prob, 3) for doc_type, prob in value.items()
            }
            # sort
            sorted_probabilities = dict(
                sorted(
                    rounded_probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            return sorted_probabilities
        return None
