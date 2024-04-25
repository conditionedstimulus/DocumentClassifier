import numpy as np
import pytesseract
import torch
from fastapi import HTTPException, UploadFile
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
)


class LiltClassifier:
    def __init__(self, model_name: str, tokenizer_name: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
        self.model.to(device)

    async def predict_from_uploadfile(self, file: UploadFile) -> str:
        word_list, bboxes = await self.preprocessing(file)
        encoded = self.tokenization(words=word_list, boxes=bboxes)
        predicted_class = self.predict(encoded)
        return predicted_class

    # Prediction steps
    def tokenization(self, words: list[str], boxes: list) -> BatchEncoding:
        encoding = self.tokenizer(words, boxes=boxes, return_tensors="pt")
        return encoding

    def predict(self, encoded: BatchEncoding) -> str:
        outputs = self.model(**encoded)
        predicted_class_idx = outputs.logits.argmax(-1).item()

        predicted_class = self.id2label[predicted_class_idx]

        return predicted_class

    # Preprocessing steps
    @staticmethod
    async def convert_pdf_to_image(file: UploadFile) -> Image.Image:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

        try:
            contents = await file.read()
            images = convert_from_bytes(contents)
            if not images:
                raise PDFPageCountError("No images found in PDF file.")
            image = images[0].convert("RGB")
            return image
        except PDFPageCountError:
            raise HTTPException(
                status_code=400, detail="The PDF file does not contain any pages."
            )
        except Exception as e:
            raise HTTPException(
                status_code=422, detail=f"An error occurred during PDF conversion: {e}"
            )

    def pytessaract_ocr_process(
        self, input_image: Image.Image
    ) -> tuple[list[str], list]:
        try:
            width, height = input_image.size

            ocr_df = pytesseract.image_to_data(input_image, output_type="data.frame")
            ocr_df = ocr_df.dropna().reset_index(drop=True)
            float_cols = ocr_df.select_dtypes("float").columns
            ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
            ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)

            coordinates = ocr_df[["left", "top", "width", "height"]]
            actual_boxes = []
            for _, row in coordinates.iterrows():
                x, y, w, h = tuple(
                    row
                )  # the row comes in (left, top, width, height) format
                actual_box = [
                    x,
                    y,
                    x + w,
                    y + h,
                ]  # we turn it into (left, top, left+width, top+height) to get the actual box
                actual_boxes.append(actual_box)

            # normalize the bounding boxes
            bboxes = []
            for box in actual_boxes:
                bboxes.append(self.normalize_box(box, width, height))

            word_list = ocr_df["text"].to_list()
            word_list = [str(w) for w in word_list]

            if len(word_list) != len(bboxes):
                raise HTTPException(
                    status_code=422,
                    detail="Number of words and the number of bounding boxes are not matching.",
                )

            return word_list, bboxes

        except Exception as e:
            raise HTTPException(
                status_code=422, detail=f"An error occurred during the OCR process: {e}"
            )

    @staticmethod
    def normalize_box(box: list, width: int, height: int) -> list[int]:
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    async def preprocessing(self, file: UploadFile) -> tuple[list[str], list]:
        try:
            converted_image = await self.convert_pdf_to_image(file)
            word_list, bboxes = self.pytessaract_ocr_process(converted_image)
            return word_list, bboxes
        except Exception as e:
            # Handle preprocessing errors
            raise HTTPException(
                status_code=422, detail=f"An error occurred during preprocessing: {e}"
            )
