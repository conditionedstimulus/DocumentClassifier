from pdf2image import convert_from_bytes
from PIL import Image
import torch
import numpy as np
from transformers import AutoTokenizer, BatchEncoding
import pytesseract
from fastapi import UploadFile, HTTPException


class LiltClassifier:

    def __init__(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
        self.model = torch.load(model_path, map_location=device)

    async def predict_from_uploadfile(self, file: UploadFile):
        word_list, bboxes = await self.preprocessing(file)
        encoded = self.tokenization(words=word_list, boxes=bboxes)
        predicted_class = self.predict(encoded)
        return predicted_class

    # Prediction steps
    def tokenization(self, words: list[str], boxes: list) -> BatchEncoding:
        encoding = self.tokenizer(words, boxes=boxes, return_tensors="pt")
        return encoding

    def predict(self, encoded):
        outputs = self.model(**encoded)
        predicted_class_idx = outputs.logits.argmax(-1).item()

        return predicted_class_idx
        # predicted_class = model.config.id2label[predicted_class_idx]

    # Preprocessing steps
    @staticmethod
    async def convert_pdf_to_image(file) -> Image:
        contents = await file.read()
        images = convert_from_bytes(contents)
        image = images[0].convert("RGB")

        return image

    def pytessaract_ocr_process(self, input_image) -> tuple[list[str], list]:
        width, height = input_image.size

        ocr_df = pytesseract.image_to_data(input_image, output_type='data.frame')
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)

        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for _, row in coordinates.iterrows():
            x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
            actual_box = [x, y, x + w, y + h]  # we turn it into (left, top, left+width, top+height) to get the actual box
            actual_boxes.append(actual_box)

        # normalize the bounding boxes
        bboxes = []
        for box in actual_boxes:
            bboxes.append(self.normalize_box(box, width, height))

        word_list = ocr_df['text'].to_list()

        if len(word_list) != len(bboxes):
            raise HTTPException(status_code=400, detail="Number of words and the number of bounding boxes are not matching.")

        return word_list, bboxes

    @staticmethod
    def normalize_box(box: list, width: int, height: int) -> tuple[int, int, int, int]:
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    async def preprocessing(self, file: UploadFile) -> tuple[list[str], list]:
        converted_image = await self.convert_pdf_to_image(file)
        word_list, bboxes = self.pytessaract_ocr_process(converted_image)

        return word_list, bboxes
