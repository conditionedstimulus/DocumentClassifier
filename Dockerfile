FROM tiangolo/uvicorn-gunicorn-fastapi

RUN apt clean
RUN apt update

RUN pip install --upgrade pip

# poppler, tessaract, and fonts for pdf2image
RUN apt-get install tesseract-ocr
RUN apt-get install poppler-utils -y
RUN apt-get install poppler-data -y

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

WORKDIR /code
