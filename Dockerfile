FROM tiangolo/uvicorn-gunicorn:python3.11

# Update package lists and upgrade existing packages
RUN apt-get update && apt-get upgrade -y

# Install dependencies for OCR and PDF processing
RUN apt-get install -y tesseract-ocr poppler-utils poppler-data

# Copy and install Python dependencies
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

# Copy the application code into the container
COPY ./app /code/app

WORKDIR /code
