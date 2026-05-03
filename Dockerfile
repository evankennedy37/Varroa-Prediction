FROM python:3.11-slim

WORKDIR /varroa_pipeline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY steps/ ./steps/
COPY setup_training_pipeline.py .
COPY setup_prediction_pipeline.py .

ENV CLEARML_API_HOST=""
ENV CLEARML_API_ACCESS_KEY=""
ENV CLEARML_API_SECRET_KEY=""

CMD ["bash", "-c", "python setup_training_pipeline.py && python setup_prediction_pipeline.py"]