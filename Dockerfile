FROM python:3.11.9-slim

WORKDIR /varroa_pipeline

#Installing requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY steps/ ./steps/
COPY setup_training_pipeline.py .
COPY setup_prediction_pipeline.py .

#Getting ClearML Credentials
ENV CLEARML_API_HOST=""
ENV CLEARML_API_ACCESS_KEY=""
ENV CLEARML_API_SECRET_KEY=""

#Running scripts to set-up ClearML Pipelines
CMD ["bash", "-c", "python setup_training_pipeline.py && python setup_prediction_pipeline.py"]