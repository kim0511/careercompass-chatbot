FROM python:3.11-slim

WORKDIR /app

COPY model_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# qa_pipeline + orgs.json 은 루트에서 복사
COPY qa_pipeline.py .
COPY orgs.json .
COPY model_server/ ./model_server/

WORKDIR /app/model_server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
