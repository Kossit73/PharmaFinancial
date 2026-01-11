FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=src

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY pharma_app.py /app/pharma_app.py
EXPOSE 8000

CMD ["uvicorn", "financial_models.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
