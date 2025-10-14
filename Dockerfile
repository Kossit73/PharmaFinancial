
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY setup.py .
COPY streamlit_app.py .

RUN pip install --no-cache-dir -e .

#  Streamlit setup
EXPOSE 8501
# ENV STREAMLIT_SERVER_HEADLESS=true
# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ENABLECORS=false

CMD ["streamlit", "run", "streamlit_app.py"]