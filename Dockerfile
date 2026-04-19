FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY services ./services
COPY shared ./shared
COPY scripts ./scripts
COPY README.md .
COPY Makefile .

CMD ["python", "-m", "uvicorn", "services.gateway_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
