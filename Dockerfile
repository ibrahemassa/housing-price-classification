FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
