FROM python:3.11.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# Install git so pip can clone dependencies from GitHub!
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY dispatcher/*.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]