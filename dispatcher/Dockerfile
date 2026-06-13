FROM python:3.11.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
COPY dispatcher/main.py .
COPY dispatcher/dispatcher.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
