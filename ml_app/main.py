import io
import time

import psutil
from fastapi import FastAPI, Request, UploadFile
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.exposition import start_http_server

from resnet_inference import ModelInference


app = FastAPI(title="ML Inference Service")
model = ModelInference()

start_http_server(9001)

REQUEST_COUNT = Counter(
    "ml_app_requests",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

CPU_USAGE = Gauge(
    "ml_app_cpu_usage_percent",
    "CPU usage percentage"
)

MEMORY_USAGE = Gauge(
    "ml_app_memory_usage_percent",
    "Memory usage percentage"
)

RESPONSE_TIME = Histogram(
    "ml_app_response_time_seconds",
    "Request response time in seconds",
    ["endpoint"]
)


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    start = time.time()

    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()

    RESPONSE_TIME.labels(endpoint=endpoint).observe(time.time() - start)
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return response


@app.get("/")
async def home():
    return {"message": "ML app is running"}


@app.post("/predict")
async def predict(image: UploadFile):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = model.transform_image(pil_image)
    prediction = model.predict(image_tensor)

    return {"prediction": prediction}
