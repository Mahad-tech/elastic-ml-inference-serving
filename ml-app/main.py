import io
import time

import psutil
from fastapi import FastAPI, Request, UploadFile
from PIL import Image
from prometheus_client.exposition import start_http_server

from config import ML_APP_METRICS_PORT, ML_APP_TITLE
from metrics import CPU_USAGE, MEMORY_USAGE, REQUEST_COUNT, RESPONSE_TIME
from resnet_inference import ModelInference


app = FastAPI(title=ML_APP_TITLE)
model = ModelInference()

start_http_server(ML_APP_METRICS_PORT)


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """
    Record request count, response latency, CPU usage, and memory usage.
    """
    method = request.method
    endpoint = request.url.path
    start_time = time.time()

    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=response.status_code,
    ).inc()

    RESPONSE_TIME.labels(endpoint=endpoint).observe(
        time.time() - start_time
    )

    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return response


@app.get("/")
async def home():
    return {"message": "ML app is running"}


@app.post("/predict")
async def predict(image: UploadFile):
    """
    Receive an uploaded image, preprocess it, run ResNet18 inference,
    and return the predicted class with confidence.
    """
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = model.transform_image(pil_image)
    prediction = model.predict(image_tensor)

    return {"prediction": prediction}