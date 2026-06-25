import io
import time
import os
import threading


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import psutil
from fastapi import FastAPI, Request, UploadFile, Response, status
from PIL import Image
from prometheus_client.exposition import start_http_server

from config import ML_APP_METRICS_PORT, ML_APP_TITLE
from metrics import CPU_USAGE, MEMORY_USAGE, REQUEST_COUNT, RESPONSE_TIME
from resnet_inference import ModelInference

app = FastAPI(title=ML_APP_TITLE)

model_ready = False
model = None

start_http_server(ML_APP_METRICS_PORT)

@app.on_event("startup")
async def initialize_and_warmup():
    """Initializes model, runs dummy image inference, and marks pod warm."""
    global model, model_ready
    
    # 1. Load weights into memory
    model = ModelInference()
    
    # 2. Warm up the model graph with a fake image so the first real request is instant
    try:
        dummy_img = Image.new('RGB', (224, 224), color='white')
        dummy_tensor = model.transform_image(dummy_img)
        _ = model.predict(dummy_tensor)
    except Exception:
        pass
        
    # 3. Flip readiness flag ONLY after graph compile is hot
    model_ready = True

# Background thread to collect CPU/Memory every second asynchronously
def collect_system_metrics():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent(interval=1))
            MEMORY_USAGE.set(psutil.virtual_memory().percent)
        except Exception:
            pass
        time.sleep(1)

threading.Thread(target=collect_system_metrics, daemon=True).start()

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """
    Record request count and clean response latency.
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

    return response

@app.get("/")
async def home():
    return {"message": "ML app is running"}

@app.get("/healthz")
async def health_check(response: Response):
    """Returns 503 while initializing/warming graph, 200 OK when ready."""
    if not model_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "warming_up"}
    return {"status": "ready"}

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