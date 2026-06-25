import asyncio
import logging
import time
import uuid
import httpx

from fastapi import FastAPI, Request, UploadFile
from prometheus_client.exposition import start_http_server

from config import (
    DISPATCHER_METRICS_PORT,
    ML_SERVICE_URL,          # K8s Service ClusterIP/NodePort URL
    REQUEST_TIMEOUT_SECONDS,
)
from dispatcher import Dispatcher
from metrics import REQUEST_COUNT, RESPONSE_TIME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dispatcher = Dispatcher()
app = FastAPI(title="Dispatcher")

start_http_server(DISPATCHER_METRICS_PORT)

HTTP_CLIENT = None

@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    # Configure client to reuse connections without recycling headers per image
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    HTTP_CLIENT = httpx.AsyncClient(limits=limits, timeout=REQUEST_TIMEOUT_SECONDS)
    logger.info("Stateless High-Concurrency Async HTTP Client initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    global HTTP_CLIENT
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

@app.middleware("http")
async def add_metrics(request: Request, call_next):
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
    return {
        "message": "This is the DISPATCHER APP",
        "queue_depth": await dispatcher.qsize(),
    }

async def forward_to_ml_pod(image_bytes: bytes) -> dict:
    """Forwards image payload instantly to the K8s load balancer service."""
    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
    # The Kubernetes Service automatically load balances this
    response = await HTTP_CLIENT.post(f"{ML_SERVICE_URL}/predict", files=files)
    response.raise_for_status()
    return response.json()

@app.post("/add_to_queue")
async def request_queue(image: UploadFile):
    """
    High-frequency non-blocking pipeline.
    Bypasses static worker restrictions to process requests concurrently.
    """
    # Read image bytes to prevent streaming timeouts
    image_bytes = await image.read()
    
    # Track the temporary metric state for custom autoscaler logic
    await dispatcher.add_to_queue(image_bytes, str(uuid.uuid4()))
    queue_size = await dispatcher.qsize()

    try:
        # Launch inference directly via the pooled network client
        result = await forward_to_ml_pod(image_bytes)
        
        # Pop from tracking queue once complete
        if not dispatcher.request_queue.empty():
            await dispatcher.request_queue.get()
            dispatcher.request_queue.task_done()
            
        return {
            "prediction": result["prediction"],
            "queue_size": queue_size,
        }
    except Exception as e:
        logger.error(f"Inference failure: {e}")
        # Clean the tracking queue on error states
        if not dispatcher.request_queue.empty():
            await dispatcher.request_queue.get()
            dispatcher.request_queue.task_done()
            
        return {
            "error": "Request processing failed or timed out",
            "queue_size": queue_size,
        }