import asyncio
import os
import uuid
import time
import httpx
import psutil
import logging

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.exposition import start_http_server
from io import BytesIO
from fastapi import FastAPI, UploadFile, Request
from dispatcher import Dispatcher


dispatcher = Dispatcher()
app = FastAPI(title="Dispatcher")

# Metrics
REQUEST_COUNT = Counter(
    "dispatcher_requests",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

QUEUE_SIZE = Gauge(
    "dispatcher_queue_size",
    "Number of tasks in the ML inference queue"
)

CPU_USAGE = Gauge(
    "dispatcher_cpu_usage_percent",
    "CPU usage percentage"
)

MEMORY_USAGE = Gauge(
    "dispatcher_memory_usage_percent",
    "Memory usage percentage"
)

RESPONSE_TIME = Histogram(
    "dispatcher_response_time_seconds",
    "Request response time in seconds",
    ["endpoint"]
)

# Read ML service URL from environment variable.
# Local default: ML app running on port 8000.
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8000")
ML_API_ENDPOINT = f"{ML_SERVICE_URL}/predict"

# Shared HTTP client for connection pooling
HTTP_CLIENT = None

# Start metrics server on port 9000
start_http_server(9000)

# Request mapping: request_id -> asyncio.Future
pending_requests = {}
pending_requests_lock = asyncio.Lock()
workers_running = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Start background consumer workers and metrics updater."""
    global workers_running, HTTP_CLIENT

    workers_running = True

    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=0
        )
    )

    for worker_id in range(1, 11):
        asyncio.create_task(consumer_worker(worker_id=worker_id))

    asyncio.create_task(update_system_metrics())

    print("Started 10 consumer workers and system metrics updater")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop workers and close HTTP client."""
    global workers_running, HTTP_CLIENT

    workers_running = False

    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()


async def update_system_metrics():
    """Update CPU, memory, and queue size metrics every second."""
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            queue_size = await dispatcher.qsize()

            CPU_USAGE.set(cpu_percent)
            MEMORY_USAGE.set(memory_percent)
            QUEUE_SIZE.set(queue_size)

            logger.info(
                f"CPU: {cpu_percent}%, "
                f"Memory: {memory_percent}%, "
                f"Queue: {queue_size}"
            )

        except Exception as e:
            logger.error(f"Error in update_system_metrics: {e}")

        await asyncio.sleep(1)


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Track request count and response time."""
    method = request.method
    endpoint = request.url.path
    start_time = time.time()

    response = await call_next(request)

    status = response.status_code

    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=status
    ).inc()

    RESPONSE_TIME.labels(endpoint=endpoint).observe(
        time.time() - start_time
    )

    return response


async def consumer_worker(worker_id: int):
    """
    Consumer worker:
    continuously takes requests from the queue,
    forwards them to the ML service,
    and returns results to the waiting HTTP request.
    """
    print(f"Worker {worker_id} started")

    while workers_running:
        try:
            result, request_id = await get_inference()
            print(f"Worker {worker_id} got result: {result}")

            async with pending_requests_lock:
                future = pending_requests.pop(request_id, None)

            if future and not future.done():
                future.set_result(result)
                print(
                    f"Worker {worker_id} delivered result "
                    f"to request {request_id[:8]}"
                )

        except Exception as e:
            print(f"Worker {worker_id} error: {e}")

            async with pending_requests_lock:
                if pending_requests:
                    request_id = next(iter(pending_requests))
                    future = pending_requests.pop(request_id)

                    if not future.done():
                        future.set_exception(e)

        await asyncio.sleep(0.1)

    print(f"Worker {worker_id} stopped")


@app.get("/")
async def home():
    return {
        "message": "This is the DISPATCHER APP",
        "queue_depth": await dispatcher.qsize()
    }


@app.post("/add_to_queue")
async def request_queue(image: UploadFile):
    """
    Producer endpoint:
    receives an image, puts it into the queue,
    and waits for a background worker to process it.
    """
    request_id = str(uuid.uuid4())

    # create and register the Future before queueing the image.
    future = asyncio.Future()

    async with pending_requests_lock:
        pending_requests[request_id] = future

    await dispatcher.add_to_queue(image, request_id)
    queue_size = await dispatcher.qsize()

    print(f"ml service url: {ML_SERVICE_URL}")
    print(f"ml api endpoint: {ML_API_ENDPOINT}")
    print(f"This is the qsize: {queue_size}")

    try:
        prediction = await asyncio.wait_for(future, timeout=60)

        print(f"Request {request_id[:8]} got result: {prediction}")

        return {
            "prediction": prediction,
            "queue_size": queue_size
        }

    except asyncio.TimeoutError:
        async with pending_requests_lock:
            pending_requests.pop(request_id, None)

        return {
            "error": "Request timeout",
            "queue_size": queue_size
        }


async def get_inference():
    """
    Consumer function:
    gets an image from the queue,
    forwards it to the ML service /predict endpoint,
    and returns prediction with request_id.
    """
    request_queue = dispatcher.request_queue

    queue_item, request_id = await request_queue.get()

    img_buffer = BytesIO()
    queue_item.save(img_buffer, format="JPEG")

    files = {
        "image": ("image.jpg", img_buffer.getvalue(), "image/jpeg")
    }

    img_buffer.close()

    response = await HTTP_CLIENT.post(
        url=ML_API_ENDPOINT,
        files=files
    )

    response_json = response.json()
    prediction = response_json["prediction"]

    print(prediction)

    return prediction, request_id