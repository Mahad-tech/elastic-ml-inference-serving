import asyncio
import logging
import time
import uuid

from fastapi import FastAPI, Request, UploadFile
from prometheus_client.exposition import start_http_server

from config import (
    DISPATCHER_METRICS_PORT,
    ML_API_ENDPOINT,
    ML_SERVICE_URL,
    NUM_WORKERS,
    REQUEST_TIMEOUT_SECONDS,
)
from dispatcher import Dispatcher
from metrics import REQUEST_COUNT, RESPONSE_TIME
from workers import (
    consumer_worker,
    create_http_client,
    update_system_metrics,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dispatcher = Dispatcher()
app = FastAPI(title="Dispatcher")

start_http_server(DISPATCHER_METRICS_PORT)

HTTP_CLIENT = None
pending_requests = {}
pending_requests_lock = asyncio.Lock()
workers_running = False


def are_workers_running() -> bool:
    return workers_running


@app.on_event("startup")
async def startup_event():
    """
    Start shared HTTP client, consumer workers, and metrics updater.
    """
    global workers_running, HTTP_CLIENT

    workers_running = True
    HTTP_CLIENT = await create_http_client()

    for worker_id in range(1, NUM_WORKERS + 1):
        asyncio.create_task(
            consumer_worker(
                worker_id=worker_id,
                dispatcher=dispatcher,
                http_client=HTTP_CLIENT,
                pending_requests=pending_requests,
                pending_requests_lock=pending_requests_lock,
                workers_running_ref=are_workers_running,
            )
        )

    asyncio.create_task(update_system_metrics(dispatcher))

    print(f"Started {NUM_WORKERS} consumer workers and system metrics updater")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Stop workers and close HTTP client.
    """
    global workers_running, HTTP_CLIENT

    workers_running = False

    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """
    Track request count and response time for every HTTP request.
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
    return {
        "message": "This is the DISPATCHER APP",
        "queue_depth": await dispatcher.qsize(),
    }


@app.post("/add_to_queue")
async def request_queue(image: UploadFile):
    """
    Producer endpoint:
    receives an image, puts it into the queue,
    and waits for a background worker to process it.
    """
    request_id = str(uuid.uuid4())

    future = asyncio.Future()

    async with pending_requests_lock:
        pending_requests[request_id] = future

    await dispatcher.add_to_queue(image, request_id)
    queue_size = await dispatcher.qsize()

    print(f"ml service url: {ML_SERVICE_URL}")
    print(f"ml api endpoint: {ML_API_ENDPOINT}")
    print(f"This is the qsize: {queue_size}")

    try:
        prediction = await asyncio.wait_for(
            future,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        print(f"Request {request_id[:8]} got result: {prediction}")

        return {
            "prediction": prediction,
            "queue_size": queue_size,
        }

    except asyncio.TimeoutError:
        async with pending_requests_lock:
            pending_requests.pop(request_id, None)

        return {
            "error": "Request timeout",
            "queue_size": queue_size,
        }