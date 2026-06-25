import asyncio
import logging
import socket
from io import BytesIO
from urllib.parse import urlparse

import httpx
import psutil

from config import (
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_TIMEOUT_SECONDS,
    ML_API_ENDPOINT,
    WORKER_SLEEP_SECONDS,
)
from metrics import CPU_USAGE, MEMORY_USAGE, QUEUE_SIZE

logger = logging.getLogger(__name__)


async def create_http_client() -> httpx.AsyncClient:
    """
    Create a shared async HTTP client for forwarding requests
    to the ML inference service, forcing IPv4 routing.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS),
        transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
        limits=httpx.Limits(
            max_connections=HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
        ),
    )


async def update_system_metrics(dispatcher):
    """
    Periodically update CPU, memory, and queue size metrics.
    """
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            queue_size = await dispatcher.qsize()

            CPU_USAGE.set(cpu_percent)
            MEMORY_USAGE.set(memory_percent)
            QUEUE_SIZE.set(queue_size)

            logger.info(
                "CPU: %.1f%%, Memory: %.1f%%, Queue: %s",
                cpu_percent,
                memory_percent,
                queue_size,
            )
        except Exception as exc:
            logger.error("Error while updating system metrics: %s", exc)

        await asyncio.sleep(1)


async def consumer_worker(
    worker_id: int,
    dispatcher,
    http_client: httpx.AsyncClient,
    pending_requests: dict,
    pending_requests_lock: asyncio.Lock,
    workers_running_ref,
):
    """
    Continuously consume requests from the queue and instantly offload them 
    as concurrent background network tasks without blocking the loop event thread.
    """
    print(f"Worker {worker_id} started")

    while workers_running_ref():
        try:
            request_queue = dispatcher.request_queue
            # Fetch from queue instantly
            queue_item, request_id = await request_queue.get()


            # ynlocks the worker loop immediately so it can pull the next queue item
            asyncio.create_task(
                process_and_deliver_inference(
                    worker_id=worker_id,
                    queue_item=queue_item,
                    request_id=request_id,
                    request_queue=request_queue,
                    http_client=http_client,
                    pending_requests=pending_requests,
                    pending_requests_lock=pending_requests_lock
                )
            )

        except Exception as exc:
            print(f"Worker {worker_id} loop encounter error: {exc}")
            await asyncio.sleep(0.1)

        if WORKER_SLEEP_SECONDS > 0:
            await asyncio.sleep(WORKER_SLEEP_SECONDS)

    print(f"Worker {worker_id} stopped")


async def process_and_deliver_inference(
    worker_id: int,
    queue_item,
    request_id: str,
    request_queue: asyncio.Queue,
    http_client: httpx.AsyncClient,
    pending_requests: dict,
    pending_requests_lock: asyncio.Lock
):
    """Handles network dispatching and future result mapping concurrently."""
    try:
        # Run network I/O bound processing task safely in the background
        result = await get_inference_for_item(queue_item, http_client)
        
        async with pending_requests_lock:
            future = pending_requests.pop(request_id, None)

        if future and not future.done():
            future.set_result(result)
            
    except Exception as exc:
        print(f"Worker {worker_id} error processing request {request_id}: {exc}")
        if request_id:
            async with pending_requests_lock:
                future = pending_requests.pop(request_id, None)
            if future and not future.done():
                future.set_exception(exc)
    finally:
        # Guarantee task completion is signaled to Prometheus tracking metrics
        request_queue.task_done()


async def get_inference_for_item(queue_item_bytes, http_client: httpx.AsyncClient):
    """
    Receive raw image bytes, resolve the endpoint host to bypass Docker MTU drops,
    and forward the payload safely to the ML inference service.
    """
    parsed_url = urlparse(ML_API_ENDPOINT)
    try:
        raw_ip = socket.gethostbyname(parsed_url.hostname)
        target_url = f"{parsed_url.scheme}://{raw_ip}:{parsed_url.port}{parsed_url.path}"
    except Exception as dns_err:
        target_url = ML_API_ENDPOINT

    files = {
        "image": ("image.jpg", queue_item_bytes, "image/jpeg")
    }
    
    response = await http_client.post(
        url=target_url,
        files=files,
    )

    if response.status_code != 200:
        print(f"Backend returned error status code {response.status_code}. Raw body: {response.text}")
        
    response.raise_for_status()
    return response.json()["prediction"]