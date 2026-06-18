import asyncio
import logging
from io import BytesIO

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
        # This fixes the dual-stack loop socket drop on Windows/Minikube:
        transport=httpx.AsyncHTTPTransport(
            local_address="0.0.0.0"
        ),
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
    Continuously consume requests from the queue, forward them to the
    ML inference service, and return results to the waiting request.
    """
    print(f"Worker {worker_id} started")

    while workers_running_ref():
        request_id = None
        try:
            # 1. Fetch from queue and isolate the request identifier first
            request_queue = dispatcher.request_queue
            queue_item, request_id = await request_queue.get()

            try:
                # 2. Process inference explicitly linked to this specific item
                result = await get_inference_for_item(queue_item, http_client)
                print(f"Worker {worker_id} got result: {result}")

                async with pending_requests_lock:
                    future = pending_requests.pop(request_id, None)

                if future and not future.done():
                    future.set_result(result)
                    print(f"Worker {worker_id} delivered result to request {request_id[:8]}")

            finally:
                request_queue.task_done()

        except Exception as exc:
            print(f"Worker {worker_id} error processing request {request_id}: {exc}")
            # 3. Only abort the specific request that actually triggered the network error!
            if request_id:
                async with pending_requests_lock:
                    future = pending_requests.pop(request_id, None)
                if future and not future.done():
                    future.set_exception(exc)

        await asyncio.sleep(WORKER_SLEEP_SECONDS)

    print(f"Worker {worker_id} stopped")


from io import BytesIO
import socket
from urllib.parse import urlparse

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
        print(f"Fallback warning: DNS resolution failed: {dns_err}")
        target_url = ML_API_ENDPOINT

    # Prepare multipart form-data payload using the stable raw bytes
    files = {
        "image": ("image.jpg", queue_item_bytes, "image/jpeg")
    }

    print(f"Forwarding image payload directly to: {target_url}")
    
    response = await http_client.post(
        url=target_url,
        files=files,
    )

    if response.status_code != 200:
        print(f"Backend returned error status code {response.status_code}. Raw body: {response.text}")
        
    response.raise_for_status()
    prediction = response.json()["prediction"]
    return prediction