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
    to the ML inference service.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS),
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
        try:
            result, request_id = await get_inference(
                dispatcher=dispatcher,
                http_client=http_client,
            )

            print(f"Worker {worker_id} got result: {result}")

            async with pending_requests_lock:
                future = pending_requests.pop(request_id, None)

            if future and not future.done():
                future.set_result(result)
                print(
                    f"Worker {worker_id} delivered result "
                    f"to request {request_id[:8]}"
                )

        except Exception as exc:
            print(f"Worker {worker_id} error: {exc}")

            async with pending_requests_lock:
                if pending_requests:
                    request_id = next(iter(pending_requests))
                    future = pending_requests.pop(request_id)

                    if not future.done():
                        future.set_exception(exc)

        await asyncio.sleep(WORKER_SLEEP_SECONDS)

    print(f"Worker {worker_id} stopped")


async def get_inference(dispatcher, http_client: httpx.AsyncClient):
    """
    Get one request from the queue, convert the image to JPEG bytes,
    forward it to the ML inference service, and return the prediction.
    """
    request_queue = dispatcher.request_queue

    queue_item, request_id = await request_queue.get()

    try:
        img_buffer = BytesIO()
        queue_item.save(img_buffer, format="JPEG")

        files = {
            "image": ("image.jpg", img_buffer.getvalue(), "image/jpeg")
        }

        img_buffer.close()

        response = await http_client.post(
            url=ML_API_ENDPOINT,
            files=files,
        )

        response.raise_for_status()

        prediction = response.json()["prediction"]
        print(prediction)

        return prediction, request_id

    finally:
        request_queue.task_done()
