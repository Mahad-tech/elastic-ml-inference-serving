import asyncio
import io

from fastapi import UploadFile
from PIL import Image


class Dispatcher:
    def __init__(self):
        # This allows the original workload to enter the dispatcher
        # without being rejected at queue entry.
        self.request_queue = asyncio.Queue()

    async def qsize(self) -> int:
        """Returns the current size of the request queue."""
        return self.request_queue.qsize()

    async def add_to_queue(self, request: UploadFile, request_id: str) -> asyncio.Queue:
        """
        Receive an uploaded image, decode it with PIL,
        and store it in the queue with a request ID.
        """
        image_bytes = await request.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        await self.request_queue.put((image, request_id))
        return self.request_queue