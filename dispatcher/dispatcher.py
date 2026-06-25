import asyncio

class Dispatcher:
    def __init__(self):
        # This allows the original workload to enter the dispatcher without being rejected at queue entry.
        self.request_queue = asyncio.Queue()

    async def qsize(self) -> int:
        """Returns the current size of the request queue."""
        return self.request_queue.qsize()

    async def add_to_queue(self, image_bytes: bytes, request_id: str) -> asyncio.Queue:
        """
        Receive pre-read raw image bytes and store them directly 
        in the queue with a request ID.
        """
        # The bytes are already extracted in main.py, so this is to put them straight into the queue
        await self.request_queue.put((image_bytes, request_id))
        return self.request_queue