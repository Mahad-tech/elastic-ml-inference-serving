import asyncio
import os
import random
import re
import time
from typing import Tuple
import aiohttp
from aiohttp import ClientTimeout, FormData

DISPATCHER_ENDPOINT = os.getenv(
    "DISPATCHER_ENDPOINT",
    "http://127.0.0.1:8001/add_to_queue",
)

IMAGE_DIR = os.getenv(
    "IMAGE_DIR",
    "./imagenet-sample-images",
)

WORKLOAD_FILE = os.getenv(
    "WORKLOAD_FILE",
    "workload.txt",
)

class AsyncImageLoadTester:
    def __init__(self, workload, endpoint, image_dir):
        self.workload = list(workload)
        self.endpoint = endpoint
        self.image_dir = image_dir
        self.image_paths = self.load_image_paths()
        
        # Statistics counters
        self.total_requests = 0
        self.successful_requests = 0
        self.class_counts = {}
        self.total_confidence = 0.0
        self.processed_count = 0
        self.request_timeout = ClientTimeout(total=80)
        
        print(f"Found {len(self.image_paths)} images for testing")

    def load_image_paths(self):
        image_paths = []
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(self.image_dir, filename))
        if not image_paths:
            raise ValueError(f"No images found in {self.image_dir}")
        return image_paths

    def get_request_data(self) -> Tuple[str, str]:
        image_path = random.choice(self.image_paths)
        image_id = os.path.basename(image_path)
        return image_id, image_path

    async def send_single_request(self, session):
        self.total_requests += 1
        image_id, image_path = self.get_request_data()
        
        try:
            with open(image_path, "rb") as image_file:
                form_data = FormData()
                form_data.add_field(
                    "image",
                    image_file,
                    filename=os.path.basename(image_path),
                    content_type="image/jpeg",
                )

                async with session.post(
                    self.endpoint,
                    data=form_data,
                    timeout=self.request_timeout,
                ) as response:
                    response_json = await response.json(content_type=None)
                    if self.process_response(image_id, response_json):
                        self.successful_requests += 1
        except asyncio.TimeoutError:
            print(f"Timeout: {image_id} exceeded {self.request_timeout.total}s")
        except Exception as exc:
            print(f"Error with {image_id}: {exc}")

    def process_response(self, image_id: str, response: dict) -> bool:
        if "prediction" not in response:
            print(f"Invalid response for {image_id}: {response}")
            return False

        prediction = response["prediction"]
        match = re.match(r"([^:]+):\s*([\d.]+)%", prediction)

        if not match:
            print(f"Cannot parse prediction '{prediction}' for {image_id}")
            return False

        class_name = match.group(1).strip()
        confidence = float(match.group(2))

        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        self.total_confidence += confidence
        self.processed_count += 1

        print(f"[{self.processed_count}] {image_id}: '{class_name}' {confidence:.1f}%")
        return True

    async def run_workload(self):
        print("\n🚀 Starting Workload Simulation...")
        start_time = time.time()
        
        # Limit concurrent connections so Windows Defender doesn't trip and drop sockets
        connector = aiohttp.TCPConnector(limit=100, force_close=False, enable_cleanup_closed=True)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            for second_idx, req_count in enumerate(self.workload):
                tick_start = time.time()

                if req_count > 0:
                    # Spawn all requests for this second concurrently
                    tasks = [self.send_single_request(session) for _ in range(req_count)]
                    await asyncio.gather(*tasks)
                
                if second_idx % 10 == 0 or req_count > 20:
                    print(f"-> Time: {second_idx}s / {len(self.workload)}s | Active Target: {req_count} req/s")

                elapsed = time.time() - tick_start
                sleep_duration = 1.0 - elapsed
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

    def display_results(self):
        total = self.total_requests
        successful = self.successful_requests
        success_rate = (successful / total * 100) if total else 0
        average_confidence = (
            self.total_confidence / self.processed_count
            if self.processed_count
            else 0
        )

        print("\n----- Test Results -----")
        print(f"Total requests sent: {total}")
        print(f"Successful replies : {successful}")
        print(f"Success rate       : {success_rate:.1f}%")
        print(f"Avg model accuracy : {average_confidence:.1f}%")


def load_workload():
    with open(WORKLOAD_FILE, "r") as file:
        return [int(value) for value in file.read().split()]


import argparse

if __name__ == "__main__":
    # Create an argument parser to read terminal flags like --url
    parser = argparse.ArgumentParser(description="Async Image Load Tester")
    parser.add_argument("--url", type=str, help="The target endpoint URL")
    parser.add_argument("--workload", type=str, help="The workload file") # Ignored for now since we use the fallback string path
    args = parser.parse_args()

    workload = load_workload()

    print(
        f"Workload Profile: {len(workload)} seconds, "
        f"{sum(workload)} total requests, "
        f"peak={max(workload)} req/s"
    )

    # Use the --url flag if provided, otherwise fall back to the environment variable path
    endpoint_url = args.url if args.url else DISPATCHER_ENDPOINT
    
    # Ensure it appends /add_to_queue if the raw domain port was passed
    if not endpoint_url.endswith("/add_to_queue"):
        endpoint_url = endpoint_url.rstrip("/") + "/add_to_queue"

    print(f" Target Endpoint: {endpoint_url}")

    tester = AsyncImageLoadTester(
        workload=workload,
        endpoint=endpoint_url,
        image_dir=IMAGE_DIR,
    )

    # Execute using the native Python asynchronous runtime loop
    asyncio.run(tester.run_workload())
    tester.display_results()