import asyncio
import os
import random
import re
from typing import Tuple

from aiohttp import ClientTimeout, FormData
from barazmoon import BarAzmoon


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


class ImageLoadTester(BarAzmoon):
    def __init__(self, *, workload, endpoint, image_dir, http_method="post", **kwargs):
        super().__init__(
            workload=workload,
            endpoint=endpoint,
            http_method=http_method,
            **kwargs,
        )

        self.image_dir = image_dir
        self.image_paths = self.load_image_paths()
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

    async def predict(self, delay, session):
        await asyncio.sleep(delay)

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
                    return 1 if self.process_response(image_id, response_json) else 0

        except asyncio.TimeoutError:
            print(f"Timeout: {image_id} exceeded {self.request_timeout.total}s")
            return 0

        except Exception as exc:
            print(f"Error with {image_id}: {exc}")
            return 0

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

        print(f"{image_id}: '{class_name}' {confidence:.1f}%")
        return True

    def display_results(self):
        total = self._BarAzmoon__counter
        successful = self._BarAzmoon__success_counter.value
        success_rate = (successful / total * 100) if total else 0
        average_confidence = (
            self.total_confidence / self.processed_count
            if self.processed_count
            else 0
        )

        print("\n----- Test Results -----")
        print(f"Total requests    : {total}")
        print(f"Successful        : {successful}")
        print(f"Success rate      : {success_rate:.1f}%")
        print(f"Avg confidence    : {average_confidence:.1f}%")

        if self.processed_count:
            print("\nClassification breakdown:")
            for class_name, count in sorted(
                self.class_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                percentage = count / self.processed_count * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")


def load_workload():
    with open(WORKLOAD_FILE, "r") as file:
        return [int(value) for value in file.read().split()]


if __name__ == "__main__":
    workload = load_workload()

    print(
        f"Workload: {len(workload)} seconds, "
        f"{sum(workload)} total requests, "
        f"peak={max(workload)} req/s"
    )

    tester = ImageLoadTester(
        workload=workload,
        endpoint=DISPATCHER_ENDPOINT,
        image_dir=IMAGE_DIR,
        timeout=30,
    )

    total, successful = tester.start()
    tester.display_results()

    success_rate = (successful / total * 100) if total else 0
    print(f"\nSummary: {successful}/{total} ({success_rate:.1f}%) successful requests")