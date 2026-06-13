import os
import random
import re
from typing import Tuple

import asyncio
from aiohttp import FormData, ClientTimeout
from barazmoon import BarAzmoon


class ImageLoadTester(BarAzmoon):
    def __init__(self, *, workload, endpoint, image_dir, http_method="post", **kwargs):
        super().__init__(
            workload=workload,
            endpoint=endpoint,
            http_method=http_method,
            **kwargs
        )

        self.image_dir = image_dir
        self.image_paths = []

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(image_dir, filename))

        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_paths)} images for testing")

        self.class_counts = {}
        self.average_confidence = 0
        self.total_confidence = 0
        self.processed_count = 0
        self.request_timeout = ClientTimeout(total=80)

    def get_request_data(self) -> Tuple[str, str]:
        image_path = random.choice(self.image_paths)
        image_id = os.path.basename(image_path)
        return image_id, image_path

    async def predict(self, delay, session):
        await asyncio.sleep(delay)
        image_id, image_path = self.get_request_data()

        file_handle = None

        try:
            file_handle = open(image_path, "rb")

            form_data = FormData()
            form_data.add_field(
                "image",
                file_handle,
                filename=os.path.basename(image_path),
                content_type="image/jpeg",
            )

            async with session.post(
                self.endpoint,
                data=form_data,
                timeout=self.request_timeout,
            ) as response:
                response_json = await response.json(content_type=None)
                is_success = self.process_response(image_id, response_json)
                return 1 if is_success else 0

        except asyncio.TimeoutError:
            print(f"Timeout: {image_id} exceeded {self.request_timeout.total}s")
            return 0

        except Exception as exc:
            print(f"Error with {image_id}: {exc}")
            return 0

        finally:
            if file_handle:
                file_handle.close()

    def process_response(self, image_id: str, response: dict) -> bool:
        try:
            print(response)

            if "prediction" not in response:
                print(f"Invalid response for {image_id}: {response}")
                return False

            prediction_str = response["prediction"]
            match = re.match(r"([^:]+):\s*([\d.]+)%", prediction_str)

            if not match:
                print(f"Cannot parse prediction '{prediction_str}' for {image_id}")
                return False

            class_name = match.group(1).strip()
            confidence = float(match.group(2))

            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
            self.total_confidence += confidence
            self.processed_count += 1
            self.average_confidence = self.total_confidence / self.processed_count

            print(f"{image_id}: '{class_name}' {confidence:.1f}%")
            return True

        except Exception as e:
            print(f"Response processing error for {image_id}: {e}")
            return False

    def display_results(self):
        print("\n----- Test Results -----")
        print(f"Total requests    : {self._BarAzmoon__counter}")
        print(f"Successful        : {self._BarAzmoon__success_counter.value}")
        print(f"Avg confidence    : {self.average_confidence:.1f}%")

        if self.processed_count > 0:
            print("\nClassification breakdown:")
            total_success = self._BarAzmoon__success_counter.value

            for cls, cnt in sorted(
                self.class_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = (cnt / total_success * 100) if total_success > 0 else 0
                print(f"  {cls}: {cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    DISPATCHER_ENDPOINT = "http://127.0.0.1:8001/add_to_queue"
    IMAGE_DIR = "./imagenet-sample-images"

    with open("workload.txt", "r") as f:
        raw = f.read().split()

    experiment_workload = [int(x) for x in raw]

    print(
        f"Workload: {len(experiment_workload)} seconds, "
        f"{sum(experiment_workload)} total requests, "
        f"peak={max(experiment_workload)} req/s"
    )

    tester = ImageLoadTester(
        workload=experiment_workload,
        endpoint=DISPATCHER_ENDPOINT,
        image_dir=IMAGE_DIR,
        timeout=30,
    )

    total, successful = tester.start()
    tester.display_results()

    success_rate = (successful / total * 100) if total > 0 else 0
    print(f"\nSummary: {successful}/{total} ({success_rate:.1f}%) successful requests")
