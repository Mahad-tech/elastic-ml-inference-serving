# Elastic ML Inference Serving

A modular machine learning inference pipeline for serving image classification requests using a ResNet18 model, a FastAPI-based inference service, an asynchronous dispatcher, and a load testing client.

The system is designed to receive image requests, queue them through a dispatcher, forward them to the inference service, and return classification results in a consistent response format.

---

## Overview

This project implements an end-to-end image inference serving pipeline with the following major components:

* **ResNet18 Inference Service**

  * FastAPI service for image classification.
  * Uses a pretrained ResNet18 model.
  * Performs image preprocessing before inference.
  * Returns predictions with class name and confidence score.

* **Dispatcher Service**

  * FastAPI service that receives client requests.
  * Uses an asynchronous request queue.
  * Starts background workers to process queued requests.
  * Forwards images to the inference service.
  * Maps prediction results back to the original request.

* **Load Tester**

  * Sends image requests to the dispatcher.
  * Uses a workload file to control request rate.
  * Tracks successful predictions and classification statistics.

* **Instrumentation**

  * Tracks request count, queue size, CPU usage, memory usage, and response latency.

---

## Project Structure

```text
elastic-ml-inference-serving/
│
├── ml_app/
│   ├── __init__.py
│   ├── config.py
│   ├── metrics.py
│   ├── main.py
│   ├── resnet_inference.py
│   └── Dockerfile
│
├── dispatcher/
│   ├── __init__.py
│   ├── config.py
│   ├── metrics.py
│   ├── dispatcher.py
│   ├── workers.py
│   ├── main.py
│   └── Dockerfile
│
├── load_tester_config/
│   ├── __init__.py
│   └── load_tester.py
│
├── requirements.txt
├── workload.txt
├── README.md
└── .gitignore
```

---

## Architecture

```text
Client / Load Tester
        |
        | POST /add_to_queue
        v
Dispatcher Service
        |
        | asyncio.Queue
        v
Background Consumer Workers
        |
        | POST /predict
        v
ResNet18 Inference Service
        |
        v
Prediction Response
```

---

## Components

### 1. ML Inference Service

The ML inference service is implemented using FastAPI and exposes the following endpoints:

```text
GET  /
POST /predict
```

The `/predict` endpoint accepts an uploaded image, preprocesses it, runs inference using a pretrained ResNet18 model, and returns the predicted class with confidence.

Example response:

```json
{
  "prediction": "tench: 97.2%"
}
```

---

### 2. Image Preprocessing

Before inference, each input image is processed using the standard ResNet18 ImageNet preprocessing pipeline:

1. Convert image to RGB.
2. Resize the image.
3. Center crop to 224x224.
4. Convert the image to a tensor.
5. Normalize using ImageNet mean and standard deviation.
6. Add a batch dimension.

The model is loaded once and reused for all requests to avoid unnecessary overhead.

---

### 3. Dispatcher Service

The dispatcher is responsible for request intake, queueing, and forwarding.

It exposes:

```text
GET  /
POST /add_to_queue
```

The `/add_to_queue` endpoint receives image requests from clients, assigns each request a unique ID, stores the request in an asynchronous queue, and waits for a background worker to process it.

Each queue item contains:

```text
(PIL image, request_id)
```

This request ID is used to match the prediction result back to the original client request.

Example response:

```json
{
  "prediction": "tench: 97.2%",
  "queue_size": 1
}
```

The `queue_size` value represents the queue depth observed after the request has been added to the queue.

---

### 4. Background Workers

The dispatcher starts multiple background workers. Each worker continuously:

1. Reads an item from the request queue.
2. Converts the image into JPEG bytes.
3. Sends the image to the ML inference service.
4. Receives the prediction response.
5. Resolves the waiting client request using the request ID.

This producer-consumer design separates request intake from inference processing and allows the dispatcher to handle concurrent workloads more effectively.

---

### 5. Load Tester

The load tester sends image requests to the dispatcher using a configurable workload file.

It reads images from a local image directory and sends each request as a multipart form upload using the field name:

```text
image
```

The load tester validates prediction responses and prints a summary containing:

* Total requests
* Successful requests
* Success rate
* Average confidence
* Classification breakdown

---

## Configuration

The project uses environment variables for runtime configuration.

### Dispatcher Configuration

| Variable                  |                 Default | Description                                          |
| ------------------------- | ----------------------: | ---------------------------------------------------- |
| `ML_SERVICE_URL`          | `http://127.0.0.1:8000` | URL of the ML inference service                      |
| `NUM_WORKERS`             |                    `10` | Number of dispatcher background workers              |
| `HTTP_TIMEOUT_SECONDS`    |                    `30` | Timeout for dispatcher-to-ML requests                |
| `REQUEST_TIMEOUT_SECONDS` |                    `60` | Maximum time a client request waits for a prediction |
| `DISPATCHER_METRICS_PORT` |                  `9000` | Dispatcher metrics port                              |

### ML App Configuration

| Variable                    | Default | Description                        |
| --------------------------- | ------: | ---------------------------------- |
| `ML_APP_METRICS_PORT`       |  `9001` | ML app metrics port                |
| `TORCH_NUM_THREADS`         |     `1` | Number of PyTorch CPU threads      |
| `TORCH_NUM_INTEROP_THREADS` |     `1` | Number of PyTorch inter-op threads |

### Load Tester Configuration

| Variable              |                              Default | Description                      |
| --------------------- | -----------------------------------: | -------------------------------- |
| `DISPATCHER_ENDPOINT` | `http://127.0.0.1:8001/add_to_queue` | Dispatcher endpoint              |
| `IMAGE_DIR`           |           `./imagenet-sample-images` | Directory containing test images |
| `WORKLOAD_FILE`       |                       `workload.txt` | Workload file path               |

---

## Local Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install CPU-only PyTorch and TorchVision:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the load testing library:

```bash
pip install git+https://github.com/reconfigurable-ml-pipeline/load_tester
```

### 3. Download sample images

```bash
git clone https://github.com/EliSchwartz/imagenet-sample-images.git
```

---

## Running the Project Locally

### Terminal 1: Start the ML inference service

```bash
cd ml_app
source ../.venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Check that it is running:

```bash
curl http://localhost:8000/
```

Expected response:

```json
{
  "message": "ML app is running"
}
```

---

### Terminal 2: Start the dispatcher

```bash
cd dispatcher
source ../.venv/bin/activate
ML_SERVICE_URL=http://127.0.0.1:8000 uvicorn main:app --host 0.0.0.0 --port 8001
```

Check that it is running:

```bash
curl http://localhost:8001/
```

Expected response:

```json
{
  "message": "This is the DISPATCHER APP",
  "queue_depth": 0
}
```

---

### Terminal 3: Send a test request

From the project root:

```bash
curl -X POST http://localhost:8001/add_to_queue \
  -F "image=@imagenet-sample-images/n01440764_tench.JPEG"
```

Expected response format:

```json
{
  "prediction": "tench: 97.2%",
  "queue_size": 1
}
```

The exact confidence score may vary.

---

## Running the Load Tester

From the project root:

```bash
source .venv/bin/activate
python load_tester_config/load_tester.py
```

Example output:

```text
Workload: 10 seconds, 10 total requests, peak=1 req/s
Found 1000 images for testing

----- Test Results -----
Total requests    : 10
Successful        : 10
Success rate      : 100.0%
Avg confidence    : 92.4%

Summary: 10/10 (100.0%) successful requests
```

---

## Docker Usage

### Build the ML inference service image

```bash
docker build -t ml-app:latest -f ml_app/Dockerfile .
```

### Build the dispatcher image

```bash
docker build -t dispatcher-app:latest -f dispatcher/Dockerfile .
```

### Verify images

```bash
docker images | grep -E "ml-app|dispatcher-app"
```

---

## Metrics

The services expose runtime metrics for observability.

### Dispatcher metrics

| Metric                             | Description                    |
| ---------------------------------- | ------------------------------ |
| `dispatcher_requests`              | Total dispatcher HTTP requests |
| `dispatcher_queue_size`            | Current dispatcher queue size  |
| `dispatcher_cpu_usage_percent`     | Dispatcher CPU usage           |
| `dispatcher_memory_usage_percent`  | Dispatcher memory usage        |
| `dispatcher_response_time_seconds` | Dispatcher response latency    |

### ML app metrics

| Metric                         | Description                |
| ------------------------------ | -------------------------- |
| `ml_app_requests`              | Total ML app HTTP requests |
| `ml_app_cpu_usage_percent`     | ML app CPU usage           |
| `ml_app_memory_usage_percent`  | ML app memory usage        |
| `ml_app_response_time_seconds` | ML app response latency    |

---

## Verification

Compile all Python files:

```bash
python3 -m py_compile \
  ml_app/config.py \
  ml_app/metrics.py \
  ml_app/main.py \
  ml_app/resnet_inference.py \
  dispatcher/config.py \
  dispatcher/metrics.py \
  dispatcher/dispatcher.py \
  dispatcher/workers.py \
  dispatcher/main.py \
  load_tester_config/load_tester.py
```

Run health checks:

```bash
curl http://localhost:8000/
curl http://localhost:8001/
```

Run an end-to-end prediction:

```bash
curl -X POST http://localhost:8001/add_to_queue \
  -F "image=@imagenet-sample-images/n01440764_tench.JPEG"
```

---

## Performance Notes

The full workload can be demanding on a single CPU-only local machine. Under high load, the inference service may become the bottleneck because ResNet18 inference is computationally expensive on CPU.

A lower success rate during heavy local load testing does not necessarily indicate a functional issue. It usually means that requests are timing out because the local machine cannot process the workload quickly enough.

For local functional verification, a short workload such as 10 requests at 1 request per second is recommended.

---

## Status

The project currently provides:

* A working FastAPI ResNet18 inference service
* A modular dispatcher with asynchronous queueing
* Background worker-based request forwarding
* A configurable load tester
* Docker support
* Runtime metrics and latency instrumentation
* Local verification commands
