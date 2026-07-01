# Elastic ML Inference Serving
**TU Ilmenau — Cloud Computing SS2026**
Mahad Ahmad [72470] · Sumit Shrivastava [70484]
Supervisors: Peter Amthor & Wenfei Huang

## What This Project Does

This project builds a complete elastic ML inference serving system on Kubernetes (Minikube). It classifies images using a ResNet18 model and automatically scales the number of inference replicas based on real-time latency — comparing a custom latency-driven autoscaler against Kubernetes' built-in Horizontal Pod Autoscaler (HPA).

## System Architecture

```
Load Tester
    |
    | POST /add_to_queue
    v
Dispatcher Service (port 8001)
    |
    | POST /predict  (via Kubernetes Service load balancing)
    v
ML App Replicas (port 8000)  [1 to 10 replicas]
    |
    v
Prediction Response

Monitoring:
Dispatcher & ML App --> Prometheus --> Grafana
                    Prometheus --> Custom Autoscaler --> scales ML App replicas
```

All components run inside a single Minikube cluster.


## Project Structure

```
elastic-ml-inference-serving/
│
├── ml-app/                         # ML inference service (source + Dockerfile)
│   ├── Dockerfile
│   ├── main.py
│   ├── resnet_inference.py
│   ├── config.py
│   ├── metrics.py
│   └── __init__.py
│
│
├── dispatcher/                     # Dispatcher service
│   ├── Dockerfile
│   ├── main.py
│   ├── dispatcher.py
│   ├── workers.py
│   ├── main_accepted_backup.py
│   ├── config.py
│   ├── metrics.py
│   └── __init__.py
│
├── custom_autoscaler/              # Latency-driven autoscaler
│   ├── Dockerfile
│   └── autoscaler.py
│
├── load_tester_config/             # Load testing client
│   ├── Dockerfile.tester
│   ├── load_tester.py
│   ├── load.py
│   ├── simple_load.py
│   └── __init__.py
│
├── manifests/                      # Kubernetes manifests
│   ├── ml-app-deployment.yaml
│   ├── dispatcher-deployment.yaml
│   ├── custom-autoscaler-deployment.yaml
│   ├── prometheus-instance.yaml
│   ├── prometheus-rbac.yaml
│   ├── grafana.yaml
│   ├── hpa-70.yaml
│   └── hpa-90.yaml
│   └── load-tester-job.yaml
│
├── workload.txt                    # Bursty workload schedule (3–18 req/s, 630s)
├── workload_2x.txt                 # 2x intensity variant
├── workload_original.txt           # Original baseline workload
├── requirements.txt
├── linux_startup.txt               # Full Linux deployment guide
├── win_startup.txt                 # Full Windows deployment guide
├── test.jpg                        # Sample image for manual testing
└── experiment_*.csv                # Collected metrics per experiment
```


## Components

### 1. ML Inference Service (ml-app)

FastAPI service running ResNet18 image classification.

Endpoints:
```
GET  /         Health message
GET  /healthz  Readiness probe (503 until model is warm, then 200)
POST /predict  Accepts uploaded image, returns predicted class + confidence
```

Key design decisions:
- Model loaded once at startup with a dummy warm-up inference to avoid cold-start latency on first real request.
- `/healthz` returns 503 until warm-up completes, keeping the pod out of Kubernetes load balancing until ready.
- `torch.set_num_threads(1)` enforced to match the 1-CPU-per-replica resource limit and avoid thread contention.
- Prometheus metrics exposed on a separate port (9001).
- Resources: 500m CPU request, 1 CPU limit, 768Mi memory limit per replica.

Example response:
```json
{"prediction": "golden retriever: 87.3%"}
```

---

### 2. Dispatcher Service (dispatcher)

FastAPI service that acts as the single entry point for all traffic.

Endpoints:
```
GET  /              Returns current queue depth
POST /add_to_queue  Accepts image, queues it, forwards to ML App, returns prediction
```

Key design decisions:
- Uses a shared `httpx.AsyncClient` with connection pooling (100 keep-alive, 500 max connections) for efficient concurrent forwarding.
- Forwards requests to `ML_SERVICE_URL` — a Kubernetes Service ClusterIP — so Kubernetes handles load balancing across ML App replicas automatically.
- Maintains an `asyncio.Queue` internally for queue-depth visibility and metrics.
- Prometheus metrics exposed on port 9000, including `dispatcher_response_time_seconds` histogram — this is the key signal used by the custom autoscaler.
- Exposed externally via NodePort 32028 (HTTP) and 32084 (metrics).

Example response:
```json
{"prediction": "golden retriever: 87.3%", "queue_size": 2}
```

---

### 3. Custom Autoscaler (custom_autoscaler)

Latency-driven autoscaler that polls Prometheus every 2 seconds and scales ML App replicas up or down based on average response latency.

Scaling logic:
- Scaling signal: cumulative average latency via PromQL on `dispatcher_response_time_seconds` histogram.
- Scale-up formula: `growth_step = ceil(replicas × (latency_ratio − 1))`, minimum +2 pods per event.
- Scale-up cooldown: 4 seconds (fast response to load spikes).
- Scale-down cooldown: 180 seconds (prevents oscillation on trailing load).
- SLO override: if latency exceeds SLO threshold during cooldown, bypass cooldown and force immediate scale-up.
- Replica range: 1 minimum, 10 maximum.

PromQL queries used:
```
Average latency:
sum(rate(dispatcher_response_time_seconds_sum[15s])) / sum(rate(dispatcher_response_time_seconds_count[15s]))

P99 latency:
histogram_quantile(0.99, sum(rate(dispatcher_response_time_seconds_bucket[15s])) by (le))
```

---

### 4. Load Tester (load_tester_config)

Async load testing client that replays a time-series workload schedule against the dispatcher.

- Reads `workload.txt` for request rate over time (3–18 req/s, 630 seconds total).
- Picks random images from `imagenet-sample-images/`.
- Fires async HTTP POSTs to dispatcher `/add_to_queue` using `aiohttp`.
- Records per-request latency for P50/P90/P99/Max calculation.
- Results saved as `experiment_*.csv` files.
- Runs as a Kubernetes Job via `manifests/load-tester-job.yaml`.

---

### 5. Monitoring (Prometheus + Grafana)

Prometheus scrapes metrics from both the dispatcher (port 9000) and ML App (port 9001) every few seconds.

Grafana dashboard available at `http://localhost:3000/d/autoscaler-comparison` (login: admin / admin).

Dashboard panels:
- P99 / P90 / P50 latency vs 0.5s SLO line
- ML App replica count over time
- Dispatcher queue size
- Average latency
- Request rate (req/s)
- Live stats summary

---

## Configuration

### ML App Environment Variables

| Variable | Default | Description |
|---|---|---|
| ML_APP_PORT | 8000 | FastAPI port |
| ML_APP_METRICS_PORT | 9001 | Prometheus metrics port |
| TORCH_NUM_THREADS | 1 | PyTorch intra-op threads |
| TORCH_NUM_INTEROP_THREADS | 1 | PyTorch inter-op threads |

### Dispatcher Environment Variables

| Variable | Default | Description |
|---|---|---|
| ML_SERVICE_URL | http://127.0.0.1:8000 | ML App Kubernetes Service URL |
| NUM_WORKERS | 10 | Background consumer workers |
| HTTP_TIMEOUT_SECONDS | 30 | Per-request HTTP timeout |
| DISPATCHER_METRICS_PORT | 9000 | Prometheus metrics port |
| HTTP_MAX_CONNECTIONS | 500 | Max HTTP connections in pool |
| HTTP_MAX_KEEPALIVE_CONNECTIONS | 100 | Max keep-alive connections |

### Load Tester Environment Variables

| Variable | Default | Description |
|---|---|---|
| DISPATCHER_ENDPOINT | http://127.0.0.1:8001/add_to_queue | Dispatcher target URL |
| IMAGE_DIR | ./imagenet-sample-images | Sample images directory |
| WORKLOAD_FILE | workload.txt | Workload schedule file |

---

## Experiments

Three experiments were run against the same 630-second bursty workload (3–18 req/s, 5868 total requests):

| Experiment | Strategy | Signal | Config |
|---|---|---|---|
| E1 | Custom Autoscaler | Avg latency | Poll 2s, SLO 0.45s, min step +2 |
| E2 | HPA 70% CPU | CPU utilisation | Target 70%, min 1, max 10 replicas |
| E3 | HPA 90% CPU | CPU utilisation | Target 90%, min 1, max 10 replicas |

Results summary (all strategies, 100% success rate):

| Strategy | Avg | P50 | P90 | P99 | Max |
|---|---|---|---|---|---|
| Custom (E1) | 0.4603s | 0.3809s | 0.8228s | 1.5091s | 2.5093s |
| HPA 70% (E2) | 0.4817s | 0.4279s | 0.8070s | 0.9219s | 1.1370s |
| HPA 90% (E3) | 0.4768s | 0.4123s | 0.8201s | 0.9021s | 1.0051s |

Custom autoscaler wins on average and median latency — meets the 0.5s SLO. HPA 90% wins on tail latency (P99/Max).

---

## Deployment (Linux)

For the full step-by-step deployment guide see `linux_startup.txt`. For Windows see `win_startup.txt`. Quick summary below.

**Prerequisites:** Docker Desktop, Minikube, kubectl, Python 3.11+

**Step 1 — Start Minikube**
```bash
minikube delete
minikube start --cpus=4 --memory=7000m --driver=docker
eval $(minikube docker-env)
```

**Step 2 — Install Prometheus Operator CRDs**
```bash
kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

**Step 3 — Create ServiceAccount for autoscaler**
```bash
kubectl create serviceaccount python-client-sa
kubectl create clusterrolebinding python-client-sa-admin-binding \
  --clusterrole=cluster-admin \
  --serviceaccount=default:python-client-sa
```

**Step 4 — Build all Docker images**
```bash
docker build -t ml-app:latest -f ml-app/Dockerfile .
docker build -t dispatcher:latest -f dispatcher/Dockerfile .
docker build -t autoscaler:latest -f custom_autoscaler/Dockerfile .
docker build -t load-tester:local -f load_tester_config/Dockerfile.tester .
```

**Step 5 — Deploy all components**
```bash
kubectl apply -f manifests/ml-app-deployment.yaml
kubectl apply -f manifests/prometheus-rbac.yaml
kubectl apply -f manifests/dispatcher-deployment.yaml
kubectl apply -f manifests/prometheus-instance.yaml
kubectl apply -f manifests/custom-autoscaler-deployment.yaml
kubectl apply -f manifests/grafana.yaml
```

**Step 6 — Port forward services (keep terminal open)**
```bash
kubectl port-forward svc/dispatcher-service 8001:8001 &
kubectl port-forward svc/prometheus-operated 9090:9090 &
kubectl port-forward svc/grafana 3000:3000 &
```

**Step 7 — Verify everything is running**
```bash
kubectl get pods -A
curl http://localhost:8001/
```

---

## Running Experiments

**Experiment 1 — Custom Autoscaler (default, already deployed)**
```bash
kubectl scale deployment ml-app-deployment --replicas=1
kubectl scale deployment custom-autoscaler-deployment --replicas=1
kubectl delete job load-tester-job --ignore-not-found
kubectl apply -f manifests/load-tester-job.yaml
kubectl logs -l job-name=load-tester-job -f
```

**Experiment 2 — HPA 70% CPU**
```bash
kubectl scale deployment custom-autoscaler-deployment --replicas=0
kubectl scale deployment ml-app-deployment --replicas=1
kubectl apply -f manifests/hpa-70.yaml
kubectl delete job load-tester-job --ignore-not-found
kubectl apply -f manifests/load-tester-job.yaml
```

**Experiment 3 — HPA 90% CPU**
```bash
kubectl delete hpa --all
kubectl scale deployment ml-app-deployment --replicas=1
kubectl apply -f manifests/hpa-90.yaml
kubectl delete job load-tester-job --ignore-not-found
kubectl apply -f manifests/load-tester-job.yaml
```

Reset between experiments: always scale ML App back to 1 replica and disable the previous autoscaler before starting the next experiment.

---

## Manual Testing

Send a single test image directly to the dispatcher:
```bash
curl -X POST http://localhost:8001/add_to_queue \
  -F "image=@test.jpg"
```

Send directly to the ML App (bypassing dispatcher):
```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@test.jpg"
```

Check readiness:
```bash
curl http://localhost:8000/healthz
curl http://localhost:8001/
```

---

## Metrics Exposed

**Dispatcher (port 9000)**

| Metric | Type | Description |
|---|---|---|
| dispatcher_requests | Counter | Total HTTP requests received |
| dispatcher_queue_size | Gauge | Current internal queue depth |
| dispatcher_cpu_usage_percent | Gauge | Dispatcher CPU usage |
| dispatcher_memory_usage_percent | Gauge | Dispatcher memory usage |
| dispatcher_response_time_seconds | Histogram | End-to-end request latency — used by autoscaler |

**ML App (port 9001)**

| Metric | Type | Description |
|---|---|---|
| ml_app_requests | Counter | Total HTTP requests received |
| ml_app_cpu_usage_percent | Gauge | ML App CPU usage |
| ml_app_memory_usage_percent | Gauge | ML App memory usage |
| ml_app_response_time_seconds | Histogram | Inference-only request latency |

---

## Debug Commands

```bash
# Check all pods
kubectl get pods -A

# Check logs
kubectl logs deployment/dispatcher-deployment --tail=20
kubectl logs deployment/ml-app-deployment --tail=30
kubectl logs deploy/custom-autoscaler-deployment --tail=30

# Rebuild and restart a component
docker build -t ml-app:latest -f ml-app/Dockerfile .
kubectl rollout restart deployment ml-app-deployment

# Query Prometheus manually
curl -s "http://localhost:9090/api/v1/query?query=sum(rate(dispatcher_response_time_seconds_sum[15s]))/sum(rate(dispatcher_response_time_seconds_count[15s]))" | python3 -m json.tool

# If Minikube is broken
minikube delete --all --purge
rm -rf ~/.minikube
```

---

## Known Issues and Fixes Applied

| # | Issue | Fix |
|---|---|---|
| 1 | Wrong env var name in dispatcher | `ML_API_ENDPOINT` renamed to `ML_SERVICE_URL` |
| 2 | Prometheus DNS failure | Hardcoded monitoring namespace changed to default namespace URL |
| 3 | ServiceAccount missing for autoscaler | Added `kubectl create serviceaccount` step |
| 4 | Prometheus not scraping dispatcher | `targetPort 9090` corrected to `9000` in Service manifest |
| 5 | Docker build failure (git missing) | Added `apt-get install -y git` to both Dockerfiles |

---

## Grafana Dashboard

URL: `http://localhost:3000/d/autoscaler-comparison`
Login: admin / admin

---