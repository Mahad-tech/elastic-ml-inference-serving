import os


ML_APP_TITLE = os.getenv("ML_APP_TITLE", "ML Inference Service")

ML_APP_HOST = os.getenv("ML_APP_HOST", "0.0.0.0")
ML_APP_PORT = int(os.getenv("ML_APP_PORT", "8000"))

ML_APP_METRICS_PORT = int(os.getenv("ML_APP_METRICS_PORT", "9001"))

TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "1"))
TORCH_NUM_INTEROP_THREADS = int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1"))