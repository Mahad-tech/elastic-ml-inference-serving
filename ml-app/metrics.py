from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNT = Counter(
    "ml_app_requests",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

CPU_USAGE = Gauge(
    "ml_app_cpu_usage_percent",
    "CPU usage percentage",
)

MEMORY_USAGE = Gauge(
    "ml_app_memory_usage_percent",
    "Memory usage percentage",
)

RESPONSE_TIME = Histogram(
    "ml_app_response_time_seconds",
    "Request response time in seconds",
    ["endpoint"],
)
