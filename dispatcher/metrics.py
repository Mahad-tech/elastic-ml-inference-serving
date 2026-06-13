from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNT = Counter(
    "dispatcher_requests",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

QUEUE_SIZE = Gauge(
    "dispatcher_queue_size",
    "Number of tasks in the ML inference queue",
)

CPU_USAGE = Gauge(
    "dispatcher_cpu_usage_percent",
    "CPU usage percentage",
)

MEMORY_USAGE = Gauge(
    "dispatcher_memory_usage_percent",
    "Memory usage percentage",
)

RESPONSE_TIME = Histogram(
    "dispatcher_response_time_seconds",
    "Request response time in seconds",
    ["endpoint"],
)
