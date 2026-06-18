import math
import time
import requests
from kubernetes import client, config

PROM_URL = "http://localhost:9090/api/v1/query"

config.load_kube_config()
apps = client.AppsV1Api()


def query_prometheus(metric):
    r = requests.get(
        PROM_URL,
        params={"query": metric}
    )

    data = r.json()["data"]["result"]

    if not data:
        return 0

    return float(data[0]["value"][1])


while True:

    queue_depth = query_prometheus(
        "dispatcher_queue_size"
    )

    required_replicas = max(
        1,
        min(
            10,
            math.ceil(queue_depth / 5)
        )
    )

    apps.patch_namespaced_deployment_scale(
        name="dispatcher",
        namespace="default",
        body={
            "spec": {
                "replicas": required_replicas
            }
        }
    )

    print(
        f"Queue={queue_depth}, "
        f"Replicas={required_replicas}"
    )

    time.sleep(10)