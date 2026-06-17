import asyncio
import httpx
import logging
from kubernetes import client, config
import time
import math

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROMETHEUS_URL = "http://prometheus-operated.monitoring.svc:9090"
DEPLOYMENT_NAME = "ml-app-deployment"
NAMESPACE = "default"
POLL_INTERVAL = 15
COOLDOWN_SECONDS = 150
MIN_REPLICAS = 1
MAX_REPLICAS = 6
DESIRED_QSIZE = 50

async def get_metric(query):
    """Get qsize from Prometheus."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5)
            response.raise_for_status()
            result = response.json()
            if result["status"] == "success" and result["data"]["result"]:
                return float(result["data"]["result"][0]["value"][1])
            return None
    except Exception as e:
        logger.error(f"Error fetching metric {query}: {e}")
        return None

async def check_replicas_ready(v1_api):
    """Check if all ml-app-deployment pods are ready."""
    try:
        pods = v1_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"app=ml-app"
        )
        for pod in pods.items:
            for condition in pod.status.conditions or []:
                if condition.type == "Ready" and condition.status != "True":
                    return False
        return True
    except client.exceptions.ApiException as e:
        logger.error(f"Error checking pod readiness: {e}")
        return False

async def scale_deployment(qsize, v1_api):
    """Scale deployment based on qsize."""
    global last_scale_time
    if time.time() - last_scale_time < COOLDOWN_SECONDS:
        logger.info("Skipping scaling due to cooldown period")
        return

    # Check if all replicas are ready
    if not await check_replicas_ready(v1_api):
        logger.info("Skipping scaling as not all pods are ready")
        return

    try:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        current_replicas = deployment.spec.replicas
    except client.exceptions.ApiException as e:
        logger.error(f"Error getting replicas: {e}")
        current_replicas = 1

    desired_replicas = current_replicas
    if qsize is not None:
        if qsize == 0:
            # Explicitly scale to MIN_REPLICAS when queue is empty
            desired_replicas = MIN_REPLICAS
        elif qsize > DESIRED_QSIZE:
            # Scale up for high queue sizes
            desired_replicas = math.ceil(current_replicas * (qsize / DESIRED_QSIZE))
        elif qsize <= DESIRED_QSIZE:
            # Gradual downscaling for low queue sizes
            desired_replicas = max(MIN_REPLICAS, math.ceil(current_replicas * (qsize / DESIRED_QSIZE)))
        # Ensure replicas stay within bounds
        desired_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, desired_replicas))

    if desired_replicas != current_replicas:
        try:
            apps_v1.patch_namespaced_deployment_scale(
                name=DEPLOYMENT_NAME,
                namespace=NAMESPACE,
                body={"spec": {"replicas": desired_replicas}}
            )
            logging.getLogger().setLevel(logging.INFO)
            logger.info(f"Scaled {DEPLOYMENT_NAME} from {current_replicas} to {desired_replicas} replicas")
            logging.getLogger().setLevel(logging.ERROR)
            last_scale_time = time.time()
        except client.exceptions.ApiException as e:
            logger.error(f"Error scaling deployment: {e}")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("No scaling action taken")
        logging.getLogger().setLevel(logging.ERROR)

async def main():
    """Run autoscaler."""
    qsize = await get_metric('dispatcher_queue_size{job="dispatcher-service", namespace="default"}')
    await scale_deployment(qsize, v1_api)

if __name__ == "__main__":
    logger.info("Entering infinite loop")
    try:
        config.load_incluster_config()
    except config.ConfigException:
        logger.error("Failed to load in-cluster config, falling back to kubeconfig")
        config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    v1_api = client.CoreV1Api()
    last_scale_time = 0
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        try:
            loop.run_until_complete(main())
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        logger.info(f"Sleeping for 15 seconds at {loop.time()}")
        loop.run_until_complete(asyncio.sleep(15))