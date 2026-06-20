import asyncio
import os
import httpx
import logging
from kubernetes import client, config
import time
import math

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-operated.default.svc:9090")
DEPLOYMENT_NAME = "ml-app-deployment"
NAMESPACE = "default"

# Timing Controls
POLL_INTERVAL = 15 

# OPTIMIZATION: Asymmetric Cooldown Targets
SCALE_UP_COOLDOWN = 15    # Fast response to allow subsequent scale-ups much quicker
SCALE_DOWN_COOLDOWN = 180  # Conservative drop to keep resources on to absorb traffic

# Scaling Boundaries
MIN_REPLICAS = 1
MAX_REPLICAS = 10
DESIRED_QSIZE = 30

async def get_metric(query):
    """Get qsize from Prometheus."""
    try:
        async with httpx.AsyncClient() as client_session:
            response = await client_session.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5)
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
            label_selector="app=ml-app"
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
    """Scale deployment based on qsize with asymmetric cooldowns."""
    global last_scale_time, current_cooldown
    
    # Check if the active cooldown safety window (dynamic based on last action direction) has passed
    if time.time() - last_scale_time < current_cooldown:
        logger.info("Skipping scaling due to active stabilization/cooldown window")
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
            desired_replicas = MIN_REPLICAS
        elif qsize > DESIRED_QSIZE:
            desired_replicas = math.ceil(current_replicas * (qsize / DESIRED_QSIZE))
        elif qsize <= DESIRED_QSIZE:
            desired_replicas = max(MIN_REPLICAS, math.ceil(current_replicas * (qsize / DESIRED_QSIZE)))
        
        desired_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, desired_replicas))

    if desired_replicas != current_replicas:
        try:
            apps_v1.patch_namespaced_deployment_scale(
                name=DEPLOYMENT_NAME,
                namespace=NAMESPACE,
                body={"spec": {"replicas": desired_replicas}}
            )
            
            # OPTIMIZATION: Dynamically set the next cooldown based on scaling direction
            if desired_replicas > current_replicas:
                current_cooldown = SCALE_UP_COOLDOWN
                direction_msg = "UP"
            else:
                current_cooldown = SCALE_DOWN_COOLDOWN
                direction_msg = "DOWN"
                
            logging.getLogger().setLevel(logging.INFO)
            logger.info(f"Scaled {DEPLOYMENT_NAME} {direction_msg} from {current_replicas} to {desired_replicas} replicas. Cooldown set to {current_cooldown}s.")
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
    current_cooldown = 0  # Starts at zero to allow immediate initial action
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        try:
            loop.run_until_complete(main())
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        logger.info(f"Sleeping for {POLL_INTERVAL} seconds")
        loop.run_until_complete(asyncio.sleep(POLL_INTERVAL))