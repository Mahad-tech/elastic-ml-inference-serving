import asyncio
import os
import httpx
import logging
from kubernetes import client, config
import time
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-operated.default.svc:9090")
DEPLOYMENT_NAME = "ml-app-deployment"
NAMESPACE = "default"

# 🏎️ Faster tracking evaluation
POLL_INTERVAL = 2 

# Asymmetric Cooldown Targets
SCALE_UP_COOLDOWN = 4      # Short cooldown allows fast subsequent steps if queue continues growing
SCALE_DOWN_COOLDOWN = 300  # 🚨 5-Minute lock: Do not scale down easily after launching pool

# Scaling Boundaries
MIN_REPLICAS = 1
MAX_REPLICAS = 15

# Core State Tracking Module Globals
last_scale_time = 0.0
current_cooldown = 0.0

async def get_metric(query):
    """Get qsize from Prometheus."""
    try:
        async with httpx.AsyncClient() as client_session:
            response = await client_session.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
            response.raise_for_status()
            result = response.json()
            if result["status"] == "success" and result["data"]["result"]:
                return float(result["data"]["result"][0]["value"][1])
            return 0.0
    except Exception as e:
        logger.error(f"Error fetching metric {query}: {e}")
        return 0.0

async def check_replicas_ready(v1_api):
    """Check if all ml-app-deployment pods are ready."""
    try:
        pods = v1_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector="app=ml-app"
        )
        for pod in pods.items:
            if pod.metadata.deletion_timestamp is not None:
                continue
            for condition in pod.status.conditions or []:
                if condition.type == "Ready" and condition.status != "True":
                    return False
        return True
    except client.exceptions.ApiException as e:
        logger.error(f"Error checking pod readiness: {e}")
        return False

async def scale_deployment(qsize, apps_v1, v1_api):
    """Scale deployment using aggressive proportional steps with cooldown overrides."""
    global last_scale_time, current_cooldown
    
    current_time = time.time()
    elapsed_time = current_time - last_scale_time
    
    try:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        current_replicas = deployment.spec.replicas
    except client.exceptions.ApiException as e:
        logger.error(f"Error getting replicas: {e}")
        return

    # --- 🚨 DYNAMIC COOLDOWN OVERRIDE ENGINE ---
    if elapsed_time < current_cooldown:
        # If the queue is growing and backing up, SHATTER the cooldown window!
        if qsize > 2 and current_replicas < MAX_REPLICAS:
            logger.warning(f"🚨 TRAFFIC SURGE DETECTED (qsize={qsize})! Overriding active cooldown window.")
        else:
            logger.info(f"Skipping scaling due to active cooldown window ({int(current_cooldown - elapsed_time)}s remaining)")
            return

    # --- AGGRESSIVE PROPORTIONAL STEP LOGIC ---
    if qsize == 0:
        if not await check_replicas_ready(v1_api):
            logger.info("Skipping scale-down as current replicas are stabilizing")
            return
        desired_replicas = max(MIN_REPLICAS, current_replicas - 1)
    else:
        # Cap the max step to +3 pods per individual cycle to smooth the CPU load
        growth_step = min(math.ceil(qsize / 2), 3)
        if growth_step < 1:
            growth_step = 1
            
        desired_replicas = current_replicas + growth_step

    # Bound replicas strictly within boundaries
    desired_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, desired_replicas))

    # --- EXECUTION ---
    if desired_replicas != current_replicas:
        try:
            apps_v1.patch_namespaced_deployment_scale(
                name=DEPLOYMENT_NAME,
                namespace=NAMESPACE,
                body={"spec": {"replicas": desired_replicas}}
            )
            
            # Set separate rules based on direction
            if desired_replicas > current_replicas:
                current_cooldown = SCALE_UP_COOLDOWN  # Fast subsequent adjustments (4s)
                direction_msg = "UP"
            else:
                current_cooldown = SCALE_DOWN_COOLDOWN  # Long protection window (300s)
                direction_msg = "DOWN"
                
            logger.info(f"Scaled {DEPLOYMENT_NAME} {direction_msg} from {current_replicas} to {desired_replicas} replicas. Queue size: {qsize}")
            last_scale_time = time.time()
        except client.exceptions.ApiException as e:
            logger.error(f"Error scaling deployment: {e}")
    else:
        logger.info(f"No scaling action taken. Queue size: {qsize}, Replicas: {current_replicas}")

async def main():
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
        
    apps_v1 = client.AppsV1Api()
    v1_api = client.CoreV1Api()
    
    logger.info("Custom Metrics Queue Autoscaler initialized successfully.")
    
    while True:
        try:
            qsize = await get_metric('dispatcher_queue_size{job="dispatcher-service", namespace="default"}')
            await scale_deployment(qsize, apps_v1, v1_api)
        except Exception as e:
            logger.error(f"Error in execution cycle: {e}")
            
        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())