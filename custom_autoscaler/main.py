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

# Evaluation intervals
POLL_INTERVAL = 2 

# Asymmetric Cooldown Targets
SCALE_UP_COOLDOWN = 4      
SCALE_DOWN_COOLDOWN = 180  # Keep resources alive to absorb trailing waves

# Scaling Boundaries & SLO Thresholds
MIN_REPLICAS = 1
MAX_REPLICAS = 8
TARGET_LATENCY = 0.45      # < 0.5s assignment threshold

# State Tracking Module Globals
last_scale_time = 0.0
current_cooldown = 0.0

async def get_metric(query):
    """Fetch calculated float values from Prometheus."""
    try:
        async with httpx.AsyncClient() as client_session:
            response = await client_session.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
            response.raise_for_status()
            result = response.json()
            if result["status"] == "success" and result["data"]["result"]:
                return float(result["data"]["result"][0]["value"][1])
            return 0.0
    except Exception as e:
        logger.error(f"Error fetching metric: {e}")
        return 0.0

async def check_replicas_ready(v1_api):
    """Check if all ml-app-deployment pods are ready."""
    try:
        pods = v1_api.list_namespaced_pod(namespace=NAMESPACE, label_selector="app=ml-app")
        for pod in pods.items:
            if pod.metadata.deletion_timestamp is not None:
                continue
            for condition in pod.status.conditions or []:
                if condition.type == "Ready" and condition.status != "True":
                    return False
        return True
    except Exception as e:
        logger.error(f"Error checking pod readiness: {e}")
        return False

async def scale_deployment(current_latency, apps_v1, v1_api):
    """Scale deployment proportionally based on target response latency."""
    global last_scale_time, current_cooldown
    
    current_time = time.time()
    elapsed_time = current_time - last_scale_time
    
    try:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        current_replicas = deployment.spec.replicas
    except Exception as e:
        logger.error(f"Error getting replicas: {e}")
        return

    # --- 🚨 DYNAMIC COOLDOWN OVERRIDE ENGINE ---
    if elapsed_time < current_cooldown:
        if current_latency > TARGET_LATENCY and current_replicas < MAX_REPLICAS:
            logger.warning(f"🚨 LATENCY SLO VIOLATION ({current_latency:.3f}s)! Overriding cooldown lock.")
        else:
            logger.info(f"Skipping scaling due to active cooldown window ({int(current_cooldown - elapsed_time)}s remaining)")
            return

    # --- LATENCY-DRIVEN PROPORTIONAL CALCULATION ---
    if current_latency <= 0.05 or current_latency < TARGET_LATENCY:
        if current_latency == 0.0 or current_latency < 0.1:
            if not await check_replicas_ready(v1_api):
                logger.info("Skipping scale-down as current replicas are stabilizing")
                return
            desired_replicas = max(MIN_REPLICAS, current_replicas - 2)
        else:
            desired_replicas = current_replicas
    else:
        # Proportional expansion based on how far latency has slipped
        latency_ratio = current_latency / TARGET_LATENCY
        growth_step = min(math.ceil(current_replicas * (latency_ratio - 1)), 4)
        
        if growth_step < 2:
            growth_step = 2  # Dynamically deploy 2 pods minimum to handle waves quickly
            
        desired_replicas = current_replicas + growth_step

    desired_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, desired_replicas))

    # --- EXECUTION ---
    if desired_replicas != current_replicas:
        try:
            apps_v1.patch_namespaced_deployment_scale(
                name=DEPLOYMENT_NAME, namespace=NAMESPACE,
                body={"spec": {"replicas": desired_replicas}}
            )
            
            if desired_replicas > current_replicas:
                current_cooldown = SCALE_UP_COOLDOWN
                direction_msg = "UP"
            else:
                current_cooldown = SCALE_DOWN_COOLDOWN
                direction_msg = "DOWN"
                
            logger.info(f"Scaled {DEPLOYMENT_NAME} {direction_msg} from {current_replicas} to {desired_replicas} replicas. Latency: {current_latency:.3f}s")
            last_scale_time = time.time()
        except Exception as e:
            logger.error(f"Error scaling deployment: {e}")
    else:
        logger.info(f"No scaling action taken. Latency: {current_latency:.3f}s, Replicas: {current_replicas}")

async def main():
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
        
    apps_v1 = client.AppsV1Api()
    v1_api = client.CoreV1Api()
    
    # Instant label-free metrics query
    latency_promql = (
        'sum(dispatcher_response_time_seconds_sum) '
        '/ sum(dispatcher_response_time_seconds_count)'
    )
    
    logger.info("Custom Latency-Driven Autoscaler initialized successfully.")
    
    while True:
        try:
            current_latency = await get_metric(latency_promql)
            if math.isnan(current_latency):
                current_latency = 0.0
            await scale_deployment(current_latency, apps_v1, v1_api)
        except Exception as e:
            logger.error(f"Error in execution cycle: {e}")
            
        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())