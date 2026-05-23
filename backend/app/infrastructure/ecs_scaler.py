"""
ECS worker auto-scaling utilities.

Scales the meetolog-worker ECS service up (desired count → 1) when a new
job is enqueued by the API, and down (desired count → 0) when the worker
has been idle long enough to consider the queue drained.

All AWS calls are fire-and-forget from the caller's perspective: any
exception is logged as a warning and swallowed so that a transient ECS
API hiccup never breaks job submission or the worker loop itself.
"""

import aioboto3
import structlog

logger = structlog.get_logger(__name__)


async def scale_worker_up(cluster: str, service: str, region: str) -> None:
    """Set the worker ECS service desired count to 1 if currently at 0.

    Calling this when the service is already running (desired count ≥ 1)
    is a no-op — we skip the UpdateService call to avoid unnecessary API
    churn and to leave any existing desired count untouched.
    """
    try:
        session = aioboto3.Session()
        async with session.client("ecs", region_name=region) as ecs:
            resp = await ecs.describe_services(cluster=cluster, services=[service])
            services = resp.get("services", [])
            if not services:
                logger.warning("ecs_service_not_found", cluster=cluster, service=service)
                return

            current_desired = services[0]["desiredCount"]
            if current_desired == 0:
                await ecs.update_service(
                    cluster=cluster, service=service, desiredCount=1
                )
                logger.info("worker_scaled_up", cluster=cluster, service=service)
            else:
                logger.debug(
                    "worker_already_running",
                    cluster=cluster,
                    service=service,
                    desired_count=current_desired,
                )
    except Exception as exc:
        # Never let ECS failures break the API response.
        logger.warning("ecs_scale_up_failed", error=str(exc))


async def scale_worker_down(cluster: str, service: str, region: str) -> None:
    """Set the worker ECS service desired count to 0.

    Called by the worker itself once it determines the queue has been
    empty long enough.  ECS will send SIGTERM to the running container
    after the service is updated; the worker's existing signal handler
    ensures a clean shutdown regardless.
    """
    try:
        session = aioboto3.Session()
        async with session.client("ecs", region_name=region) as ecs:
            await ecs.update_service(
                cluster=cluster, service=service, desiredCount=0
            )
            logger.info("worker_scaled_down", cluster=cluster, service=service)
    except Exception as exc:
        logger.warning("ecs_scale_down_failed", error=str(exc))
