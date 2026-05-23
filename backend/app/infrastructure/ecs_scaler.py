"""
ECS worker auto-scaling and task-launching utilities.

Handles two distinct operations:

1. **Service scaling** — bump the ``meetolog-worker`` service desired count
   between 0 and 1 so a persistent splitter task wakes when work arrives
   and sleeps when the queue drains.

2. **One-off RunTask calls** — launch ephemeral ``chunk_worker`` and
   ``assembler`` Fargate tasks for each job.  These tasks exit naturally
   when their work is done; they are not managed by a service.

Network configuration for RunTask (subnets + security groups) is fetched
from the running worker service itself, so no extra SSM parameters or
hardcoded values are needed.

All AWS calls are fire-and-forget from the caller's perspective: any
exception is logged as a warning and swallowed so that a transient ECS
hiccup never crashes the worker or blocks the API.
"""

import uuid

import aioboto3
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Service-level scaling (splitter wake / sleep)
# ---------------------------------------------------------------------------

async def scale_worker_up(cluster: str, service: str, region: str) -> None:
    """Set the worker ECS service desired count to 1 if currently at 0."""
    try:
        session = aioboto3.Session()
        async with session.client("ecs", region_name=region) as ecs:
            resp = await ecs.describe_services(cluster=cluster, services=[service])
            services = resp.get("services", [])
            if not services:
                logger.warning("ecs_service_not_found", cluster=cluster, service=service)
                return

            if services[0]["desiredCount"] == 0:
                await ecs.update_service(
                    cluster=cluster, service=service, desiredCount=1
                )
                logger.info("worker_scaled_up", cluster=cluster, service=service)
            else:
                logger.debug(
                    "worker_already_running",
                    cluster=cluster,
                    service=service,
                    desired_count=services[0]["desiredCount"],
                )
    except Exception as exc:
        logger.warning("ecs_scale_up_failed", error=str(exc))


async def scale_worker_down(cluster: str, service: str, region: str) -> None:
    """Set the worker ECS service desired count to 0."""
    try:
        session = aioboto3.Session()
        async with session.client("ecs", region_name=region) as ecs:
            await ecs.update_service(
                cluster=cluster, service=service, desiredCount=0
            )
            logger.info("worker_scaled_down", cluster=cluster, service=service)
    except Exception as exc:
        logger.warning("ecs_scale_down_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Network config (shared by both RunTask calls)
# ---------------------------------------------------------------------------

async def _get_service_network_config(
    cluster: str, service: str, region: str
) -> dict | None:
    """Return the awsvpcConfiguration from the running worker service.

    Reusing the service's own subnet + security-group list means we never
    need to store those values separately in config or SSM.
    """
    try:
        session = aioboto3.Session()
        async with session.client("ecs", region_name=region) as ecs:
            resp = await ecs.describe_services(cluster=cluster, services=[service])
            services = resp.get("services", [])
            if not services:
                logger.warning(
                    "ecs_service_not_found_for_network_config",
                    cluster=cluster,
                    service=service,
                )
                return None
            net = (
                services[0]
                .get("networkConfiguration", {})
                .get("awsvpcConfiguration")
            )
            if not net:
                logger.warning(
                    "ecs_service_missing_network_config",
                    cluster=cluster,
                    service=service,
                )
            return net
    except Exception as exc:
        logger.warning("ecs_get_network_config_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# One-off RunTask helpers
# ---------------------------------------------------------------------------

async def run_chunk_workers(
    *,
    cluster: str,
    service: str,
    task_definition: str,
    job_id: uuid.UUID,
    num_workers: int,
    detected_language: str | None,
    region: str,
) -> None:
    """Launch *num_workers* ephemeral chunk-worker Fargate tasks for *job_id*.

    Each task runs the same image as the worker service but with
    ``SERVICE_TYPE=chunk_worker`` and carries the job ID as an env var so
    it knows which ``job_chunks`` rows to claim.

    Chunk workers use 0.5 vCPU / 2 GB — half the splitter's footprint —
    so up to 12 can run simultaneously within the default 6 vCPU Fargate
    limit (minus the 0.25 vCPU API and 1 vCPU splitter).

    Failures are logged as warnings; a missing network config causes all
    RunTask calls to be skipped (the chunks stay ``pending`` and will be
    picked up when the worker service restarts — rare edge case).
    """
    net_config = await _get_service_network_config(cluster, service, region)
    if not net_config:
        logger.error(
            "chunk_workers_not_launched_missing_network_config",
            job_id=str(job_id),
        )
        return

    env_overrides = [
        {"name": "SERVICE_TYPE", "value": "chunk_worker"},
        {"name": "JOB_ID", "value": str(job_id)},
    ]
    if detected_language:
        env_overrides.append({"name": "DETECTED_LANGUAGE", "value": detected_language})

    try:
        session = aioboto3.Session()
        async with session.client("ecs", region_name=region) as ecs:
            for _ in range(num_workers):
                await ecs.run_task(
                    cluster=cluster,
                    taskDefinition=task_definition,
                    launchType="FARGATE",
                    networkConfiguration={"awsvpcConfiguration": net_config},
                    overrides={
                        "cpu": "512",
                        "memory": "2048",
                        "containerOverrides": [
                            {
                                "name": "meetolog-worker",
                                "environment": env_overrides,
                            }
                        ],
                    },
                )
        logger.info(
            "chunk_workers_launched",
            job_id=str(job_id),
            num_workers=num_workers,
        )
    except Exception as exc:
        logger.warning(
            "chunk_workers_launch_failed",
            job_id=str(job_id),
            error=str(exc),
        )


async def run_assembler(
    *,
    cluster: str,
    service: str,
    task_definition: str,
    job_id: uuid.UUID,
    region: str,
) -> None:
    """Launch a single ephemeral assembler Fargate task for *job_id*.

    The assembler uses the full 1 vCPU / 4 GB profile because it runs
    LLM extraction and PDF generation, which need the same memory budget
    as the old monolithic worker.
    """
    net_config = await _get_service_network_config(cluster, service, region)
    if not net_config:
        logger.error(
            "assembler_not_launched_missing_network_config",
            job_id=str(job_id),
        )
        return

    try:
        session = aioboto3.Session()
        async with session.client("ecs", region_name=region) as ecs:
            await ecs.run_task(
                cluster=cluster,
                taskDefinition=task_definition,
                launchType="FARGATE",
                networkConfiguration={"awsvpcConfiguration": net_config},
                overrides={
                    "cpu": "1024",
                    "memory": "4096",
                    "containerOverrides": [
                        {
                            "name": "meetolog-worker",
                            "environment": [
                                {"name": "SERVICE_TYPE", "value": "assembler"},
                                {"name": "JOB_ID", "value": str(job_id)},
                            ],
                        }
                    ],
                },
            )
        logger.info("assembler_launched", job_id=str(job_id))
    except Exception as exc:
        logger.warning(
            "assembler_launch_failed",
            job_id=str(job_id),
            error=str(exc),
        )
