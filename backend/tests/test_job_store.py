"""Tests for RedisJobStore using fakeredis async backend."""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
import fakeredis.aioredis

from app.infrastructure.job_store import RedisJobStore
from app.models import (
    JobResponse,
    MeetingArtifacts,
    ProcessingStatus,
    UserStory,
    Priority,
)


SAMPLE_JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest_asyncio.fixture
async def redis():
    """Provide an isolated fakeredis async client with decode_responses=True."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest_asyncio.fixture
async def store(redis):
    """Provide a RedisJobStore backed by fakeredis."""
    return RedisJobStore(redis=redis)


@pytest.fixture
def pending_job() -> JobResponse:
    return JobResponse(
        job_id=SAMPLE_JOB_ID,
        status=ProcessingStatus.UPLOADING,
        message="Queued",
        progress=0,
    )


# save / load

@pytest.mark.asyncio
async def test_save_and_load_roundtrip(store, pending_job):
    await store.save(
        SAMPLE_JOB_ID,
        pending_job,
        file_path="/uploads/audio.wav",
        file_name="audio.wav",
        file_size=4096,
    )

    loaded = await store.load(SAMPLE_JOB_ID)
    assert loaded is not None
    assert loaded.job_id == SAMPLE_JOB_ID
    assert loaded.status == ProcessingStatus.UPLOADING
    assert loaded.progress == 0


@pytest.mark.asyncio
async def test_load_returns_none_for_missing_job(store):
    result = await store.load(uuid4())
    assert result is None


# update

@pytest.mark.asyncio
async def test_update_changes_status_and_progress(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)

    updated = await store.update(
        SAMPLE_JOB_ID,
        status=ProcessingStatus.TRANSCRIBING,
        progress=30,
        message="Transcribing...",
    )

    assert updated is not None
    assert updated.status == ProcessingStatus.TRANSCRIBING
    assert updated.progress == 30
    assert updated.message == "Transcribing..."


@pytest.mark.asyncio
async def test_update_returns_none_for_missing_job(store):
    result = await store.update(uuid4(), progress=50)
    assert result is None


# exists / delete

@pytest.mark.asyncio
async def test_exists_returns_true_for_saved_job(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)
    assert await store.exists(SAMPLE_JOB_ID) is True


@pytest.mark.asyncio
async def test_exists_returns_false_for_missing_job(store):
    assert await store.exists(uuid4()) is False


@pytest.mark.asyncio
async def test_delete_removes_job_and_cached_data(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)
    await store.cache_transcript(SAMPLE_JOB_ID, "Transcript text")

    result = await store.delete(SAMPLE_JOB_ID)
    assert result is True
    assert await store.exists(SAMPLE_JOB_ID) is False
    assert await store.get_cached_transcript(SAMPLE_JOB_ID) is None


@pytest.mark.asyncio
async def test_delete_returns_false_for_missing_job(store):
    result = await store.delete(uuid4())
    assert result is False


# Transcript caching

@pytest.mark.asyncio
async def test_cache_and_retrieve_transcript(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)
    await store.cache_transcript(SAMPLE_JOB_ID, "Full transcript content")

    cached = await store.get_cached_transcript(SAMPLE_JOB_ID)
    assert cached == "Full transcript content"


@pytest.mark.asyncio
async def test_get_cached_transcript_returns_none_when_absent(store):
    result = await store.get_cached_transcript(uuid4())
    assert result is None


# Artifact caching

@pytest.mark.asyncio
async def test_cache_and_retrieve_artifacts(store, pending_job, sample_artifacts):
    await store.save(SAMPLE_JOB_ID, pending_job)
    await store.cache_artifacts(SAMPLE_JOB_ID, sample_artifacts)

    cached = await store.get_cached_artifacts(SAMPLE_JOB_ID)
    assert cached is not None
    assert isinstance(cached, MeetingArtifacts)
    assert cached.meeting_title == sample_artifacts.meeting_title
    assert len(cached.user_stories) == len(sample_artifacts.user_stories)


@pytest.mark.asyncio
async def test_get_cached_artifacts_returns_none_when_absent(store):
    result = await store.get_cached_artifacts(uuid4())
    assert result is None


# Chunk transcript storage

@pytest.mark.asyncio
async def test_chunk_transcript_save_and_assemble(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)

    await store.save_chunk_transcript(SAMPLE_JOB_ID, 0, "Hello", 3)
    await store.save_chunk_transcript(SAMPLE_JOB_ID, 1, "World", 3)
    await store.save_chunk_transcript(SAMPLE_JOB_ID, 2, "End", 3)

    assembled = await store.assemble_transcript_from_chunks(SAMPLE_JOB_ID)
    assert assembled == "Hello World End"


@pytest.mark.asyncio
async def test_get_completed_chunk_indices(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)

    await store.save_chunk_transcript(SAMPLE_JOB_ID, 0, "A", 2)
    await store.save_chunk_transcript(SAMPLE_JOB_ID, 1, "B", 2)

    completed, total = await store.get_completed_chunk_indices(SAMPLE_JOB_ID)
    assert completed == {0, 1}
    assert total == 2


@pytest.mark.asyncio
async def test_delete_chunk_data_cleans_up(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)

    await store.save_chunk_transcript(SAMPLE_JOB_ID, 0, "Text", 1)
    await store.delete_chunk_data(SAMPLE_JOB_ID)

    completed, total = await store.get_completed_chunk_indices(SAMPLE_JOB_ID)
    assert completed == set()
    assert total == 0


# Audio storage

@pytest.mark.asyncio
async def test_store_and_retrieve_audio(store, pending_job, redis):
    await store.save(SAMPLE_JOB_ID, pending_job)
    audio_data = b"\xff\xd8" * 100

    stored = await store.store_audio(SAMPLE_JOB_ID, audio_data)
    assert stored is True

    # Verify key exists (raw byte retrieval requires decode_responses=False,
    # which conflicts with the hash-based methods that need decoded strings).
    assert await redis.exists(f"job:{SAMPLE_JOB_ID}:audio")


@pytest.mark.asyncio
async def test_store_audio_rejects_oversized_data(store, pending_job):
    await store.save(SAMPLE_JOB_ID, pending_job)
    oversized = b"\x00" * (16 * 1024 * 1024)

    stored = await store.store_audio(SAMPLE_JOB_ID, oversized)
    assert stored is False


# mark_job_failed

@pytest.mark.asyncio
async def test_mark_job_failed_sets_status(store, pending_job, redis):
    await store.save(SAMPLE_JOB_ID, pending_job)
    await store.mark_job_failed(SAMPLE_JOB_ID, "Out of memory")

    status = await redis.hget(f"job:{SAMPLE_JOB_ID}", "status")
    assert status == "failed"

    error = await redis.hget(f"job:{SAMPLE_JOB_ID}", "error")
    assert error == "Out of memory"
