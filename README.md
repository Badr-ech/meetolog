# Meetolog — Meeting to Backlog

Transform meeting audio recordings into structured Agile artifacts using AI-powered semantic extraction.

---

## Overview

Meetolog accepts audio uploads or in-browser recordings, transcribes them with OpenAI Whisper, extracts Agile artifacts via Google Gemini (or OpenAI), and produces a downloadable PDF report. Extracted artifacts are editable inline before generating the final PDF.

**Pipeline:**

1. Accept audio (MP3, WAV, M4A, OGG, WebM) via upload or in-browser recording
2. Upload audio directly to AWS S3 via a presigned POST URL (client-side, stateless API)
3. Transcribe speech to text (OpenAI Whisper, local model)
4. Extract Agile artifacts via LLM (Gemini or OpenAI):
   - User Stories (with acceptance criteria)
   - Tasks (with assignments and priorities)
   - Decisions (with rationale)
   - Blockers (with resolution plans)
   - Action Items
   - Execution Tasks (AI-inferred actionable work items with owner roles, priorities, and dependency tracking)
5. Assign confidence scores (LLM-provided or deterministic heuristic fallback)
6. Edit artifacts inline in the browser
7. Generate and download a PDF summary
8. Export artifacts as Jira-compatible JSON for bulk import

---

## Production Architecture (AWS)

The production backend runs on **AWS ECS Fargate**. The API and workers are deployed as separate ECS services from the same Docker image (selected via `SERVICE_TYPE` env var). The frontend is deployed to Vercel.

```
                        ┌──────────────┐
                        │    Vercel     │
                        │  (Frontend)   │
                        └──────┬───────┘
                               │ HTTPS
                 ┌─────────────▼─────────────┐
                 │  Application Load Balancer │
                 │       (ALB, port 443)      │
                 └─────────────┬─────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
  ┌───────▼────────┐  ┌───────▼────────┐   ┌───────▼────────┐
  │ ECS Task (API)  │  │ ECS Task (API)  │   │   S3 Bucket    │
  │ 0.25 vCPU       │  │ 0.25 vCPU       │   │  (presigned    │
  │ 512 MB RAM      │  │ 512 MB RAM      │   │   uploads)     │
  └───────┬────────┘  └───────┬────────┘   └────────────────┘
          │                    │
          └────────┬───────────┘
                   │
          ┌────────▼────────┐
          │  RDS PostgreSQL  │
          │  (db.t3.micro)   │
          └────────┬────────┘
                   │
  ┌────────────────┼────────────────┐
  │                │                │
  ┌────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
  │ ECS Worker    │ │ ECS Worker  │ │ ECS Worker  │
  │ 1 vCPU        │ │ 1 vCPU      │ │ 1 vCPU      │
  │ 2 GB RAM      │ │ 2 GB RAM    │ │ 2 GB RAM    │
  │ Fargate Spot  │ │ Fargate Spot│ │ Fargate Spot│
  └───────────────┘ └─────────────┘ └─────────────┘
```

| Component | AWS Service | Details |
|-----------|-------------|---------|
| **API** | ECS Fargate | Stateless FastAPI. Streams uploads to S3, reads/writes PostgreSQL. |
| **Worker** | ECS Fargate Spot | Runs Whisper `tiny` + ffmpeg chunking. Spot interruptions handled by PostgreSQL retry logic. |
| **Database** | RDS PostgreSQL | Job queue (`SELECT … FOR UPDATE SKIP LOCKED`), metadata, artifact storage. |
| **Object Storage** | S3 | Audio uploads, PDFs, artifact JSON. Presigned POST for direct browser uploads. |
| **Secrets** | SSM Parameter Store | `DATABASE_URL`, `AWS_S3_BUCKET`, `GEMINI_API_KEY`, `CORS_ORIGINS` injected into ECS tasks. |
| **Container Registry** | ECR | Single image for both API and worker roles. |
| **Logs** | CloudWatch Logs | Structured JSON logs from `structlog`. 30-day retention. |
| **Load Balancer** | ALB | TLS termination, health checks (`/health`), HTTP→HTTPS redirect. |
| **Frontend** | Vercel | Next.js deployed separately. `NEXT_PUBLIC_API_URL` points to the ALB. |

Workers use **Fargate Spot** for up to 70% cost savings. ECS tasks run in public subnets with auto-assigned public IPs (no NAT Gateway). The IAM task role grants S3 access — no static AWS credentials needed in production.

---

## Key Features

### Artifact Editing

After processing, all extracted artifacts are rendered as inline-editable form fields. Client-side validation prevents saving with empty required fields. Edits are saved via `PUT /artifacts/{job_id}` with full `MeetingArtifacts` Pydantic validation. The PDF is always generated from the latest artifacts.

### Confidence Scores

Every artifact carries an optional `confidence_score` (0.0–1.0). If the LLM omits a score, a deterministic heuristic computes a fallback based on field completeness, action verb presence, and ambiguity detection. The frontend renders colour-coded indicators (green ≥ 0.8, amber ≥ 0.5, red < 0.5).

### Explicit vs Inferred Badges

Execution Tasks carry a `task_source` field (`"Explicit"` or `"Inferred"`) indicating whether the task was directly stated in the meeting or AI-derived. The frontend renders a colour-coded pill badge next to each title.

### Granular Progress States

Jobs transition through 6 stages: `uploading` → `transcribing` → `extracting` → `generating_pdf` → `completed` / `failed`. Each transition writes status and progress atomically in a single SQL `UPDATE`.

### Jira Export

Completed jobs export as Jira-compatible bulk-import JSON via `GET /export/jira/{job_id}`. Artifact types map to Jira issue types (Story, Task, Bug). Priorities, labels, and summaries are translated automatically.

### Chunked Transcription

Multi-hour recordings are split into 5-minute WAV chunks via ffmpeg. The Whisper model is loaded once per process. Chunks are transcribed sequentially with `gc.collect()` after each to reclaim memory. Progress callbacks write per-chunk updates to PostgreSQL for real-time frontend visibility.

---

## Observability & Reliability

- **Structured Logging** — All components emit JSON logs via `structlog` with UTC timestamps, log levels, and contextual IDs (`job_id`, `worker_id`). Logs are captured by AWS CloudWatch.
- **Transient Failure Retries** — S3 calls use `tenacity` exponential-backoff retries (up to 4 attempts). LLM calls retry up to 3 times. Only transient errors trigger retries.
- **Timeout Protection** — LLM extraction uses `asyncio.timeout(60)`. Whisper runs in a thread pool with inherited OS-level timeout.
- **Stale-Lock Recovery** — The PostgreSQL queue reclaims jobs stuck in `processing` when `locked_at` exceeds 2 hours. Crashed workers' jobs are automatically retried.
- **Uncrashable Worker Loop** — Each job is wrapped in `try … except Exception`. Failures mark the job as `failed` and the worker continues polling.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info (version, status, LLM provider) |
| `GET` | `/health` | Health check (database, queue depth) |
| `POST` | `/upload/presign` | Presigned S3 POST URL for direct browser upload |
| `POST` | `/jobs/enqueue` | Enqueue a transcription job for a file already in S3 |
| `POST` | `/upload` | Legacy single-request upload (suitable for test mode) |
| `GET` | `/status/{job_id}` | Job progress, status, and artifacts when complete |
| `GET` | `/artifacts/{job_id}` | Extracted artifacts JSON |
| `PUT` | `/artifacts/{job_id}` | Replace artifacts (full payload, Pydantic-validated) |
| `GET` | `/download/{job_id}` | Download generated PDF (presigned S3 URL) |
| `GET` | `/export/jira/{job_id}` | Jira-compatible bulk-import JSON |

---

## Project Structure

```
meetolog/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI endpoints
│   │   ├── config.py                  # Pydantic settings (AWS, DB, LLM, Whisper)
│   │   ├── worker.py                  # Async Postgres-polling background worker
│   │   ├── interfaces.py              # Abstract base classes (JobStore, Transcriber, LLMExtractor)
│   │   ├── models/
│   │   │   ├── schemas.py             # Pydantic models (MeetingArtifacts, JobResponse, etc.)
│   │   │   ├── metadata.py            # FileMetadata SQLAlchemy ORM
│   │   │   └── db_models.py           # JobRecord SQLAlchemy ORM
│   │   ├── core/
│   │   │   └── logger.py              # structlog configuration (JSON / console)
│   │   ├── infrastructure/
│   │   │   ├── db.py                  # Async SQLAlchemy engine + session
│   │   │   ├── postgres_job_store.py  # PostgresJobStore (read/update)
│   │   │   └── postgres_queue.py      # PostgresJobQueue (SKIP LOCKED queue)
│   │   ├── services/
│   │   │   ├── storage.py             # S3StorageService (upload/download/presign)
│   │   │   ├── transcription.py       # WhisperTranscriber (chunked)
│   │   │   ├── llm_extraction.py      # GeminiExtractor
│   │   │   ├── llm_engine.py          # LLM provider abstraction (Gemini/OpenAI)
│   │   │   ├── heuristics.py          # Deterministic confidence scoring
│   │   │   ├── pdf_generator.py       # ReportLab PDF generation
│   │   │   ├── jira_mapper.py         # Jira bulk-import JSON mapper
│   │   │   └── mock_services.py       # Mock services for testing
│   │   └── utils/
│   │       └── audio.py               # ffmpeg audio splitting & duration probe
│   ├── alembic/                       # Database migrations
│   ├── tests/
│   ├── Dockerfile                     # Multi-stage production image
│   ├── start.sh                       # Container entrypoint
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                   # Main page (upload, progress, editor)
│   │   ├── layout.tsx
│   │   └── components/
│   │       ├── ArtifactEditor.tsx      # Inline-editable artifact cards
│   │       ├── JobProgress.tsx         # Stage-based progress bar
│   │       ├── ui/ArtifactBadge.tsx    # Reusable Explicit/Inferred badge
│   │       └── recorder/VoiceRecorder.tsx
│   ├── lib/
│   │   ├── api.ts                     # Backend API client
│   │   └── audio.ts                   # Client-side audio processing
│   ├── types/index.ts
│   ├── package.json
│   └── next.config.js
│
├── aws/
│   ├── ecs-task-def-api.json          # ECS Fargate task definition (API)
│   ├── ecs-task-def-worker.json       # ECS Fargate task definition (Worker)
│   ├── ecs-trust-policy.json          # ECS trust policy for IAM roles
│   ├── iam-execution-role-policy.json # Execution role (ECR, SSM, CloudWatch)
│   └── iam-task-role-policy.json      # Task role (S3, CloudWatch)
│
├── docker-compose.yml                 # Local dev stack (PostgreSQL, MinIO, Redis)
└── LICENSE
```

---

## Architecture Decisions

- **Stateless API & Workers** — Audio inputs are fetched from S3. All outputs (PDF, JSON) are uploaded back to S3. Workers hold zero persistent local state. Scale freely via `docker compose up --scale worker=N` or `aws ecs update-service --desired-count N`.
- **Direct-to-S3 Uploads** — The browser requests a presigned POST from `POST /upload/presign`, uploads directly to S3 (bypassing the API), then calls `POST /jobs/enqueue`. The API never touches raw audio bytes.
- **PostgreSQL as Job Queue** — `SELECT … FOR UPDATE SKIP LOCKED` enables safe concurrent job claiming. Failed jobs auto-retry with exponential backoff. Stale locks from crashed workers are reclaimed automatically.
- **SSM Parameter Store** — Production secrets are injected into ECS tasks by the execution role. No static credentials in environment variables or task definitions.
- **Structured JSON Logging** — `structlog` emits machine-readable JSON to stdout, captured by CloudWatch Logs via the `awslogs` driver.
- **Interface-Based Services** — `JobStore`, `Transcriber`, `LLMExtractor` are abstract base classes. Implementations swap between production and mock via `TEST_MODE`.
- **Pydantic Settings** — All configuration loaded from environment variables with validation via `pydantic-settings`. The `.env` file is used only for local development.

---

## Environment Variables

All variables are loaded via Pydantic Settings from environment variables (or `.env` for local development). In production, secrets are injected by AWS SSM Parameter Store through the ECS task definition.

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_MODE` | `false` | Mock all external services |
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_API_KEY` | `""` | Google Gemini API key |
| `OPENAI_API_KEY` | `""` | OpenAI API key |
| `WHISPER_MODEL` | `tiny` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `DATABASE_URL` | `""` | PostgreSQL async DSN: `postgresql+asyncpg://user:pass@host/db` |
| `AWS_ACCESS_KEY_ID` | `""` | AWS IAM access key (local dev only; production uses IAM task role) |
| `AWS_SECRET_ACCESS_KEY` | `""` | AWS IAM secret key (local dev only) |
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_S3_BUCKET` | `""` | S3 bucket name |
| `AWS_ENDPOINT_URL` | `None` | Custom S3 endpoint (e.g. `http://minio:9000` for local MinIO) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `MAX_UPLOAD_SIZE_MB` | `100` | Max upload size |

The frontend reads:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for browser requests and Next.js rewrites |

---

## Deployment Constraints

- **PostgreSQL required.** Job lifecycle, persistent queue, file metadata. Production uses AWS RDS. Run `alembic upgrade head` to apply migrations.
- **AWS S3 required.** Audio, PDFs, and artifact JSON. The ECS task role provides access — no static IAM credentials in production.
- **Background worker required.** Run as a separate ECS service (`SERVICE_TYPE=worker`) or locally via `python -m app.worker`.
- **Horizontal scaling.** Workers are stateless. Scale via `aws ecs update-service --desired-count N`. The `SKIP LOCKED` mechanism prevents duplicate processing.

---

## License

MIT License. See [LICENSE](LICENSE).
