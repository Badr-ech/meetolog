# Meetolog - Meeting to Backlog

Transform meeting audio recordings into structured Agile artifacts using AI-powered semantic extraction.

## Overview

Meetolog records or accepts audio uploads, transcribes them with OpenAI Whisper, extracts Agile artifacts via Google Gemini (or OpenAI), and produces a downloadable PDF report. Extracted artifacts are editable inline before generating the final PDF.

**Pipeline:**
1. Accept audio (MP3, WAV, M4A, OGG, WebM) via upload or in-browser recording
2. Transcribe speech to text (OpenAI Whisper, local model)
3. Extract Agile artifacts via LLM (Gemini or OpenAI):
   - User Stories (with acceptance criteria)
   - Tasks (with assignments and priorities)
   - Decisions (with rationale)
   - Blockers (with resolution plans)
   - Action Items
   - Execution Tasks (AI-inferred actionable work items with owner roles, priorities, and dependency tracking)
4. Assign confidence scores (LLM-provided or deterministic heuristic fallback)
5. Edit artifacts inline in the browser
6. Generate and download a PDF summary
7. Export artifacts as Jira-compatible JSON for bulk import

---

## System Requirements

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Python 3.12+** | Backend runtime | [python.org](https://www.python.org/downloads/) |
| **Node.js 18+** | Frontend runtime | [nodejs.org](https://nodejs.org/) |
| **Redis 7+** | Job state, task queue, caching | [redis.io](https://redis.io/docs/getting-started/) or Docker (see below) |
| **ffmpeg** | Audio processing for Whisper | `apt install ffmpeg` / `brew install ffmpeg` / [ffmpeg.org](https://ffmpeg.org/download.html) |

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 4 GB | 8+ GB (Whisper medium model) |
| GPU | Not required | CUDA GPU (10x faster transcription) |
| Storage | 5 GB | 20+ GB |

**Whisper model sizes:**

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~1 GB | Fastest | Lower (default) |
| `base` | ~1 GB | Fast | Good |
| `small` | ~2 GB | Moderate | Better |
| `medium` | ~5 GB | Slow | High |
| `large` | ~10 GB | Slowest | Highest |

---

## Quick Start

You need **three terminal windows**: Redis, backend (API + worker), and frontend.

### 1. Start Redis

If you have Docker:

```bash
docker run -d --name meetolog-redis -p 6379:6379 redis:7-alpine
```

Or install Redis natively and run `redis-server`.

### 2. Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```dotenv
TEST_MODE=true
REDIS_URL=redis://localhost:6379
```

This runs in test mode (no API keys needed, deterministic mock data). To use real transcription and extraction, see [Environment Variables](#environment-variables) below.

Start the API server and background worker in two separate terminals (both from `backend/` with the venv activated):

**Terminal A — API server:**
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000 --workers 1
```

**Terminal B — ARQ worker:**
```bash
cd backend
venv\Scripts\activate
arq app.worker.WorkerSettings
```

Both processes are required. The API server handles HTTP requests; the ARQ worker processes audio jobs from the Redis queue.

Backend: `http://localhost:8000`

### 3. Frontend

```bash
cd frontend

npm install
npm run dev
```

Frontend: `http://localhost:3000`

### 4. Use it

1. Open `http://localhost:3000`
2. Record audio or upload a file
3. Wait for processing (progress bar updates via polling)
4. Edit the extracted artifacts inline
5. Click "Save Changes" to persist edits
6. Download the PDF

---

## Environment Variables

All variables are set in `backend/.env`. Nothing is required when running in test mode.

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_MODE` | `false` | Mock all external services. No API keys or ffmpeg needed. |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_JOB_TTL_DAYS` | `7` | Days before job data expires in Redis |
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_API_KEY` | `""` | Google Gemini API key. Falls back to mock if empty. |
| `OPENAI_API_KEY` | `""` | OpenAI API key (when `LLM_PROVIDER=openai`) |
| `WHISPER_MODEL` | `tiny` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `MAX_UPLOAD_SIZE_MB` | `100` | Max upload size |
| `UPLOAD_DIR` | `uploads` | Temp upload directory |
| `OUTPUT_DIR` | `outputs` | PDF output directory |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed CORS origins (comma-separated) |
| `DEBUG` | `false` | Debug logging |

The frontend reads one optional variable from the shell environment (or `.env.local`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used for direct browser requests and Next.js rewrites |

### Test Mode

With `TEST_MODE=true`:
- MockTranscriber returns a hardcoded transcript instantly
- MockExtractor returns deterministic artifacts (including Execution Tasks) **without** confidence scores
- The heuristic scoring engine (`heuristics.py`) automatically backfills scores, exercising the full scoring pipeline
- No Whisper model loading (fast startup)
- No LLM API calls (no API key required)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info and Redis health |
| `GET` | `/health` | Detailed health check (Redis, ARQ queue) |
| `POST` | `/upload` | Upload audio file, returns job ID |
| `GET` | `/status/{job_id}` | Job progress, status, and artifacts when complete |
| `GET` | `/artifacts/{job_id}` | Extracted artifacts as JSON (completed jobs only) |
| `PUT` | `/artifacts/{job_id}` | Replace artifacts for a completed job (full payload) |
| `GET` | `/download/{job_id}` | Download generated PDF |
| `GET` | `/export/jira/{job_id}` | Download Jira-compatible bulk-import JSON |

### Examples

**Upload:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@meeting.mp3"
```

**Check status:**
```bash
curl http://localhost:8000/status/{job_id}
```

**Edit artifacts:**
```bash
curl -X PUT http://localhost:8000/artifacts/{job_id} \
  -H "Content-Type: application/json" \
  -d @edited_artifacts.json
```

The `PUT` endpoint expects the complete `MeetingArtifacts` schema. Partial updates are not supported — this guarantees Pydantic validates the entire payload on every save and prevents schema drift.

---

## Project Structure

```
meetolog/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI endpoints
│   │   ├── config.py               # Pydantic settings
│   │   ├── models.py               # Pydantic models (MeetingArtifacts, etc.)
│   │   ├── interfaces.py           # Abstract base classes
│   │   ├── dependencies.py         # Service factory / DI
│   │   ├── worker.py               # ARQ background job processing
│   │   ├── utils/
│   │   │   └── audio.py            # ffmpeg audio splitting & duration probe
│   │   ├── infrastructure/
│   │   │   ├── job_store.py        # RedisJobStore
│   │   │   └── redis.py            # Redis connection pool
│   │   └── services/
│   │       ├── transcription.py    # WhisperTranscriber (chunked, v1.1)
│   │       ├── jira_mapper.py      # Jira bulk-import JSON mapper
│   │       ├── llm_extraction.py   # GeminiExtractor
│   │       ├── llm_engine.py       # LLM provider abstraction
│   │       ├── heuristics.py       # Deterministic confidence scoring
│   │       ├── pdf_generator.py    # ReportLab PDF generation
│   │       ├── job_store.py        # Legacy LocalJobStore
│   │       └── mock_services.py    # Mock services for testing
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── start.sh                    # Container entrypoint
│   └── start-combined.sh           # Single-container API + worker
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                # Main page (upload, progress, editor)
│   │   ├── globals.css
│   │   ├── page.module.css
│   │   └── components/
│   │       ├── ArtifactEditor.tsx   # Inline-editable artifact cards
│   │       ├── ArtifactEditor.module.css
│   │       ├── JobProgress.tsx      # Stage-based progress bar (v1.1)
│   │       ├── JobProgress.module.css
│   │       ├── ui/
│   │       │   ├── ArtifactBadge.tsx          # Reusable Explicit/Inferred badge
│   │       │   └── ArtifactBadge.module.css
│   │       └── recorder/
│   │           ├── VoiceRecorder.tsx
│   │           └── VoiceRecorder.module.css
│   ├── lib/
│   │   ├── api.ts                  # Backend API client
│   │   └── audio.ts                # Client-side audio processing
│   ├── types/
│   │   └── index.ts                # Shared TS types (BadgeVariant, ConfidenceLevel, etc.)
│   ├── tests/
│   ├── package.json
│   ├── next.config.js
│   └── tsconfig.json
│
├── docker-compose.prod.yml
├── render.yaml
├── LICENSE
└── README.md
```

---

## Artifact Editing

After processing completes, all extracted artifacts are displayed as inline-editable form fields. Users can modify any field — titles, descriptions, priorities, assignees, acceptance criteria, etc.

**How it works:**

1. The frontend renders artifacts in editable input/textarea/select fields.
2. Client-side validation prevents saving with empty required fields (titles, descriptions).
3. Clicking "Save Changes" performs an optimistic UI update: local state updates immediately, then a `PUT /artifacts/{job_id}` request fires in the background.
4. If the request fails, the UI reverts to the previous state and shows an error toast.
5. The backend validates the full `MeetingArtifacts` schema via Pydantic. Invalid payloads get a 422 response.
6. Only completed jobs can be edited (400 if the job is still processing).
7. PDF generation always reads the latest artifacts from Redis, so edits made before downloading are reflected.

### Explicit vs Inferred Badges

Execution Tasks carry a `task_source` field (`"Explicit"` or `"Inferred"`) indicating whether the task was directly stated in the meeting or AI-derived. The frontend renders a colour-coded pill badge next to each Execution Task title:

| Badge | Meaning | Style |
|-------|---------|-------|
| **Explicit** | Task was directly stated in the meeting | Blue pill |
| **Inferred** | Task was logically derived by the AI | Purple pill |
| **Unknown** | Missing or unexpected `task_source` value | Gray pill (graceful fallback) |

The `<ArtifactBadge />` component (`frontend/app/components/ui/ArtifactBadge.tsx`) is fully reusable and accepts a `variant`, `label`, and optional `confidenceScore` prop for future expansion.

### Confidence Scores

Every extracted artifact carries an optional `confidence_score` (0.0 – 1.0) representing how explicitly the item was discussed in the meeting.

**Scoring pipeline:**

1. The LLM prompt requests a `confidence_score` for each artifact.
2. If the LLM provides a valid float in [0.0, 1.0], it is used as-is.
3. If the score is missing, `null`, or unparseable, a deterministic heuristic (`backend/app/services/heuristics.py`) computes a fallback:
   - **Base score:** 0.2
   - **+0.2** if an explicit owner/assignee is present
   - **+0.2** if a priority is explicitly set
   - **+0.2** if a strong action verb is in the title/description
   - **−0.2** if ambiguous phrases are detected ("maybe", "probably", etc.)
   - **+0.2** if all expected schema fields are populated
   - Final score is clamped to [0.0, 1.0]
4. Weights adjust per artifact type (e.g., Blockers don’t penalise for missing “assignee” — they check “owner” instead).

**Frontend display:**

| Score | Colour | Meaning |
|-------|--------|---------|
| $\ge$ 0.8 | Green | High confidence |
| $\ge$ 0.5 | Amber | Medium confidence |
| $<$ 0.5 | Red | Low confidence |
| N/A | Gray | Score not available |

The `<ConfidenceIndicator />` component renders a colour-coded pill badge next to the title of every artifact in the editor. The PDF also includes confidence percentages in all artifact sections.

**Backward compatibility:** `confidence_score` is `Optional[float]` with a `None` default, so older cached JSON payloads and existing API consumers are unaffected.

---

### Granular Progress States (v1.1)

Jobs now transition through **6 granular stages** instead of the original 4-state model (`pending` → `processing` → `completed` / `failed`). The frontend polls `GET /status/{job_id}` and renders a stage-based progress bar with dot indicators.

| State | Label | Progress |
|-------|-------|----------|
| `uploading` | Uploading Audio… | 10% |
| `transcribing` | Transcribing Audio… | 25% |
| `extracting` | Extracting Artifacts… | 50% |
| `generating_pdf` | Generating PDF… | 75% |
| `completed` | Processing Complete! | 100% |
| `failed` | Processing Failed | 0% |

**Key design decisions:**

- **Atomic state transitions:** Each stage change writes both `status` and `progress` in a single Redis `HSET` command (via `RedisJobStore.update_job_stage()`). This prevents race conditions where the frontend reads a new status but a stale progress value.
- **Backward compatibility:** Legacy `"pending"` and `"processing"` values are mapped to `uploading` and `transcribing` respectively via `parse_processing_status()`. Old cached jobs parse without errors.
- **Frontend mapping:** The `PROGRESS_MAPPING` constant in `frontend/types/index.ts` maps each state to a user-friendly label and fallback percentage. The `<JobProgress />` component renders the bar and stage-indicator dots.

### Jira Export (v1.1)

Completed jobs can be exported as a Jira-compatible bulk-import JSON file via `GET /export/jira/{job_id}`. The response forces a file download (`Content-Disposition: attachment`).

**Type mapping:**

| Meetolog Artifact | Jira Issue Type | Notes |
|-------------------|-----------------|-------|
| `UserStory` | Story | Includes acceptance criteria in description |
| `ExecutionTask` | Task | Labels include `explicit` / `inferred` source |
| `Blocker` | Bug | Priority forced to Highest |
| `Decision` | Task | Summary prefixed with `[Decision]` |
| `ActionItem` | Task | Summary prefixed with `[Action Item]` |
| `Task` | Task | Summary prefixed with `[Task]` |

**Priority mapping:** `Critical` → `Highest`, `High` → `High`, `Medium` → `Medium`, `Low` → `Low`. Unknown values fall back to `Medium`.

**Edge cases:**
- Missing titles default to `"(No title)"`.
- Summaries exceeding Jira's 255-character limit are truncated with `"…"`.
- All labels include `meetolog` for easy filtering after import.
- The exported JSON is validated against an internal Pydantic model before download, guaranteeing well-formed output.

### Chunked Transcription (v1.1)

Long recordings are split into fixed-duration chunks (default: 5 minutes) before being fed to Whisper. This prevents out-of-memory kills on CPU-only machines and allows the frontend to show per-chunk progress.

**How it works:**

1. `WhisperTranscriber.transcribe()` calls `split_audio_into_chunks()` (in `backend/app/utils/audio.py`) which uses the ffmpeg *segment muxer* to split the audio into 16 kHz mono WAV files on disk — nothing is held in RAM.
2. The Whisper model is loaded **once** (cached across invocations) and each chunk is transcribed **sequentially** — no parallelism.
3. After every chunk, `gc.collect()` is called to release any transient memory used by the model.
4. All temporary chunk files are cleaned up in a `try / finally` block, even if transcription fails mid-way.
5. Chunk transcripts are concatenated with a single-space separator to form the final full transcript.
6. The worker attaches a progress callback (`on_chunk_complete`) so the frontend shows real-time per-chunk progress (e.g. "Transcribing chunk 3/8…").

**Mock mode:** `MockTranscriber` inherits the default `transcribe_chunk()` from the `Transcriber` interface (delegates to `transcribe()`) and never invokes ffmpeg or chunking logic. `TEST_MODE=true` requires neither ffmpeg nor a Whisper model.

---

## Docker Compose (Production-Like)

Run the full Redis + API + Worker stack:

```bash
docker compose -f docker-compose.prod.yml up --build
```

Then run the frontend separately:

```bash
cd frontend
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

---

## Architecture

- **Job state and queue**: Redis (Hashes for job metadata, separate keys for transcript/artifact caching, 7-day TTL)
- **Granular progress states (v1.1)**: Jobs transition through 6 stages — `uploading` → `transcribing` → `extracting` → `generating_pdf` → `completed` / `failed`. Each stage transition is written atomically via a single `HSET` to prevent race conditions between status and progress fields. The frontend polls `GET /status/{job_id}` and maps each state to a user-friendly label and progress percentage via `PROGRESS_MAPPING`. Legacy `"pending"` and `"processing"` values are mapped transparently for backward compatibility.
- **Chunked transcription (v1.1)**: `WhisperTranscriber.transcribe()` splits audio into 5-minute chunks via ffmpeg, transcribes each sequentially, and merges results. `gc.collect()` after every chunk prevents OOM on CPU-only machines. A `ChunkProgressCallback` property lets the worker report per-chunk progress to Redis. `MockTranscriber` bypasses chunking entirely.
- **Background processing**: ARQ (async Redis queue). The API enqueues jobs; the worker executes the transcription → extraction → PDF pipeline.
- **Service abstraction**: Interface-based (`JobStore`, `Transcriber`, `LLMExtractor`). Implementations swap between production and mock via `TEST_MODE` and config.
- **Frontend**: Next.js 16 with App Router, TypeScript, CSS Modules. API calls proxy through Next.js rewrites (`/api/*` → backend) with direct fetch for uploads.
- **Single instance**: This MVP uses local file storage for uploads and PDFs. Do not run multiple uvicorn workers or container replicas.

---

## Deployment Constraints

- **Single instance only.** Local file storage for uploads and PDFs means multiple workers/replicas cause data loss.
- **Redis required.** Job state, the ARQ queue, transcript caching, and artifact caching all live in Redis.
- **ARQ worker required.** Without it, uploaded files sit in the queue and never process.

---

## License

MIT License. See [LICENSE](LICENSE).
