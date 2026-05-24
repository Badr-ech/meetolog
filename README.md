# Meetolog — Meeting to Backlog

Transform meeting audio recordings into structured Agile artifacts using AI-powered semantic extraction.

---

## Overview

Meetolog accepts audio uploads or in-browser recordings, identifies speakers via diarization, transcribes them with OpenAI Whisper, and runs a **unified intelligence pipeline** (hierarchical summarization → context compression → RAG retrieval → structured extraction) to produce Agile artifacts via Google Gemini or OpenAI. Extracted artifacts are editable inline before generating a downloadable PDF report.

**Pipeline:**

1. Accept audio (MP3, WAV, M4A, OGG, WebM) via upload or in-browser recording
2. Upload audio directly to AWS S3 via a presigned POST URL (client-side, stateless API)
3. **Identify speakers** via pyannote speaker diarization (global timeline, optional — requires `HF_TOKEN`)
4. Transcribe speech to text (OpenAI Whisper, local model)
5. **Align** chunked Whisper segments with the global speaker timeline to produce a speaker-labelled transcript
6. Extract Agile artifacts via LLM (Gemini or OpenAI):
   - User Stories (with acceptance criteria)
   - Tasks (with assignments, priorities, and contextual reasoning)
   - Decisions (with rationale, decision summary, and rejected alternatives)
   - Blockers (with resolution plans)
   - Action Items (with title, context, and priority)
   - Ideas & Suggestions (with proposer, potential impact, and confidence)
   - Execution Tasks (AI-inferred actionable work items with owner roles, priorities, and dependency tracking)
7. Assign confidence scores (LLM-provided or deterministic heuristic fallback)
8. Edit artifacts inline in the browser
9. Generate and download a PDF summary
10. Export artifacts as Jira-compatible JSON for bulk import

---

## Production Architecture (AWS)

The backend runs on **AWS ECS Fargate**. All three worker roles share a single Docker image; `SERVICE_TYPE` selects the role at runtime. The frontend is deployed to Vercel.

### Worker pipeline (parallel transcription)

When a job is enqueued the worker service acts as a **splitter**: it downloads the audio, cuts it into 5-minute chunks with ffmpeg, uploads each chunk to S3, and fires up to `MAX_PARALLEL_CHUNKS` (default 6) ephemeral **chunk-worker** tasks via `ecs:RunTask`. Each chunk worker claims chunks from the `job_chunks` queue with `SELECT … FOR UPDATE SKIP LOCKED`, transcribes with Whisper, and exits when no chunks remain. The last worker transitions the job to `assembling` and launches a single **assembler** task, which joins the transcripts, runs LLM extraction, generates the PDF, and marks the job `completed`.

```
Job enqueued
     │
     ▼
┌────────────────────┐   1 vCPU / 4 GB
│  Splitter          │   (ECS service, scales to 0 when idle)
│  - split audio     │
│  - upload chunks   │
│  - detect language │
└────────┬───────────┘
         │  ecs:RunTask × N  (up to MAX_PARALLEL_CHUNKS)
         │
    ┌────┴───────────────────────────────┐
    ▼          ▼                         ▼
┌──────────┐ ┌──────────┐   …   ┌──────────────┐   0.5 vCPU / 2 GB each
│ Chunk    │ │ Chunk    │       │ Chunk worker │   (ephemeral RunTask)
│ worker 1 │ │ worker 2 │       │ worker N     │
│ chunk 0  │ │ chunk 1  │       │ …            │
└──────────┘ └──────────┘       └──────┬───────┘
                                        │ last worker wins CAS
                                        ▼
                               ┌─────────────────┐   1 vCPU / 4 GB
                               │   Assembler      │   (ephemeral RunTask)
                               │ - join transcripts│
                               │ - LLM extraction  │
                               │ - PDF + S3        │
                               └─────────────────┘
```

A 90-minute meeting (18 × 5-minute chunks) with 6 parallel workers completes transcription in roughly the time of 3 sequential chunks — about a 6× speedup at the same total compute cost.

Language is detected from the first chunk by the splitter and passed to all chunk workers via an env-var override in the `RunTask` call, preventing Whisper from re-running its 30-second language probe on every segment.

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
  │ 0.25 vCPU       │  │ 0.25 vCPU       │   │  uploads/      │
  │ 512 MB RAM      │  │ 512 MB RAM      │   │  chunks/       │
  └───────┬────────┘  └───────┬────────┘   │  results/      │
          │                    │             └────────────────┘
          └────────┬───────────┘
                   │
          ┌────────▼────────┐
          │  RDS PostgreSQL  │
          │  job_records     │
          │  job_chunks      │
          └─────────────────┘
```

| Component | AWS Service | Details |
|-----------|-------------|---------|
| **API** | ECS Fargate | Stateless FastAPI. Streams uploads to S3, reads/writes PostgreSQL. |
| **Splitter** | ECS Fargate (service) | Polls job queue, splits audio, launches chunk workers. Scales to 0 when idle. |
| **Chunk worker** | ECS Fargate (RunTask) | Transcribes one job's chunks in a loop. 0.5 vCPU / 2 GB. Exits when done. |
| **Assembler** | ECS Fargate (RunTask) | Joins transcripts, runs LLM pipeline, generates PDF. 1 vCPU / 4 GB. Exits when done. |
| **Database** | RDS PostgreSQL | `job_records` job queue + `job_chunks` chunk queue, both using `SELECT … FOR UPDATE SKIP LOCKED`. |
| **Object Storage** | S3 | Audio uploads (`uploads/`), chunk WAVs (`chunks/`), PDFs + artifacts JSON (`results/`). |
| **Secrets** | SSM Parameter Store | `DATABASE_URL`, `AWS_S3_BUCKET`, `GEMINI_API_KEY`, `CORS_ORIGINS` injected into ECS tasks. |
| **Container Registry** | ECR | Single image for API, splitter, chunk worker, and assembler roles. |
| **Logs** | CloudWatch Logs | Structured JSON logs from `structlog`. 30-day retention. |
| **Load Balancer** | ALB | TLS termination, health checks (`/health`), HTTP→HTTPS redirect. |
| **Frontend** | Vercel | Next.js deployed separately. `NEXT_PUBLIC_API_URL` points to the ALB. |

The IAM task role requires `ecs:RunTask` and `iam:PassRole` (scoped to the worker task definition) in addition to the existing S3 and SSM permissions, so the splitter can launch chunk workers and the assembler.

---

## Key Features

### Artifact Editing

After processing, all extracted artifacts are rendered as inline-editable form fields. Client-side validation prevents saving with empty required fields. Edits are saved via `PUT /artifacts/{job_id}` with full `MeetingArtifacts` Pydantic validation. The PDF is always generated from the latest artifacts.

### Confidence Scores

Every artifact carries an optional `confidence_score` (0.0–1.0). If the LLM omits a score, a deterministic heuristic computes a fallback based on field completeness, action verb presence, and ambiguity detection. The frontend renders colour-coded indicators (green ≥ 0.8, amber ≥ 0.5, red < 0.5).

### Explicit vs Inferred Badges

Execution Tasks carry a `task_source` field (`"Explicit"` or `"Inferred"`) indicating whether the task was directly stated in the meeting or AI-derived. The frontend renders a colour-coded pill badge next to each title.

### Granular Progress States

Jobs transition through these stages: `uploading` → `splitting` → `transcribing` → `assembling` → `extracting` → `generating_pdf` → `completed` / `failed`. Each transition writes status and progress atomically in a single SQL `UPDATE`. The `diarizing` stage is inserted between `uploading` and `splitting` only when `HF_TOKEN` is configured.

During the `extracting` stage the worker receives granular sub-stage callbacks from the intelligence pipeline and writes them to PostgreSQL in real time:

| Sub-stage | Progress | Detail |
|-----------|----------|--------|
| **summarizing** | 52% | Map-Reduce hierarchical summarization |
| **compressing** | 60% | Semantic filtering + budget selection |
| **retrieving** | 63% | Per-artifact RAG segment retrieval |
| **extracting** | 66% | LLM structured artifact extraction |

Short transcripts (below `HIERARCHICAL_TOKEN_THRESHOLD`) skip directly to the extraction sub-stage.

### Unified Intelligence Pipeline

All AI features — speaker diarization, hierarchical summarization, transcript indexing, RAG retrieval, context compression, and structured extraction — are orchestrated by a single `HierarchicalExtractor` class in `backend/app/services/llm_extraction.py`. The pipeline is provider-agnostic: any `LLMProvider` (Gemini, OpenAI, or mock) is wrapped by the extractor, which decides the optimal path based on transcript length.

```
 Audio file
      │
      ▼
┌──────────────────┐
│ Speaker Diarize   │  (optional — requires HF_TOKEN)
│ (pyannote 3.1)    │  Global timeline → speaker labels
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Transcribe        │  Whisper (chunked, 5-min segments)
│                   │  Speaker-aligned if diarized
└────────┬─────────┘
         │
    token count
    ≤ threshold?──── YES ───► Direct Extraction ───► Artifacts
         │
         NO  (long transcript)
         │
         ├──────────────────────────────┐
         │  (concurrent via asyncio)    │
         ▼                              ▼
┌──────────────────┐       ┌───────────────────┐
│ Map-Reduce        │       │ RAG Index Build    │
│ Summarization     │       │ (chunk → embed →   │
│ (chunk → LLM →    │       │  NumPy / pgvector) │
│  merge)           │       └─────────┬─────────┘
└────────┬─────────┘                 │
         ▼                            │
┌──────────────────┐                 │
│ Context           │                 │
│ Compression       │                 │
│ (semantic filter   │                 │
│  + budget select)  │                 │
└────────┬─────────┘                 │
         │                            │
         └──────────┬─────────────────┘
                    ▼
          ┌───────────────────┐
          │ Per-artifact RAG   │  7 queries — one per
          │ Retrieval (top-K)  │  artifact category
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │ RAG-Augmented      │  Condensed summary +
          │ Extraction Prompt  │  retrieved segments
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │ Structured Artifact│  MeetingArtifacts JSON
          │ JSON + Validation  │  (Pydantic v2)
          └───────────────────┘
```

Each sub-stage fires an async callback that the worker writes to PostgreSQL, giving the frontend real-time progress visibility (see **Granular Progress States** above). The detailed design of each component is documented in the sections below.

### Speaker Diarization

When `HF_TOKEN` is set, the worker runs a **global speaker diarization** pass before transcription using `pyannote/speaker-diarization-3.1`. The pipeline:

1. **Convert** the uploaded audio to a 16 kHz mono WAV track via ffmpeg.
2. **Diarize** the full track with pyannote to produce a global timeline of speaker turns (`SPEAKER_00`, `SPEAKER_01`, …).
3. **Free** the diarization model (`del pipeline; gc.collect()`) to reclaim memory before Whisper loads.
4. **Transcribe** chunks as usual, but capture per-segment timestamps from Whisper.
5. **Align** each Whisper segment to the global timeline by mapping the segment's midpoint to the overlapping speaker turn. Consecutive segments by the same speaker are merged into a single paragraph.
6. **Output** a labelled transcript: `Speaker 1: … \n Speaker 2: …` that is passed to the LLM.

Because diarization runs over the full recording, speaker identities remain consistent across chunk boundaries. The diarization and Whisper models are never resident in memory simultaneously (sequential load / unload), keeping peak RSS within the 2 GB Fargate Spot budget.

**Prerequisites:** Create a free HuggingFace account and accept the user conditions for both [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0). Then generate an access token at https://huggingface.co/settings/tokens and set `HF_TOKEN` in your `.env`.

### Jira Export

Completed jobs export as Jira-compatible bulk-import JSON via `GET /export/jira/{job_id}`. Artifact types map to Jira issue types (Story, Task, Bug). Priorities, labels, and summaries are translated automatically.

### Hierarchical Summarization (Map-Reduce)

Transcripts from meetings longer than ~20 minutes can exceed the context window sweet-spot of most LLMs, causing the "lost in the middle" phenomenon — details in the centre of the text are silently dropped or hallucinated. Meetolog mitigates this with a **Hierarchical Summarization** pipeline that automatically activates for transcripts exceeding a configurable token threshold (default: 12 000 tokens).

```
┌──────────────────────────────────────────────────────────────┐
│                    Full Transcript (N tokens)                │
└──────────────────┬───────────────────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   ┌────────┐ ┌────────┐ ┌────────┐   ← Token-bounded chunks
   │Chunk 1 │ │Chunk 2 │ │Chunk K │     with overlap
   └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │
       ▼          ▼          ▼        ← MAP (Level 1):
   ┌────────┐ ┌────────┐ ┌────────┐    concurrent chunk
   │Summary │ │Summary │ │Summary │    summarization
   │   1    │ │   2    │ │   K    │
   └───┬────┘ └───┬────┘ └───┬────┘
       └──────────┼──────────┘
                  ▼
          ┌──────────────┐            ← REDUCE (Level 2+):
          │Merged Summary│              recursive merge until
          │  (condensed) │              within token budget
          └──────┬───────┘
                 ▼
         ┌───────────────┐            ← EXTRACT:
         │ Structured     │             existing extraction
         │ Artifact JSON  │             prompt runs on
         └───────────────┘             condensed text
```

**Chunking Strategy:**
- Transcripts are split into blocks along speaker-turn and paragraph boundaries — never mid-sentence.
- Each chunk is capped at `HIERARCHICAL_CHUNK_MAX_TOKENS` (default 6 000) tokens measured via `tiktoken`.
- A sliding overlap of `HIERARCHICAL_CHUNK_OVERLAP_TOKENS` (default 200) tokens is prepended to each subsequent chunk to maintain conversational context across boundaries.

**Map Phase (Level 1):**
- Each chunk is sent to the LLM with a summarization prompt that **strictly enforces** retention of all actionable items (tasks, decisions, blockers, user stories, action items).
- Chunk summaries run concurrently via `asyncio.gather`, bounded by a semaphore (`HIERARCHICAL_CONCURRENCY_LIMIT`, default 3) to respect API rate limits.

**Reduce Phase (Level 2+):**
- Chunk summaries are concatenated. If the combined token count still exceeds `HIERARCHICAL_MAX_SUMMARY_TOKENS` (default 12 000), the pipeline re-chunks the summaries and runs another round of summarization + merge.
- This recursive process continues until the text fits within the extraction budget or cannot be reduced further.

**Final Extraction:**
- The condensed summary — optionally augmented with RAG-retrieved context (see below) — is fed to the existing structured extraction prompt, producing the same `MeetingArtifacts` JSON schema the frontend expects. No downstream changes are required.

### RAG Transcript Retrieval

Even after hierarchical summarization, fine-grained details (specific acceptance criteria, exact assignees, resolution plans) can be compressed away during the Map-Reduce pipeline. The **RAG Transcript Retrieval** layer mitigates this by performing a targeted semantic search over the *original full-length transcript* at extraction time, injecting the most relevant verbatim segments alongside the condensed summary.

```
┌───────────────────────────────────────────────────────────────────┐
│                    Full Transcript (N tokens)                     │
└──────────┬────────────────────────────────────────┬───────────────┘
           │                                        │
           ▼                                        ▼
  ┌─────────────────────┐               ┌──────────────────────┐
  │ Hierarchical         │  (parallel)   │ RAG Indexing Pipeline │
  │ Map-Reduce           │               │                      │
  │ Summarization        │               │ 1. Chunk (1 500 tok) │
  │                      │               │ 2. Embed (batched)   │
  │                      │               │ 3. Store in NumPy    │
  └──────────┬──────────┘               └──────────┬───────────┘
             │                                      │
             ▼                                      ▼
  ┌──────────────────┐               ┌──────────────────────────┐
  │ Condensed Summary │               │ Per-artifact retrieval:  │
  │ (global context)  │               │ top-K cosine similarity  │
  └──────────┬───────┘               │ for each artifact type   │
             │                        └──────────┬───────────────┘
             │                                   │
             └──────────────┬────────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ RAG-Augmented Prompt │
                 │ (summary + segments) │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Structured Artifact  │
                 │ JSON Extraction      │
                 └─────────────────────┘
```

**How it works:**

1. **Index Build** — The original transcript is split into small chunks (default 1 500 tokens, 100-token overlap) and each chunk is embedded via the configured provider (OpenAI `text-embedding-3-small` or Gemini `text-embedding-004`). Embeddings are batched to respect rate limits. The resulting vectors are stored in a plain NumPy array — no external vector database is needed.

2. **Parallel Execution** — Index building runs concurrently with hierarchical summarization via `asyncio.gather`. Since they operate on independent API endpoints (embedding vs. text generation), this adds negligible latency.

3. **Per-Artifact Retrieval** — Before extraction, the pipeline issues seven semantic search queries (one per artifact category: User Stories, Tasks, Decisions, Blockers, Action Items, Execution Tasks, Ideas). Each query returns the top-K (default 5) most relevant transcript segments ranked by cosine similarity.

4. **Prompt Injection** — The retrieved segments are injected into the extraction prompt as a dedicated ``RETRIEVED TRANSCRIPT SEGMENTS`` section. The prompt instructs the LLM to treat these segments as **primary evidence** while using the condensed summary for global context. A configurable token budget (`RAG_MAX_CONTEXT_TOKENS`, default 3 000) prevents context window overflow.

5. **Statelessness** — The NumPy vector index lives in-process memory for the duration of the job. When the worker's `TemporaryDirectory` context manager exits, all references are garbage-collected. No persistent storage is written.

**When RAG activates:**
- Only for transcripts exceeding `HIERARCHICAL_TOKEN_THRESHOLD` (same trigger as hierarchical summarization).
- Short transcripts are extracted directly — the full text already fits in the LLM context window.
- In `TEST_MODE` (mock provider), RAG is skipped since mock providers don't support embeddings.

### Transcript Indexing Architecture

The Transcript Indexing system is the foundation that powers both the RAG retrieval layer and future cross-meeting search. It exposes a pluggable storage backend selected via the `RAG_STORAGE_BACKEND` environment variable.

**Storage Backends:**

| Backend | Setting | Storage | Lifecycle | Use Case |
|---------|---------|---------|-----------|----------|
| **In-Memory (NumPy)** | `RAG_STORAGE_BACKEND=memory` (default) | NumPy `float32` arrays | Ephemeral — destroyed when the worker's temp directory exits | Single-job extraction; lowest latency; no external dependencies |
| **Persistent (pgvector)** | `RAG_STORAGE_BACKEND=pgvector` | PostgreSQL `vector` column via the pgvector extension | Persistent — survives worker restarts; queryable from any process | Re-querying indexed transcripts; cross-meeting search; audit trails |

**Indexing Pipeline (both backends):**

```
┌──────────────────────────────────────────────────────────────────┐
│                  Full Transcript (N tokens)                       │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │ 1. Context-Aware       │  Split on speaker turns and
          │    Chunking            │  paragraph boundaries, never
          │    (1 500 tok/chunk,   │  mid-sentence. Sliding overlap
          │     100 tok overlap)   │  prevents context loss.
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │ 2. Async Batched       │  Chunks batched to respect
          │    Embedding           │  provider rate limits.
          │    (OpenAI or Gemini)  │  RAG_EMBEDDING_BATCH_SIZE
          └────────────┬───────────┘  controls batch width.
                       │
                       ▼
          ┌────────────────────────┐
          │ 3. Vector Storage      │  memory → NumPy float32 array
          │    (configurable)      │  pgvector → INSERT into
          │                        │  transcript_embeddings table
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │ 4. Cosine Similarity   │  memory → NumPy dot product
          │    Search              │  pgvector → ORDER BY <=>
          │    (top-K per query)   │  operator (exact kNN)
          └────────────────────────┘
```

**Embedding Storage Format:**

- **In-Memory:** Embeddings are stored as a contiguous `np.float32` matrix of shape `(num_chunks, embedding_dim)`. Cosine similarity is computed via normalised matrix multiplication — a single NumPy operation over the full index. No serialisation overhead.
- **pgvector:** Each chunk is stored as a row in the `transcript_embeddings` table:

  | Column | Type | Description |
  |--------|------|-------------|
  | `id` | `UUID` | Primary key (auto-generated) |
  | `job_id` | `UUID` | Foreign key scoping embeddings to a specific job |
  | `chunk_index` | `INTEGER` | Positional order within the transcript |
  | `chunk_text` | `TEXT` | Verbatim chunk content |
  | `embedding` | `vector` | Dimensionless pgvector column (768-d for Gemini, 1536-d for OpenAI) |
  | `created_at` | `TIMESTAMPTZ` | Insertion timestamp |

  Retrieval uses pgvector's native `<=>` cosine distance operator with an `ORDER BY … LIMIT` query scoped to `job_id`. For per-job workloads (hundreds of chunks), exact kNN is fast without an HNSW or IVFFlat index.

**End-to-End Retrieval Workflow:**

1. A long transcript enters the worker pipeline and triggers hierarchical summarization.
2. **Concurrently**, the indexing system chunks the *original full-length transcript* into 1 500-token blocks with 100-token sliding overlap, embeds each chunk via the configured provider, and stores the vectors in the selected backend.
3. After summarization completes, the pipeline issues seven semantic search queries — one per artifact category (User Stories, Tasks, Decisions, Blockers, Action Items, Execution Tasks, Ideas) — each retrieving the top-K most relevant verbatim transcript segments.
4. Retrieved segments are assembled into a `RETRIEVED TRANSCRIPT SEGMENTS` block, token-budget-capped at `RAG_MAX_CONTEXT_TOKENS` (default 3 000), and injected alongside the condensed summary into the extraction prompt.
5. The LLM receives both **global context** (condensed summary) and **fine-grained evidence** (retrieved segments), producing structured `MeetingArtifacts` JSON with maximal detail retention.

**pgvector Setup:**

To use the persistent backend:

1. Install the `vector` extension on your PostgreSQL server (RDS: enable in `shared_preload_libraries`; self-hosted: `apt install postgresql-16-pgvector` or build from source).
2. Run the migration: `alembic upgrade head` (or `alembic upgrade c7d9e1f3a5b2` for just this table).
3. Set `RAG_STORAGE_BACKEND=pgvector` in your environment.

The application also creates the table lazily on first use if the migration has not been applied.

### Context Compression

Even after hierarchical summarization, the condensed summary may contain conversational filler, hedging language, and low-density segments that consume tokens without contributing actionable information. The **Context Compression** layer sits between the Map-Reduce output and the final extraction prompt, reducing the token footprint while strictly preserving high-value semantic content.

```
┌──────────────────────────────────┐
│  Condensed Summary (from Map-   │
│  Reduce hierarchical pipeline)  │
└───────────────┬──────────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  1. Segment Splitting      │  Speaker turns, bullets,
   │                            │  paragraphs
   └────────────┬───────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  2. Semantic Filtering     │  Remove filler, pleasantries,
   │                            │  verbal tics, pure acks
   └────────────┬───────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  3. Chunk Prioritization   │  Score segments on weighted
   │                            │  feature set:
   │   • Decision language  3.0 │
   │   • Temporal markers   2.5 │
   │   • Assignment patterns 2.5│
   │   • Blocker language   2.5 │
   │   • Action verbs       2.0 │
   │   • Quantitative data  1.5 │
   │   • Named entities     1.0 │
   └────────────┬───────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  4. Budget Selection       │  Rank by score, select top
   │                            │  segments up to token budget,
   │                            │  restore original order
   └────────────┬───────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  Compressed Context        │  Reduced token count,
   │  (ready for extraction)    │  all actionable items preserved
   └────────────────────────────┘
```

**Strategy: Semantic Filtering + Chunk Prioritization**

The compression layer uses a purely algorithmic approach — no additional LLM calls, no added latency or API cost.

1. **Semantic Filtering** — Regex-based pattern matching removes greetings, meeting logistics filler ("can everyone hear me"), verbal tics ("um", "uh"), standalone acknowledgements ("okay", "got it"), and non-verbal annotations ("[laughs]"). These patterns are compiled once at module load time.

2. **Chunk Prioritization** — Each surviving segment is scored against a weighted feature set designed to surface actionable content:
   - *Decision language* ("agreed", "confirmed", "we'll go with") scores highest at 3.0×
   - *Temporal markers* (deadlines, sprint references, dates) score 2.5×
   - *Assignment patterns* ("assigned to", "Tom will", "owner") score 2.5×
   - *Blocker language* ("blocked", "dependency", "waiting on") scores 2.5×
   - *Action verbs* ("implement", "deploy", "fix") score 2.0×
   - *Quantitative data* (story points, percentages, durations) scores 1.5×
   - *Named entities* (multi-word proper nouns, @mentions) score 1.0×

3. **Budget-Constrained Selection** — Segments are ranked by score (descending) and greedily selected until the configurable token budget is filled (`COMPRESSION_TARGET_BUDGET_TOKENS`, default 8 000). Selected segments are returned in their original document order to preserve narrative coherence.

**Artifact Accuracy Guarantees:**
- Named entities, assignees, and technical terms are **never altered** — the compressor only removes or keeps whole segments, never rewrites text.
- The scoring system is calibrated to heavily favour segments containing decisions, assignments, blockers, and deadlines — exactly the content the downstream extraction prompt needs.
- A baseline score (0.1) is assigned to every non-filler segment, ensuring low-keyword but contextually relevant segments are still considered when budget allows.

**Performance Impact:**
- **Token Reduction:** ~30–50% reduction in tokens sent to the final extraction prompt (varies by meeting style — highly conversational meetings see larger gains).
- **Latency:** Zero added LLM latency — the compression pass runs in-process on CPU in O(n) time.
- **Cost Savings:** Proportional to token reduction — fewer input tokens per extraction API call.
- **Accuracy:** The scoring system preserves all high-density actionable content. Combined with RAG-retrieved verbatim segments (which are injected separately), fine-grained details remain available to the extraction LLM.

**When compression activates:**
- Only for transcripts that trigger hierarchical summarization (exceeding `HIERARCHICAL_TOKEN_THRESHOLD`).
- Disabled when `COMPRESSION_ENABLED=false`.
- Skipped when the condensed summary already fits within `COMPRESSION_TARGET_BUDGET_TOKENS`.

### Chunked Transcription

Multi-hour recordings are split into 5-minute WAV chunks via ffmpeg. The Whisper model is loaded once per process. Chunks are transcribed sequentially with `gc.collect()` after each to reclaim memory. Progress callbacks write per-chunk updates to PostgreSQL for real-time frontend visibility.

### Prompt Engineering System

The LLM extraction pipeline uses a **template-driven prompt architecture** with strict structured output validation and anti-hallucination safeguards. All prompt templates live in `backend/app/core/prompts.py` and follow a consistent five-section structure:

```
┌─────────────────────────────────────────────────┐
│  1. ROLE         — Expert persona assignment     │
│                    + anti-hallucination rules     │
│  2. CONTEXT      — Transcript / summary input    │
│  3. INSTRUCTIONS — Numbered extraction rules     │
│                    with Chain-of-Thought guidance │
│  4. EXAMPLES     — Few-shot JSON exemplar        │
│  5. SCHEMA       — Exact JSON schema to follow   │
└─────────────────────────────────────────────────┘
```

**Anti-hallucination safeguards** built into the role persona:
- Extract ONLY what is explicitly stated or strongly implied in the transcript
- Every field must be traceable to a specific passage — do not fabricate details
- Prefer empty/null fields over invented content

**Prompt domains:**

| Domain | Builder | Persona | Technique |
|--------|---------|---------|-----------|
| **Artifact Extraction** | `build_extraction_prompt()` | Elite Agile Business Analyst | Few-shot exemplar, full JSON schema, anti-hallucination rules |
| **Task Detection** | `build_task_detection_prompt()` | Agile Task Analyst | Explicit vs. Inferred classification |
| **Decision Detection** | `build_decision_detection_prompt()` | Requirements Analyst | Chain-of-Thought 5-step reasoning |
| **Summarization** | `build_summarization_prompt()` | Technical Meeting Analyst | Structured section output |

Two additional templates (`CHUNK_SUMMARIZATION_PROMPT`, `MERGE_SUMMARIZATION_PROMPT`) drive the hierarchical Map-Reduce pipeline.

**Artifact extraction categories (7 types):**

1. **User Stories** — Role, desire, goal, acceptance criteria, story points
2. **Decisions** — Title, description, rationale, `decision_summary`, `alternatives_rejected`, made-by
3. **Tasks** — Title, description, assignee, priority, due date, `context` (reasoning)
4. **Blockers** — Title, description, owner, resolution plan, affected tasks
5. **Action Items** — `title`, description, assignee, due date, `context`, `priority`
6. **Execution Tasks** — AI-inferred tasks with Explicit/Inferred source tracking
7. **Ideas & Suggestions** — `idea_description`, `proposed_by`, `potential_impact`

**Validation pipeline:**

Raw LLM text passes through a strict 4-stage sanitization and validation pipeline defined in `backend/app/models/artifacts.py`:

```
 LLM raw text
      │
      ▼
 ┌──────────────────────┐
 │ 1. Sanitize text      │  Strip ```json fencing,
 │                        │  remove trailing commas,
 │                        │  extract JSON from prose
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 2. JSON parse         │  json.loads() first;
 │    + repair           │  json_repair.loads() fallback
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 3. Pydantic v2        │  LLMExtractionResponse
 │    validation          │  model_validate() with
 │    + model_validators  │  pre-validation cleaning
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 4. Domain conversion  │  to_meeting_artifacts()
 │    + normalization     │  → MeetingArtifacts
 └──────────────────────┘
```

The `sanitize_json_string()` function chains three recovery stages: markdown fencing removal, trailing comma cleanup via regex, and brace-boundary extraction that discards any LLM prose surrounding the JSON object. The `json-repair` library then recovers from remaining issues (missing quotes, unescaped characters) before Pydantic validation runs. All Pydantic models use `@model_validator(mode="before")` decorators that strip whitespace from strings, normalize dates, and clamp confidence scores to [0.0, 1.0].

**Retry strategy:**

If validation fails (malformed JSON or schema mismatch), the pipeline **retries at progressively lower temperatures** with an explicit re-prompt on the final attempt:

1. **Attempt 1:** `temperature=0.1` — standard extraction
2. **Attempt 2:** `temperature=0.0` — maximum determinism retry
3. **Attempt 3:** `temperature=0.0` with explicit re-prompt instructing the LLM to return only valid JSON

This three-temperature approach is separate from the `tenacity` retry logic that handles transient API errors (connection failures, timeouts). Both layers combine to provide robust end-to-end reliability.

---

## Observability & Reliability

- **Structured Logging** — All components emit JSON logs via `structlog` with UTC timestamps, log levels, and contextual IDs (`job_id`, `worker_id`). Logs are captured by AWS CloudWatch.
- **Transient Failure Retries** — S3 calls use `tenacity` exponential-backoff retries (up to 4 attempts). LLM calls retry up to 3 times. Only transient errors trigger retries.
- **Timeout Protection** — LLM extraction uses `asyncio.timeout(60)`. Whisper runs in a thread pool with inherited OS-level timeout.
- **Stale-Lock Recovery** — The PostgreSQL queue reclaims jobs stuck in `processing` when `locked_at` exceeds 2 hours. Crashed workers' jobs are automatically retried.
- **Uncrashable Worker Loop** — Each job is wrapped in `try … except Exception`. Failures mark the job as `failed` and the worker continues polling.
- **Graceful Cancellation** — See below.

### Graceful Job Cancellation

Meetolog supports user-initiated cancellation of any queued or in-progress transcription job. Cancellation is coordinated entirely via a database status flag — no inter-process signals, message queues, or shared memory are required. This keeps the API and worker fully stateless and compatible with horizontal scaling.

**State machine addition:**

```
pending ──────────────────────────────────────► cancelled
    │                                               ▲
    └─► processing ──► completed (terminal)         │
              │                                     │
              └──────────────────────────────────────┘
              │
              └─► failed (retryable or terminal)
```

`cancelled` is a permanent terminal state. A `cancelled` job cannot transition to `completed`, `failed`, or back to `pending`. Attempting to cancel a `completed` or `failed` job returns HTTP 409.

**How cancellation works end-to-end:**

1. **Frontend trigger (explicit)** — The user clicks "Cancel Processing" during an active polling session. The client immediately stops the poll loop, optimistically updates local state to `cancelled`, and calls `POST /api/jobs/{job_id}/cancel`. The server response replaces the optimistic state with the authoritative job record.

2. **Frontend trigger (page abandonment)** — When the user closes the tab or reloads while a job is in flight, the `beforeunload` event handler fires `navigator.sendBeacon("/api/jobs/{job_id}/cancel")`. `sendBeacon` is guaranteed to complete the POST even as the page unloads; no response is awaited or needed.

3. **API layer** — `POST /jobs/{job_id}/cancel` executes an atomic SQL `UPDATE` guarded by `WHERE status NOT IN ('completed', 'failed')`. This prevents clobbering a job that races to a terminal state between the status read and the write. The `cancelled_at` timestamp is recorded for observability. Worker lock fields (`locked_at`, `locked_by`) are cleared.

4. **Worker polling** — At every significant processing boundary the worker calls `queue.fetch_job_status(job_id)` (a non-locking `SELECT`). The boundaries are:
   - After S3 audio download (before any CPU work begins)
   - Before each 5-minute Whisper transcription chunk (via the progress callback)
   - After transcription completes, before LLM extraction begins
   - Before each LLM pipeline sub-stage (summarising → compressing → retrieving → extracting)
   - After artifact extraction, before PDF generation
   - After PDF generation, before S3 upload (last opportunity to avoid writing permanent artefacts)

5. **Worker abort** — When the status reads `cancelled`, the worker raises an internal `_JobCancelledError` sentinel. This propagates out of the `with tempfile.TemporaryDirectory(...)` block, which automatically deletes the `/tmp` working directory. The dedicated `except _JobCancelledError` handler logs `job_cancelled_gracefully` and exits cleanly — it does **not** call `mark_job_failed`, leaving the database status as `cancelled` without triggering the retry mechanism.

**Guarantees:**
- A `pending` job cancelled before any worker claims it will never be processed — the `claim_next_job` query's `WHERE status = 'pending'` clause excludes cancelled rows.
- A `processing` job will be cancelled within at most one Whisper chunk boundary (≤ 5 minutes for standard recordings).
- No S3 artefacts (PDF, artifact JSON) are written for cancelled jobs.
- The `/tmp` session directory is always cleaned up, regardless of which checkpoint triggers the abort.

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
│   │   │   ├── artifacts.py           # LLM response validation & JSON repair pipeline
│   │   │   ├── metadata.py            # FileMetadata SQLAlchemy ORM
│   │   │   └── db_models.py           # JobRecord SQLAlchemy ORM
│   │   ├── core/
│   │   │   ├── logger.py              # structlog configuration (JSON / console)
│   │   │   └── prompts.py             # Template-driven prompt system (6 domains)
│   │   ├── infrastructure/
│   │   │   ├── db.py                  # Async SQLAlchemy engine + session
│   │   │   ├── postgres_job_store.py  # PostgresJobStore (read/update)
│   │   │   └── postgres_queue.py      # PostgresJobQueue (SKIP LOCKED queue)
│   │   ├── services/
│   │   │   ├── storage.py             # S3StorageService (upload/download/presign)
│   │   │   ├── transcription.py       # WhisperTranscriber (chunked + segment-level)
│   │   │   ├── diarization.py         # SpeakerDiarizer (pyannote global timeline)
│   │   │   ├── llm_extraction.py      # Unified Intelligence Pipeline (HierarchicalExtractor)
│   │   │   ├── llm_engine.py          # LLM provider abstraction (Gemini/OpenAI)
│   │   │   ├── rag_retrieval.py       # RAG transcript retrieval (embed, index, search)
│   │   │   ├── transcript_index.py    # PgVectorIndex — persistent pgvector storage backend
│   │   │   ├── compression.py         # Context compression (semantic filtering + prioritization)
│   │   │   ├── heuristics.py          # Deterministic confidence scoring
│   │   │   ├── pdf_generator.py       # ReportLab PDF generation
│   │   │   ├── jira_mapper.py         # Jira bulk-import JSON mapper
│   │   │   └── mock_services.py       # Mock services for testing
│   │   └── utils/
│   │       ├── audio.py               # ffmpeg audio splitting & duration probe
│   │       └── text_chunking.py       # Token-aware transcript chunking
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
├── docker-compose.yml                 # Local dev stack (PostgreSQL, MinIO, S3-compatible storage)
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
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model used for extraction |
| `OPENAI_API_KEY` | `""` | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model used for extraction |
| `HF_TOKEN` | `""` | HuggingFace token enabling pyannote speaker diarization |
| `WHISPER_MODEL` | `tiny` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `DATABASE_URL` | `""` | PostgreSQL async DSN: `postgresql+asyncpg://user:pass@host/db` |
| `AWS_ACCESS_KEY_ID` | `""` | AWS IAM access key (local dev only; production uses IAM task role) |
| `AWS_SECRET_ACCESS_KEY` | `""` | AWS IAM secret key (local dev only) |
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_S3_BUCKET` | `""` | S3 bucket name |
| `AWS_ENDPOINT_URL` | `None` | Custom S3 endpoint (e.g. `http://minio:9000` for local MinIO) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `MAX_UPLOAD_SIZE_MB` | `100` | Max upload size |
| `HIERARCHICAL_TOKEN_THRESHOLD` | `12000` | Token count above which hierarchical summarization activates |
| `HIERARCHICAL_CHUNK_MAX_TOKENS` | `6000` | Maximum tokens per chunk in the Map phase |
| `HIERARCHICAL_CHUNK_OVERLAP_TOKENS` | `200` | Overlap tokens between consecutive chunks |
| `HIERARCHICAL_MAX_SUMMARY_TOKENS` | `12000` | Trigger an additional Reduce pass if merged summaries exceed this |
| `HIERARCHICAL_CONCURRENCY_LIMIT` | `3` | Max concurrent LLM calls during the Map phase |
| `RAG_CHUNK_MAX_TOKENS` | `1500` | Maximum tokens per chunk for RAG indexing |
| `RAG_CHUNK_OVERLAP_TOKENS` | `100` | Overlap tokens between consecutive RAG chunks |
| `RAG_TOP_K` | `5` | Number of top-K chunks to retrieve per artifact category |
| `RAG_MAX_CONTEXT_TOKENS` | `3000` | Token budget for RAG context injected into extraction prompt |
| `RAG_EMBEDDING_BATCH_SIZE` | `64` | Batch size for embedding API calls during RAG indexing |
| `RAG_STORAGE_BACKEND` | `memory` | RAG vector storage: `memory` (ephemeral NumPy) or `pgvector` (persistent PostgreSQL) |
| `COMPRESSION_ENABLED` | `true` | Enable context compression before final extraction |
| `COMPRESSION_TARGET_BUDGET_TOKENS` | `8000` | Target token budget for compressed context |
| `ECS_CLUSTER` | `meetolog-cluster` | ECS cluster name |
| `ECS_WORKER_SERVICE` | `meetolog-worker-svc` | ECS service name for the splitter (scales to 0 when idle) |
| `ECS_WORKER_TASK_DEFINITION` | `meetolog-worker` | Task definition used when launching chunk workers and assembler via RunTask |
| `MAX_PARALLEL_CHUNKS` | `6` | Max simultaneous chunk-worker tasks per job (0.5 vCPU × N, default fits within 6 vCPU Fargate limit) |
| `WORKER_IDLE_SHUTDOWN_POLLS` | `6` | Consecutive empty polls before the splitter scales itself to 0 (5 s each → 30 s idle window) |

The frontend reads:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for browser requests and Next.js rewrites |

---

## Deployment Constraints

- **PostgreSQL required.** Job lifecycle, `job_records` queue, `job_chunks` parallel-chunk queue, file metadata. Production uses AWS RDS. Run `alembic upgrade head` to apply all migrations.
- **AWS S3 required.** Audio uploads (`uploads/`), chunk WAVs (`chunks/`), PDFs and artifact JSON (`results/`). The ECS task role provides access — no static IAM credentials in production.
- **IAM permissions for RunTask.** The ECS task role (`meetolog-ecs-task-role`) needs `ecs:RunTask` and `iam:PassRole` scoped to the worker task definition. Without these the splitter cannot launch chunk workers; jobs would get stuck in `transcribing`.
- **Splitter runs as ECS service.** `SERVICE_TYPE=worker` (or `splitter`). Scales to 0 when idle; the API wakes it on job enqueue. Chunk workers and the assembler are ephemeral `RunTask` calls — they are not ECS services.
- **SKIP LOCKED prevents duplicates.** Both `job_records` and `job_chunks` use `SELECT … FOR UPDATE SKIP LOCKED`. Safe to run multiple splitters or chunk workers simultaneously.
- **SERVICE_TYPE values.** `web` → FastAPI server. `worker` / `splitter` → splitter polling loop. `chunk_worker` → single-job chunk transcription loop. `assembler` → transcript assembly + extraction + PDF.

---

## License

MIT License. See [LICENSE](LICENSE).
