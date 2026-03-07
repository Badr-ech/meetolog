# Meetolog — Capstone Internship Project Report

**Project Title:** Meetolog — Meeting to Backlog: AI-Powered Meeting Artifact Extraction System

**Student:** [Student Full Name]

**Academic Supervisor:** [Supervisor Full Name]

**Company/Organization:** CodeVista Innovations

**Internship Period:** [Start Date] — [End Date]

**Date of Submission:** February 21, 2026

---

## Table of Contents

1. [Project Definition](#1-project-definition)
   - 1.1 [Context](#11-context)
   - 1.2 [Scope](#12-scope)
   - 1.3 [Overall Description](#13-overall-description)
   - 1.4 [Objectives](#14-objectives)
2. [Project Organization](#2-project-organization)
   - 2.1 [Selected Software Engineering Model](#21-selected-software-engineering-model)
   - 2.2 [Planning of Activities](#22-planning-of-activities)
3. [Requirements Engineering](#3-requirements-engineering)
   - 3.1 [Requirements Elicitation](#31-requirements-elicitation)
   - 3.2 [System Models](#32-system-models)
   - 3.3 [Formal Specifications](#33-formal-specifications)

---

## 1. Project Definition

### 1.1 Context

The present capstone project was carried out within the framework of an internship at **CodeVista Innovations**, a software development company specializing in building innovative digital solutions that leverage cutting-edge technologies such as artificial intelligence, cloud computing, and modern web frameworks to address real-world business challenges. CodeVista Innovations fosters a collaborative engineering culture that encourages interns and developers to prototype novel solutions and bring them from concept to deployment.

In today's fast-paced Agile software development environments, teams conduct a significant number of meetings on a daily and weekly basis — including sprint planning sessions, daily stand-ups, retrospectives, and stakeholder reviews. These meetings produce a wealth of actionable information: user stories, tasks, decisions, impediments, and action items. However, the process of manually capturing, transcribing, and organizing these meeting outcomes into structured backlog artifacts remains a time-consuming, error-prone, and often inconsistent endeavor. Studies in project management literature indicate that poor meeting documentation leads to knowledge loss, misaligned priorities, and duplicated effort across development teams (Schwaber & Sutherland, 2020).

The business need that prompted this capstone project was to explore the feasibility of an AI-powered system capable of automating the transformation of unstructured meeting audio recordings into structured, actionable Agile artifacts. CodeVista Innovations identified this as an opportunity to build a Minimum Viable Product (MVP) that demonstrates the integration of state-of-the-art speech-to-text transcription and Large Language Model (LLM) semantic extraction to solve a recurring pain point in the Agile software development workflow.

The resulting system, **Meetolog** (a portmanteau of "Meeting" and "Backlog"), was conceived, designed, and implemented during the internship period as a full-stack web application.

### 1.2 Scope

#### In Scope

The following elements fall within the boundaries of this project:

- **Audio Input Pipeline:** Support for audio file uploads in standard formats (MP3, WAV, M4A, OGG, WebM) as well as in-browser microphone recording with real-time waveform visualization.
- **Client-Side Audio Optimization:** Automatic downsampling and conversion of audio to 16 kHz mono WAV format on the client side to minimize network payload and optimize transcription accuracy.
- **Speech-to-Text Transcription:** Integration of OpenAI's Whisper model running locally on the server for offline, privacy-preserving speech-to-text conversion, with configurable model sizes (tiny, base, small, medium, large) to balance accuracy against hardware constraints.
- **AI-Powered Semantic Extraction:** Utilization of Google Gemini (and optionally OpenAI GPT) Large Language Models to extract structured Agile artifacts from raw transcripts, including:
  - User Stories with acceptance criteria and story point estimates
  - Tasks with assignee, priority, and due date
  - Decisions with rationale
  - Blockers with affected tasks and resolution plans
  - Action Items
  - Execution Tasks — AI-inferred actionable work items derived from both explicit statements and logical implications, with owner roles, priorities, and dependency tracking
- **PDF Report Generation:** Automated generation of downloadable, professionally formatted PDF meeting summaries using the ReportLab library, including a dedicated Execution Tasks page.
- **Asynchronous Job Processing:** A Redis-backed asynchronous job queue (ARQ) enabling non-blocking background processing of audio files with real-time progress polling from the frontend.
- **Deployment Infrastructure:** Docker containerization and deployment configurations for Render (cloud PaaS) and local Docker Compose environments, including a Blueprint Infrastructure-as-Code specification.
- **Test Mode & CI/CD Support:** A comprehensive mock service layer with deterministic outputs to enable testing, CI/CD pipelines, and demo environments without external API dependencies.

#### Out of Scope

The following elements are explicitly excluded from the current MVP:

- User authentication and multi-user support.
- Persistent relational database storage (e.g., PostgreSQL); the current system relies on Redis with configurable TTL.
- Cloud object storage for uploads and generated PDFs (e.g., Amazon S3 or Cloudflare R2).
- Direct integration with project management tools such as Jira, Azure DevOps, or Trello.
- Real-time WebSocket-based progress updates (the current implementation uses HTTP polling).
- Multi-language meeting transcription (although Whisper natively supports this, the LLM prompt engineering is currently optimized for English).
- GPU acceleration deployment configuration.
- Horizontal scaling beyond a single worker instance (as documented in deployment constraints).

### 1.3 Overall Description

**Meetolog** is a full-stack web application structured as a decoupled client–server architecture. The system is composed of three principal subsystems:

1. **Frontend (Presentation Layer):** A modern single-page application built with **Next.js 16** (React framework with App Router) and **TypeScript**, providing a responsive user interface for audio recording, file uploading, progress monitoring, and artifact visualization. The frontend employs CSS Modules for scoped styling and communicates with the backend via a RESTful API. Client-side audio processing is performed using the Web Audio API (`OfflineAudioContext`) to downsample recordings to 16 kHz mono WAV format before transmission.

2. **Backend API (Application Layer):** A **FastAPI**-based REST API serving as the system's entry point for file uploads, job status queries, artifact retrieval, and PDF downloads. The API is fully asynchronous, leveraging Python's `asyncio` for non-blocking I/O operations. The backend employs a clean, interface-based architecture with dependency injection and the Factory Pattern for service instantiation, enabling seamless switching between production and mock implementations.

3. **Background Worker (Processing Layer):** An **ARQ** (Async Redis Queue) worker process that executes the core processing pipeline in the background:
   - **Stage 1 — Transcription (10–40%):** The uploaded audio is split into chunks via `ffmpeg` and transcribed incrementally using the local Whisper model, with partial results cached in Redis for restart resilience.
   - **Stage 2 — Extraction (40–75%):** The complete transcript is submitted to the configured LLM provider (Google Gemini or OpenAI) with a carefully engineered prompt to extract structured Agile artifacts as JSON.
   - **Stage 3 — PDF Generation (75–95%):** A professional PDF report is generated using ReportLab with styled tables, bullet lists, and a dedicated Execution Tasks section.
   - **Stage 4 — Completion (100%):** The job status is updated, and artifacts/PDF are made available for retrieval.

**Redis** serves as the central shared state store, providing job metadata persistence (via Redis Hashes with configurable TTL), transcript and artifact caching (for pipeline resumability), compressed audio backup (for restart resilience), and the ARQ task queue.

The system integrates into the development workflow as follows: a team member uploads or records a meeting, and the system automatically produces a set of structured backlog items and a downloadable PDF summary, which can then be manually imported into the team's project management tool.

### 1.4 Objectives

The objectives of the Meetolog project are classified into three categories: technical, business, and academic.

#### 1.4.1 Technical Objectives

| ID | Objective | Measurable Criterion |
|----|-----------|---------------------|
| T1 | Build a functional end-to-end audio processing pipeline | System accepts audio input (upload or recording), transcribes it, extracts artifacts, and generates a PDF without manual intervention |
| T2 | Achieve accurate speech-to-text transcription | Whisper model produces intelligible transcripts from standard meeting audio recordings |
| T3 | Extract structured Agile artifacts using LLM-based semantic analysis | The system correctly identifies and outputs User Stories, Tasks, Decisions, Blockers, Action Items, and Execution Tasks in validated Pydantic data models |
| T4 | Implement an asynchronous, resilient processing architecture | Jobs are processed in the background via ARQ, with Redis-backed state persistence and pipeline resumability after restarts |
| T5 | Design a clean, extensible backend with interface-based architecture | All core services (Transcriber, LLM Extractor, Job Store) implement abstract interfaces, enabling strategy swapping and test mocking via dependency injection |
| T6 | Deliver a responsive, user-friendly frontend experience | The Next.js frontend provides in-browser recording (with waveform visualization), file upload, real-time progress tracking, structured artifact display, and PDF download |
| T7 | Containerize and deploy the application to a cloud PaaS | The system is deployable via Docker Compose (local) and Render Blueprint (cloud) with Infrastructure-as-Code configuration |

#### 1.4.2 Business Objectives

| ID | Objective | Measurable Criterion |
|----|-----------|---------------------|
| B1 | Reduce the time required to document meeting outcomes | The system converts a 30–45 minute meeting recording into structured artifacts within minutes, compared to 30+ minutes of manual documentation |
| B2 | Improve consistency and completeness of backlog items | AI extraction ensures standardized formatting (e.g., "As a... I want... So that...") and captures items that may be overlooked in manual note-taking |
| B3 | Demonstrate the viability of AI-assisted Agile tooling | The MVP serves as a proof-of-concept for CodeVista Innovations' potential product offering in the AI-powered DevOps and project management space |

#### 1.4.3 Academic Objectives

| ID | Objective | Measurable Criterion |
|----|-----------|---------------------|
| A1 | Apply software engineering principles in a real-world project lifecycle | Demonstrate proficiency in requirements engineering, system design, implementation, and deployment within a professional environment |
| A2 | Practice the Agile Scrum methodology in an authentic internship setting | Participate in sprint ceremonies, manage a product backlog, and iteratively deliver increments |
| A3 | Integrate and evaluate multiple AI/ML technologies | Gain hands-on experience with speech-to-text (Whisper), Large Language Models (Gemini/OpenAI), and prompt engineering |
| A4 | Design and implement a production-grade full-stack application | Demonstrate competence in modern web technologies (FastAPI, Next.js, Redis, Docker) and software architecture patterns (Strategy, Factory, Dependency Injection) |

---

## 2. Project Organization

### 2.1 Selected Software Engineering Model

#### 2.1.1 Model Identification

The **Agile Scrum** framework was selected as the Software Development Life Cycle (SDLC) model for this capstone project. Scrum is an iterative and incremental framework for managing complex product development, as defined by Schwaber and Sutherland in *The Scrum Guide* (2020). It organizes work into fixed-length iterations called **Sprints**, during which cross-functional teams deliver potentially shippable product increments.

#### 2.1.2 Academic Justification

The selection of Agile Scrum over alternative SDLC models (such as Waterfall, V-Model, or Spiral) is justified by the following project-specific factors:

1. **Evolving Requirements:** The nature of this MVP project involved significant uncertainty regarding the optimal combination of AI models, prompt engineering strategies, and user interface design. Requirements were expected to evolve as the team gained a deeper understanding of LLM capabilities and the quality of Whisper transcriptions. Scrum's embrace of change through iterative refinement is well-suited to this context, as argued by Pressman and Maxim (2020) in *Software Engineering: A Practitioner's Approach*.

2. **Short Development Timeline:** The internship period imposed a constrained timeline. Scrum's time-boxed Sprints (typically 1–2 weeks) ensured that working increments were delivered early and frequently, enabling rapid feedback loops and course correction. This aligns with the Agile Manifesto's principle of delivering "working software frequently, from a couple of weeks to a couple of months" (Beck et al., 2001).

3. **Stakeholder Engagement:** The internship structure required regular demonstrations to the company supervisor and academic advisor. Scrum's Sprint Review ceremony provided a natural cadence for stakeholder engagement, ensuring alignment between the delivered product and stakeholder expectations.

4. **Technical Experimentation:** The project required significant prototyping and experimentation — evaluating Whisper model sizes, iterating on LLM prompts, comparing Gemini vs. OpenAI providers, and tuning chunked transcription parameters. Scrum's Sprint-based iteration allowed these experiments to be conducted incrementally, with findings informing subsequent Sprint backlogs.

5. **Single-Developer Adaptation:** Although Scrum is traditionally designed for teams of 3–9 members, its ceremonies and artifacts (Product Backlog, Sprint Backlog, Sprint Review, Retrospective) can be meaningfully adapted for a single developer working under a supervisor who acts as the Product Owner and Scrum Master. This adaptation is consistent with recommendations in the literature on solo Scrum practices (Pagotto et al., 2016).

6. **Risk Mitigation:** The Spiral model was considered for its explicit risk management, but Scrum's empirical process control (transparency, inspection, and adaptation) was deemed sufficient for this project's risk profile while introducing less process overhead.

#### 2.1.3 Comparison with Alternative Models

| Criterion | Waterfall | Spiral | Agile Scrum (Selected) |
|-----------|-----------|--------|----------------------|
| Requirements Stability | Stable, well-defined | Partially defined | Evolving, flexible |
| Feedback Frequency | End of lifecycle | Per iteration | Every 1–2 weeks (Sprint) |
| Risk Management | Low (late detection) | High (explicit risk phases) | Moderate (iterative inspection) |
| Stakeholder Involvement | Minimal (after phase completion) | Periodic | Continuous (Sprint Reviews) |
| Suitability for MVP | Low | Moderate | **High** |
| Overhead for Small Teams | Low process overhead | High process overhead | **Low to moderate** |

### 2.2 Planning of Activities

#### 2.2.1 Sprint Structure

The project was organized into **[5–6] Sprints**, each spanning **[1–2 weeks]**. The planning followed a progressive delivery approach, starting from core infrastructure and building toward the complete end-to-end pipeline.

| Sprint | Duration | Focus Area | Key Deliverables |
|--------|----------|------------|------------------|
| Sprint 0 | [Week 1] | **Project Setup & Research** | Technology evaluation (Whisper model comparison, Gemini vs. OpenAI), environment setup, architectural design, initial backlog creation |
| Sprint 1 | [Weeks 2–3] | **Core Backend & Transcription** | FastAPI project structure, Pydantic data models, abstract interfaces (`Transcriber`, `LLMExtractor`, `JobStore`), Whisper integration with chunked transcription, mock services for testing |
| Sprint 2 | [Weeks 3–4] | **LLM Extraction & Prompt Engineering** | Gemini API integration, extraction prompt design and iteration, JSON parsing with validation, artifact schema refinement (addition of Execution Tasks), OpenAI provider as alternative |
| Sprint 3 | [Weeks 5–6] | **Frontend Development & Integration** | Next.js application with VoiceRecorder component, in-browser recording with waveform visualization, client-side audio downsampling (Web Audio API), API client implementation with polling |
| Sprint 4 | [Weeks 6–7] | **PDF Generation & Async Architecture** | ReportLab PDF generator, Redis integration for job persistence, ARQ background worker, pipeline resumability with transcript/audio caching, progress tracking |
| Sprint 5 | [Weeks 7–8] | **Deployment, Testing & Documentation** | Dockerfile (multi-stage, CPU-only PyTorch), Docker Compose configuration, Render Blueprint (IaC), end-to-end testing in test mode, README documentation |

#### 2.2.2 Ceremonies

The following Scrum ceremonies were practiced during the internship:

- **Sprint Planning:** At the beginning of each Sprint, the intern and the company supervisor (acting as Product Owner) reviewed the Product Backlog, prioritized items, and selected the Sprint Goal and Sprint Backlog items using story point estimation (Fibonacci sequence).
- **Daily Stand-ups:** Brief daily check-ins (adapted to the solo-developer context) were conducted either in-person or via Slack/messaging to communicate progress, identify impediments, and align on priorities.
- **Sprint Review/Demo:** At the conclusion of each Sprint, a working demonstration of the increment was presented to the company supervisor, showcasing new functionality and gathering feedback.
- **Sprint Retrospective:** A brief self-reflection was conducted to identify what went well, what could be improved, and action items for the next Sprint. Findings were documented in internal notes.

#### 2.2.3 Resource Management

- **Human Resources:** The project was executed by a single intern developer under the guidance of a company supervisor (Product Owner) and academic supervisor (advisor). Code reviews were conducted by the company supervisor.
- **Infrastructure Resources:** Development was performed on a local machine meeting the minimum hardware requirements (4+ cores, 8+ GB RAM). Cloud deployment was managed via Render's free tier for demonstration purposes.
- **Time Management:** Tasks were tracked using a product backlog, with user stories and technical tasks sized using Fibonacci story points. Velocity was monitored Sprint-over-Sprint to inform planning.

---

## 3. Requirements Engineering

### 3.1 Requirements Elicitation

#### 3.1.1 Methodology

Requirements were gathered through a combination of the following elicitation techniques, as recommended by Sommerville (2016) in *Software Engineering*:

1. **Stakeholder Interviews:** Semi-structured interviews were conducted with the company supervisor and senior developers at CodeVista Innovations to identify pain points related to meeting documentation in Agile teams. Key findings included:
   - Manual meeting note-taking is inconsistent across team members.
   - Action items from meetings are frequently lost or not tracked.
   - Formatting backlog items (especially user stories with proper "As a / I want / So that" structure) is tedious.
   - Teams desire a tool that captures both **explicit** action items and **implicit** work items derivable from decisions and blockers.

2. **Domain Research & Observation:** The intern observed sprint planning and retrospective meetings to understand the typical flow, vocabulary, and artifact types produced. Academic literature on Agile practices and meeting management was reviewed.

3. **User Stories:** Requirements were expressed as user stories following the standard template: *"As a [role], I want [action], so that [benefit]."* These were refined iteratively through Sprint Planning sessions and grooming exercises with acceptance criteria.

4. **Prototyping & Feedback:** Rapid prototyping was used to validate requirements. Early prototypes of the audio upload flow, extraction output format, and PDF layout were demonstrated to the supervisor, and feedback was incorporated into subsequent Sprint backlogs.

5. **Technology Spike Results:** Technical spikes (time-boxed investigations) were conducted during Sprint 0 to evaluate Whisper model accuracy across different sizes, Gemini vs. OpenAI extraction quality, and chunked vs. monolithic transcription strategies. The findings directly informed non-functional requirements related to performance and accuracy.

#### 3.1.2 Stakeholder Identification

| Stakeholder | Role | Interest |
|-------------|------|----------|
| Company Supervisor | Product Owner | Defines business requirements, validates deliverables, approves Sprint increments |
| Academic Supervisor | Advisor | Ensures academic rigor, evaluates engineering methodology and documentation |
| Agile Team Members (target users) | End Users | Primary beneficiaries — use the system to automate meeting documentation |
| Development Team (intern) | Developer | Designs, implements, tests, and deploys the system |

### 3.2 System Models

This section describes the conceptual models of the Meetolog system. Descriptions are provided in sufficient detail to facilitate the generation of formal UML diagrams.

#### 3.2.1 Use Case Diagram

**Description:** The Use Case Diagram identifies the primary actors and their interactions with the system.

**Actors:**

- **Meeting Participant (Primary Actor):** A team member who records or uploads a meeting audio file and consumes the output artifacts and PDF report.
- **Whisper Model (System Actor):** The local speech-to-text engine that performs transcription.
- **LLM Provider (System Actor):** The external AI service (Google Gemini or OpenAI GPT) that performs semantic extraction.
- **Redis (System Actor):** The persistence and queue infrastructure.

**Use Cases:**

| Use Case ID | Use Case Name | Actor(s) | Description |
|-------------|---------------|----------|-------------|
| UC-01 | Record Meeting Audio | Meeting Participant | The user records audio via the in-browser microphone recorder with live waveform visualization |
| UC-02 | Upload Audio File | Meeting Participant | The user uploads a pre-recorded audio file (MP3, WAV, M4A, OGG, or WebM) |
| UC-03 | Monitor Processing Progress | Meeting Participant | The user views real-time progress updates (pending → transcribing → extracting → generating PDF → completed) via polling |
| UC-04 | View Extracted Artifacts | Meeting Participant | The user views the structured artifacts (user stories, tasks, decisions, blockers, action items, execution tasks) in the web interface |
| UC-05 | Download PDF Summary | Meeting Participant | The user downloads a formatted PDF report containing all extracted artifacts |
| UC-06 | Transcribe Audio | Whisper Model | The system transcribes the audio into text using chunked processing |
| UC-07 | Extract Agile Artifacts | LLM Provider | The system sends the transcript to the LLM, which returns structured JSON artifacts |
| UC-08 | Persist Job State | Redis | The system stores and retrieves job metadata, transcripts, artifacts, and compressed audio in Redis |

**Relationships:**

- UC-01 and UC-02 are alternatives (both «include» a common sub-use-case: "Submit Audio for Processing").
- "Submit Audio for Processing" «include» → UC-06 (Transcribe Audio).
- UC-06 «include» → UC-07 (Extract Agile Artifacts).
- UC-07 «include» → "Generate PDF Report."
- UC-03, UC-04, and UC-05 are available after processing is complete.

> **[Placeholder: Insert Use Case Diagram here]**

#### 3.2.2 Architecture / Component Diagram

**Description:** The Architecture Diagram illustrates the high-level system components and their interconnections.

**Components:**

1. **«Component» Frontend (Next.js)**
   - Sub-components:
     - `VoiceRecorder` — Handles in-browser microphone recording (MediaRecorder API), waveform visualization (AnalyserNode), and recorded audio conversion to WAV.
     - `FileUploader` — Handles file selection, client-side audio downsampling via `OfflineAudioContext`, and upload submission.
     - `ProgressTracker` — Polls the backend `/status/{job_id}` endpoint and renders a progress bar.
     - `ResultsView` — Renders extracted artifacts (user stories, tasks, decisions, blockers, action items) with badges and structured formatting.
     - `API Client (lib/api.ts)` — HTTP client module for all backend communication, including upload, status polling, and PDF download URL generation.
     - `Audio Processor (lib/audio.ts)` — Client-side WAV encoding and downsampling utilities.

2. **«Component» Backend API (FastAPI)**
   - Sub-components:
     - `main.py` — Application entry point, CORS middleware, REST endpoint definitions (`/upload`, `/status/{job_id}`, `/artifacts/{job_id}`, `/download/{job_id}`, `/health`).
     - `config.py` — Pydantic Settings for environment variable management with validation.
     - `models.py` — Pydantic data models: `MeetingArtifacts`, `UserStory`, `Task`, `Decision`, `Blocker`, `ActionItem`, `ActionableTask`, `JobResponse`, `ProcessingStatus`.
     - `interfaces.py` — Abstract Base Classes: `JobStore`, `Transcriber`, `LLMExtractor`.
     - `dependencies.py` — Factory Pattern and Dependency Injection functions for creating service instances based on configuration (test mode vs. production).

3. **«Component» Background Worker (ARQ)**
   - Sub-components:
     - `worker.py` — ARQ worker definition with the `process_audio_job` task, implementing the four-stage processing pipeline (Transcription → Extraction → PDF Generation → Completion).

4. **«Component» Service Layer**
   - Sub-components:
     - `WhisperTranscriber` — Implements `Transcriber` interface; loads and caches the Whisper model, performs chunked audio transcription via `ffmpeg` segmentation.
     - `GeminiExtractor` — Implements `LLMExtractor` interface; calls the Google Gemini API with a structured extraction prompt.
     - `OpenAIExtractor` — Implements `LLMExtractor` interface; alternative provider using the OpenAI GPT API with function calling.
     - `LLMProvider` (abstract engine) — Unified abstraction over Gemini and OpenAI with common prompt template and response parsing.
     - `PDFGeneratorService` — Uses ReportLab to generate styled PDF reports with sections for each artifact type.
     - `MockTranscriber` / `MockExtractor` — Deterministic mock implementations for testing and CI/CD.

5. **«Component» Infrastructure Layer**
   - Sub-components:
     - `RedisJobStore` — Implements `JobStore` interface using Redis Hashes; manages job metadata, transcript caching, artifact caching, and compressed audio storage with configurable TTL.
     - `Redis Connection Pool` — Async Redis client management with connection pooling via `redis-py` and `hiredis`.

6. **«Component» External Services**
   - Google Gemini API (LLM extraction)
   - OpenAI API (alternative LLM extraction)

**Connectors:**

- Frontend → Backend API: HTTP/REST (JSON over HTTPS; upload via `multipart/form-data`)
- Backend API → Redis: TCP (Redis protocol) — Job creation, status queries
- Backend API → ARQ Queue: TCP (Redis protocol) — Job enqueueing
- ARQ Worker → Redis: TCP (Redis protocol) — Job state updates, transcript/artifact caching
- ARQ Worker → Whisper Model: In-process function call (Python)
- ARQ Worker → LLM Provider: HTTPS (Google/OpenAI API)
- ARQ Worker → File System: Local disk I/O (audio files, PDF output)

> **[Placeholder: Insert Component/Architecture Diagram here]**

#### 3.2.3 Sequence Diagram — Audio Upload and Processing Flow

**Description:** This sequence diagram illustrates the end-to-end flow from audio upload to artifact delivery.

**Participants:** User (Browser), Frontend (Next.js), Backend API (FastAPI), Redis, ARQ Queue, Worker Process, Whisper Model, Gemini API, PDF Generator.

**Flow:**

1. **User** → **Frontend**: Selects/records audio file.
2. **Frontend** (Audio Processor): Downsamples audio to 16 kHz mono WAV.
3. **Frontend** → **Backend API**: `POST /upload` (multipart/form-data with audio file).
4. **Backend API**: Validates file type and size.
5. **Backend API**: Generates `job_id` (UUID), saves file to disk.
6. **Backend API** → **Redis**: Creates job record (status: `pending`, progress: 0).
7. **Backend API** → **ARQ Queue**: Enqueues `process_audio_job(job_id, file_path, ...)`.
8. **Backend API** → **Frontend**: Returns `JobResponse { job_id, status: "pending", progress: 0 }`.
9. **Frontend**: Begins polling `GET /status/{job_id}` every 1 second.
10. **ARQ Worker** picks up job from queue.
11. **Worker** → **Redis**: Updates status to `transcribing`, progress: 10.
12. **Worker**: Compresses audio via `ffmpeg` and stores backup in Redis.
13. **Worker**: Splits audio into 5-minute chunks via `ffmpeg segment`.
14. **Worker** → **Whisper Model**: Transcribes each chunk incrementally.
15. **Worker** → **Redis**: Caches each chunk transcript (for resumability); updates progress (10–40%).
16. **Worker** → **Redis**: Stores complete transcript; updates status to `extracting`, progress: 40.
17. **Worker** → **Gemini API**: Sends transcript with extraction prompt.
18. **Gemini API** → **Worker**: Returns structured JSON artifacts.
19. **Worker**: Parses JSON into Pydantic models; validates schema.
20. **Worker** → **Redis**: Caches artifacts; updates progress: 75.
21. **Worker** → **Redis**: Updates status to `generating_pdf`, progress: 80.
22. **Worker** → **PDF Generator**: Generates styled PDF from `MeetingArtifacts`.
23. **Worker**: Saves PDF to disk.
24. **Worker** → **Redis**: Updates status to `completed`, progress: 100, sets `pdf_url` and `artifacts`.
25. **Frontend** (polling): Receives `status: "completed"` response with artifacts.
26. **Frontend**: Renders `ResultsView` with all extracted artifacts.
27. **User** → **Frontend**: Clicks "Download PDF".
28. **Frontend** → **Backend API**: `GET /download/{job_id}`.
29. **Backend API**: Returns PDF file as `application/pdf`.

> **[Placeholder: Insert Sequence Diagram here]**

#### 3.2.4 Class Diagram — Core Domain Model

**Description:** The class diagram captures the key domain entities and their relationships.

**Classes:**

- **`MeetingArtifacts`** (aggregate root)
  - Attributes: `meeting_id: UUID`, `meeting_title: str`, `meeting_date: datetime`, `duration_minutes: int?`, `participants: list[str]`, `summary: str`, `transcript: str`
  - Associations:
    - `1` → `*` `UserStory`
    - `1` → `*` `Task`
    - `1` → `*` `Decision`
    - `1` → `*` `Blocker`
    - `1` → `*` `ActionItem`
    - `1` → `*` `ActionableTask` (Execution Tasks)

- **`UserStory`**
  - Attributes: `id: UUID`, `title: str`, `as_a: str`, `i_want: str`, `so_that: str`, `acceptance_criteria: list[str]`, `priority: Priority`, `story_points: int?`

- **`Task`**
  - Attributes: `id: UUID`, `title: str`, `description: str`, `assignee: str?`, `priority: Priority`, `status: TaskStatus`, `due_date: str?`

- **`Decision`**
  - Attributes: `id: UUID`, `title: str`, `description: str`, `made_by: str?`, `rationale: str`, `timestamp: str?`

- **`Blocker`**
  - Attributes: `id: UUID`, `title: str`, `description: str`, `affected_tasks: list[str]`, `owner: str?`, `resolution_plan: str`

- **`ActionItem`**
  - Attributes: `id: UUID`, `description: str`, `assignee: str?`, `due_date: str?`

- **`ActionableTask`** (Execution Task)
  - Attributes: `title: str`, `description: str`, `owner_role: str`, `priority: Literal["High", "Medium", "Low"]`, `task_source: Literal["Explicit", "Inferred"]`, `dependencies: list[str]`

- **`JobResponse`**
  - Attributes: `job_id: UUID`, `status: ProcessingStatus`, `message: str`, `progress: int`, `artifacts: MeetingArtifacts?`, `pdf_url: str?`, `error: str?`

- **Enumerations:**
  - `Priority` { LOW, MEDIUM, HIGH, CRITICAL }
  - `TaskStatus` { TODO, IN_PROGRESS, BLOCKED, DONE }
  - `ProcessingStatus` { PENDING, TRANSCRIBING, EXTRACTING, GENERATING_PDF, COMPLETED, FAILED }

- **Interfaces (Abstract Base Classes):**
  - `«interface» JobStore`: `save()`, `load()`, `update()`, `exists()`, `delete()`
  - `«interface» Transcriber`: `transcribe(audio_path) → str`, `preprocess_transcript(raw) → str`
  - `«interface» LLMExtractor`: `extract_artifacts(transcript) → MeetingArtifacts`, `is_mock: bool`

- **Implementations:**
  - `RedisJobStore` implements `JobStore`
  - `WhisperTranscriber` implements `Transcriber`
  - `MockTranscriber` implements `Transcriber`
  - `GeminiExtractor` implements `LLMExtractor`
  - `OpenAIExtractor` implements `LLMExtractor`
  - `MockExtractor` implements `LLMExtractor`

> **[Placeholder: Insert Class Diagram here]**

### 3.3 Formal Specifications

#### 3.3.1 Functional Requirements

| Req. ID | Requirement | Priority | Description |
|---------|-------------|----------|-------------|
| FR-01 | Audio File Upload | High | The system shall accept audio file uploads in MP3, WAV, M4A, OGG, and WebM formats, with file size validation up to a configurable maximum (default: 100 MB). |
| FR-02 | In-Browser Audio Recording | High | The system shall provide an in-browser microphone recording interface with real-time waveform visualization, recording timer, and automatic conversion to 16 kHz mono WAV. |
| FR-03 | Client-Side Audio Downsampling | Medium | The frontend shall downsample uploaded audio files to 16 kHz mono WAV using the Web Audio API (`OfflineAudioContext`) before transmission to minimize payload size. |
| FR-04 | Speech-to-Text Transcription | High | The system shall transcribe uploaded audio into text using the OpenAI Whisper model running locally, with support for configurable model sizes (tiny, base, small, medium, large). |
| FR-05 | Chunked Transcription | Medium | The system shall split audio into fixed-duration chunks (default: 5 minutes) using `ffmpeg` and transcribe each chunk independently for resilience and lower peak memory usage. |
| FR-06 | Transcript Caching | Medium | The system shall cache partial and complete transcripts in Redis to enable pipeline resumability after worker restarts. |
| FR-07 | LLM-Based Artifact Extraction | High | The system shall submit the transcript to a configured LLM provider (Google Gemini or OpenAI GPT) and extract structured Agile artifacts as strictly valid JSON. |
| FR-08 | User Story Extraction | High | The system shall extract User Stories with the following fields: title, "As a" (role), "I want" (action), "So that" (benefit), acceptance criteria, priority, and story points (Fibonacci). |
| FR-09 | Task Extraction | High | The system shall extract Tasks with: title, description, assignee, priority (low/medium/high/critical), status, and due date. |
| FR-10 | Decision Extraction | High | The system shall extract Decisions with: title, description, decision maker, and rationale. |
| FR-11 | Blocker Extraction | High | The system shall extract Blockers with: title, description, affected tasks, owner, and resolution plan. |
| FR-12 | Action Item Extraction | High | The system shall extract Action Items with: description, assignee, and due date. |
| FR-13 | Execution Task Inference | High | The system shall generate AI-inferred Execution Tasks derived from both explicit statements and logical implications in the transcript, with owner role, priority, task source (Explicit/Inferred), and dependency tracking. The LLM shall be instructed to avoid hallucinating tasks for features not discussed. |
| FR-14 | PDF Report Generation | High | The system shall generate a downloadable PDF summary containing styled sections for meeting metadata, summary, user stories, tasks (tabular), decisions, blockers, action items, and execution tasks (with visual differentiation between explicit and inferred tasks). |
| FR-15 | Asynchronous Job Processing | High | Audio processing shall be performed asynchronously in a background worker (ARQ), with job state persisted in Redis. The API shall return immediately with a `job_id` for status polling. |
| FR-16 | Real-Time Progress Tracking | High | The system shall expose a `/status/{job_id}` endpoint returning the current processing stage and progress percentage (0–100%), which the frontend polls at 1-second intervals. |
| FR-17 | Artifact Retrieval via API | Medium | The system shall expose a `/artifacts/{job_id}` endpoint returning extracted artifacts as JSON for programmatic consumption. |
| FR-18 | Test Mode | Medium | The system shall support a `TEST_MODE` flag that replaces all external service calls (Whisper, Gemini) with deterministic mock implementations, enabling testing and CI/CD without API keys. |
| FR-19 | Multiple LLM Provider Support | Low | The system shall support both Google Gemini and OpenAI GPT as LLM providers, selectable via the `LLM_PROVIDER` configuration variable. |
| FR-20 | Health Check Endpoints | Low | The system shall expose `/` and `/health` endpoints returning the overall system health status, Redis connectivity, ARQ queue status, and configuration summary. |
| FR-21 | Graceful Degradation | Medium | If the configured LLM API key is missing, the system shall gracefully fall back to mock extraction with a warning, rather than failing. |

#### 3.3.2 Non-Functional Requirements

| Req. ID | Requirement | Category | Description |
|---------|-------------|----------|-------------|
| NFR-01 | Response Time | Performance | The API shall respond to upload requests within 2 seconds (excluding audio processing). Job status polling responses shall be returned within 200 ms. |
| NFR-02 | Processing Throughput | Performance | The system shall complete the full processing pipeline (transcription + extraction + PDF generation) for a 30-minute meeting recording within 10 minutes using the Whisper `base` model on a 4-core CPU. |
| NFR-03 | Memory Efficiency | Performance | Chunked transcription shall limit peak memory usage to below 4 GB during Whisper inference by processing 5-minute segments independently. |
| NFR-04 | Audio Compression | Performance | The system shall compress audio for Redis backup using Opus at 32 kbps, reducing storage requirements (e.g., 42 MB WAV → 5–8 MB). |
| NFR-05 | Extensibility | Maintainability | The system architecture shall use abstract interfaces and dependency injection to allow new service implementations (e.g., a Deepgram transcriber, a PostgreSQL job store) without modifying existing code, adhering to the Open/Closed Principle. |
| NFR-06 | Modularity | Maintainability | The codebase shall be organized into clearly separated layers (API, services, infrastructure) with single-responsibility modules. |
| NFR-07 | Configuration Management | Maintainability | All configurable parameters (API keys, model sizes, file size limits, directories, CORS origins, Redis URL) shall be managed via environment variables with Pydantic validation, default values, and type checking. |
| NFR-08 | Containerization | Deployability | The backend shall be fully containerizable via a multi-stage Dockerfile that produces a minimal runtime image with CPU-only PyTorch (saving approximately 3 GB compared to the CUDA-enabled distribution). |
| NFR-09 | Infrastructure as Code | Deployability | The system shall provide a `render.yaml` Blueprint for one-click deployment to Render PaaS, and a `docker-compose.prod.yml` for local production-like environments. |
| NFR-10 | Resilience | Reliability | The background worker shall support pipeline resumability: if a worker restarts, it shall resume from the last cached transcript chunk or completed stage, rather than restarting from scratch. |
| NFR-11 | Data Retention | Reliability | Job data in Redis shall have a configurable TTL (default: 7 days for completed jobs, 3 days for failed jobs), after which it is automatically evicted. |
| NFR-12 | Error Handling | Reliability | The system shall provide descriptive error messages for invalid file types, oversized uploads, missing API keys, Redis unavailability, and LLM parsing failures. The frontend shall display user-friendly error messages and handle polling failures with exponential back-off (up to 30 consecutive errors before reporting server loss). |
| NFR-13 | CORS Security | Security | The backend shall enforce CORS policies, restricting API access to configured allowed origins (default: `localhost:3000` for development; `meetolog.vercel.app` for production). |
| NFR-14 | API Key Protection | Security | API keys shall be loaded from environment variables or `.env` files and shall never be logged in their entirety. The health endpoint shall redact the Redis URL password. |
| NFR-15 | Frontend Responsiveness | Usability | The frontend user interface shall be responsive and functional on modern desktop browsers (Chrome, Firefox, Edge, Safari). |
| NFR-16 | Client-Side Privacy | Security | Audio downsampling shall occur client-side, ensuring that the raw high-fidelity audio is not transmitted over the network unnecessarily. |

---

## Technology Stack Summary

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Frontend Framework** | Next.js | 16.x | React-based SSR/SPA framework with App Router |
| **Frontend Language** | TypeScript | 5.3+ | Type-safe frontend development |
| **Frontend Styling** | CSS Modules | — | Scoped component styling |
| **Backend Framework** | FastAPI | 0.109+ | Async Python REST API framework |
| **Backend Language** | Python | 3.12+ | Backend runtime |
| **Data Validation** | Pydantic | 2.5+ | Type-safe data models and configuration |
| **Speech-to-Text** | OpenAI Whisper | 20231117+ | Local speech-to-text transcription |
| **LLM (Primary)** | Google Gemini (gemini-2.5-flash-lite) | Latest | Semantic artifact extraction |
| **LLM (Alternative)** | OpenAI GPT | Latest | Alternative extraction provider |
| **PDF Generation** | ReportLab | 4.0+ | Styled PDF report creation |
| **Message Queue** | ARQ (Async Redis Queue) | 0.26+ | Background job processing |
| **State Store** | Redis | 7+ | Job persistence, caching, and queue |
| **Audio Processing** | FFmpeg | Latest | Audio splitting, compression, format conversion |
| **Containerization** | Docker | — | Multi-stage production builds |
| **Orchestration** | Docker Compose | — | Local multi-service environment |
| **Cloud PaaS** | Render | — | Production deployment (Blueprint IaC) |
| **Frontend Hosting** | Vercel | — | Next.js frontend deployment |

---

## References

- Beck, K., Beedle, M., van Bennekum, A., et al. (2001). *Manifesto for Agile Software Development*. Retrieved from https://agilemanifesto.org
- Pagotto, T., Fabri, J. A., Lerario, A., & Gonçalves, J. A. (2016). Scrum Solo: Software Process for Individual Development. *2016 11th Iberian Conference on Information Systems and Technologies (CISTI)*, 1–6.
- Pressman, R. S., & Maxim, B. R. (2020). *Software Engineering: A Practitioner's Approach* (9th ed.). McGraw-Hill.
- Schwaber, K., & Sutherland, J. (2020). *The Scrum Guide*. Scrum.org.
- Sommerville, I. (2016). *Software Engineering* (10th ed.). Pearson.

---

*This report was prepared as part of the capstone internship project at CodeVista Innovations. All technical details are based on the Meetolog source code repository and project documentation.*
