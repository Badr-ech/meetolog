# Meetolog - Meeting to Backlog

Transform meeting audio recordings into structured Agile artifacts using AI-powered semantic extraction.

## 🎯 Overview

Meetolog is an MVP system that:
1. **Accepts** audio uploads (MP3, WAV, M4A, OGG, WebM)
2. **Transcribes** speech to text using OpenAI Whisper (local model)
3. **Extracts** Agile artifacts using Google Gemini LLM:
   - User Stories (with acceptance criteria)
   - Tasks (with assignments and priorities)
   - Decisions (with rationale)
   - Blockers (with resolution plans)
   - Action Items
4. **Generates** a downloadable PDF summary

---

## ⚠️ Deployment Constraints (IMPORTANT)

### 1. Single Instance Only
This MVP **must run as a single instance** (no horizontal scaling). The application uses:
- In-memory job state with local file backup
- Local file storage for uploads and generated PDFs

**Do not** use `workers > 1` in uvicorn or deploy multiple pods/containers. Data loss and race conditions will occur.

### 2. System Dependencies
The following must be installed on the host system:

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **ffmpeg** | Audio processing for Whisper | `apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) / [ffmpeg.org](https://ffmpeg.org/download.html) (Windows) |
| **Python 3.12+** | Runtime | [python.org](https://www.python.org/downloads/) |
| **Node.js 18+** | Frontend | [nodejs.org](https://nodejs.org/) |

### 3. Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 4 GB | 8+ GB (for Whisper medium model) |
| **GPU** | Not required | CUDA-compatible GPU (10x faster transcription) |
| **Storage** | 5 GB | 20+ GB (for audio uploads and models) |

**Whisper Model Sizes:**
| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~1 GB | Fastest | Lower |
| `base` | ~1 GB | Fast | Good (default) |
| `small` | ~2 GB | Moderate | Better |
| `medium` | ~5 GB | Slow | High |
| `large` | ~10 GB | Slowest | Highest |

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TEST_MODE` | No | `false` | Enable mock services (no API calls, deterministic output) |
| `GEMINI_API_KEY` | No* | `""` | Google Gemini API key for LLM extraction |
| `WHISPER_MODEL` | No | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `DEBUG` | No | `false` | Enable debug logging |
| `MAX_UPLOAD_SIZE_MB` | No | `100` | Maximum audio file upload size |
| `UPLOAD_DIR` | No | `uploads` | Temporary upload directory |
| `OUTPUT_DIR` | No | `outputs` | PDF and job state output directory |

\* If `GEMINI_API_KEY` is not set, the application will automatically use mock extraction (deterministic test data).

### Test Mode (CI/CD)

Set `TEST_MODE=true` to run the application without any external API calls:

```bash
# Enable test mode
export TEST_MODE=true

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In test mode:
- **MockTranscriber** returns a realistic hardcoded transcript instantly
- **MockExtractor** returns deterministic Agile artifacts matching the schema
- No Whisper model loading (faster startup)
- No Gemini API calls (no API key required)
- Perfect for CI/CD pipelines and automated testing

---

## 🏗️ Project Structure

```
meetolog/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI entry point & endpoints
│   │   ├── config.py            # Environment configuration
│   │   ├── models.py            # Pydantic models for Agile artifacts
│   │   ├── interfaces.py        # Abstract base classes (JobStore, Transcriber, LLMExtractor)
│   │   ├── dependencies.py      # Factory pattern & dependency injection
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── transcription.py    # WhisperTranscriber (production)
│   │       ├── llm_extraction.py   # GeminiExtractor (production)
│   │       ├── pdf_generator.py    # ReportLab PDF generation
│   │       ├── job_store.py        # LocalJobStore (in-memory + file backup)
│   │       └── mock_services.py    # MockTranscriber & MockExtractor (testing)
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx           # Root layout
│   │   ├── page.tsx             # Main upload & results page
│   │   ├── page.module.css      # Page styles
│   │   └── globals.css          # Global styles
│   ├── lib/
│   │   └── api.ts               # Backend API utilities
│   ├── package.json
│   ├── next.config.js           # API rewrites for CORS
│   └── tsconfig.json
│
├── AI_CONTEXT.md                # AI coding context
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- ffmpeg (for audio processing)
- Google Gemini API key (optional - works with mock data without it)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and configure:
#   GEMINI_API_KEY=your_api_key_here  (optional)
#   TEST_MODE=true                     (for testing without APIs)

# Run the server (IMPORTANT: single worker only!)
uvicorn app.main:app --reload --port 8000 --workers 1
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload audio file, returns job ID |
| `GET` | `/status/{job_id}` | Get processing status |
| `GET` | `/artifacts/{job_id}` | Get extracted artifacts as JSON |
| `GET` | `/download/{job_id}` | Download PDF summary |

### Example: Upload Audio

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@meeting.mp3"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "File uploaded, processing starting...",
  "progress": 0
}
```

### Example: Check Status

```bash
curl "http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000"
```

## 🔧 Configuration

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

> **Note:** The system works without an API key using mock extraction for testing. Set `TEST_MODE=true` or simply omit the API key.

---

## 🧪 Testing the MVP

### With Mock Services (Recommended for First Test)

1. Set `TEST_MODE=true` in your `.env` file
2. Start both backend and frontend
3. Open `http://localhost:3000`
4. Upload any audio file (transcription is mocked)
5. Watch the progress as it processes
6. View the deterministic extracted artifacts
7. Download the PDF summary

### With Real Services

1. Ensure `ffmpeg` is installed
2. Set `GEMINI_API_KEY` in your `.env` file
3. Remove or set `TEST_MODE=false`
4. Upload a real meeting audio file
5. Wait for Whisper transcription (may take a few minutes)
6. View the AI-extracted artifacts

---

## 📦 Key Dependencies

### Backend
- **FastAPI** - Modern async web framework
- **Pydantic** - Data validation with type hints
- **OpenAI Whisper** - Local speech-to-text model
- **google-generativeai** - Gemini LLM integration
- **ReportLab** - PDF generation
- **aiofiles** - Async file I/O

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **CSS Modules** - Scoped styling

---

## 🔮 Future Enhancements (Version 2)

- [ ] **RedisJobStore** - Replace LocalJobStore for horizontal scaling
- [ ] **User authentication** - Multi-user support
- [ ] **PostgreSQL persistence** - Replace local file storage
- [ ] **Export to Jira/Azure DevOps** - Direct integration
- [ ] **Real-time WebSocket updates** - Replace polling
- [ ] **Multiple language support** - Whisper already supports this
- [ ] **GPU acceleration** - Faster Whisper transcription

---

## 📝 Architecture Notes

### Service Abstraction Layer
The backend uses an interface-based architecture for easy testing and future extensibility:

- **`JobStore`** (interface) → `LocalJobStore` (MVP) → `RedisJobStore` (V2)
- **`Transcriber`** (interface) → `WhisperTranscriber` (production) / `MockTranscriber` (test)
- **`LLMExtractor`** (interface) → `GeminiExtractor` (production) / `MockExtractor` (test)

### Factory Pattern
Services are instantiated via `dependencies.py` based on configuration:
- `TEST_MODE=true` → All mock services
- `GEMINI_API_KEY` missing → Mock LLM extractor with warning
- Production mode → Real Whisper + Gemini

### Async Processing
The backend uses FastAPI's `BackgroundTasks` for non-blocking audio processing. Job state is persisted to disk for crash recovery.

### CORS Handling
The frontend uses Next.js rewrites in `next.config.js` to proxy API calls, avoiding CORS issues during development.

---

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for the full text.

Note: you indicated you will keep the repository private when publishing — that is the recommended approach
if you plan to sell the software or keep full commercial control. The `LICENSE` file grants permission
for others to use and redistribute the code if the repository is ever made public.
