# AI Video Summarizer

AI Video Summarizer is a local-first application that generates structured summaries from uploaded videos using Whisper transcription and Ollama-based LLM summarization.

## What it does

- Accepts video uploads through the existing UI and API.
- Extracts audio with ffmpeg.
- Transcribes speech locally with Whisper.
- Builds a grounded summary using transcript plus lightweight visual context.
- Returns a strict JSON response used directly by the frontend.

## Current backend behavior

- Endpoint is unchanged: `POST /api/analyze-video`.
- Alias endpoint is available: `POST /api/upload`.
- Summary response format:

```json
{
	"main_idea": "...",
	"detailed_summary": "...",
	"key_insights": ["...", "..."],
	"highlights": ["...", "..."],
	"final_takeaway": "..."
}
```

## Local run (one URL)

```powershell
cd "c:\Users\Saloni\OneDrive\Desktop\AI Video Summarizer"
.\scripts\start-one-url.ps1
```

Open:

- `http://localhost:3000`
- `http://localhost:3000/api/docs`

Stop services:

```powershell
cd "c:\Users\Saloni\OneDrive\Desktop\AI Video Summarizer"
.\scripts\stop-one-url.ps1
```

## Manual backend run

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

## Core environment variables

- `WHISPER_MODEL` (default: `base`)
- `OLLAMA_MODEL` (default: `llama3`)
- `OLLAMA_API_URL` (default: `http://localhost:11434`)
- `FFMPEG_PATH` (optional absolute path to ffmpeg)
- `FFPROBE_PATH` (optional absolute path to ffprobe)
- `PIPELINE_TIMEOUT_SECONDS` (default: `300`)
- `TRANSCRIPTION_TIMEOUT_SECONDS` (default: `240`)
- `OLLAMA_TIMEOUT_SECONDS` (default: `120`)

## API quick check

```powershell
curl.exe -X POST "http://localhost:3000/api/upload" -F "file=@backend/tests/data/sample_test_video.mp4"
```

## Notes

- If a video has no speech/audio, the backend still returns a visual-only summary.
- The backend always returns a response (success or JSON error) to avoid infinite loading in the UI.
- No frontend changes are required for the current backend contract.