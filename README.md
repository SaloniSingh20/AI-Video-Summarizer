# SUMMAR-AI

## Local-First Video Summarization System

SUMMAR-AI is a local-first AI application that generates structured summaries from video content using Whisper for speech transcription and LLaMA 3 (via Ollama) for summarization. The system runs entirely offline, ensuring data privacy, low latency, and zero dependency on external APIs.

---

## Overview

The platform processes uploaded videos through a multimodal pipeline. It extracts audio, transcribes speech into text, and generates context-aware summaries using a locally hosted large language model. The output follows a strict JSON schema, enabling seamless and direct frontend integration.

---

## Key Features

* End-to-end local AI pipeline with no external API calls
* Audio extraction and preprocessing using FFmpeg
* Speech-to-text transcription using Whisper
* Context-aware summarization using LLaMA 3 via Ollama
* Structured JSON output for direct UI consumption
* Graceful handling of videos with no audio input
* Timeout-controlled execution for stability and reliability
* Modular backend design for easy scalability and upgrades

---

## System Workflow

1. Video uploaded via UI or API
2. Audio extracted using FFmpeg
3. Whisper transcribes speech into text
4. Transcript is processed and contextualized
5. Ollama (LLaMA 3) generates structured summary
6. JSON response returned to frontend

---

## Tech Stack

### Backend

* Python
* FastAPI (REST API framework)
* Uvicorn (ASGI server)

### AI / Machine Learning

* Whisper (speech recognition and transcription)
* Ollama (local LLM serving framework)
* LLaMA 3 (text summarization model)

### Media Processing

* FFmpeg (audio extraction and preprocessing)
* FFprobe (video metadata handling)

### Frontend

* React / Next.js (pre-built interface)
* REST API integration

### DevOps / Tooling

* PowerShell scripts for one-command startup
* Python virtual environment (.venv)
* Local model hosting (no cloud dependency)

---

## API

**POST /api/analyze-video**
**POST /api/upload**

**Response:**

```json id="h6y1bn"
{
  "main_idea": "...",
  "detailed_summary": "...",
  "key_insights": ["...", "..."],
  "highlights": ["...", "..."],
  "final_takeaway": "..."
}
```

---

## Run Locally

```powershell id="5t3knt"
cd "c:\Users\Saloni\OneDrive\Desktop\AI Video Summarizer"
.\scripts\start-one-url.ps1
```

---

## Impact

* Eliminated dependency on external APIs, reducing operational cost to zero
* Built a fully local multimodal AI pipeline combining audio processing and LLM inference
* Designed structured outputs enabling seamless frontend integration
* Improved system reliability with timeout handling and fault-tolerant execution

---

## Resume Points

* Developed a local-first video summarization system using Whisper and LLaMA 3
* Built a FastAPI backend integrating audio processing, transcription, and LLM inference
* Designed structured JSON output pipeline for direct UI consumption
* Implemented a fault-tolerant architecture with timeout controls and reliable API responses

---

## Author

Saloni Singh
