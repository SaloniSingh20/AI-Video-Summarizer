from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

BASE_PROMPT = """You are an expert video analyst.

Analyze the video content (transcript + visual cues) and generate a rich, human-like description.

IMPORTANT:

* Only use the provided transcript and visual context.
* Do NOT guess missing information.
* If the available data is insufficient, explicitly say that details are limited.

IMPORTANT RULES:

* Focus on REAL-WORLD objects and environment (e.g., waterfall, mountains, trees, sky)
* Describe the setting, atmosphere, and motion
* DO NOT mention colors numerically (no RGB or technical stats)
* DO NOT mention timestamps
* DO NOT generate generic or technical output

SPECIAL CASE:

* If no audio is detected, mention ONLY ONCE: 'No audio detected in the video.'
* Do not repeat this multiple times

OUTPUT FORMAT:

1. Main Idea
2. Detailed Summary (very descriptive, natural language)
3. Key Insights (content-based, not technical)
4. Highlights (scene-based, not timestamps)
5. Final Takeaway

Make the output feel like a human describing the video.

Transcript:
{transcript}

Visual Cues:
{visual_cues}

No Audio:
{no_audio_hint}
"""

JSON_INSTRUCTION = (
    "Return ONLY valid JSON with keys: main_idea, detailed_summary, key_insights, highlights, final_takeaway.\n"
    "Requirements:\n"
    "- Ground the summary strictly in the provided transcript and visual cues.\n"
    "- Only use provided transcript/context and do not infer unseen entities.\n"
    "- If context is insufficient, clearly state that details are limited.\n"
    "- Use concrete details (entities, actions, setting, motion) from this exact input.\n"
    "- Do not mention timestamps.\n"
    "- Do not include numeric color values, RGB-like values, or technical visual metrics.\n"
    "- Keep language natural and human-like.\n"
    "- key_insights and highlights must be arrays of strings.\n"
    "- Do not include any keys other than the required five keys.\n\n"
)


class SummarizationService:
    def __init__(self) -> None:
        self.ollama_url = (os.getenv("OLLAMA_API_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3").strip() or "llama3"
        self.ollama_timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120") or "120")

    def summarize(self, transcript: str, visual_cues: list[str] | None = None, no_audio: bool = False) -> dict:
        visual_context = "\n".join(f"- {item}" for item in (visual_cues or [])) or "- No visual cues available."
        no_audio_hint = "No audio detected in the video." if no_audio else "Audio transcript is available."

        if no_audio:
            return self._build_visual_only_summary(visual_cues or [])

        prompt = BASE_PROMPT.format(
            transcript=transcript.strip() or "",
            visual_cues=visual_context,
            no_audio_hint=no_audio_hint,
        )

        logger.info("summary started model=%s", self.model)
        logger.info("summarization_provider=ollama model=%s", self.model)
        summary = self._summarize_with_ollama(prompt)
        if no_audio:
            summary = self._apply_no_audio_note_once(summary)
        logger.info("summary completed")
        return summary

    @staticmethod
    def _build_visual_only_summary(visual_cues: list[str]) -> dict:
        filtered = [cue.strip() for cue in visual_cues if cue and cue.strip()]
        if not filtered:
            filtered = ["The available visual context is limited."]

        main_idea = "Visual-only summary based on detected scene context"
        detailed_summary = "No audio detected in the video. " + " ".join(filtered)
        key_insights = [
            "The summary is derived from sampled visual context only.",
            filtered[0],
        ]
        highlights = filtered[:2]
        final_takeaway = "The video can be described only from visible scene cues because no speech transcript is available."

        return {
            "main_idea": main_idea,
            "detailed_summary": detailed_summary,
            "key_insights": key_insights,
            "highlights": highlights,
            "final_takeaway": final_takeaway,
        }

    @staticmethod
    def _apply_no_audio_note_once(summary: dict) -> dict:
        note = "No audio detected in the video."

        detailed_summary = str(summary.get("detailed_summary") or "").strip()
        if note.lower() not in detailed_summary.lower():
            summary["detailed_summary"] = f"{note} {detailed_summary}".strip()

        for key in ["main_idea", "final_takeaway"]:
            text = str(summary.get(key) or "").strip()
            if note.lower() in text.lower():
                summary[key] = text.replace(note, "").strip(" .")

        def _remove_note_from_list(items: list[str]) -> list[str]:
            cleaned: list[str] = []
            for item in items:
                text = str(item).strip()
                if note.lower() in text.lower():
                    continue
                if text:
                    cleaned.append(text)
            return cleaned

        summary["key_insights"] = _remove_note_from_list(summary.get("key_insights") or [])
        summary["highlights"] = _remove_note_from_list(summary.get("highlights") or [])

        if not summary["key_insights"]:
            summary["key_insights"] = ["The visual sequence carries the core meaning of the video."]
        if not summary["highlights"]:
            summary["highlights"] = ["The scene progression is conveyed primarily through visual storytelling."]

        return summary

    def _summarize_with_ollama(self, prompt: str) -> dict:
        request_payload: dict[str, Any] = {
            "model": self.model,
            "prompt": f"{JSON_INSTRUCTION}{prompt}",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "repeat_penalty": 1.1,
            },
        }

        body = json.dumps(request_payload).encode("utf-8")
        request = Request(
            url=f"{self.ollama_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.ollama_timeout_seconds) as response:
                status_code = response.status
                response_text = response.read().decode("utf-8", errors="replace")
        except URLError as exc:
            raise RuntimeError(f"Failed to connect to Ollama at {self.ollama_url}: {exc}") from exc

        if status_code >= 400:
            raise RuntimeError(f"Ollama request failed ({status_code}): {response_text[:500]}")

        try:
            response_json = json.loads(response_text)
        except ValueError as exc:
            raise RuntimeError("Ollama returned non-JSON response") from exc

        raw_text = (response_json.get("response") or "").strip()
        logger.info("ollama_response=%s", raw_text)

        return self._parse_model_json(raw_text)

    def _parse_model_json(self, raw_text: str) -> dict:
        cleaned_text = raw_text.strip()
        if "```" in cleaned_text:
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()

        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned_text = cleaned_text[start : end + 1]

        try:
            payload = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse model JSON response: {exc}") from exc

        main_idea = str(payload.get("main_idea") or "").strip()
        detailed_summary = str(payload.get("detailed_summary") or "").strip()
        final_takeaway = str(payload.get("final_takeaway") or "").strip()

        def _to_string_list(value: Any) -> list[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str):
                lines = [line.strip(" -\t") for line in value.splitlines()]
                return [line for line in lines if line]
            return []

        key_insights = _to_string_list(payload.get("key_insights"))
        highlights = _to_string_list(payload.get("highlights"))

        if not main_idea:
            raise RuntimeError("Model response missing main_idea")
        if not detailed_summary:
            raise RuntimeError("Model response missing detailed_summary")
        if not final_takeaway:
            raise RuntimeError("Model response missing final_takeaway")
        if not key_insights:
            raise RuntimeError("Model response missing key_insights")
        if not highlights:
            raise RuntimeError("Model response missing highlights")

        return {
            "main_idea": main_idea,
            "detailed_summary": detailed_summary,
            "key_insights": key_insights,
            "highlights": highlights,
            "final_takeaway": final_takeaway,
        }
