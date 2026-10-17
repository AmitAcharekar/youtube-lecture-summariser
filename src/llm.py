"""Thin wrapper around a local Ollama model.

Everything runs on your own machine: no API key, no account, no per-token cost.
Start Ollama (the desktop app, or `ollama serve`) and pull a model first:

    ollama pull llama3.1:8b

Only `requests` is needed, so there are no vendor SDKs to install.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import requests

DEFAULT_MODEL = "llama3.1:8b"

# Models worth trying, roughly fastest to best. Any model you have pulled works.
SUGGESTED_MODELS = [
    "llama3.2:3b",   # fastest, fine for notes, weaker quiz questions
    "llama3.1:8b",   # good balance on an M-series laptop
    "qwen2.5:7b",
    "mistral:7b",
    "gemma2:9b",
]


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns something unusable."""


def extract_json(text: str) -> dict | list:
    """Pull a JSON object out of a model response that may be wrapped in prose or fences."""
    if not text or not text.strip():
        raise LLMError("Model returned an empty response.")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Could not parse JSON from model output: {text[:300]}...")


@dataclass
class LLM:
    """A local Ollama model, used through its chat endpoint."""

    model: str | None = None
    temperature: float = 0.2
    timeout: int = 600
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.model = self.model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.base_url = (self.base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    # ------------------------------------------------------------------ public

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        """Send one prompt, return the raw text response."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"  # Ollama's native JSON mode

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
                if response.status_code == 404:
                    raise LLMError(
                        f"Ollama does not have the model '{self.model}'. Pull it first: ollama pull {self.model}"
                    )
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "")
            except requests.ConnectionError as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))  # Ollama may still be starting up
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))

        raise LLMError(
            f"Could not reach Ollama at {self.base_url} after 3 attempts. "
            f"Is the Ollama app running? ({last_error})"
        )

    def json(self, system: str, user: str, retries: int = 2) -> dict | list:
        """Send one prompt and insist on valid JSON back."""
        prompt = user
        for attempt in range(retries + 1):
            raw = self.complete(system, prompt, json_mode=True)
            try:
                return extract_json(raw)
            except LLMError:
                if attempt == retries:
                    raise
                # Smaller models sometimes add commentary; ask again, more firmly.
                prompt = user + "\n\nIMPORTANT: respond with raw JSON only. No prose, no markdown fences."
        raise LLMError("Unreachable")  # pragma: no cover

    def health_check(self) -> tuple[bool, str]:
        """Used by the UI to report problems before the user waits several minutes."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"Cannot reach Ollama at {self.base_url}. Open the Ollama app "
                f"(or run `ollama serve`) and try again. ({exc})"
            )

        installed = [m.get("name", "") for m in response.json().get("models", [])]
        if not installed:
            return False, f"Ollama is running but has no models. Run: ollama pull {self.model}"
        if self.model not in installed and not any(m.split(":")[0] == self.model.split(":")[0] for m in installed):
            return False, (
                f"Ollama is running but '{self.model}' is not pulled. "
                f"Run `ollama pull {self.model}`, or pick one of: {', '.join(installed)}"
            )
        return True, f"Ollama ready — {self.model}"

    def list_models(self) -> list[str]:
        """Models currently pulled on this machine, for the UI dropdown."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return sorted(m.get("name", "") for m in response.json().get("models", []) if m.get("name"))
        except Exception:  # noqa: BLE001
            return []
