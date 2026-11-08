"""OpenAI-compatible LLM adapter (Groq, OpenRouter, OpenAI, or any compatible endpoint).

Configuration (environment variables, never committed):
  LLM_PROVIDER   groq | openrouter | openai | custom | none     (default: auto-detect from keys)
  LLM_API_KEY    or GROQ_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY
  LLM_MODEL      default per provider (Groq: llama-3.3-70b-versatile)
  LLM_BASE_URL   required for provider=custom
If no key is configured the system runs in deterministic offline mode (template synthesis).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

log = logging.getLogger("agent.llm")

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
}


class LLM:
    def __init__(self):
        provider = os.environ.get("LLM_PROVIDER", "").lower()
        if not provider:
            provider = next((p for p, (_, _, k) in PROVIDERS.items() if os.environ.get(k)), "none")
        self.provider = provider
        base, model, key_env = PROVIDERS.get(provider, (os.environ.get("LLM_BASE_URL"), None, "LLM_API_KEY"))
        self.base_url = os.environ.get("LLM_BASE_URL") or base
        self.model = os.environ.get("LLM_MODEL") or model
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get(key_env or "", "")
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
        self.client = None
        if provider != "none" and self.api_key and self.base_url and self.model:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45, max_retries=2)

    @property
    def available(self) -> bool:
        return self.client is not None

    def describe(self) -> dict:
        return {"provider": self.provider if self.available else "none (offline template mode)",
                "model": self.model if self.available else None}

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 900,
             json_mode: bool = False):
        if not self.client:
            raise RuntimeError("LLM not configured")
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": max_tokens,
                  "seed": int(os.environ.get("SEED", "42"))}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "seed" in str(exc).lower() or "response_format" in str(exc).lower():
                kwargs.pop("seed", None)
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        log.info("LLM call %.0f ms", (time.time() - t0) * 1000)
        return resp.choices[0].message

    def complete_json(self, system: str, user: str, max_tokens: int = 400) -> dict | None:
        try:
            msg = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                            max_tokens=max_tokens, json_mode=True)
            text = msg.content or ""
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM JSON extraction failed: %s", exc)
            return None
