"""
LLM client wrappers for Claude, GPT, Gemini, and DeepSeek.
"""

import httpx
import json
from typing import Optional
import config


async def call_claude(prompt: str, system: Optional[str] = None) -> str:
    """Call Anthropic Claude API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": config.CLAUDE_MODEL,
            "max_tokens": 2048,
            "messages": messages
        }
        if system:
            payload["system"] = system
        
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json=payload
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


async def call_gpt(prompt: str, system: Optional[str] = None) -> str:
    """Call OpenAI GPT API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": config.GPT_MODEL,
                "messages": messages,
                "max_tokens": 2048
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


async def call_gemini(prompt: str) -> str:
    """Call Google Gemini API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": config.GOOGLE_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 2048}
            }
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


async def call_deepseek(prompt: str, system: Optional[str] = None) -> str:
    """Call DeepSeek API (OpenAI-compatible)."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": config.DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": 2048
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())
