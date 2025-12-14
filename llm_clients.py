"""
LLM client wrappers using official SDKs.
"""

import json
import httpx
from openai import OpenAI
from anthropic import Anthropic
import config


# Initialize clients
anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


async def call_claude(prompt: str, system: str = None) -> str:
    """Call Anthropic Claude API."""
    kwargs = {
        "model": config.CLAUDE_MODEL,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}]
    }
    if system:
        kwargs["system"] = system
    
    response = anthropic_client.messages.create(**kwargs)
    return response.content[0].text


async def call_gpt(prompt: str, system: str = None) -> str:
    """Call OpenAI GPT API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = openai_client.chat.completions.create(
        model=config.GPT_MODEL,
        messages=messages,
        max_tokens=2048
    )
    return response.choices[0].message.content


async def call_gemini(prompt: str) -> str:
    """Call Google Gemini API (still using httpx - no official sync SDK)."""
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


async def call_deepseek(prompt: str, system: str = None) -> str:
    """Call DeepSeek API (OpenAI-compatible)."""
    deepseek_client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = deepseek_client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        max_tokens=2048
    )
    return response.choices[0].message.content


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
