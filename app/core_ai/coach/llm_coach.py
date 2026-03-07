import os
import json
import httpx
import logging
from dotenv import load_dotenv, find_dotenv
from typing import AsyncGenerator

# Load nearest .env
_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(_dotenv_path, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL = os.getenv("XAI_MODEL", "")

if GEMINI_API_KEY and "REPLACE" not in GEMINI_API_KEY:
    _base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    _api_key = GEMINI_API_KEY
    _model = GEMINI_MODEL
elif XAI_API_KEY and "REPLACE" not in XAI_API_KEY:
    _base_url = "https://api.x.ai/v1"
    _api_key = XAI_API_KEY
    _model = XAI_MODEL or "grok-3-mini-beta"
elif GROQ_API_KEY and "REPLACE" not in GROQ_API_KEY:
    _base_url = "https://api.groq.com/openai/v1"
    _api_key = GROQ_API_KEY
    _model = "llama-3.1-8b-instant"
else:
    _base_url = ""
    _api_key = ""
    _model = ""

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an energetic, high-performance AI fitness coach.
Provide rapid, proactive, and actionable feedback based on real-time data.
Be motivating, direct, and slightly competitive. Push the user to their limits.
Keep responses lightning-fast (max 2 sentences), no markdown, no lists. Just raw, high-impact verbal cues.
"""

PERSONA_PROMPTS = {
    "general": SYSTEM_PROMPT,
    "drill_sergeant": """
You are a hardcore Drill Sergeant AI Coach. No excuses. No mercy.
Short, aggressive, high-energy commands. Demand 100% effort.
Max 2 sentences. Use forceful language.
""",
    "zen_coach": """
You are a mindful Zen Alignment Coach. Focus on fluid energy and balance.
Calm but firm. Guide with steady, encouraging flow.
Max 2 sentences.
""",
}


async def ask_llm_async(
    fitness_summary: str, user_message: str = None, persona: str = "general"
) -> str:
    if not _api_key:
        raise RuntimeError("No AI API key configured for LLM coach")

    system_prompt = PERSONA_PROMPTS.get(persona, SYSTEM_PROMPT)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Fitness summary: {fitness_summary}\nUser message: {user_message}",
        },
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": _model, "messages": messages, "temperature": 0.6},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


async def stream_llm_async(
    fitness_summary: str, user_message: str = None, persona: str = "general"
) -> AsyncGenerator[str, None]:
    if not _api_key:
        raise RuntimeError("No AI API key configured for LLM coach")

    system_prompt = PERSONA_PROMPTS.get(persona, SYSTEM_PROMPT)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Fitness summary: {fitness_summary}\nUser message: {user_message}",
        },
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            f"{_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": _model, "messages": messages, "temperature": 0.6, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        pass
