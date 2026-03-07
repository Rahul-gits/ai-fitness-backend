import os
import sys
import httpx
import json
import logging
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any

# Load the nearest .env, overriding any already-set vars
_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(_dotenv_path, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

logger = logging.getLogger(__name__)

sys.stderr.write(
    f"[lifestyle_bot] GEMINI={'SET' if GEMINI_API_KEY else 'NOT SET'} model={GEMINI_MODEL}\n"
)
sys.stderr.flush()


class AIClient:
    """Unified async AI client. Uses Gemini native REST API as primary."""

    def __init__(self):
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        self._gemini_ok = bool(GEMINI_API_KEY and "REPLACE" not in GEMINI_API_KEY)
        self._xai_ok = bool(XAI_API_KEY and "REPLACE" not in XAI_API_KEY)
        self._groq_ok = bool(GROQ_API_KEY and "REPLACE" not in GROQ_API_KEY)

        if self._gemini_ok:
            self.provider = "gemini"
            self.model = GEMINI_MODEL
        elif self._xai_ok:
            self.provider = "xai"
            self.model = os.getenv("XAI_MODEL", "grok-3-mini-beta")
        elif self._groq_ok:
            self.provider = "groq"
            self.model = "llama-3.1-8b-instant"
        else:
            self.provider = None
            self.model = ""

        sys.stderr.write(f"[lifestyle_bot] provider={self.provider} model={self.model}\n")
        sys.stderr.flush()

    @property
    def available(self) -> bool:
        return self.provider is not None

    async def chat_completion(
        self, messages, temperature=0.7, max_tokens=500, response_format=None, json_mode=False
    ):
        if not self.available:
            raise Exception("No AI API client initialized")

        if self.provider == "gemini":
            return await self._gemini_request(messages, temperature, max_tokens, json_mode)
        else:
            return await self._openai_compat_request(messages, temperature, max_tokens, response_format)

    async def chat_stream(self, messages, temperature=0.7, max_tokens=500):
        if not self.available:
            yield "AI Client unavailable."
            return

        if self.provider == "gemini":
            # Native Gemini stream not implemented here for brevity, fallback or implement if needed
            res = await self._gemini_request(messages, temperature, max_tokens)
            yield res["choices"][0]["message"]["content"]
        else:
            async for chunk in self._openai_compat_stream(messages, temperature, max_tokens):
                yield chunk

    async def _openai_compat_stream(self, messages, temperature, max_tokens):
        if self.provider == "xai":
            base_url = "https://api.x.ai/v1"
            api_key = XAI_API_KEY
        else:  # groq
            base_url = "https://api.groq.com/openai/v1"
            api_key = GROQ_API_KEY

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}

        async with self.httpx_client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.trim() == "[DONE]": break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta: yield delta
                    except: pass


    async def _gemini_request(self, messages, temperature, max_tokens, json_mode=False):
        """Use the native Gemini generateContent API."""
        # Convert OpenAI-style messages to Gemini format
        system_text = ""
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_text = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={GEMINI_API_KEY}"

        try:
            response = await self.httpx_client.post(url, json=payload)

            with open("ai_debug.txt", "a") as f:
                f.write(f"provider=gemini model={self.model} status={response.status_code}\n")
                if response.status_code != 200:
                    f.write(f"body={response.text[:600]}\n")
                f.write("\n")

            if response.status_code != 200:
                logger.error(f"Gemini Error {response.status_code}: {response.text}")
            response.raise_for_status()

            # Convert Gemini response to OpenAI-compatible format
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"choices": [{"message": {"content": text}}]}

        except Exception as e:
            with open("ai_debug.txt", "a") as f:
                f.write(f"Exception: {e}\n\n")
            logger.error(f"Gemini request exception: {e}", exc_info=True)
            raise

    async def _openai_compat_request(self, messages, temperature, max_tokens, response_format):
        """Used for Groq / x.ai — both are OpenAI-compatible."""
        if self.provider == "xai":
            base_url = "https://api.x.ai/v1"
            api_key = XAI_API_KEY
        else:  # groq
            base_url = "https://api.groq.com/openai/v1"
            api_key = GROQ_API_KEY

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            response = await self.httpx_client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            with open("ai_debug.txt", "a") as f:
                f.write(f"provider={self.provider} model={self.model} status={response.status_code}\n")
                if response.status_code != 200:
                    f.write(f"body={response.text[:500]}\n")
                f.write("\n")

            if response.status_code != 200:
                logger.error(f"AI Core Error {response.status_code}: {response.text}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            with open("ai_debug.txt", "a") as f:
                f.write(f"Exception: {e}\n\n")
            logger.error(f"Chat completion exception: {e}", exc_info=True)
            raise


ai_client = AIClient()

# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """
You are a high-energy, proactive AI Fitness & Lifestyle Coach for {username}.
Your mission is to drive results through hyper-personalized, data-driven advice.

USER PROFILE:
- Age: {age} | Height: {height_cm}cm | Weight: {weight_kg}kg
- Body Type: {body_type} | Activity: {activity_level}
- Goal: {diet_goal}
- Daily Targets: Sleep {daily_sleep_goal}h | Water {daily_water_goal}ml
- Injuries: {injuries} | Diet: {dietary_preferences}

GUIDELINES:
1. Be PROACTIVE. Don't just answer; suggest the next step.
2. High energy. Use motivating, punchy language.
3. Rapid responses. Keep it under 3 sentences unless detailing a specific drill.
4. Data-driven. Reference their profile constantly.
5. Safety first: If injuries are noted, suggest explosive but safe alternatives.

Redirect off-topic chatter instantly. You are here to build athletes.
"""


async def generate_diet_plan(
    user_context: Dict[str, Any], stats_summary: Dict[str, Any]
) -> Dict[str, str]:
    if not ai_client.available:
        return {
            "pre_workout": "Not available",
            "post_workout": "Not available",
            "analysis": "LLM client not initialized",
            "management_suggestion": "Please check API configuration",
        }

    prompt = f"""
You are an expert Sports Nutritionist.

USER PROFILE:
- Age: {user_context.get("age", "N/A")} | Height: {user_context.get("height_cm", "N/A")}cm | Weight: {user_context.get("weight_kg", "N/A")}kg
- Body Type: {user_context.get("body_type", "N/A")}
- Activity Level: {user_context.get("activity_level", "N/A")}
- Main Goal: {user_context.get("diet_goal", "N/A")}
- Dietary Preferences: {user_context.get("dietary_preferences", "None")}
- Injuries: {user_context.get("injuries", "None")}

PERFORMANCE TODAY:
- Workout Duration: {stats_summary.get("minutes", 0)} mins
- Calories Burned: {stats_summary.get("calories", 0)} kcal

Return ONLY valid JSON with exactly these 4 keys (no markdown, no explanation):
{{"pre_workout": "...", "post_workout": "...", "analysis": "...", "management_suggestion": "..."}}
"""

    try:
        response_data = await ai_client.chat_completion(
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
            json_mode=True,
        )
        content = response_data["choices"][0]["message"]["content"].strip()
        # Strip markdown if any
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        logger.error(f"Diet Plan Error: {e}")
        return {
            "pre_workout": "Complex carbs (oats/toast) 1-2h before",
            "post_workout": "Protein + fast carbs immediately after",
            "analysis": "Focus on whole foods and hydration.",
            "management_suggestion": "Track macros if possible.",
        }


async def ask_lifestyle_bot(user_context: Dict[str, Any], user_message: str) -> str:
    if not ai_client.available:
        return "I'm currently offline (no AI API key configured)."

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        username=user_context.get("username", "Athlete"),
        age=user_context.get("age", "N/A"),
        height_cm=user_context.get("height_cm", "N/A"),
        weight_kg=user_context.get("weight_kg", "N/A"),
        body_type=user_context.get("body_type", "N/A"),
        activity_level=user_context.get("activity_level", "N/A"),
        diet_goal=user_context.get("diet_goal", "N/A"),
        daily_sleep_goal=user_context.get("daily_sleep_goal", "N/A"),
        daily_water_goal=user_context.get("daily_water_goal", "N/A"),
        injuries=user_context.get("injuries") or "None",
        dietary_preferences=user_context.get("dietary_preferences") or "None",
    )

    try:
        response_data = await ai_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "I'm having trouble connecting right now. Please try again in a moment."


FORMATTED_PLAN_INSTRUCTIONS = """
You are an AI fitness coach. Return a structured response with EXACTLY these 4 sections:

### 🏋️ Workout Plan
- Warm-up: (short description)
- Main Exercises:
  - Exercise 1 (sets x reps)
  - Exercise 2 (sets x reps)
- Cool-down: (short description)

### 🥗 Diet Plan
- Breakfast: (short)
- Lunch: (short)
- Dinner: (short)
- Snacks: (optional)

### 💧 Hydration
- Daily Water Intake: (in liters)
- Tips: (1-2 short points)

### 📊 Explanation
- 2-3 bullet points explaining WHY this plan suits the user

RULES: Bullet points only. Short lines. No extra text.
"""


async def ask_coach_formatted(
    user_context: Dict[str, Any], preds: Dict[str, Any], user_query: str
) -> str:
    if not ai_client.available:
        water_l = preds.get("water", 2.5)
        return (
            "### 🏋️ Workout Plan\n"
            "- Warm-up: 5 min brisk walk + mobility\n"
            "- Main Exercises:\n"
            "  - Bodyweight squats 3x12\n"
            "  - Push-ups 3x10\n"
            "  - Lunges 3x10/leg\n"
            "- Cool-down: 5 min stretches\n\n"
            "### 🥗 Diet Plan\n"
            "- Breakfast: Greek yogurt, berries, oats\n"
            "- Lunch: Chicken, quinoa, mixed greens\n"
            "- Dinner: Salmon, veggies, sweet potato\n"
            "- Snacks: Cottage cheese, apple\n\n"
            "### 💧 Hydration\n"
            f"- Daily Water Intake: {round(float(water_l), 2)} L\n"
            "- Tips: Sip evenly throughout the day\n\n"
            "### 📊 Explanation\n"
            "- Matches goals and current fitness level\n"
            "- Balances cardio and strength safely\n"
            "- High protein supports recovery\n"
        )

    prompt = (
        FORMATTED_PLAN_INSTRUCTIONS
        + "\n\nUSER DATA:\n"
        + str({k: user_context.get(k) for k in [
            "username", "age", "height_cm", "weight_kg", "body_type",
            "activity_level", "diet_goal", "injuries", "dietary_preferences"
        ]})
        + f"\n\nPredicted Calories: {preds.get('calories')}"
        + f"\nPredicted Water: {preds.get('water')}"
        + f"\nPredicted Intensity: {preds.get('intensity')}"
        + f"\n\nUSER QUERY: {user_query}"
    )

    try:
        response_data = await ai_client.chat_completion(
            messages=[
                {"role": "system", "content": "Return ONLY the specified markdown sections and bullets."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM Formatted Error: {e}")
        return (
            "### 🏋️ Workout Plan\n- Warm-up: 5 min brisk walk\n"
            "- Main Exercises:\n  - Bodyweight squats 3x12\n  - Lunges 3x10/leg\n"
            "- Cool-down: stretches\n\n"
            "### 🥗 Diet Plan\n- Breakfast: Yogurt + oats\n- Lunch: Chicken + quinoa\n"
            "- Dinner: Fish + veggies\n- Snacks: Fruit\n\n"
            "### 💧 Hydration\n- Daily Water Intake: 2.5 L\n- Tips: Sip evenly\n\n"
            "### 📊 Explanation\n- Safe starter plan\n- Supports fat loss\n- Matches experience level\n"
        )
