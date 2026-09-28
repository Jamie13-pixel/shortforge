import os
import json
import re

from google import genai
from google.genai import types


DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the server. "
            "Add GEMINI_API_KEY to your deployment environment "
            "variables and restart the application."
        )

    return genai.Client(api_key=api_key)


def target_word_count(duration: int) -> int:
    if duration == 30:
        return 70

    if duration == 45:
        return 105

    if duration == 60:
        return 140

    raise ValueError(
        "Duration must be 30, 45, or 60 seconds."
    )


def clean_response(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def extract_json(text: str):
    text = clean_response(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(
            "Gemini returned an invalid script response."
        )

    try:
        return json.loads(
            text[start:end + 1]
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON for the video script."
        ) from exc


def generate_script(
    topic: str,
    duration: int = 30
):
    if not topic or not topic.strip():
        raise ValueError(
            "Video topic cannot be empty."
        )

    topic = topic.strip()

    if duration not in (30, 45, 60):
        raise ValueError(
            "Duration must be 30, 45, or 60 seconds."
        )

    word_count = target_word_count(duration)

    client = get_gemini_client()

    prompt = f"""
You are the professional script-writing engine for
Clip Pirate .ai, an AI short-video generator.

Create a high-retention short-form video narration.

TOPIC:
{topic}

REQUESTED DURATION:
{duration} seconds

TARGET WORD COUNT:
Approximately {word_count} spoken words.

REQUIREMENTS:

- Start with a strong hook.
- Keep the viewer engaged.
- Use natural spoken language.
- Use short, clear sentences.
- Make it suitable for text-to-speech.
- Make it informative and engaging.
- Avoid unnecessary filler.
- Do not include camera directions.
- Do not include scene directions.
- Do not include timestamps.
- Do not include sound effects.
- Do not include production instructions.
- Do not mention AI.
- End naturally.
- Keep the narration close to the requested word count.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "script": "Complete narration as one string.",
    "dialogue": [
        "First spoken sentence.",
        "Second spoken sentence.",
        "Third spoken sentence."
    ]
}}
"""

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=500
            )
        )

    except Exception as exc:
        error_text = str(exc)

        if "429" in error_text:
            raise RuntimeError(
                "Gemini API rate limit or quota reached. "
                "Please try again later."
            ) from exc

        if "401" in error_text or "403" in error_text:
            raise RuntimeError(
                "Gemini API authentication failed. "
                "Check your GEMINI_API_KEY."
            ) from exc

        raise RuntimeError(
            f"Gemini script generation failed: {error_text}"
        ) from exc

    content = getattr(
        response,
        "text",
        None
    )

    if not content:
        raise RuntimeError(
            "Gemini returned an empty script."
        )

    data = extract_json(content)

    if not isinstance(data, dict):
        raise RuntimeError(
            "Gemini returned an invalid script structure."
        )

    script_text = str(
        data.get("script", "")
    ).strip()

    if not script_text:
        raise RuntimeError(
            "Gemini returned no script text."
        )

    dialogue = data.get(
        "dialogue",
        []
    )

    if not isinstance(dialogue, list):
        raise RuntimeError(
            "Gemini returned an invalid dialogue format."
        )

    dialogue = [
        str(sentence).strip()
        for sentence in dialogue
        if str(sentence).strip()
    ]

    if not dialogue:
        raise RuntimeError(
            "Gemini returned no dialogue."
        )

    return script_text, dialogue