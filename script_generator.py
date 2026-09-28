import os
import json
import re
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def target_word_count(duration: int) -> int:
    """
    Target narration length for each supported video duration.
    """

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
    """
    Clean common Markdown formatting returned by the model.
    """

    text = text.strip()

    text = re.sub(
        r"^```(?:json|text)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def generate_script(
    topic: str,
    duration: int = 30,
):
    """
    Generate a script appropriate for the requested duration.

    Returns:
        script_text, dialogue
    """

    word_count = target_word_count(duration)

    prompt = f"""
Create a high-retention short-form video script about:

{topic}

The final video duration is {duration} seconds.

Write approximately {word_count} spoken words.

Requirements:

- Start with a strong hook.
- Keep the viewer engaged.
- Use natural spoken language.
- Use relatively short sentences.
- Do not include camera directions.
- Do not include scene directions.
- Do not include sound effects.
- Do not include timestamps.
- Do not mention that AI generated the script.
- End naturally.

Return ONLY valid JSON using exactly this structure:

{{
    "script": "the complete narration",
    "dialogue": [
        "first spoken sentence",
        "second spoken sentence",
        "third spoken sentence"
    ]
}}
"""

    response = client.chat.completions.create(
        model=os.getenv(
            "OPENAI_SCRIPT_MODEL",
            "gpt-4o-mini",
        ),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional short-form "
                    "video script writer."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.8,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The AI returned an empty script."
        )

    content = clean_response(content)

    try:
        data = json.loads(content)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The AI returned invalid script JSON."
        ) from exc

    script_text = str(
        data.get("script", "")
    ).strip()

    dialogue = data.get(
        "dialogue",
        []
    )

    if not script_text:
        raise RuntimeError(
            "No script was generated."
        )

    if not isinstance(dialogue, list):
        raise RuntimeError(
            "Invalid dialogue format."
        )

    dialogue = [
        str(sentence).strip()
        for sentence in dialogue
        if str(sentence).strip()
    ]

    if not dialogue:
        raise RuntimeError(
            "No dialogue was generated."
        )

    return script_text, dialogue