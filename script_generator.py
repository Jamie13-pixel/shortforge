import re
import time
from groq import Groq

client = Groq(
    timeout=45.0,
    max_retries=1
)

SYSTEM_PROMPT = """
You are an expert viral content writer for TikTok, Instagram Reels,
and YouTube Shorts.

Create a highly engaging dialogue between HOST and GUEST.

RULES:

- Exactly 8 dialogue lines.
- Alternate HOST and GUEST.
- First line must be a strong hook.
- Include surprising facts.
- End with a call-to-action.
- Keep each line under 12 words.

Output ONLY:

HOST: text
GUEST: text
HOST: text
GUEST: text
HOST: text
GUEST: text
HOST: text
GUEST: text
"""

LINE_PATTERN = re.compile(
    r"^[\s>*_#\-]*(HOST|GUEST)[\s*_]*:[\s*_]*(.+)$",
    re.IGNORECASE,
)


class ScriptGenerationError(Exception):
    pass


def parse_dialogue(raw_text):
    dialogue = []

    for line in raw_text.splitlines():
        match = LINE_PATTERN.match(line)

        if match:
            speaker = match.group(1).upper()
            text = match.group(2).strip()

            if text:
                dialogue.append((speaker, text))

    return dialogue


def fact_check_dialogue(topic, dialogue):
    script_text = "\n".join(
        f"{speaker}: {text}"
        for speaker, text in dialogue
    )

    prompt = f"""
Topic: {topic}

Fact-check and improve this dialogue.

Requirements:
- Keep HOST/GUEST format.
- Keep 8 lines.
- Correct factual errors.
- Improve weak lines.
- Return dialogue only.

{script_text}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content

        if content:
            checked = parse_dialogue(content)

            if checked:
                return checked

    except Exception as e:
        print("[fact_check]", e)

    return dialogue


def generate_script(topic):

    user_prompt = f"""
Create a viral HOST/GUEST dialogue.

Topic: {topic}

Requirements:
- Exactly 8 lines.
- Strong hook.
- Surprising facts.
- Fast pacing.
- Strong ending.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=1.0,
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        text = response.choices[0].message.content.strip()

        dialogue = parse_dialogue(text)

        if not dialogue:
            raise ScriptGenerationError(
                "Failed to parse HOST/GUEST dialogue"
            )

        dialogue = fact_check_dialogue(
            topic,
            dialogue
        )

        clean_text = "\n".join(
            f"{speaker}: {line}"
            for speaker, line in dialogue
        )

        return clean_text, dialogue

    except Exception as e:
        raise ScriptGenerationError(
            f"Script generation failed: {e}"
        )