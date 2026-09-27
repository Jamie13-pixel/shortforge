from groq import Groq

client = Groq()

SYSTEM_PROMPT = (
    "You write short, punchy scripts for 30-45 second faceless short-form "
    "videos (TikTok/Reels/Shorts style). Output ONLY the spoken script text, "
    "no stage directions, no markdown, no headers. Keep sentences short and "
    "punchy for text-to-speech pacing."
)


def _fallback_script(topic: str) -> str:
    return (
        f"Here are three surprising facts about {topic}. "
        "Fact one. Fact two. Fact three. "
        "Follow for more."
    )


def generate_script(topic: str) -> str:
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=300,
            temperature=0.8,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Write a short video script about: {topic}. "
                        "Structure: a hook line, three surprising facts, "
                        "then a call-to-action to follow for more."
                    ),
                },
            ],
        )
        text = response.choices[0].message.content.strip()
        if not text:
            raise ValueError("Empty response from model")
        return text
    except Exception as e:
        print(f"[script_generator] Groq call failed, using fallback: {e}")
        return _fallback_script(topic)