import re
from groq import Groq

client = Groq()

SYSTEM_PROMPT = (
    "You write short, punchy DIALOGUE scripts for 15-25 second faceless short-form "
    "videos (TikTok/Reels/Shorts style), between two characters: HOST and GUEST. "
    "Output ONLY the dialogue, one line per turn, in this EXACT format:\n"
    "HOST: <line>\n"
    "GUEST: <line>\n"
    "...\n"
    "No stage directions, no markdown, no headers, no narration outside the HOST/GUEST lines. "
    "Keep each line SHORT (under 12 words) for text-to-speech pacing. "
    "Exactly 4 exchanges total (8 lines): a hook, two surprising facts, and a call-to-action to follow for more."
)


def _fallback_dialogue(topic: str):
    return [
        ("HOST", f"Did you know these facts about {topic}?"),
        ("GUEST", "Tell me more!"),
        ("HOST", "Here's fact one."),
        ("GUEST", "Whoa, really?"),
        ("HOST", "Here's fact two."),
        ("GUEST", "Follow for more like this!"),
    ]


def parse_dialogue(raw_text: str):
    dialogue = []
    pattern = re.compile(r"^\s*(HOST|GUEST)\s*:\s*(.+)$", re.IGNORECASE)
    for line in raw_text.splitlines():
        match = pattern.match(line)
        if match:
            speaker = match.group(1).upper()
            text = match.group(2).strip()
            if text:
                dialogue.append((speaker, text))
    return dialogue


def generate_script(topic: str):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=1500,
            temperature=0.8,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Write a short dialogue script