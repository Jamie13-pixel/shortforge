import re
import time
from groq import Groq

client = Groq(timeout=45.0, max_retries=1)

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

# Each attempt is (model, max_tokens). Later attempts give the model more room
# for its hidden reasoning, and the last one switches to a larger model.
ATTEMPTS = [
    ("openai/gpt-oss-20b", 1500),
    ("openai/gpt-oss-20b", 3000),
    ("openai/gpt-oss-120b", 3000),
]

MIN_LINES_STRICT = 6
MIN_LINES_LENIENT = 4
MAX_LINES = 12
MAX_WORDS_PER_LINE = 25

LINE_PATTERN = re.compile(r"^[\s>*_#\-]*(HOST|GUEST)[\s*_]*:[\s*_]*(.+)$", re.IGNORECASE)


class ScriptGenerationError(Exception):
    pass


def parse_dialogue(raw_text):
    dialogue = []
    for line in raw_text.splitlines():
        match = LINE_PATTERN.match(line)
        if match:
            speaker = match.group(1).upper()
            text = match.group(2).replace("*", "").strip()
            if text:
                dialogue.append((speaker, text))
    return dialogue


def check_dialogue(dialogue, strict):
    """Returns (ok, reason). Strict mode is used on early attempts."""
    if not dialogue:
        return False, "no HOST/GUEST lines could be parsed"

    speakers = set(spk for spk, _ in dialogue)
    if speakers != {"HOST", "GUEST"}:
        return False, "missing a speaker (found: " + ", ".join(sorted(speakers)) + ")"

    min_lines = MIN_LINES_STRICT if strict else MIN_LINES_LENIENT
    if len(dialogue) < min_lines:
        return False, "only " + str(len(dialogue)) + " lines (need " + str(min_lines) + ")"

    if any(len(text.split()) < 2 for _, text in dialogue):
        return False, "contains a line with fewer than 2 words"

    if strict:
        if len(dialogue) > MAX_LINES:
            return False, "too many lines (" + str(len(dialogue)) + ")"
        if any(len(text.split()) > MAX_WORDS_PER_LINE for _, text in dialogue):
            return False, "contains an overly long line"

    return True, ""


def generate_script(topic):
    user_prompt = "Write a short dialogue script about: " + topic + "."
    last_error = "unknown error"

    for attempt, (model, max_tokens) in enumerate(ATTEMPTS, start=1):
        strict = attempt < len(ATTEMPTS)

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.8,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            choice = response.choices[0]
            content = choice.message.content
            text = content.strip() if content else ""

            if not text:
                last_error = "empty response (finish_reason=" + str(choice.finish_reason) + ")"
            else:
                dialogue = parse_dialogue(text)
                ok, reason = check_dialogue(dialogue, strict)
                if ok:
                    dialogue = dialogue[:MAX_LINES]
                    clean_text = "\n".join(spk + ": " + line for spk, line in dialogue)
                    print("[script_generator] Success on attempt " + str(attempt) + " using " + model)
                    return clean_text, dialogue
                last_error = reason
                print("[script_generator] Raw output was: " + repr(text[:500]))

        except Exception as e:
            last_error = type(e).__name__ + ": " + str(e)

        print(
            "[script_generator] Attempt " + str(attempt) + "/" + str(len(ATTEMPTS))
            + " (" + model + ", max_tokens=" + str(max_tokens) + ") failed: " + last_error
        )

        if attempt < len(ATTEMPTS):
            time.sleep(attempt)

    raise ScriptGenerationError(
        "Could not generate a script for '" + topic + "' after "
        + str(len(ATTEMPTS)) + " attempts (" + last_error + "). Please try again."
    )