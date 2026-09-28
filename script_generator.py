import re
import time
from groq import Groq

client = Groq(timeout=45.0, max_retries=1)

# (model, max_tokens) for each generation attempt. Later attempts leave more
# room for hidden reasoning, and the last one switches to a larger model.
GEN_ATTEMPTS = [
    ("openai/gpt-oss-20b", 1500),
    ("openai/gpt-oss-20b", 3000),
    ("openai/gpt-oss-120b", 3000),
]
CHECK_MODEL = "openai/gpt-oss-120b"
CHECK_MAX_TOKENS = 3000

EXPECTED_LINES = 8
MAX_WORDS = 14  # prompt asks for under 12; small slack before we reject

SYSTEM_PROMPT = """
You are an expert viral content writer for TikTok, Instagram Reels,
and YouTube Shorts.

Create a highly engaging dialogue between HOST and GUEST.

RULES:

- Exactly 8 dialogue lines.
- Alternate HOST and GUEST, starting with HOST.
- First line must be a strong hook.
- Include surprising facts. Only state facts that are well established,
  and avoid precise numbers unless you are certain of them.
- Every line must stand on its own: never refer back to an earlier line
  with words like "those", "that" or "they".
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
            text = match.group(2).replace("*", "").strip()
            if text:
                dialogue.append((speaker, text))
    return dialogue


def format_dialogue(dialogue):
    return "\n".join(speaker + ": " + text for speaker, text in dialogue)


def validate_dialogue(dialogue, strict=True):
    """Returns (ok, reason). Non-strict mode is used on the last attempt."""
    if not dialogue:
        return False, "no HOST/GUEST lines could be parsed"

    if set(speaker for speaker, _ in dialogue) != {"HOST", "GUEST"}:
        return False, "both HOST and GUEST must speak"

    count = len(dialogue)
    if strict and count != EXPECTED_LINES:
        return False, str(count) + " lines, expected " + str(EXPECTED_LINES)
    if not strict and not (6 <= count <= 10):
        return False, str(count) + " lines, expected 6 to 10"

    limit = MAX_WORDS if strict else MAX_WORDS + 6
    longest = max(len(text.split()) for _, text in dialogue)
    if longest > limit:
        return False, "a line has " + str(longest) + " words (limit " + str(limit) + ")"

    if any(len(text.split()) < 2 for _, text in dialogue):
        return False, "a line has fewer than 2 words"

    return True, ""


def _chat(model, max_tokens, temperature, messages):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise ValueError("empty response (finish_reason=" + str(choice.finish_reason) + ")")
    return text


def fact_check_dialogue(topic, dialogue):
    prompt = (
        "Topic: " + topic + "\n\n"
        "Fact-check this dialogue. Fix factual errors, and replace any claim "
        "you cannot confirm with a safer, well-established fact.\n\n"
        "Requirements:\n"
        "- Do NOT make lines longer. Every line must stay under 12 words.\n"
        "- No hedges, parentheticals or extra clauses.\n"
        "- Keep exactly 8 lines, alternating HOST and GUEST, HOST first.\n"
        "- Every line must stand on its own (no 'those', 'that' or 'they' "
        "referring to an earlier line).\n"
        "- Return the dialogue only, in the same HOST:/GUEST: format.\n\n"
        + format_dialogue(dialogue)
    )

    try:
        text = _chat(
            CHECK_MODEL,
            CHECK_MAX_TOKENS,
            0.2,
            [{"role": "user", "content": prompt}],
        )
        checked = parse_dialogue(text)
        ok, reason = validate_dialogue(checked, strict=True)

        if not ok:
            print("[fact_check] Rejected checked version (" + reason + "); keeping original")
            return dialogue

        if checked != dialogue:
            print("[fact_check] Before:\n" + format_dialogue(dialogue))
            print("[fact_check] After:\n" + format_dialogue(checked))
        else:
            print("[fact_check] No changes")
        return checked

    except Exception as e:
        print("[fact_check] Skipped: " + type(e).__name__ + ": " + str(e))
        return dialogue


def generate_script(topic):
    user_prompt = (
        "Create a viral HOST/GUEST dialogue.\n\n"
        "Topic: " + topic + "\n\n"
        "Requirements:\n"
        "- Exactly 8 lines.\n"
        "- Strong hook.\n"
        "- Surprising facts.\n"
        "- Fast pacing.\n"
        "- Strong ending.\n"
    )
    last_error = "unknown error"

    for attempt, (model, max_tokens) in enumerate(GEN_ATTEMPTS, start=1):
        strict = attempt < len(GEN_ATTEMPTS)
        try:
            text = _chat(
                model,
                max_tokens,
                0.8,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            dialogue = parse_dialogue(text)
            ok, reason = validate_dialogue(dialogue, strict)

            if ok:
                print("[script_generator] Generated on attempt " + str(attempt) + " with " + model)
                dialogue = fact_check_dialogue(topic, dialogue)
                return format_dialogue(dialogue), dialogue

            last_error = reason
            print("[script_generator] Raw output: " + repr(text[:500]))

        except Exception as e:
            last_error = type(e).__name__ + ": " + str(e)

        print(
            "[script_generator] Attempt " + str(attempt) + "/" + str(len(GEN_ATTEMPTS))
            + " (" + model + ") failed: " + last_error
        )
        if attempt < len(GEN_ATTEMPTS):
            time.sleep(attempt)

    raise ScriptGenerationError(
        "Could not generate a script for '" + topic + "' after "
        + str(len(GEN_ATTEMPTS)) + " attempts (" + last_error + "). Please try again."
    )