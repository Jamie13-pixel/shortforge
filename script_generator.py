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

GOAL:
Make viewers stay until the final second.

RULES:

1. Exactly 8 dialogue lines.

2. Alternate speakers exactly:

HOST
GUEST
HOST
GUEST
HOST
GUEST
HOST
GUEST

3. Structure:

Line 1:
A shocking or curiosity-driven hook.

Lines 2-3:
Build curiosity.

Lines 4-6:
Reveal surprising facts or insights.

Lines 7-8:
Strong payoff and call-to-action.

4. Every line must add new information.

5. Avoid filler:
- Wow
- Really
- Amazing
- Interesting
- That's crazy

6. Fast-paced conversational style.

7. Keep each line under 12 words.

8. Use real facts whenever possible.

OUTPUT FORMAT:

HOST: text
GUEST: text
HOST: text
GUEST: text
HOST: text
GUEST: text
HOST: text
GUEST: text

No markdown.
No narration.
No stage directions.
No extra text.
"""

ATTEMPTS = [
    ("openai/gpt-oss-20b", 1500),
    ("openai/gpt-oss-20b", 3000),
    ("openai/gpt-oss-120b", 3000),
]

MIN_LINES_STRICT = 8
MIN_LINES_LENIENT = 6
MAX_LINES = 8
MAX_WORDS_PER_LINE = 12

LINE_PATTERN = re.compile(
    r"^[\s>*_#\-]*(HOST|GUEST)[\s*_]*:[\s*_]*(.+)$",
    re.IGNORECASE,
)