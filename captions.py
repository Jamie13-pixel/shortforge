import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux (Railway/Docker)
    "arial.ttf",  # Windows fallback
    "DejaVuSans-Bold.ttf",
]


def _load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    print("[captions] WARNING: no TTF font found, falling back to default bitmap font")
    return ImageFont.load_default()

def render_caption_image(text, video_width=1080, font_size=70, max_chars_per_line=22):
    font = _load_font(font_size)
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line) or [text]

    dummy_img = Image.new("RGBA", (video_width, 10))
    draw = ImageDraw.Draw(dummy_img)

    line_heights = []
    line_widths = []
    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=4)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = 15
    total_height = sum(line_heights) + line_spacing * (len(wrapped_lines) - 1) + 40

    img = Image.new("RGBA", (video_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = 20
    for line, lh, lw in zip(wrapped_lines, line_heights, line_widths):
        x = (video_width - lw) / 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill="white",
            stroke_width=4,
            stroke_fill="black",
        )
        y += lh + line_spacing

    return np.array(img)


def split_script_into_chunks(script_text, words_per_chunk=6):
    words = script_text.replace("\n", " ").split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i:i + words_per_chunk]))
    return chunks


def build_caption_clips(script_text, total_duration, video_width=1080, video_height=1920):
    from moviepy import ImageClip

    chunks = split_script_into_chunks(script_text)
    if not chunks:
        return []

    word_counts = [len(c.split()) for c in chunks]
    total_words = sum(word_counts)

    clips = []
    t = 0.0
    for chunk, wc in zip(chunks, word_counts):
        duration = total_duration * (wc / total_words)
        img_array = render_caption_image(chunk, video_width=video_width)
        clip = (
            ImageClip(img_array)
            .with_duration(duration)
            .with_start(t)
            .with_position(("center", int(video_height * 0.72)))
        )
        clips.append(clip)
        t += duration

    return clips