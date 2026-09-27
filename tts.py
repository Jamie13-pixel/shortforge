import os
import uuid
import edge_tts
from moviepy import AudioFileClip, concatenate_audioclips

VOICE_MAP = {
    "HOST": "en-US-GuyNeural",
    "GUEST": "en-US-AriaNeural",
}
DEFAULT_VOICE = "en-US-AriaNeural"


async def _synthesize_line(text, voice, path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)


async def create_dialogue_voice(dialogue, output_file, temp_dir="data/temp_clips"):
    """
    dialogue: list of (speaker, text) tuples.
    Writes combined audio to output_file.
    Returns a timeline: [{"speaker", "text", "start", "duration"}, ...]
    """
    os.makedirs(temp_dir, exist_ok=True)
    line_paths = []

    try:
        for speaker, text in dialogue:
            voice = VOICE_MAP.get(speaker.upper(), DEFAULT_VOICE)
            path = os.path.join(temp_dir, f"line_{uuid.uuid4().hex}.mp3")
            await _synthesize_line(text, voice, path)
            line_paths.append((speaker, text, path))

        clips = []
        timeline = []
        current_time = 0.0

        for speaker, text, path in line_paths:
            clip = AudioFileClip(path)
            duration = clip.duration
            timeline.append({
                "speaker": speaker,
                "text": text,
                "start": current_time,
                "duration": duration,
            })
            clips.append(clip)
            current_time += duration

        final_audio = concatenate_audioclips(clips)
        final_audio.write_audiofile(output_file, logger=None)

        for clip in clips:
            clip.close()

        return timeline

    finally:
        for _, _, path in line_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass