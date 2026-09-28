import os
import uuid
import edge_tts
from moviepy import AudioFileClip, concatenate_audioclips

VOICE_MAPS = {
    "professional": {"HOST": "en-US-GuyNeural", "GUEST": "en-US-AriaNeural"},
    "energetic": {"HOST": "en-US-ChristopherNeural", "GUEST": "en-US-AvaNeural"},
    "calm": {"HOST": "en-US-AndrewNeural", "GUEST": "en-US-EmmaNeural"},
}
DEFAULT_VOICE = "professional"

async def _synthesize_line(text, voice, path):
    await edge_tts.Communicate(text, voice).save(path)

async def create_dialogue_voice(dialogue, output_file, voice_profile=DEFAULT_VOICE, temp_dir="data/temp_clips"):
    """Synthesize dialogue and return timing metadata."""
    voices = VOICE_MAPS.get(voice_profile, VOICE_MAPS[DEFAULT_VOICE])
    os.makedirs(temp_dir, exist_ok=True)
    line_paths = []
    try:
        for speaker, text in dialogue:
            voice = voices.get(speaker.upper(), voices.get("GUEST"))
            path = os.path.join(temp_dir, f"line_{uuid.uuid4().hex}.mp3")
            await _synthesize_line(text, voice, path)
            line_paths.append((speaker, text, path))

        if not line_paths:
            raise ValueError("No dialogue was generated.")

        clips, timeline = [], []
        current_time = 0.0
        for speaker, text, path in line_paths:
            clip = AudioFileClip(path)
            duration = clip.duration
            timeline.append({"speaker": speaker, "text": text, "start": current_time, "duration": duration})
            clips.append(clip)
            current_time += duration

        final_audio = concatenate_audioclips(clips)
        final_audio.write_audiofile(output_file, logger=None)
        final_audio.close()
        return timeline
    finally:
        for _, _, path in line_paths:
            try:
                if os.path.exists(path): os.remove(path)
            except Exception:
                pass
