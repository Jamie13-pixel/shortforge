import os
import uuid
import edge_tts
from moviepy import AudioFileClip, concatenate_audioclips, AudioClip

VOICE_MAP = {
    "HOST": "en-US-GuyNeural",
    "GUEST": "en-US-AriaNeural",
}
DEFAULT_VOICE = "en-US-AriaNeural"

VOICE_PROFILES = {
    "professional": {"HOST": "en-US-GuyNeural", "GUEST": "en-US-AriaNeural"},
    "energetic": {"HOST": "en-US-ChristopherNeural", "GUEST": "en-US-JennyNeural"},
    "calm": {"HOST": "en-US-EricNeural", "GUEST": "en-US-AriaNeural"},
}


async def _synthesize_line(text, voice, path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)


async def create_dialogue_voice(dialogue, output_file, temp_dir="data/temp_clips", voice_profile="professional", target_duration=None):
    """
    dialogue: list of (speaker, text) tuples.
    Writes combined audio to output_file.
    Returns a timeline: [{"speaker", "text", "start", "duration"}, ...]
    """
    os.makedirs(temp_dir, exist_ok=True)
    line_paths = []
    selected_map = VOICE_PROFILES.get(str(voice_profile).lower(), VOICE_MAP)

    try:
        for speaker, text in dialogue:
            voice = selected_map.get(speaker.upper(), DEFAULT_VOICE)
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
        if target_duration is not None:
            target_duration = float(target_duration)
            if target_duration not in (30, 45, 60):
                raise ValueError("target_duration must be 30, 45, or 60 seconds")
            if final_audio.duration > target_duration + 0.02:
                final_audio = final_audio.subclipped(0, target_duration)
            elif final_audio.duration < target_duration - 0.02:
                silence = AudioClip(lambda t: 0.0, duration=target_duration-final_audio.duration, fps=44100)
                final_audio = concatenate_audioclips([final_audio, silence])

        final_audio.write_audiofile(output_file, logger=None)

        # Keep caption timing aligned with the actual requested audio window.
        if target_duration is not None:
            clipped_timeline = []
            for entry in timeline:
                start = float(entry["start"])
                if start >= target_duration:
                    continue
                clipped = min(float(entry["duration"]), target_duration - start)
                if clipped > 0:
                    clipped_entry = dict(entry)
                    clipped_entry["duration"] = clipped
                    clipped_timeline.append(clipped_entry)
            timeline = clipped_timeline

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