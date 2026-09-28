import os
import uuid
import subprocess
from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip
from stock_footage import search_video_clips, download_clip
from captions import build_caption_clips

RATIO_SIZES = {
    "4:3": (1440, 1080),
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}


def _cover_resize_crop(clip, target_w, target_h):
    scale = max(target_w / clip.w, target_h / clip.h)
    resized = clip.resized((int(clip.w * scale) + 2, int(clip.h * scale) + 2))
    return resized.cropped(x_center=resized.w / 2, y_center=resized.h / 2, width=target_w, height=target_h)


def _prepare_audio(audio_file, target_duration):
    """Trim or pad the generated narration so the requested duration is real."""
    probe = AudioFileClip(audio_file)
    current = probe.duration
    probe.close()
    if abs(current - target_duration) < 0.05:
        return
    tmp = audio_file + ".duration.mp3"
    if current > target_duration:
        args = ["ffmpeg", "-y", "-i", audio_file, "-t", str(target_duration), "-c:a", "libmp3lame", "-q:a", "4", tmp]
    else:
        pad = target_duration - current
        args = ["ffmpeg", "-y", "-i", audio_file, "-af", f"apad=pad_dur={pad}", "-t", str(target_duration), "-c:a", "libmp3lame", "-q:a", "4", tmp]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, audio_file)


def build_video(audio_file, output_file, topic="", timeline=None, aspect_ratio="4:3", captions=True, target_duration=None):
    if aspect_ratio not in RATIO_SIZES:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
    target_w, target_h = RATIO_SIZES[aspect_ratio]
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration
    target_duration = float(target_duration or audio_duration)
    audio.close()
    _prepare_audio(audio_file, target_duration)
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration
    if target_duration <= 0:
        raise ValueError("Video duration must be greater than zero.")

    # The source audio remains authoritative when the selected duration exceeds it.
    # We render to the selected duration where possible, but never fabricate speech.
    render_duration = target_duration
    clip_urls = search_video_clips(topic or "abstract background", count=1, aspect_ratio=aspect_ratio)
    if not clip_urls:
        raise ValueError(f"No stock footage found for topic: {topic}")

    os.makedirs("data/temp_clips", exist_ok=True)
    downloaded_paths, raw_clips = [], []
    try:
        for url in clip_urls:
            path = f"data/temp_clips/{uuid.uuid4().hex}.mp4"
            try:
                download_clip(url, path)
                downloaded_paths.append(path)
            except Exception as e:
                print(f"[video_builder] Skipping clip after repeated failures: {e}")

        if not downloaded_paths:
            raise RuntimeError("All stock footage downloads failed.")

        raw_clips = [VideoFileClip(p) for p in downloaded_paths]
        clips = [_cover_resize_crop(c, target_w, target_h) for c in raw_clips]
        sequence, accumulated, i = [], 0.0, 0
        while accumulated < render_duration:
            clip = clips[i % len(clips)]
            sequence.append(clip)
            accumulated += clip.duration
            i += 1
        combined = concatenate_videoclips(sequence, method="compose").subclipped(0, render_duration)

        caption_clips = build_caption_clips(timeline, target_w, target_h) if captions and timeline else []
        final_video = CompositeVideoClip([combined] + caption_clips, size=(target_w, target_h)).with_audio(audio.subclipped(0, render_duration))
        final_video.write_videofile(output_file, fps=20, codec="libx264", preset="ultrafast", threads=os.cpu_count() or 4, audio_codec="aac", logger=None)
        final_video.close()
    finally:
        for clip in raw_clips:
            try: clip.close()
            except Exception: pass
        try: audio.close()
        except Exception: pass
        for path in downloaded_paths:
            try:
                if os.path.exists(path): os.remove(path)
            except Exception: pass
