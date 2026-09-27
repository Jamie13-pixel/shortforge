import os
import uuid

from moviepy import (
    AudioFileClip,
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
)

from stock_footage import search_video_clips, download_clip
from captions import build_caption_clips

TARGET_W, TARGET_H = 720, 1280


def _cover_resize_crop(clip, target_w=TARGET_W, target_h=TARGET_H):
    scale = max(target_w / clip.w, target_h / clip.h)
    resized = clip.resized((int(clip.w * scale) + 2, int(clip.h * scale) + 2))
    return resized.cropped(
        x_center=resized.w / 2,
        y_center=resized.h / 2,
        width=target_w,
        height=target_h,
    )


def build_video(audio_file, output_file, topic="", script_text=""):
    audio = AudioFileClip(audio_file)
    target_duration = audio.duration

    clip_urls = search_video_clips(topic or "abstract background", count=1)
    if not clip_urls:
        raise ValueError(f"No stock footage found for topic: {topic}")

    os.makedirs("data/temp_clips", exist_ok=True)

    downloaded_paths = []
    for url in clip_urls:
        path = f"data/temp_clips/{uuid.uuid4().hex}.mp4"
        try:
            download_clip(url, path)
            downloaded_paths.append(path)
        except Exception as e:
            print(f"[video_builder] Skipping clip after repeated failures: {e}")

    raw_clips = []

    try:
        if not downloaded_paths:
            raise RuntimeError("All stock footage downloads failed.")

        raw_clips = [VideoFileClip(p) for p in downloaded_paths]
        clips = [_cover_resize_crop(c) for c in raw_clips]

        sequence = []
        accumulated = 0
        i = 0
        while accumulated < target_duration:
            clip = clips[i % len(clips)]
            sequence.append(clip)
            accumulated += clip.duration
            i += 1

        combined = concatenate_videoclips(sequence, method="compose")
        combined = combined.subclipped(0, target_duration)

        caption_clips = (
            build_caption_clips(script_text, target_duration, TARGET_W, TARGET_H)
            if script_text else []
        )

        layers = [combined] + caption_clips
        final_video = CompositeVideoClip(layers, size=(TARGET_W, TARGET_H))
        final = final_video.with_audio(audio)

        final.write_videofile(
            output_file,
            fps=24,
            codec="libx264",
            preset="ultrafast",
            threads=os.cpu_count() or 4,
            audio_codec="aac",
            logger=None,
        )

    finally:
        for clip in raw_clips:
            try:
                clip.close()
            except Exception:
                pass

        for path in downloaded_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"[video_builder] Failed to delete temp file {path}: {e}")

        try:
            audio.close()
        except Exception:
            pass