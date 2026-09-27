import os
import uuid
from concurrent.futures import ThreadPoolExecutor

from moviepy import (
    AudioFileClip,
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
)

from stock_footage import search_video_clips, download_clip
from captions import build_caption_clips

TARGET_W, TARGET_H = 1080, 1920


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

    clip_urls = search_video_clips(topic or "abstract background", count=2)
    if not clip_urls:
        raise ValueError(f"No stock footage found for topic: {topic}")

    os.makedirs("temp_clips", exist_ok=True)

    def _dl(url):
        path = f"temp_clips/{uuid.uuid4().hex}.mp4"
        download_clip(url, path)
        return path

    downloaded_paths = []
    raw_clips = []

    try:
        with ThreadPoolExecutor(max_workers=len(clip_urls)) as ex:
            downloaded_paths = list(ex.map(_dl, clip_urls))

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
        # Always runs, whether the render succeeded or raised an exception
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