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

ASPECT_SIZES = {
    "4:3": (1440, 1080),
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}
DEFAULT_ASPECT_RATIO = "4:3"


def _cover_resize_crop(clip, target_w, target_h):
    scale = max(target_w / clip.w, target_h / clip.h)
    resized = clip.resized((int(clip.w * scale) + 2, int(clip.h * scale) + 2))
    return resized.cropped(
        x_center=resized.w / 2,
        y_center=resized.h / 2,
        width=target_w,
        height=target_h,
    )


def _fit_audio_to_duration(audio, target_duration):
    """Return an audio clip whose duration is exactly target_duration.

    If TTS is short, pad with silence; if it is long, trim it. This makes the
    requested duration authoritative without changing speech speed.
    """
    from moviepy import AudioClip, concatenate_audioclips

    if target_duration <= 0:
        raise ValueError("target_duration must be positive")

    if audio.duration > target_duration + 0.02:
        return audio.subclipped(0, target_duration)

    if audio.duration < target_duration - 0.02:
        silence_duration = target_duration - audio.duration
        silence = AudioClip(lambda t: 0.0, duration=silence_duration, fps=44100)
        return concatenate_audioclips([audio, silence])

    return audio


def build_video(audio_file, output_file, topic="", timeline=None, target_duration=None, aspect_ratio=DEFAULT_ASPECT_RATIO, captions=True):
    audio = AudioFileClip(audio_file)
    target_duration = float(target_duration) if target_duration is not None else audio.duration
    if target_duration not in (30, 45, 60):
        raise ValueError("target_duration must be 30, 45, or 60 seconds")

    if aspect_ratio not in ASPECT_SIZES:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    target_w, target_h = ASPECT_SIZES[aspect_ratio]
    fitted_audio = _fit_audio_to_duration(audio, target_duration)

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
        clips = [_cover_resize_crop(c, target_w, target_h) for c in raw_clips]

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

        caption_clips = build_caption_clips(timeline, target_w, target_h) if captions and timeline else []

        layers = [combined] + caption_clips
        final_video = CompositeVideoClip(layers, size=(target_w, target_h)).with_duration(target_duration)
        final = final_video.with_audio(fitted_audio)

        final.write_videofile(
            output_file,
            fps=20,
            codec="libx264",
            preset="ultrafast",
            threads=os.cpu_count() or 4,
            audio_codec="aac",
            logger=None,
        )

        # Verify the encoded MP4 rather than trusting the requested timeline.
        verification = VideoFileClip(output_file)
        try:
            actual_duration = verification.duration
            if abs(actual_duration - target_duration) > 0.15:
                raise RuntimeError(
                    f"Rendered duration mismatch: requested {target_duration:.2f}s, "
                    f"got {actual_duration:.2f}s"
                )
        finally:
            verification.close()

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
            if fitted_audio is not audio:
                try:
                    fitted_audio.close()
                except Exception:
                    pass
            audio.close()
        except Exception:
            pass