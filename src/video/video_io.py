import logging
import subprocess

from pathlib import Path


class VideoIO:
    logger = logging.getLogger(__name__)

    @staticmethod
    def extract_audio_from_video(input_path: Path, output_path: Path):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-vn",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "1",
                output_path,
            ],
            check=True,
            stderr=subprocess.PIPE,
        )

    @staticmethod
    def combine_audio_video(video_path: Path, audio_path: Path, output_path: Path):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                output_path,
            ],
            check=True,
            stderr=subprocess.PIPE,
        )
