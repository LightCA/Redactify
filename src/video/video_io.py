import logging
import subprocess

from pathlib import Path


class VideoIO:
    logger = logging.getLogger(__name__)

    @classmethod
    def _run_ffmpeg(cls, args: list):
        result = subprocess.run(args, capture_output=True, text=True)
        if result.stderr:
            cls.logger.debug(f"ffmpeg output:\n{result.stderr}")
        if result.returncode != 0:
            cls.logger.error(f"ffmpeg failed with exit code {result.returncode}")
            raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)

    @classmethod
    def extract_audio_from_video(cls, input_path: Path, output_path: Path):
        cls._run_ffmpeg(
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
            ]
        )

    @classmethod
    def combine_audio_video(cls, video_path: Path, audio_path: Path, output_path: Path):
        cls._run_ffmpeg(
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
            ]
        )
