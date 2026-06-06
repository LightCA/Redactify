import os
import shutil
import logging
import threading

from pathlib import Path

from cli import RedactifyCLI
from video.video_io import VideoIO
from audio.censor import AudioCensor, AudioCensorConfig
from video.censor import VideoCensor, VideoCensorConfig


class Redactify:
    _init_logger: bool = False
    _init_working_dirs: bool = False
    logger: logging.Logger
    input_dir: Path
    intermediate_dir: Path
    valid_extensions = [".mp4", ".avi", ".webm", ".ogv", ".mkv", ".wmv"]

    def __init__(self):
        self._configure_logger()
        self._create_working_dirs()

    def _configure_logger(self):
        if not Redactify._init_logger:
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            file_handler = logging.FileHandler(log_dir / "app.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            Redactify.logger = logging.getLogger(__name__)

            Redactify._init_logger = True

    @staticmethod
    def _create_working_dirs():
        if not Redactify._init_working_dirs:
            working_dir = Path(os.environ["WORKING_DIRECTORY"])

            Redactify.input_dir = working_dir / "input"
            Redactify.intermediate_dir = working_dir / "intermediate"

            os.makedirs(Redactify.input_dir, exist_ok=True)
            os.makedirs(Redactify.intermediate_dir, exist_ok=True)
            Redactify._init_working_dirs = True

    def run_audio_pipeline(self, input_path: Path, output_path: Path, audio_config: AudioCensorConfig | None = None):
        try:
            self.logger.info("Starting audio pipeline")

            working_input_path = self.input_dir / f"{input_path.stem}.wav"

            VideoIO.extract_audio_from_video(input_path, working_input_path)

            AudioCensor.run(working_input_path, output_path, audio_config)
        except Exception as ex:
            self.logger.error(ex)
            raise (ex)
        finally:
            if os.path.exists(working_input_path):
                os.remove(working_input_path)

            self.logger.info("Finished audio pipeline")

    def run_video_pipeline(self, input_path: Path, output_path: Path, video_config: VideoCensorConfig | None = None):
        try:
            self.logger.info("Starting video pipeline")

            working_input_path = self.input_dir / input_path.name

            shutil.copy2(input_path, working_input_path)

            VideoCensor.run(working_input_path, output_path, video_config)
        except Exception as ex:
            self.logger.error(ex)
            raise (ex)
        finally:
            if os.path.exists(working_input_path):
                os.remove(working_input_path)

            self.logger.info("Finished video pipeline")

    def run(self, input_path: Path, output_path: Path, **kwargs):
        try:
            self.logger.info("Starting Redactify")

            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            if not os.access(input_path, os.R_OK):
                raise PermissionError(f"Input file is not readable: {input_path}")

            file_extension = input_path.suffix
            if file_extension not in self.valid_extensions:
                valid = ", ".join(self.valid_extensions)
                raise ValueError(f"Unsupported file type '{file_extension}', expected one of: {valid}")

            working_video_intermediate_path = self.intermediate_dir / f"{input_path.stem}.mp4"
            working_audio_intermediate_path = self.intermediate_dir / f"{input_path.stem}.wav"

            audio_censor_config = AudioCensorConfig(**RedactifyCLI.config_kwargs(AudioCensorConfig, kwargs))
            video_censor_config = VideoCensorConfig(**RedactifyCLI.config_kwargs(VideoCensorConfig, kwargs))

            video_thread = threading.Thread(
                target=self.run_video_pipeline, args=(input_path, working_video_intermediate_path, video_censor_config)
            )
            audio_thread = threading.Thread(
                target=self.run_audio_pipeline, args=(input_path, working_audio_intermediate_path, audio_censor_config)
            )

            threads = [audio_thread, video_thread]
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            self.logger.info("Combining video with audio into final output")

            VideoIO.combine_audio_video(working_video_intermediate_path, working_audio_intermediate_path, output_path)

            self.logger.info(f"Censored video saved to: {output_path}")
            self.logger.info(f"Redactify finished")
        except Exception as ex:
            self.logger.error(ex)
            raise ex
        finally:
            try:
                for path in [working_video_intermediate_path, working_audio_intermediate_path]:
                    if os.path.exists(path):
                        os.remove(path)
            except:
                pass
