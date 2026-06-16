import argparse

from pathlib import Path

from dataclasses import fields as dataclass_fields


class RedactifyCLI:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description="Redactify - Censor faces and profanity in video files")
        self._add_arguments()

    def _add_arguments(self):
        self._parser.add_argument(
            "input_path", nargs="?", type=Path, help="Path to input video file (opens file picker if omitted)"
        )
        self._parser.add_argument(
            "output_path", nargs="?", type=Path, help="Path to output video file (defaults to WORKING_DIRECTORY/output)"
        )

        # Shared
        self._parser.add_argument("--device", choices=["auto", "cpu", "cuda"])

        # Audio config
        self._parser.add_argument("--compute-type", dest="compute_type", choices=["auto", "float16", "int8"])
        self._parser.add_argument("--whisper-model", dest="whisper_model")
        self._parser.add_argument("--language")
        self._parser.add_argument("--asr-batch-size", dest="asr_batch_size", type=int)
        self._parser.add_argument("--alignment-padding", dest="alignment_padding", type=float)
        self._parser.add_argument("--censor-frequency", dest="censor_frequency", type=int)
        self._parser.add_argument("--censor-amplitude", dest="censor_amplitude", type=float)
        self._parser.add_argument("--censor-padding", dest="censor_padding", type=float)

        # Video config
        self._parser.add_argument("--fd-input-max-size", dest="fd_input_max_size", type=int)
        self._parser.add_argument("--fd-score-threshold", dest="fd_score_threshold", type=float)
        self._parser.add_argument("--fd-nms-threshold", dest="fd_nms_threshold", type=float)
        self._parser.add_argument("--fd-top-k", dest="fd_top_k", type=int)
        self._parser.add_argument("--blur-expansion", dest="blur_expansion", type=float)
        self._parser.add_argument("--blur-feather-inner", dest="blur_feather_inner", type=float)
        self._parser.add_argument("--blur-feather-outer", dest="blur_feather_outer", type=float)
        self._parser.add_argument("--blur-downsample-resolution", dest="blur_downsample_resolution", type=int)

    def parse(self) -> argparse.Namespace:
        return self._parser.parse_args()

    @staticmethod
    def config_kwargs(dataclass, kwargs: dict) -> dict:
        valid = {f.name for f in dataclass_fields(dataclass)}
        return {k: v for k, v in kwargs.items() if k in valid and v is not None}
