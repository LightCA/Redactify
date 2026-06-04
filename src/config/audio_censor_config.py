import torch

from dataclasses import dataclass


@dataclass
class AudioCensorConfig:
    _init: bool = False
    device: str = "auto"
    compute_type: str = "auto"

    # Model sources
    whisper_model: str = "large-v3"
    sep_model_source: str = "speechbrain/sepformer-whamr16k"

    # Audio
    sample_rate: int = 16000

    # Transcription
    asr_batch_size: int = 16
    language: str | None = None

    # Alignment
    alignment_padding: float = 0.1

    # Censoring
    censor_frequency: int = 1000
    censor_amplitude: float = 0.1
    censor_padding: float = 0.1

    def __post_init__(self) -> None:
        if not self._init:
            self._init = True
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            elif self.device not in ["cpu", "cuda"]:
                raise ValueError(f"Invalid device: {self.device}")
            if self.compute_type == "auto":
                self.compute_type = "float16" if self.device == "cuda" else "int8"
