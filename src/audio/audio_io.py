import logging
import torch
import torchaudio
import torchcodec  # noqa: F401  # must precede speechbrain: torch._dynamo init during torchcodec import calls inspect.getframeinfo which triggers speechbrain's k2_fsa lazy-load
import torchaudio.functional as func

from pathlib import Path


class AudioIO:
    logger = logging.getLogger(__name__)

    @staticmethod
    def load(input_path: Path, sample_rate: int = 0) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(input_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate > 0 and sr != sample_rate:
            waveform = func.resample(waveform, sr, sample_rate)
        return waveform, sample_rate
