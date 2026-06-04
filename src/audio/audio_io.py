import torchcodec  # noqa: F401  # must precede speechbrain: torch._dynamo init during torchcodec import calls inspect.getframeinfo which triggers speechbrain's k2_fsa lazy-load
import torch
import torchaudio
import torchaudio.functional as func


class AudioIO:
    @staticmethod
    def load(path: str, sample_rate: int = 0) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate > 0 and sr != sample_rate:
            waveform = func.resample(waveform, sr, sample_rate)
        return waveform, sample_rate
