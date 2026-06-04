import os
import torch

from functools import cached_property
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.separation import SepformerSeparation

from config.audio_censor_config import AudioCensorConfig


class SpeechSeparator:
    def __init__(self, config: AudioCensorConfig):
        self.config = config

    @cached_property
    def _model(self):
        savedir = os.path.join(os.environ["MODELS_DIRECTORY"], self.config.sep_model_source.rsplit("/", 1)[-1])
        sepformer = SepformerSeparation.from_hparams(
            source=self.config.sep_model_source, savedir=savedir, local_strategy=LocalStrategy.COPY
        )
        sepformer.to(self.config.device)
        return sepformer

    def separate(self, waveform: torch.Tensor, start: float = 0, end: float = -1) -> list[torch.Tensor]:
        sr = self.config.sample_rate
        if start == 0:
            if end == -1:
                chunk = waveform.to(self.config.device)
            else:
                chunk = waveform[:, int(start * sr) :].to(self.config.device)
        else:
            chunk = waveform[:, int(start * sr) : int(end * sr)].to(self.config.device)

        with torch.no_grad():
            est = self._model.separate_batch(chunk)  # [1, T, n_spk]
        return [est[0, :, i].cpu() for i in range(est.shape[-1])]

    @staticmethod
    def rms(track: torch.Tensor) -> float:
        return float(torch.sqrt((track**2).mean() + 1e-12).item())
