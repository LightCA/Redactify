import torch
import logging
import numpy as np

from typing import Any
from functools import cached_property
from config.audio_censor_config import AudioCensorConfig
from faster_whisper.vad import VadOptions
from faster_whisper import BatchedInferencePipeline, WhisperModel


class BasicTranscriber:
    logger = logging.getLogger(__name__)
    
    def __init__(self, config: AudioCensorConfig):
        self.config = config

    @cached_property
    def _model(self):
        base = WhisperModel(self.config.whisper_model, device=self.config.device, compute_type=self.config.compute_type)
        return BatchedInferencePipeline(model=base)

    def transcribe(
        self, audio: np.ndarray | torch.Tensor, *, time_offset: float = 0.0, from_separation: bool = False
    ) -> tuple[list[dict[str, Any]], Any]:
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy().astype(np.float32)


        vad = VadOptions(
            threshold=0.5, neg_threshold=0.35, min_speech_duration_ms=30, min_silence_duration_ms=80, speech_pad_ms=0
        )
        gen, info = self._model.transcribe(
            audio,
            batch_size=self.config.asr_batch_size,
            multilingual=True,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=vad,
            condition_on_previous_text=False,
            language=self.config.language,
        )

        segments: list[dict[str, Any]] = []
        for seg in gen:
            text = seg.text.strip()
            if not text:
                continue
            out: dict[str, Any] = {
                "start": seg.start + time_offset,
                "end": seg.end + time_offset,
                "text": text,
                "words": [
                    {"start": w.start + time_offset, "end": w.end + time_offset, "word": w.word, "score": w.probability}
                    for w in (seg.words or [])
                ],
            }
            if from_separation:
                out["from_separation"] = True
            segments.append(out)
        return segments, info
