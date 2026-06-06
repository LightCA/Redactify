import re
import torch
import logging

from typing import Any
from functools import cached_property

from torchaudio.pipelines import MMS_FA as bundle
from config.audio_censor_config import AudioCensorConfig


class ForcedAligner:
    logger = logging.getLogger(__name__)

    _digit_map = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    def __init__(self, config: AudioCensorConfig):
        self.config = config

    @cached_property
    def _bundle(self):
        try:
            model = bundle.get_model().to(self.config.device).eval()
            return {
                "model": model,
                "tokenizer": bundle.get_tokenizer(),
                "aligner": bundle.get_aligner(),
                "sample_rate": bundle.sample_rate,
                "dictionary": set(bundle.get_dict().keys()),
            }
        except Exception as ex:
            self.logger.error(
                f"Forced alignment disabled (could not load torchaudio MMS_FA bundle): {ex}\nRequires torchaudio>=2.1. Falling back to Whisper's native word timestamps."
            )
            return None

    def normalize_word(self, word: str) -> str:
        # Normalizes a word by lowercasing, replacing digits with their word forms, then removing non-alphabetic characters (except apostrophes and spaces).
        word = re.sub(r"\d", lambda m: self._digit_map[m.group()], word.strip().lower())
        return re.sub(r"[^a-z' ]", "", word)

    def align(self, audio: torch.Tensor, segment: dict[str, Any]) -> dict[str, Any]:
        words = segment.get("words") or []
        if not words or self._bundle is None:
            return segment

        dictionary = self._bundle["dictionary"]

        normalized: list[str] = []
        keep_idx: list[int] = []
        for i, w in enumerate(words):
            norm = self.normalize_word(w["word"])
            if norm and all(c in dictionary for c in norm):
                normalized.append(norm)
                keep_idx.append(i)
        if not normalized:
            return segment

        sample_rate = self.config.sample_rate
        padding = self.config.alignment_padding
        chunk_start = max(0.0, segment["start"] - padding)
        chunk_end = min(audio.shape[-1] / sample_rate, segment["end"] + padding)
        chunk = audio[int(chunk_start * sample_rate) : int(chunk_end * sample_rate)]
        if chunk.numel() < sample_rate // 10:
            return segment

        try:
            audio_waveform = chunk.unsqueeze(0).to(self.config.device)
            with torch.inference_mode():
                emission, _ = self._bundle["model"](audio_waveform)
            spans = self._bundle["aligner"](emission[0], self._bundle["tokenizer"](normalized))

            frame_to_time = (audio_waveform.shape[1] / emission.shape[1]) / sample_rate
            for word_idx, word_spans in zip(keep_idx, spans):
                if not word_spans:
                    continue
                words[word_idx]["start"] = chunk_start + word_spans[0].start * frame_to_time
                words[word_idx]["end"] = chunk_start + word_spans[-1].end * frame_to_time
                words[word_idx]["align_score"] = float(sum(s.score for s in word_spans) / len(word_spans))
        except Exception as ex:
            self.logger.error(f"Alignment failed for '{segment.get('text','')[:40]}': {ex}")
            raise ex

        segment["start"] = words[0]["start"]
        segment["end"] = words[-1]["end"]
        return segment
