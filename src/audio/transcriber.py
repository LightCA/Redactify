import logging

from typing import Any
from pathlib import Path

from .audio_io import AudioIO
from .basic_transcriber import BasicTranscriber
from config.audio_censor_config import AudioCensorConfig
from .aligner import ForcedAligner
from .separator import SpeechSeparator


class Transcriber:
    logger = logging.getLogger(__name__)

    def __init__(self, config: AudioCensorConfig):
        self.config = config
        self.separator = SpeechSeparator(config)
        self.transcriber = BasicTranscriber(config)
        self.aligner = ForcedAligner(config)

    def run(self, input_path: Path) -> list[dict[str, Any]]:
        waveform, _ = AudioIO.load(input_path, self.config.sample_rate)
        wf_voices = self.separator.separate(waveform)

        results = []
        for wf_voice in wf_voices:
            wf_voice_transcriptions, info = self.transcriber.transcribe(wf_voice)
            for wf_voice_transcription in wf_voice_transcriptions:
                aligned_words = self.aligner.align(wf_voice, wf_voice_transcription)
                res_dct = {"language": info.language, "text": wf_voice_transcription["text"], "words": aligned_words["words"]}
                results.append(res_dct)

        return results
