from typing import Any

from .audio_io import AudioIO
from .basic_transcriber import BasicTranscriber
from config.audio_censor_config import AudioCensorConfig
from .aligner import ForcedAligner
from .separator import SpeechSeparator


class Transcriber:
    def __init__(self, config: AudioCensorConfig):
        self.config = config
        self.asr = BasicTranscriber(config)
        self.aligner = ForcedAligner(config)
        self.separator = SpeechSeparator(config)

    def run(self, audio_path: str) -> list[dict[str, Any]]:
        waveform, _ = AudioIO.load(audio_path, self.config.sample_rate)
        wf_voices = self.separator.separate(waveform)

        results = []
        for wf_voice in wf_voices:
            wf_voice_transcriptions, info = self.asr.transcribe(wf_voice)
            for wf_voice_transcription in wf_voice_transcriptions:
                mfa_words = self.aligner.align(wf_voice, wf_voice_transcription)
                res_dct = {"language": info.language, "text": wf_voice_transcription["text"], "words": mfa_words["words"]}
                results.append(res_dct)

        return results
