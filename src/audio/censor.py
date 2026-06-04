from text.pii_detector import PIIDetector
from .transcriber import Transcriber
from .censor_beep import CensorBeep
from config.audio_censor_config import AudioCensorConfig


class AudioCensor:
    @staticmethod
    def run(input_path: str, output_path: str, config: AudioCensorConfig):
        transcriber = Transcriber(config)

        t_res = transcriber.run(input_path)

        pii_timestamps: list[tuple[float, float]] = []
        for t_seg in t_res:
            detector = PIIDetector(language=t_seg["language"])
            t_word_lookup = []
            for i in range(len(t_seg["words"])):
                t_word_lookup.extend([i] * len(t_seg["words"][i]["word"]))
            t_word_lookup = t_word_lookup[1:]
            pii_res = detector.detect(t_seg["text"])

            for pii_entity in pii_res["entities"]:
                t_word_start = t_seg["words"][t_word_lookup[pii_entity["offset"] - 1]]
                t_word_end = t_seg["words"][t_word_lookup[pii_entity["offset"] + pii_entity["length"] - 1]]
                pii_timestamps.append((t_word_start["start"], t_word_end["end"]))

        CensorBeep.run(input_path=input_path, output_path=output_path, timestamps=pii_timestamps, config=config)
