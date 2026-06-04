import numpy as np
from scipy.io import wavfile
from config.audio_censor_config import AudioCensorConfig


class CensorBeep:
    @staticmethod
    def run(input_path: str, output_path: str, timestamps: list[tuple[float, float]], config: AudioCensorConfig) -> None:
        sample_rate, audio = wavfile.read(input_path)
        original_dtype = audio.dtype

        audio = audio.astype(np.float32)
        if np.issubdtype(original_dtype, np.integer):
            peak = float(np.iinfo(original_dtype).max)
        else:
            peak = 1.0

        n_samples = audio.shape[0]
        n_channels = audio.shape[1] if audio.ndim > 1 else 1

        ranges = []
        for start, end in timestamps:
            s = max(0, min(int((start - config.censor_padding) * sample_rate), n_samples))
            e = max(0, min(int((end + config.censor_padding) * sample_rate), n_samples))
            if s < e:
                ranges.append((s, e))
            else:
                print(f"Censor end was before start, skipping... ({start}, {end})")
        ranges.sort()

        merged: list[tuple[int, int]] = []
        for s, e in ranges:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        for s, e in merged:
            t = np.arange(e - s) / sample_rate
            beep = config.censor_amplitude * peak * np.sin(2 * np.pi * config.censor_frequency * t)
            if n_channels > 1:
                beep = np.repeat(beep[:, None], n_channels, axis=1)
            audio[s:e] = beep.astype(np.float32)

        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            audio = np.clip(audio, info.min, info.max)
        audio = audio.astype(original_dtype)

        wavfile.write(output_path, sample_rate, audio)
