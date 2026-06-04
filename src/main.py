import os
import shutil
import threading

import tkinter as tk
from tkinter import filedialog as fd

from video.video_io import VideoIO
import audio.censor as audio_censor
import video.censor as video_censor

VALID_EXTENSIONS = [".mp4", ".avi", ".webm", ".ogv", ".mkv", ".wmv"]

tk.Tk().withdraw()
file_path = fd.askopenfilename(
    title="Select video file to censor",
    filetypes=[("Video files", " ".join(f"*{ext}" for ext in VALID_EXTENSIONS)), ("All files", "*.*")],
)
file_extension = file_path[-4:]
if file_extension not in VALID_EXTENSIONS:
    print(f"Unsupported file type: {file_extension}")
    exit(1)

file_name = os.path.splitext(os.path.basename(file_path))[0]
base_dir = os.environ["BASE_DIRECTORY"]

input_dir = os.path.join(base_dir, "input")
pending_dir = os.path.join(base_dir, "pending")
output_dir = os.path.join(base_dir, "output")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(pending_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

input_audio_path = os.path.join(input_dir, f"{file_name}.wav")
input_video_path = os.path.join(input_dir, f"{file_name}{file_extension}")
pending_audio_path = os.path.join(pending_dir, f"{file_name}.wav")
pending_video_path = os.path.join(pending_dir, f"{file_name}.mp4")
output_video_path = os.path.join(output_dir, f"{file_name}.mp4")


def run_audio_pipeline():
    audio_config = audio_censor.AudioCensorConfig()
    VideoIO.extract_audio_from_video(file_path, input_audio_path)
    audio_censor.AudioCensor.run(input_audio_path, pending_audio_path, audio_config)


def run_video_pipeline():
    shutil.copy2(file_path, input_video_path)
    video_config = video_censor.VideoCensorConfig(device="cpu")
    video_censor.VideoCensor.run(input_video_path, pending_video_path, video_config)


try:
    audio_thread = threading.Thread(target=run_audio_pipeline)
    video_thread = threading.Thread(target=run_video_pipeline)

    audio_thread.start()
    video_thread.start()
    audio_thread.join()
    video_thread.join()

    VideoIO.combine_audio_video(pending_video_path, pending_audio_path, output_video_path)
except Exception as e:
    print(f"Error: {e}")
finally:
    for path in [input_audio_path, input_video_path, pending_audio_path, pending_video_path]:
        if os.path.exists(path):
            os.remove(path)
