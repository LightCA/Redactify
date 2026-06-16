import os
import cv2
import logging
import numpy as np

from pathlib import Path
from huggingface_hub import hf_hub_download

from config.video_censor_config import VideoCensorConfig


class FaceDetector:
    logger = logging.getLogger(__name__)

    def __init__(self, config: VideoCensorConfig):
        self.config = config
        hf_token = os.environ["HF_TOKEN"]
        models_dir = Path(os.environ["MODELS_DIRECTORY"])
        local_dir = models_dir / self.config.fd_repo_id.rsplit("/", 1)[-1]
        model_path = hf_hub_download(
            repo_id=self.config.fd_repo_id, filename=self.config.fd_file_name, local_dir=local_dir, token=hf_token
        )

        if self.config.device == "cuda":
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA
        else:
            backend_id = cv2.dnn.DNN_BACKEND_DEFAULT
            target_id = cv2.dnn.DNN_TARGET_CPU

        self.logger.info(f"FaceDetector using device: {self.config.device}")

        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(1, 1),
            score_threshold=self.config.fd_score_threshold,
            nms_threshold=self.config.fd_nms_threshold,
            top_k=self.config.fd_top_k,
            backend_id=backend_id,
            target_id=target_id,
        )

        self._last_input_size: tuple[int, int] | None = None
        self._resize_buffer: np.ndarray | None = None

    def run(self, input: cv2.typing.MatLike) -> np.ndarray:
        max_resolution = self.config.fd_input_max_size
        frame_h, frame_w = input.shape[0], input.shape[1]

        # Calculate the resolution to resize the frame
        if (max_resolution > 0) and ((frame_w > max_resolution) or (frame_h > max_resolution)):
            frame_ratio = frame_w / frame_h
            if frame_ratio > 1:
                scaled_w = max_resolution
                scaled_h = max(1, int(max_resolution / frame_ratio))
                scale = max_resolution / frame_w
            else:
                scaled_w = max(1, int(max_resolution * frame_ratio))
                scaled_h = max_resolution
                scale = max_resolution / frame_h

            if self._resize_buffer is None or self._resize_buffer.shape[:2] != (scaled_h, scaled_w):
                self._resize_buffer = np.empty((scaled_h, scaled_w, input.shape[2]), dtype=input.dtype)

            cv2.resize(input, (scaled_w, scaled_h), dst=self._resize_buffer, interpolation=cv2.INTER_AREA)
            detect_input = self._resize_buffer
        else:
            detect_input = input
            scale = 1.0

        input_size = (detect_input.shape[1], detect_input.shape[0])
        if self._last_input_size != input_size:
            self.detector.setInputSize(input_size)
            self._last_input_size = input_size

        _, faces = self.detector.detect(detect_input)

        if faces is not None and len(faces) > 0:
            if scale < 1:
                faces[:, :4] /= scale
            faces = np.column_stack((faces[:, :4], faces[:, -1]))

        return faces
