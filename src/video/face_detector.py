import os
import cv2
import numpy as np

from huggingface_hub import hf_hub_download

from config.video_censor_config import VideoCensorConfig


class FaceDetector:
    def __init__(self, config: VideoCensorConfig):
        self.config = config
        hf_token = os.environ["HF_TOKEN"]
        local_dir = os.path.join(os.environ["MODELS_DIRECTORY"], self.config.fd_repo_id.rsplit("/", 1)[-1])
        model_path = hf_hub_download(
            repo_id=self.config.fd_repo_id, filename=self.config.fd_file_name, local_dir=local_dir, token=hf_token
        )

        if self.config.device == "cuda":
            backend_id = cv2.dnn.DNN_BACKEND_CUDA
            target_id = cv2.dnn.DNN_TARGET_CUDA
        else:
            backend_id = cv2.dnn.DNN_BACKEND_DEFAULT
            target_id = cv2.dnn.DNN_TARGET_CPU

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

    def run(self, input: cv2.typing.MatLike) -> np.ndarray:
        input_size = (input.shape[1], input.shape[0])
        if self.detector.getInputSize() != input_size:
            self.detector.setInputSize(input_size)

        _, faces = self.detector.detect(input)
        return faces
