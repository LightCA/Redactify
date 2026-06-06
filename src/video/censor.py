import cv2
import logging

from pathlib import Path

from config.video_censor_config import VideoCensorConfig
from video.face_detector import FaceDetector
from image.mask_blur import MaskCensor


class VideoCensor:
    logger = logging.getLogger(__name__)

    @staticmethod
    def run(input_path: Path, output_path: Path, config: VideoCensorConfig | None = None) -> None:
        if config is None:
            config = VideoCensorConfig()

        detector = FaceDetector(config=config)

        cap = cv2.VideoCapture(input_path)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        blur = MaskCensor(config=config)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(frame_w, frame_h), isColor=True)

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                VideoCensor.logger.error("Error: Couldn't retrieve frame")
                raise RuntimeError("Couldn't retrieve frame")

            faces = detector.run(frame)

            if faces is not None and faces.any():
                frame = blur.run(frame, faces)

            out.write(frame)

        cap.release()
        out.release()
