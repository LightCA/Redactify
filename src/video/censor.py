import cv2
import logging

from pathlib import Path

from config.video_censor_config import VideoCensorConfig
from image.face_detector import FaceDetector
from image.mask_blur import MaskCensor


class VideoCensor:
    logger = logging.getLogger(__name__)

    @classmethod
    def run(cls, input_path: Path, output_path: Path, config: VideoCensorConfig | None = None) -> None:
        if config is None:
            config = VideoCensorConfig()

        cap = cv2.VideoCapture(input_path)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(frame_w, frame_h), isColor=True)

        try:
            detector = FaceDetector(config=config)
            blur = MaskCensor(config=config)

            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    cls.logger.error("Error: Couldn't retrieve frame")
                    break

                faces = detector.run(frame)

                if faces is not None and faces.any():
                    frame = blur.run(frame, faces)

                out.write(frame)
        except Exception as ex:
            cls.logger.error(ex)
            raise ex
        finally:
            cap.release()
            out.release()
