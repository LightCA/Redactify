import cv2

from dataclasses import dataclass


@dataclass
class VideoCensorConfig:
    _init: bool = False
    device: str = "auto"

    # Face detection parameters
    fd_repo_id: str = "opencv/face_detection_yunet"
    fd_file_name: str = "face_detection_yunet_2023mar.onnx"
    fd_score_threshold: float = 0.3
    fd_nms_threshold: float = 0.3
    fd_top_k: int = 5000
    fd_input_max_size: int = 0  # 0 disables rescaling

    # blur parameters
    blur_downsample_resolution: int = 10
    blur_downsample_interpolation: int = cv2.INTER_AREA
    blur_upsample_interpolation: int = cv2.INTER_CUBIC
    blur_expansion: float = 0.2

    # blur feathering parameters
    blur_feather_inner: float = 0.75
    blur_feather_outer: float = 1.0

    def __post_init__(self) -> None:
        if not self._init:
            self._init = True
            if self.device == "auto":
                self.device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
            elif self.device not in ["cpu", "cuda"]:
                raise ValueError(f"Invalid device: {self.device}")
