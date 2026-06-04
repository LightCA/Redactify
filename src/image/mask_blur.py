import cv2
import numpy as np

from config.video_censor_config import VideoCensorConfig


class MaskCoordinates:
    def __init__(self, x1, y1, x2, y2, downsample_width, downsample_height):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.downsample_width = downsample_width
        self.downsample_height = downsample_height


class MaskCensor:
    def __init__(self, config: VideoCensorConfig):
        if config.blur_downsample_resolution <= 0:
            raise ValueError("blur_downsample_resolution must be greater than 0")

        self.config = config

        if config.device == "cuda":
            self.stream = cv2.cuda.Stream()

    @staticmethod
    def cv_type(img):
        depth = {
            np.dtype("uint8"): 0,
            np.dtype("int8"): 1,
            np.dtype("uint16"): 2,
            np.dtype("int16"): 3,
            np.dtype("int32"): 4,
            np.dtype("float32"): 5,
            np.dtype("float64"): 6,
        }[img.dtype]
        channels = 1 if img.ndim == 2 else img.shape[2]
        return depth + ((channels - 1) << 3)

    @staticmethod
    def create_feathered_alpha(width=720, height=1280, inner=0.0, outer=1.0):
        y, x = np.ogrid[:height, :width]
        nx = (x - width / 2) / (width / 2)
        ny = (y - height / 2) / (height / 2)
        t = np.clip((np.hypot(nx, ny) - inner) / (outer - inner), 0, 1).astype(np.float32)

        u = 1 - t
        alpha = u**3 + 3 * u**2 * t
        return alpha[..., None]

    def calculate_indices(self, mask, frame_w, frame_h) -> MaskCoordinates | None:
        mx, my, mw, mh = mask[:4]
        if mw <= 0 or mh <= 0:
            return None

        # Add margin to the bounding box
        margin = int(max(mw, mh) * self.config.blur_expansion)
        x1 = int(max(0, mx - margin))
        y1 = int(max(0, my - margin))
        x2 = int(min(frame_w, mx + mw + margin))
        y2 = int(min(frame_h, my + mh + margin))

        # Calculate the resolution to resize the masked area to for censoring
        if (self.config.blur_downsample_resolution > 0) and (
            (frame_w > self.config.blur_downsample_resolution) or (frame_h > self.config.blur_downsample_resolution)
        ):
            frame_ratio = mw / mh
            if frame_ratio > 1:
                mask_censor_width = self.config.blur_downsample_resolution
                mask_censor_height = max(1, int(self.config.blur_downsample_resolution / frame_ratio))
            else:
                mask_censor_width = max(1, int(self.config.blur_downsample_resolution * frame_ratio))
                mask_censor_height = self.config.blur_downsample_resolution
        else:
            return None

        return MaskCoordinates(x1, y1, x2, y2, mask_censor_width, mask_censor_height)

    def _blur_cuda(self, input_image, masks, frame_w, frame_h) -> cv2.typing.MatLike:
        type = self.cv_type(input_image)
        gpu_input = cv2.cuda.GpuMat(arr=input_image)
        gpu_masked_buffer = cv2.cuda.GpuMat(
            size=(self.config.blur_downsample_resolution, self.config.blur_downsample_resolution), type=type
        )

        # Sort by area (w * h) descending to overlap based on the size of the mask
        for mask in sorted(masks, key=lambda m: m[2] * m[3]):
            mask_coords = self.calculate_indices(mask, frame_w, frame_h)
            if mask_coords is None:
                continue

            # Create gpu mat reference to subsection of the input image and masked buffer
            gpu_censored = cv2.cuda.GpuMat(
                gpu_masked_buffer, (0, 0, mask_coords.downsample_width, mask_coords.downsample_height)
            )
            gpu_image_roi = cv2.cuda.GpuMat(
                gpu_input, (mask_coords.x1, mask_coords.y1, mask_coords.x2 - mask_coords.x1, mask_coords.y2 - mask_coords.y1)
            )

            # Downsample the masked area, then upsample back to the original size to achieve blur effect, and copy back to the gpu image
            cv2.cuda.resize(
                src=gpu_image_roi,
                dst=gpu_censored,
                dsize=(mask_coords.downsample_width, mask_coords.downsample_height),
                interpolation=self.config.blur_downsample_interpolation,
                stream=self.stream,
            )
            cv2.cuda.resize(
                src=gpu_censored,
                dst=gpu_image_roi,
                dsize=(mask_coords.x2 - mask_coords.x1, mask_coords.y2 - mask_coords.y1),
                interpolation=self.config.blur_upsample_interpolation,
                stream=self.stream,
            )
            self.stream.waitForCompletion()

        return gpu_input.download()

    def _blur_cpu(self, input_image: cv2.typing.MatLike, masks: np.ndarray, frame_w, frame_h) -> cv2.typing.MatLike:
        result = input_image.copy()
        resized_masked_buffer = np.zeros(
            (self.config.blur_downsample_resolution, self.config.blur_downsample_resolution, 3), dtype=np.uint8
        )
        masked_buffer = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Sort by area (w * h) descending to overlap based on the size of the mask
        for mask in sorted(masks, key=lambda m: m[2] * m[3]):
            mask_coords = self.calculate_indices(mask, frame_w, frame_h)
            if mask_coords is None:
                continue

            # Downsample the masked area, then upsample back to its original size to achieve blur effect, and copy back to the image using a feathered alpha
            cv2.resize(
                src=input_image[mask_coords.y1 : mask_coords.y2, mask_coords.x1 : mask_coords.x2],
                dst=resized_masked_buffer[: mask_coords.downsample_height, : mask_coords.downsample_width],
                dsize=(mask_coords.downsample_width, mask_coords.downsample_height),
                interpolation=self.config.blur_downsample_interpolation,
            )
            cv2.resize(
                src=resized_masked_buffer[: mask_coords.downsample_height, : mask_coords.downsample_width],
                dst=masked_buffer[mask_coords.y1 : mask_coords.y2, mask_coords.x1 : mask_coords.x2],
                dsize=(mask_coords.x2 - mask_coords.x1, mask_coords.y2 - mask_coords.y1),
                interpolation=self.config.blur_upsample_interpolation,
            )

            blur_alpha = self.create_feathered_alpha(
                mask_coords.x2 - mask_coords.x1,
                mask_coords.y2 - mask_coords.y1,
                inner=self.config.blur_feather_inner,
                outer=self.config.blur_feather_outer,
            )

            blended = masked_buffer[mask_coords.y1 : mask_coords.y2, mask_coords.x1 : mask_coords.x2] * blur_alpha + result[
                mask_coords.y1 : mask_coords.y2, mask_coords.x1 : mask_coords.x2
            ] * (1.0 - blur_alpha)

            result[mask_coords.y1 : mask_coords.y2, mask_coords.x1 : mask_coords.x2] = np.rint(blended).astype(np.uint8)

        return result

    def run(self, input_image: cv2.typing.MatLike, masks: np.ndarray) -> cv2.typing.MatLike:
        try:
            if masks is None or len(masks) == 0:
                return input_image

            frame_w = input_image.shape[1]
            frame_h = input_image.shape[0]

            if self.config.device == "cuda":
                result = self._blur_cuda(input_image, masks, frame_w, frame_h)
            else:
                result = self._blur_cpu(input_image, masks, frame_w, frame_h)

        except Exception as ex:
            print(f"Error in mask_blur: {ex}")
            raise

        return result
