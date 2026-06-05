import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLACT_ROOT = REPO_ROOT / "yolact_edge"

if str(YOLACT_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLACT_ROOT))

from yolact_edge.data import COLORS, cfg, set_cfg, set_dataset  # noqa: E402
from yolact_edge.layers.output_utils import postprocess  # noqa: E402
from yolact_edge.utils.augmentations import BaseTransform, FastBaseTransform  # noqa: E402
from yolact_edge.utils.tensorrt import convert_to_tensorrt  # noqa: E402
from yolact_edge.yolact import Yolact  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_runtime_args(
    weights: str,
    score_threshold: float,
    top_k: int,
    disable_tensorrt: bool,
    use_cpu: bool,
) -> argparse.Namespace:
    cuda = not use_cpu and torch.cuda.is_available()
    return SimpleNamespace(
        trained_model=weights,
        weights=weights,
        score_threshold=score_threshold,
        top_k=top_k,
        disable_tensorrt=disable_tensorrt,
        cuda=cuda,
        cpu=use_cpu,
        fast_nms=True,
        display_lincomb=False,
        mask_proto_debug=False,
        crop=True,
        use_fp16_tensorrt=False,
        use_tensorrt_safe_mode=False,
        trt_batch_size=1,
        calib_images=None,
        drop_weights=None,
        coco_transfer=False,
        yolact_transfer=False,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def timestamp_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"


def fit_for_display(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def combine_views(original_bgr: np.ndarray, result_bgr: np.ndarray) -> np.ndarray:
    if original_bgr.shape[:2] != result_bgr.shape[:2]:
        result_bgr = cv2.resize(
            result_bgr,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    separator = np.full((original_bgr.shape[0], 12, 3), 40, dtype=np.uint8)
    return np.hstack([original_bgr, separator, result_bgr])


class YolactInferenceBackend:
    def __init__(
        self,
        weights: str,
        config: str = "yolact_edge_config",
        dataset: str = "coco2017_dataset",
        score_threshold: float = 0.3,
        top_k: int = 15,
        disable_tensorrt: bool = True,
        use_cpu: bool = False,
    ) -> None:
        self.weights = str(weights)
        self.config = config
        self.dataset = dataset
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.disable_tensorrt = disable_tensorrt
        self.use_cpu = use_cpu
        self.runtime_args = build_runtime_args(
            self.weights,
            self.score_threshold,
            self.top_k,
            self.disable_tensorrt,
            self.use_cpu,
        )
        self.device = torch.device(
            "cpu" if self.runtime_args.cpu or not torch.cuda.is_available() else "cuda"
        )
        self.transform = FastBaseTransform()
        self.color_cache = {}
        self.mask_alpha = 0.45
        self._setup_model()

    def _setup_model(self) -> None:
        set_cfg(self.config)
        set_dataset(self.dataset)
        cfg.eval_mask_branch = True
        cfg.mask_proto_debug = False

        torch.set_grad_enabled(False)
        if self.device.type == "cuda":
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        net = Yolact(training=False)
        net.load_weights(self.weights, args=self.runtime_args)
        net.eval()

        if not self.disable_tensorrt:
            convert_to_tensorrt(net, cfg, self.runtime_args, transform=BaseTransform())

        self.net = net.to(self.device)
        self.net.detect.use_fast_nms = True

    def _get_color(self, color_idx: int, bgr: bool) -> Tuple[int, int, int]:
        cache_key = (color_idx, bgr)
        cached = self.color_cache.get(cache_key)
        if cached is not None:
            return cached

        color = COLORS[color_idx % len(COLORS)]
        if bgr:
            color = (int(color[2]), int(color[1]), int(color[0]))
        else:
            color = (int(color[0]), int(color[1]), int(color[2]))
        self.color_cache[cache_key] = color
        return color

    def predict(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        frame_tensor = torch.from_numpy(frame_bgr).to(self.device).float()
        batch = self.transform(frame_tensor.unsqueeze(0))
        extras = {
            "backbone": "full",
            "interrupt": False,
            "keep_statistics": False,
            "moving_statistics": None,
        }
        preds = self.net(batch, extras=extras)["pred_outs"]
        result_bgr = self._draw_predictions(
            preds,
            frame_tensor,
            frame_bgr.shape[0],
            frame_bgr.shape[1],
        )
        combined = combine_views(frame_bgr, result_bgr)
        return result_bgr, combined

    def process_image_file(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        result_bgr, combined = self.predict(image)
        return image, result_bgr, combined

    def export_video(
        self,
        input_path: str,
        output_path: str,
        save_side_by_side: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_size = (width * 2 + 12, height) if save_side_by_side else (width, height)
        output_file = Path(output_path)
        ensure_parent(output_file)

        writer = cv2.VideoWriter(
            str(output_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            output_size,
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create output video: {output_path}")

        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result_bgr, combined = self.predict(frame)
                writer.write(combined if save_side_by_side else result_bgr)
                processed += 1

                if progress_callback is not None:
                    progress_callback(processed, frame_count)
        finally:
            cap.release()
            writer.release()

    def _draw_predictions(
        self,
        dets_out,
        frame_tensor: torch.Tensor,
        height: int,
        width: int,
    ) -> np.ndarray:
        processed = postprocess(
            dets_out,
            width,
            height,
            crop_masks=True,
            score_threshold=self.score_threshold,
        )

        classes = processed[0][: self.top_k].detach().cpu().numpy()
        scores = processed[1][: self.top_k].detach().cpu().numpy()
        boxes = processed[2][: self.top_k].detach().cpu().numpy()
        masks = processed[3][: self.top_k] if cfg.eval_mask_branch else None

        count = min(self.top_k, len(classes))
        for idx in range(count):
            if scores[idx] < self.score_threshold:
                count = idx
                break

        base = frame_tensor.detach().cpu().numpy().astype(np.uint8)
        output = base.copy()

        if count == 0:
            return output

        if masks is not None:
            for idx in range(count):
                mask = masks[idx].detach().cpu().numpy()
                color = np.array(self._get_color(idx * 5, bgr=True), dtype=np.float32)
                colored = np.zeros_like(output, dtype=np.float32)
                colored[:, :] = color
                mask_3c = mask[:, :, None].astype(np.float32)
                output = (
                    output.astype(np.float32) * (1 - mask_3c * self.mask_alpha)
                    + colored * (mask_3c * self.mask_alpha)
                ).astype(np.uint8)

        for idx in range(count):
            x1, y1, x2, y2 = boxes[idx, :].astype(int)
            color = self._get_color(idx * 5, bgr=True)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            class_name = cfg.dataset.class_names[int(classes[idx])]
            label = f"{class_name}: {scores[idx]:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                1,
            )
            text_y = y1 - 8 if y1 - 8 > text_h else y1 + text_h + 8
            box_top = text_y - text_h - 6
            box_bottom = text_y + 2
            cv2.rectangle(output, (x1, box_top), (x1 + text_w + 8, box_bottom), color, -1)
            cv2.putText(
                output,
                label,
                (x1 + 4, text_y - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output
