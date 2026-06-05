import argparse
from pathlib import Path

import cv2

from inference_backend import (
    DEFAULT_OUTPUT_DIR,
    YolactInferenceBackend,
    ensure_parent,
    fit_for_display,
    timestamp_name,
)


WINDOW_NAME = "YOLACT Edge Visual Terminal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLACT Edge 终端可视化工具。")
    parser.add_argument("--mode", choices=["image", "video", "camera"], required=True, help="运行模式。")
    parser.add_argument("--input", help="图片或视频路径。")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引。")
    parser.add_argument(
        "--weights",
        default=str(Path(__file__).resolve().parents[1] / "yolact_edge" / "weights" / "yolact_edge_54_800000.pth"),
        help="模型权重路径。",
    )
    parser.add_argument("--config", default="yolact_edge_config", help="模型配置名。")
    parser.add_argument("--dataset", default="coco2017_dataset", help="数据集配置名。")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="置信度阈值。")
    parser.add_argument("--top-k", type=int, default=15, help="最多显示目标数量。")
    parser.add_argument("--disable-tensorrt", action="store_true", help="禁用 TensorRT。")
    parser.add_argument("--output", help="导出图片或视频路径。")
    parser.add_argument("--save-side-by-side", action="store_true", help="导出原图和结果拼接图。")
    parser.add_argument("--display-width", type=int, default=1600, help="窗口最大显示宽度。")
    parser.add_argument("--display-height", type=int, default=900, help="窗口最大显示高度。")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 推理。")
    return parser.parse_args()


def save_image(path: Path, image) -> None:
    ensure_parent(path)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"保存图片失败: {path}")


def run_image_mode(args: argparse.Namespace, backend: YolactInferenceBackend) -> None:
    if not args.input:
        raise ValueError("图片模式必须提供 --input")

    _, result_bgr, combined = backend.process_image_file(args.input)

    if args.output:
        output_path = Path(args.output)
        save_image(output_path, combined if args.save_side_by_side else result_bgr)
        print(f"已导出图片: {output_path}")

    print("快捷键: s 保存当前结果, q 或 Esc 退出")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        display = fit_for_display(combined, args.display_width, args.display_height)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord("s"):
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = DEFAULT_OUTPUT_DIR / timestamp_name("image_result", ".jpg")
            save_image(output_path, combined if args.save_side_by_side else result_bgr)
            print(f"已保存图片: {output_path}")

    cv2.destroyAllWindows()


def run_video_or_camera_mode(args: argparse.Namespace, backend: YolactInferenceBackend) -> None:
    if args.mode == "video":
        if not args.input:
            raise ValueError("视频模式必须提供 --input")
        cap = cv2.VideoCapture(str(args.input))
        source_name = args.input
    else:
        cap = cv2.VideoCapture(args.camera_index)
        source_name = f"camera:{args.camera_index}"

    if not cap.isOpened():
        raise RuntimeError(f"打开输入源失败: {source_name}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    export_path = Path(args.output) if args.output else None
    if export_path is not None:
        ensure_parent(export_path)
        output_size = (width * 2 + 12, height) if args.save_side_by_side else (width, height)
        writer = cv2.VideoWriter(
            str(export_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            output_size,
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"创建输出视频失败: {export_path}")
        print(f"已开始录制: {export_path}")

    recording = writer is not None
    print("快捷键: q 退出, s 保存当前帧, r 开始或暂停录制")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result_bgr, combined = backend.predict(frame)
            display_frame = fit_for_display(combined, args.display_width, args.display_height)
            cv2.imshow(WINDOW_NAME, display_frame)

            if recording and writer is not None:
                writer.write(combined if args.save_side_by_side else result_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                output_path = DEFAULT_OUTPUT_DIR / timestamp_name("frame_result", ".jpg")
                save_image(output_path, combined if args.save_side_by_side else result_bgr)
                print(f"已保存当前帧: {output_path}")
            if key == ord("r"):
                if writer is None:
                    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    export_path = DEFAULT_OUTPUT_DIR / timestamp_name("record_result", ".mp4")
                    output_size = (width * 2 + 12, height) if args.save_side_by_side else (width, height)
                    writer = cv2.VideoWriter(
                        str(export_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        output_size,
                    )
                    if not writer.isOpened():
                        writer = None
                        print("启动录制失败")
                    else:
                        recording = True
                        print(f"已开始录制: {export_path}")
                else:
                    recording = not recording
                    print("继续录制" if recording else "暂停录制")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            if export_path is not None:
                print(f"已保存视频: {export_path}")
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"未找到模型权重: {args.weights}")

    backend = YolactInferenceBackend(
        weights=args.weights,
        config=args.config,
        dataset=args.dataset,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        disable_tensorrt=args.disable_tensorrt,
        use_cpu=args.cpu,
    )

    if args.mode == "image":
        run_image_mode(args, backend)
    else:
        run_video_or_camera_mode(args, backend)


if __name__ == "__main__":
    main()
