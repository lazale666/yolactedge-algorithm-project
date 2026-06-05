import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from inference_backend import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    YolactInferenceBackend,
    ensure_parent,
)


class VisualSegApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLACT Edge 可视化界面")
        self.geometry("1380x860")
        self.minsize(1200, 760)

        self.weights_var = tk.StringVar(
            value=str(REPO_ROOT / "yolact_edge" / "weights" / "yolact_edge_54_800000.pth")
        )
        self.score_var = tk.DoubleVar(value=0.3)
        self.side_by_side_var = tk.BooleanVar(value=False)
        self.disable_tensorrt_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="准备就绪，请先加载模型。")

        self.backend: Optional[YolactInferenceBackend] = None
        self.video_cap = None
        self.video_path: Optional[str] = None
        self.current_original = None
        self.current_result = None
        self.current_combined = None
        self.current_photo_original = None
        self.current_photo_result = None
        self.video_running = False
        self.export_thread = None

        self._build_layout()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        control_frame = ttk.Frame(self, padding=12)
        control_frame.grid(row=0, column=0, sticky="ew")
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="权重").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(control_frame, textvariable=self.weights_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(control_frame, text="浏览", command=self.choose_weights).grid(row=0, column=2, padx=6)
        ttk.Button(control_frame, text="加载模型", command=self.load_model).grid(row=0, column=3)

        ttk.Label(control_frame, text="阈值").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Scale(
            control_frame,
            from_=0.05,
            to=0.95,
            orient="horizontal",
            variable=self.score_var,
        ).grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(
            control_frame,
            text="禁用 TensorRT",
            variable=self.disable_tensorrt_var,
        ).grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            control_frame,
            text="导出拼接图",
            variable=self.side_by_side_var,
        ).grid(row=1, column=3, sticky="w", pady=(10, 0))

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(12, 0))

        ttk.Button(button_frame, text="打开图片", command=self.open_image).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(button_frame, text="保存图片", command=self.save_current_image).grid(row=0, column=1, padx=6)
        ttk.Button(button_frame, text="打开视频", command=self.open_video).grid(row=0, column=2, padx=6)
        ttk.Button(button_frame, text="播放视频", command=self.play_video).grid(row=0, column=3, padx=6)
        ttk.Button(button_frame, text="停止播放", command=self.stop_video).grid(row=0, column=4, padx=6)
        ttk.Button(button_frame, text="导出视频", command=self.export_video).grid(row=0, column=5, padx=6)

        viewer_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        viewer_frame.grid(row=1, column=0, sticky="nsew")
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.columnconfigure(1, weight=1)
        viewer_frame.rowconfigure(1, weight=1)

        ttk.Label(viewer_frame, text="原始画面").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(viewer_frame, text="实例分割结果").grid(row=0, column=1, sticky="w", pady=(0, 6))

        self.original_label = ttk.Label(viewer_frame, anchor="center", relief="solid")
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.result_label = ttk.Label(viewer_frame, anchor="center", relief="solid")
        self.result_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(12, 6))
        status_bar.grid(row=2, column=0, sticky="ew")

    def choose_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="选择模型权重",
            filetypes=[("PyTorch 权重", "*.pth"), ("所有文件", "*.*")],
        )
        if path:
            self.weights_var.set(path)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.update_idletasks()

    def ensure_backend(self) -> bool:
        if self.backend is not None:
            return True
        return self.load_model()

    def load_model(self) -> bool:
        weights = self.weights_var.get().strip()
        if not weights:
            messagebox.showerror("错误", "权重路径为空。")
            return False
        if not Path(weights).exists():
            messagebox.showerror("错误", f"未找到模型权重：\n{weights}")
            return False

        self.set_status("正在加载模型，请稍候……")
        self.update()
        try:
            self.backend = YolactInferenceBackend(
                weights=weights,
                score_threshold=float(self.score_var.get()),
                disable_tensorrt=self.disable_tensorrt_var.get(),
            )
        except Exception as exc:
            self.backend = None
            self.set_status("模型加载失败。")
            messagebox.showerror("模型加载失败", str(exc))
            return False

        self.set_status("模型已加载。")
        return True

    def render_panels(self, original_bgr, result_bgr) -> None:
        self.current_original = original_bgr
        self.current_result = result_bgr

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        left_image = self.prepare_tk_image(original_rgb, 620, 620)
        right_image = self.prepare_tk_image(result_rgb, 620, 620)

        self.current_photo_original = left_image
        self.current_photo_result = right_image

        self.original_label.configure(image=left_image)
        self.result_label.configure(image=right_image)

    def prepare_tk_image(self, rgb_image, max_width: int, max_height: int):
        image = Image.fromarray(rgb_image)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def open_image(self) -> None:
        if not self.ensure_backend():
            return

        path = filedialog.askopenfilename(
            title="打开图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        try:
            original_bgr, result_bgr, combined = self.backend.process_image_file(path)
        except Exception as exc:
            messagebox.showerror("图片处理错误", str(exc))
            return

        self.current_combined = combined
        self.render_panels(original_bgr, result_bgr)
        self.set_status(f"已加载图片：{path}")

    def save_current_image(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("提示", "当前没有可保存的处理结果。")
            return

        target = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".jpg",
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png"), ("所有文件", "*.*")],
        )
        if not target:
            return

        image_to_save = self.current_combined if self.side_by_side_var.get() else self.current_result
        ensure_parent(Path(target))
        if not cv2.imwrite(target, image_to_save):
            messagebox.showerror("保存失败", f"保存图片失败：\n{target}")
            return

        self.set_status(f"已保存图片：{target}")

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="打开视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        self.stop_video()
        self.video_path = path
        self.set_status(f"已选择视频：{path}")

    def play_video(self) -> None:
        if not self.ensure_backend():
            return
        if not self.video_path:
            messagebox.showinfo("提示", "请先选择一个视频文件。")
            return

        self.stop_video()
        self.video_cap = cv2.VideoCapture(self.video_path)
        if not self.video_cap.isOpened():
            self.video_cap = None
            messagebox.showerror("视频错误", f"打开视频失败：\n{self.video_path}")
            return

        self.video_running = True
        self.set_status(f"正在播放视频：{self.video_path}")
        self.after(1, self.update_video_frame)

    def update_video_frame(self) -> None:
        if not self.video_running or self.video_cap is None or self.backend is None:
            return

        ok, frame = self.video_cap.read()
        if not ok:
            self.stop_video()
            self.set_status("视频播放结束。")
            return

        try:
            result_bgr, combined = self.backend.predict(frame)
        except Exception as exc:
            self.stop_video()
            messagebox.showerror("推理错误", str(exc))
            return

        self.current_combined = combined
        self.render_panels(frame, result_bgr)
        self.after(1, self.update_video_frame)

    def stop_video(self) -> None:
        self.video_running = False
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

    def export_video(self) -> None:
        if not self.ensure_backend():
            return
        if not self.video_path:
            messagebox.showinfo("提示", "请先选择一个视频文件。")
            return
        if self.export_thread is not None and self.export_thread.is_alive():
            messagebox.showinfo("提示", "视频导出任务正在进行中。")
            return

        target = filedialog.asksaveasfilename(
            title="导出视频",
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4"), ("所有文件", "*.*")],
        )
        if not target:
            return

        self.set_status("正在导出视频……")

        def progress_callback(processed: int, total: int) -> None:
            if total > 0:
                message = f"正在导出视频：{processed}/{total}"
            else:
                message = f"正在导出视频：{processed}"
            self.after(0, lambda: self.set_status(message))

        def run_export() -> None:
            try:
                self.backend.export_video(
                    self.video_path,
                    target,
                    save_side_by_side=self.side_by_side_var.get(),
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("导出失败", str(exc)))
                self.after(0, lambda: self.set_status("视频导出失败。"))
                return

            self.after(0, lambda: self.set_status(f"已保存视频：{target}"))

        self.export_thread = threading.Thread(target=run_export, daemon=True)
        self.export_thread.start()

    def destroy(self) -> None:
        self.stop_video()
        super().destroy()


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = VisualSegApp()
    app.mainloop()


if __name__ == "__main__":
    main()
