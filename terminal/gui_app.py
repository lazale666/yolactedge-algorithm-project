import threading
import time
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from inference_backend import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    YolactInferenceBackend,
    ensure_parent,
    timestamp_name,
)


class VisualSegApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLACT Edge 可视化界面")
        self.geometry("1480x920")
        self.minsize(1260, 800)

        self.weights_var = tk.StringVar(
            value=str(REPO_ROOT / "yolact_edge" / "weights" / "yolact_edge_54_800000.pth")
        )
        self.score_var = tk.DoubleVar(value=0.3)
        self.side_by_side_var = tk.BooleanVar(value=False)
        self.disable_tensorrt_var = tk.BooleanVar(value=True)
        self.preview_scale_var = tk.StringVar(value="640")
        self.camera_index_var = tk.StringVar(value="0")
        self.mode_var = tk.StringVar(value="当前模式：空闲")
        self.status_var = tk.StringVar(value="准备就绪，程序启动后将自动加载默认模型。")

        self.backend: Optional[YolactInferenceBackend] = None
        self.current_source_kind: Optional[str] = None
        self.current_source_name: Optional[str] = None
        self.video_path: Optional[str] = None

        self.capture = None
        self.preview_thread = None
        self.preview_queue = Queue(maxsize=1)
        self.preview_session_id = 0
        self.preview_frame_count = 0
        self.preview_start_time = 0.0
        self.preview_running = False

        self.export_thread = None
        self.recording_writer = None
        self.recording_path: Optional[Path] = None
        self.recording_active = False

        self.current_original = None
        self.current_result = None
        self.current_combined = None
        self.current_photo_original = None
        self.current_photo_result = None

        self._build_layout()
        self.update_button_states()
        self.after(300, self.try_auto_load_model)

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

        ttk.Label(control_frame, text="预览边长").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            control_frame,
            textvariable=self.preview_scale_var,
            values=["480", "640", "800", "960", "原始"],
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="w", pady=(10, 0))

        ttk.Label(control_frame, text="摄像头索引").grid(row=2, column=2, sticky="e", pady=(10, 0))
        ttk.Entry(control_frame, textvariable=self.camera_index_var, width=8).grid(
            row=2, column=3, sticky="w", pady=(10, 0)
        )

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 0))

        ttk.Button(button_frame, text="打开图片", command=self.open_image).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(button_frame, text="保存图片", command=self.save_current_image).grid(row=0, column=1, padx=6)
        ttk.Button(button_frame, text="打开视频", command=self.open_video).grid(row=0, column=2, padx=6)
        self.play_video_button = ttk.Button(button_frame, text="播放视频", command=self.play_video)
        self.play_video_button.grid(row=0, column=3, padx=6)
        self.start_camera_button = ttk.Button(button_frame, text="启动摄像头", command=self.start_camera)
        self.start_camera_button.grid(row=0, column=4, padx=6)
        self.stop_button = ttk.Button(button_frame, text="停止预览", command=self.stop_preview)
        self.stop_button.grid(row=0, column=5, padx=6)
        ttk.Button(button_frame, text="导出视频", command=self.export_video).grid(row=0, column=6, padx=6)

        tool_frame = ttk.Frame(control_frame)
        tool_frame.grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.snapshot_button = ttk.Button(tool_frame, text="截图当前帧", command=self.snapshot_current_frame)
        self.snapshot_button.grid(row=0, column=0, padx=(0, 6))
        self.record_button = ttk.Button(tool_frame, text="开始录制", command=self.toggle_recording)
        self.record_button.grid(row=0, column=1, padx=6)
        ttk.Label(tool_frame, textvariable=self.mode_var).grid(row=0, column=2, padx=(16, 0), sticky="w")

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

    def try_auto_load_model(self) -> None:
        if self.backend is None:
            self.load_model(show_success=False)

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

    def set_mode(self, mode_text: str) -> None:
        self.mode_var.set("当前模式：%s" % mode_text)

    def update_button_states(self) -> None:
        if self.preview_running:
            self.play_video_button.configure(state="disabled")
            self.start_camera_button.configure(state="disabled")
            self.snapshot_button.configure(state="normal")
            self.record_button.configure(state="normal")
            self.stop_button.configure(state="normal")
        else:
            self.play_video_button.configure(state="normal")
            self.start_camera_button.configure(state="normal")
            self.snapshot_button.configure(state="normal" if self.current_result is not None else "disabled")
            self.record_button.configure(state="disabled")
            self.stop_button.configure(state="normal")

        if self.recording_active:
            self.record_button.configure(text="停止录制")
        else:
            self.record_button.configure(text="开始录制")

    def ensure_backend(self) -> bool:
        if self.backend is not None:
            return True
        return self.load_model()

    def load_model(self, show_success: bool = True) -> bool:
        weights = self.weights_var.get().strip()
        if not weights:
            messagebox.showerror("错误", "权重路径为空。")
            return False
        if not Path(weights).exists():
            messagebox.showerror("错误", "未找到模型权重：\n%s" % weights)
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
        if show_success:
            self.set_mode("空闲")
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
        self.update_button_states()

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

        self.current_source_kind = "image"
        self.current_source_name = path
        self.current_combined = combined
        self.render_panels(original_bgr, result_bgr)
        self.set_mode("图片")
        self.set_status("已加载图片：%s" % path)

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
            messagebox.showerror("保存失败", "保存图片失败：\n%s" % target)
            return

        self.set_status("已保存图片：%s" % target)

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

        self.stop_preview()
        self.video_path = path
        self.current_source_kind = "video"
        self.current_source_name = path
        self.set_mode("视频")
        self.set_status("已选择视频：%s" % path)

    def play_video(self) -> None:
        if not self.ensure_backend():
            return
        if not self.video_path:
            messagebox.showinfo("提示", "请先选择一个视频文件。")
            return
        if self.preview_running:
            self.set_status("当前已有预览在运行，请先停止。")
            return

        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            messagebox.showerror("视频错误", "打开视频失败：\n%s" % self.video_path)
            return

        self.current_source_kind = "video"
        self.current_source_name = self.video_path
        self.start_preview_session(capture)

    def start_camera(self) -> None:
        if not self.ensure_backend():
            return
        if self.preview_running:
            self.set_status("当前已有预览在运行，请先停止。")
            return

        try:
            camera_index = int(self.camera_index_var.get().strip())
        except ValueError:
            messagebox.showerror("摄像头错误", "摄像头索引必须是整数。")
            return

        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            messagebox.showerror("摄像头错误", "打开摄像头失败，索引为：%s" % camera_index)
            return

        self.current_source_kind = "camera"
        self.current_source_name = "camera:%s" % camera_index
        self.start_preview_session(capture)

    def start_preview_session(self, capture) -> None:
        self.stop_preview()
        self.capture = capture
        self.preview_session_id += 1
        session_id = self.preview_session_id
        self.preview_running = True
        self.preview_frame_count = 0
        self.preview_start_time = time.time()
        self.clear_preview_queue()
        self.update_button_states()

        if self.current_source_kind == "camera":
            self.set_mode("摄像头")
            self.set_status("正在预览摄像头：%s" % self.current_source_name)
        else:
            self.set_mode("视频")
            self.set_status("正在播放视频：%s" % self.current_source_name)

        self.preview_thread = threading.Thread(
            target=self.preview_loop,
            args=(session_id, capture, self.get_preview_max_side()),
            daemon=True,
        )
        self.preview_thread.start()
        self.after(15, lambda: self.consume_preview_queue(session_id))

    def get_preview_max_side(self) -> Optional[int]:
        value = self.preview_scale_var.get().strip()
        if value == "原始":
            return None
        try:
            return int(value)
        except ValueError:
            return 640

    def clear_preview_queue(self) -> None:
        while True:
            try:
                self.preview_queue.get_nowait()
            except Empty:
                break

    def preview_loop(self, session_id: int, capture, preview_max_side: Optional[int]) -> None:
        if capture is None or self.backend is None:
            return

        while self.preview_running and session_id == self.preview_session_id:
            ok, frame = capture.read()
            if not ok:
                self.push_preview_item(session_id, "done", None)
                return

            try:
                result_bgr, combined = self.backend.predict(
                    frame,
                    inference_max_side=preview_max_side,
                )
            except Exception as exc:
                self.push_preview_item(session_id, "error", str(exc))
                return

            self.preview_frame_count += 1
            elapsed = max(time.time() - self.preview_start_time, 1e-6)
            fps = self.preview_frame_count / elapsed
            self.push_preview_item(session_id, "frame", (frame, result_bgr, combined, fps))

    def push_preview_item(self, session_id: int, item_type: str, payload) -> None:
        if self.preview_queue.full():
            try:
                self.preview_queue.get_nowait()
            except Empty:
                pass

        try:
            self.preview_queue.put_nowait((session_id, item_type, payload))
        except Exception:
            pass

    def consume_preview_queue(self, session_id: int) -> None:
        if not self.preview_running or session_id != self.preview_session_id:
            return

        latest_item = None
        while True:
            try:
                latest_item = self.preview_queue.get_nowait()
            except Empty:
                break

        if latest_item is not None:
            item_session_id, item_type, payload = latest_item
            if item_session_id != self.preview_session_id:
                self.after(15, lambda: self.consume_preview_queue(session_id))
                return

            if item_type == "frame":
                original_bgr, result_bgr, combined, fps = payload
                self.current_combined = combined
                self.render_panels(original_bgr, result_bgr)
                self.write_recording_frame()
                if self.current_source_kind == "camera":
                    self.set_status("正在预览摄像头：%s | 预览 FPS: %.2f" % (self.current_source_name, fps))
                else:
                    self.set_status("正在播放视频：%s | 预览 FPS: %.2f" % (self.current_source_name, fps))
            elif item_type == "done":
                self.stop_preview()
                if self.current_source_kind == "camera":
                    self.set_status("摄像头预览已停止。")
                else:
                    self.set_status("视频播放结束。")
                return
            elif item_type == "error":
                self.stop_preview()
                messagebox.showerror("推理错误", payload)
                return

        self.after(15, lambda: self.consume_preview_queue(session_id))

    def snapshot_current_frame(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("提示", "当前没有可截图的画面。")
            return

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / timestamp_name("snapshot", ".jpg")
        image_to_save = self.current_combined if self.side_by_side_var.get() else self.current_result
        ensure_parent(output_path)
        if not cv2.imwrite(str(output_path), image_to_save):
            messagebox.showerror("截图失败", "保存截图失败：\n%s" % output_path)
            return

        self.set_status("已保存截图：%s" % output_path)

    def toggle_recording(self) -> None:
        if not self.preview_running:
            messagebox.showinfo("提示", "请先启动视频或摄像头预览。")
            return

        if self.recording_active:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("提示", "当前没有可录制的画面。")
            return

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / timestamp_name("record", ".mp4")
        frame_to_write = self.current_combined if self.side_by_side_var.get() else self.current_result
        height, width = frame_to_write.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (width, height),
        )
        if not writer.isOpened():
            messagebox.showerror("录制失败", "无法创建录制文件：\n%s" % output_path)
            return

        self.recording_writer = writer
        self.recording_path = output_path
        self.recording_active = True
        self.update_button_states()
        self.set_status("开始录制：%s" % output_path)

    def write_recording_frame(self) -> None:
        if not self.recording_active or self.recording_writer is None or self.current_result is None:
            return

        frame_to_write = self.current_combined if self.side_by_side_var.get() else self.current_result
        try:
            self.recording_writer.write(frame_to_write)
        except Exception:
            self.stop_recording()
            messagebox.showerror("录制失败", "写入录制视频时出错。")

    def stop_recording(self) -> None:
        if self.recording_writer is not None:
            try:
                self.recording_writer.release()
            except Exception:
                pass
        saved_path = self.recording_path
        self.recording_writer = None
        self.recording_path = None
        self.recording_active = False
        self.update_button_states()
        if saved_path is not None:
            self.set_status("已保存录制视频：%s" % saved_path)

    def stop_preview(self) -> None:
        self.preview_session_id += 1
        self.preview_running = False
        self.update_button_states()
        self.stop_recording()
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None
        self.clear_preview_queue()
        if self.current_source_kind in ("video", "camera"):
            self.set_mode("空闲")

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
                message = "正在导出视频：%d/%d" % (processed, total)
            else:
                message = "正在导出视频：%d" % processed
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

            self.after(0, lambda: self.set_status("已保存视频：%s" % target))

        self.export_thread = threading.Thread(target=run_export, daemon=True)
        self.export_thread.start()

    def destroy(self) -> None:
        self.stop_preview()
        super().destroy()


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = VisualSegApp()
    app.mainloop()


if __name__ == "__main__":
    main()
