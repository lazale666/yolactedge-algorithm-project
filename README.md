# yolactedge-algorithm-project

这是一个基于 `YOLACT Edge` 的实例分割工作区，当前主要面向 Windows 本地推理、结果可视化和图片/视频导出。

仓库目前包含三部分：

- `cocoapi/`
  `pycocotools` 源码依赖
- `yolact_edge/`
  `YOLACT Edge` 主项目源码、配置、权重、原始推理脚本
- `terminal/`
  新增的本地可视化工具，包括终端版和桌面 GUI

如果你的目标是：

- 选择一张图片做实例分割
- 界面同时显示原图和分割结果
- 选择视频并预览处理效果
- 一键导出处理后图片或视频

优先使用 `terminal/gui_app.py`，不建议再直接手敲长命令。

**目录结构**
```text
D:\All Program\yolactedge-algorithm-project
├─ .gitignore
├─ README.md
├─ cocoapi
│  ├─ common
│  ├─ PythonAPI
│  └─ ...
├─ yolact_edge
│  ├─ eval.py
│  ├─ INSTALL.md
│  ├─ README.md
│  ├─ README_CN.md
│  ├─ setup.py
│  ├─ weights
│  ├─ results
│  ├─ photo
│  └─ yolact_edge
│     ├─ data
│     ├─ layers
│     ├─ utils
│     ├─ inference.py
│     └─ yolact.py
└─ terminal
   ├─ inference_backend.py
   ├─ gui_app.py
   ├─ run_gui.bat
   ├─ visual_terminal.py
   └─ outputs
```

**关键目录说明**

- `yolact_edge/weights/`
  模型权重目录。当前仓库里已经有 `yolact_edge_54_800000.pth`，可直接作为默认推理模型。

- `terminal/inference_backend.py`
  可复用的推理后端。终端版和 GUI 都复用这套逻辑。

- `terminal/gui_app.py`
  本地桌面 GUI。支持选图、选视频、预览、导出。

- `terminal/run_gui.bat`
  GUI 一键启动脚本。优先用这个启动。

- `terminal/visual_terminal.py`
  终端版交互工具。适合调试或简化场景，不是主要入口。

**推荐环境**

- Windows
- Python `3.8`
- PyTorch `1.8.1+cu111`
- TorchVision `0.9.1+cu111`
- NVIDIA CUDA GPU

如果只是先把功能跑通，建议关闭 TensorRT。

**安装顺序**

1. 创建虚拟环境
```powershell
cd "D:\All Program\yolactedge-algorithm-project"
python -m venv .venv38
```

2. 安装 PyTorch
```powershell
.\.venv38\Scripts\python.exe -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

3. 安装基础依赖
```powershell
.\.venv38\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv38\Scripts\python.exe -m pip install cython numpy opencv-python pillow matplotlib GitPython termcolor tensorboard colorama
```

4. 安装 `pycocotools`
```powershell
cd "D:\All Program\yolactedge-algorithm-project\cocoapi\PythonAPI"
..\..\.venv38\Scripts\python.exe -m pip install .
```

5. 安装 `yolact_edge`
```powershell
cd "D:\All Program\yolactedge-algorithm-project\yolact_edge"
..\.venv38\Scripts\python.exe -m pip install .
```

更详细的安装文档见 [INSTALL.md](/D:/All%20Program/yolactedge-algorithm-project/yolact_edge/INSTALL.md:1)。

**GUI 启动方式**

最简单的方式：

```powershell
cd "D:\All Program\yolactedge-algorithm-project\terminal"
.\run_gui.bat
```

也可以直接用 Python：

```powershell
cd "D:\All Program\yolactedge-algorithm-project"
& "D:\All Program\yolactedge-algorithm-project\yolact_edge\.venv38\Scripts\python.exe" terminal\gui_app.py
```

**GUI 功能**

当前 GUI 支持：

- `Load Model`
  加载权重

- `Open Image`
  选择图片并显示原图和实例分割结果

- `Save Image`
  导出当前处理结果

- `Open Video`
  选择视频文件

- `Play Video`
  实时播放并做逐帧实例分割

- `Stop Video`
  停止播放

- `Export Video`
  一键导出处理后的视频

界面参数：

- `Weights`
  模型权重路径

- `Score`
  置信度阈值

- `Disable TensorRT`
  Windows 下建议默认勾选

- `Export side-by-side`
  导出时保存“原图 + 分割图”拼接结果；不勾选则只导出分割结果

**终端版启动方式**

如果你还想用终端版：

图片模式：

```powershell
cd "D:\All Program\yolactedge-algorithm-project"
& "D:\All Program\yolactedge-algorithm-project\yolact_edge\.venv38\Scripts\python.exe" terminal\visual_terminal.py --mode image --input yolact_edge\test.jpg --disable-tensorrt
```

视频模式：

```powershell
cd "D:\All Program\yolactedge-algorithm-project"
& "D:\All Program\yolactedge-algorithm-project\yolact_edge\.venv38\Scripts\python.exe" terminal\visual_terminal.py --mode video --input your_video.mp4 --disable-tensorrt --output terminal\outputs\result.mp4
```

**原项目原生命令**

原始命令行入口仍然在 [yolact_edge/eval.py](/D:/All%20Program/yolactedge-algorithm-project/yolact_edge/eval.py:1)。

例如：

```powershell
cd "D:\All Program\yolactedge-algorithm-project\yolact_edge"
..\.venv38\Scripts\python.exe eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --image=test.jpg
```

但这个入口更适合原始调试，不适合日常可视化操作。

**setup.py 是否有用**

有用，建议保留。

[yolact_edge/setup.py](/D:/All%20Program/yolactedge-algorithm-project/yolact_edge/setup.py:1) 负责：

- 支持 `pip install .`
- 编译 `cython_nms` 扩展

删掉它会直接影响安装和后续二次开发。

**当前清理和忽略规则**

- 根目录 `.gitignore` 已补充
- 已忽略虚拟环境、编译产物、`weights/`、结果目录和本地大数据目录
- `terminal/outputs/` 会被自动忽略

**已验证**

当前新增内容已做这些检查：

- `terminal/inference_backend.py` 语法检查通过
- `terminal/visual_terminal.py` 语法检查通过
- `terminal/gui_app.py` 语法检查通过
- 当前 `.venv38` 中 `tkinter` 可用
- 当前 `.venv38` 中 `Pillow` 可用

**注意**

如果你双击 GUI 后无法推理，优先检查：

- 权重路径是否存在
- 是否已正确安装 `yolact_edge`
- CUDA / PyTorch 是否可用
- 是否保持 `Disable TensorRT` 勾选
