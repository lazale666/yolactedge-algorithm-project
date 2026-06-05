# yolactedge-algorithm-project

这是一个基于 `YOLACT Edge` 的实例分割项目工作区，当前主要面向 Windows 本地推理、可视化展示、图片/视频导出，以及电脑摄像头实时识别。

本仓库不是单纯保留原始 `YOLACT Edge` 源码，而是在其基础上补了一套本地可视化工具，包含：

- 桌面 GUI
- 终端交互工具
- 统一推理后端
- 图片/视频导出能力
- 电脑自带摄像头实时识别能力

如果你的目标是“选图、选视频、调用电脑摄像头、实时看分割结果、再导出结果”，优先使用 `terminal/gui_app.py`。

**目录结构**
```text
D:\All Program\yolactedge-algorithm-project
├─ .git
│  用途：Git 版本控制目录。
├─ .gitignore
│  用途：忽略虚拟环境、编译产物、结果目录、权重目录和本地大数据目录。
├─ README.md
│  用途：当前项目总说明文档，包含结构说明、启动方式、GUI/终端/摄像头入口。
├─ cocoapi
│  用途：pycocotools 依赖源码目录。
│
│  ├─ common
│  │  用途：COCO 掩码与底层公共 C/C++ 代码。
│  ├─ PythonAPI
│  │  用途：Python 版 pycocotools 安装源码入口。
│  ├─ LuaAPI
│  │  用途：Lua 版本接口，当前项目基本不使用。
│  ├─ MatlabAPI
│  │  用途：Matlab 版本接口，当前项目基本不使用。
│  ├─ README.md
│  │  用途：cocoapi 原始说明文档。
│  └─ README.txt
│     用途：cocoapi 简要文本说明。
│
├─ terminal
│  用途：当前项目新增的本地可视化工具目录。
│
│  ├─ gui_app.py
│  │  用途：桌面 GUI 主程序。
│  │  功能：图片实例分割、视频实例分割、电脑摄像头实时识别、图片导出、视频导出。
│  ├─ inference_backend.py
│  │  用途：统一推理后端。
│  │  功能：加载模型、执行推理、绘制实例分割结果、导出处理后视频。
│  ├─ visual_terminal.py
│  │  用途：终端版可视化入口。
│  │  功能：通过命令行启动图片/视频/摄像头推理，适合调试和轻量使用。
│  ├─ run_gui.bat
│  │  用途：Windows 下一键启动 GUI。
│  ├─ outputs
│  │  用途：终端版默认输出目录，保存图片、视频或中间导出结果。
│  └─ __pycache__
│     用途：Python 字节码缓存目录，可忽略。
│
├─ yolact_edge
│  用途：YOLACT Edge 主项目目录。
│
│  ├─ .venv38
│  │  用途：你当前实际在用的 Python 3.8 虚拟环境。
│  ├─ .vscode
│  │  用途：VS Code 工作区配置目录。
│  ├─ build
│  │  用途：本地编译产物目录，主要来自扩展模块构建。
│  ├─ data
│  │  用途：项目内置示例资源、脚本和部分数据目录。
│  ├─ docker
│  │  用途：原项目 Docker / Jetson 相关部署资料。
│  ├─ photo
│  │  用途：你的本地测试图片目录，不属于核心源码。
│  ├─ results
│  │  用途：原始 eval.py 运行时结果目录。
│  ├─ weights
│  │  用途：模型权重目录。
│  │  当前可直接使用：
│  │  - yolact_edge_54_800000.pth
│  │  - yolact_edge_vid_847_50000.pth
│  ├─ yolact_edge
│  │  用途：YOLACT Edge 核心 Python 包源码目录。
│  │
│  │  ├─ data
│  │  │  用途：模型配置、数据集配置、类别名等。
│  │  ├─ layers
│  │  │  用途：检测头、后处理、NMS、输出处理等底层模块。
│  │  ├─ scripts
│  │  │  用途：原项目附带的训练/评估辅助脚本。
│  │  ├─ utils
│  │  │  用途：增强、计时、TensorRT、cython_nms 等工具模块。
│  │  ├─ backbone.py
│  │  │  用途：骨干网络定义。
│  │  ├─ inference.py
│  │  │  用途：原项目提供的 Python 推理封装。
│  │  ├─ yolact.py
│  │  │  用途：YOLACT Edge 主模型定义。
│  │  └─ __init__.py
│  │     用途：Python 包入口。
│  ├─ yolact_edge.egg-info
│  │  用途：本地安装生成的包元数据目录。
│  ├─ eval.py
│  │  用途：原项目主推理/评估入口。
│  │  功能：单图、批量图片、视频、摄像头、评估。
│  ├─ INSTALL.md
│  │  用途：中文安装文档。
│  ├─ LICENSE
│  │  用途：许可证文件。
│  ├─ pkg_usage.py
│  │  用途：原项目包调用示例。
│  ├─ README.md
│  │  用途：原项目英文说明。
│  ├─ README_CN.md
│  │  用途：原项目中文说明。
│  ├─ run_coco_eval.py
│  │  用途：COCO 评估辅助脚本。
│  ├─ setup.py
│  │  用途：安装 yolact_edge 包并编译 cython_nms。
│  ├─ test.jpg
│  │  用途：单图推理示例图片。
│  ├─ train.py
│  │  用途：训练入口。
│  └─ 配置说明.md
│     用途：Windows 下的本地配置记录与问题说明。
```

**推荐环境**

- Windows
- Python `3.8`
- PyTorch `1.8.1+cu111`
- TorchVision `0.9.1+cu111`
- NVIDIA CUDA GPU

当前这台机器已经确认 `CUDA` 可用，GPU 为：

- `NVIDIA GeForce RTX 3050 Ti Laptop GPU`

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

更详细的安装说明请看 [INSTALL.md](/D:/All%20Program/yolactedge-algorithm-project/yolact_edge/INSTALL.md:1)。

**GUI 启动方式**

推荐直接双击或命令行启动：

```powershell
cd "D:\All Program\yolactedge-algorithm-project\terminal"
.\run_gui.bat
```

也可以直接：

```powershell
cd "D:\All Program\yolactedge-algorithm-project"
& "D:\All Program\yolactedge-algorithm-project\yolact_edge\.venv38\Scripts\python.exe" terminal\gui_app.py
```

**GUI 支持的功能**

- `加载模型`
  加载 `weights` 中的模型权重

- `打开图片`
  对单张图片做实例分割

- `保存图片`
  导出当前处理结果

- `打开视频`
  选择视频文件进行预览和后续导出

- `播放视频`
  实时播放视频并显示实例分割结果

- `启动摄像头`
  调用电脑自带摄像头进行实时实例分割

- `停止预览`
  停止视频或摄像头预览

- `导出视频`
  一键导出处理后的视频

**电脑摄像头实时识别**

现在已经支持电脑自带摄像头实时识别。

使用方式：

1. 启动 GUI
2. 点击 `加载模型`
3. 确认 `摄像头索引` 默认为 `0`
4. 点击 `启动摄像头`

如果你的电脑有多个摄像头：

- 主摄像头一般是 `0`
- 外接摄像头可能是 `1`
- 其他视频采集设备可能是 `2` 或更高

如果 `0` 打不开，就改成 `1` 再试。

建议实时预览时：

- 保持 `禁用 TensorRT` 勾选
- `预览边长` 先选 `480` 或 `640`

这样更容易保证流畅度。

**终端版启动方式**

如果你仍然想用终端版，可使用 [terminal/visual_terminal.py](/D:/All%20Program/yolactedge-algorithm-project/terminal/visual_terminal.py:1)。

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

电脑摄像头模式：

```powershell
cd "D:\All Program\yolactedge-algorithm-project"
& "D:\All Program\yolactedge-algorithm-project\yolact_edge\.venv38\Scripts\python.exe" terminal\visual_terminal.py --mode camera --camera-index 0 --disable-tensorrt
```

**原项目命令行入口**

原始命令行主入口仍然是 [yolact_edge/eval.py](/D:/All%20Program/yolactedge-algorithm-project/yolact_edge/eval.py:1)。

例如单图推理：

```powershell
cd "D:\All Program\yolactedge-algorithm-project\yolact_edge"
..\.venv38\Scripts\python.exe eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --image=test.jpg
```

这个入口更适合原始调试，不如 GUI 适合日常使用。

**setup.py 是否有用**

有用，建议保留。

[yolact_edge/setup.py](/D:/All%20Program/yolactedge-algorithm-project/yolact_edge/setup.py:1) 当前负责：

- 支持 `pip install .`
- 编译 `cython_nms`

删掉它会直接影响安装和本地推理。

**当前项目已做的增强**

- 补充了根目录 `.gitignore`
- 重写了根目录 `README.md`
- 新增了统一推理后端
- 新增了终端版实例分割工具
- 新增了桌面 GUI
- 新增了电脑摄像头实时识别
- 修复了 Python 3.8 注解兼容问题
- 修复了视频播放重复点击导致卡死的问题
- 修复了视频里颜色跳动的问题，使颜色按类别固定

**已验证**

当前新增代码已通过：

- `terminal/inference_backend.py` 语法检查
- `terminal/gui_app.py` 语法检查
- `terminal/visual_terminal.py` 语法检查

当前环境已确认：

- `tkinter` 可用
- `Pillow` 可用
- `CUDA` 可用

**常见问题**

- `加载模型失败`
  先检查权重路径是否存在、环境是否装好、是否保持禁用 TensorRT。

- `视频预览很慢`
  优先把 `预览边长` 设为 `480` 或 `640`。

- `摄像头打不开`
  把 `摄像头索引` 从 `0` 改成 `1` 再试。

- `导出视频很慢`
  这是正常现象，导出是完整逐帧处理，通常会比实时预览慢。
