# 安装说明

本文档基于当前仓库的实际使用方式整理，重点面向 Windows 本地推理场景。原始英文文档偏向 Linux / Jetson / TensorRT 部署；如果你的目标是先把图片或视频实例分割跑通，建议优先按本文执行。

## 1. 适用范围

- 目标：本地完成图片、视频、摄像头实例分割推理
- 平台：Windows
- 推荐 Python：`3.8`
- 推荐 PyTorch：`1.8.1+cu111`
- 推荐 TorchVision：`0.9.1+cu111`
- 推荐硬件：NVIDIA CUDA GPU

如果你只是要先验证功能，不建议一开始就折腾 TensorRT。先使用普通 PyTorch 推理，确认模型加载、单图推理、视频导出都正常，再考虑加速优化。

## 2. 环境准备

### 2.1 创建虚拟环境

如果系统没有合适的 `py` 启动器，也可以直接用现有 Python 创建虚拟环境：

```powershell
python -m venv .venv38
```

验证：

```powershell
.\.venv38\Scripts\python.exe -V
```

### 2.2 安装 PyTorch

建议安装：

```powershell
.\.venv38\Scripts\python.exe -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

验证：

```powershell
.\.venv38\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

### 2.3 安装基础依赖

```powershell
.\.venv38\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv38\Scripts\python.exe -m pip install cython numpy opencv-python pillow matplotlib GitPython termcolor tensorboard colorama
```

## 3. 安装 pycocotools

当前仓库中已经包含 `cocoapi` 源码。Windows 下建议直接在本地源码基础上安装，不必再从 GitHub 额外拉取。

进入 `cocoapi/PythonAPI`，确认 `setup.py` 中没有 Linux / GCC 专用编译参数后执行：

```powershell
cd cocoapi\PythonAPI
..\..\.venv38\Scripts\python.exe -m pip install .
```

如果你的 PowerShell 对相对路径解析不方便，也可以直接使用完整相对路径：

```powershell
cd cocoapi\PythonAPI
..\..\.venv38\Scripts\python.exe -m pip install .
```

安装完成后验证：

```powershell
cd ..\..
.\.venv38\Scripts\python.exe -c "from pycocotools.coco import COCO; print('COCO ok')"
.\.venv38\Scripts\python.exe -c "from pycocotools.ytvos import YTVOS; print('YTVOS ok')"
```

说明：

- `pycocotools` 是本项目评估、掩码编码等流程的依赖
- Windows 下最常见问题是编译参数不兼容，当前仓库中的 `cocoapi/PythonAPI/setup.py` 已是适合本地 MSVC 编译的简化版本

## 4. 安装 yolact_edge

在 `yolact_edge` 目录中执行：

```powershell
cd yolact_edge
..\.venv38\Scripts\python.exe -m pip install .
```

验证：

```powershell
..\.venv38\Scripts\python.exe -c "import yolact_edge; print('yolact_edge ok')"
```

## 5. `setup.py` 有什么用

`setup.py` 目前是有用的，不建议删除。

它的作用主要有两点：

1. 让 `pip install .` 能把 `yolact_edge` 安装成可导入的 Python 包
2. 编译项目中的 `cython_nms` 扩展模块

当前仓库的 [setup.py](./setup.py) 里已经包含：

- `find_packages(...)`：用于收集并安装 Python 包
- `Extension("cython_nms", ...)`：用于编译 `yolact_edge/utils/cython_nms.pyx`
- `include_dirs=[np.get_include()]`：用于让 Windows 下编译时能找到 NumPy 头文件

如果删掉它：

- `pip install .` 这条安装路径会失效
- `cython_nms` 的本地编译会变麻烦
- 你后续要做 GUI 封装或二次开发时，也更不方便复用本项目

所以结论是：`setup.py` 保留。

## 6. 模型权重

下载与你配置对应的权重文件，并放到：

```text
yolact_edge\weights\
```

常见示例：

- `yolact_edge_54_800000.pth`
- `yolact_edge_resnet50_54_800000.pth`

注意权重名和配置名要对应。

## 7. 首次推理建议

首次运行建议先关闭 TensorRT：

```powershell
cd yolact_edge
..\.venv38\Scripts\python.exe eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --image=test.jpg
```

保存处理后的图片：

```powershell
..\.venv38\Scripts\python.exe eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --image=test.jpg:output.jpg
```

处理视频：

```powershell
..\.venv38\Scripts\python.exe eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --video=input.mp4:output.mp4
```

摄像头实时推理：

```powershell
..\.venv38\Scripts\python.exe eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --video=0
```

## 8. TensorRT 说明

原始项目文档对 TensorRT 的说明主要面向 Linux / Jetson。当前仓库在 Windows 下更适合先走非 TensorRT 路线。

建议顺序：

1. 先安装并跑通普通 PyTorch 推理
2. 确认单图、视频、导出都正常
3. 再评估是否需要 TensorRT 优化

如果你只是做本地可视化演示项目，完全可以先不启用 TensorRT。

## 9. 可选内容

以下内容不是“本地推理必须项”：

- Docker 相关目录
- Linux / Jetson 部署脚本
- COCO / YouTube VIS 全量训练数据下载
- TensorRT INT8 校准数据集

只有在你后续要做训练、评估或特定平台部署时，才需要补这些内容。

## 10. 常见问题

### 10.1 `cl` 无法识别

说明你当前终端没有加载 Visual Studio C++ 编译环境。Windows 下编译 Cython 扩展时通常需要 MSVC。

### 10.2 `numpy/arrayobject.h` 找不到

这是编译扩展时没找到 NumPy 头文件。当前仓库中的 `yolact_edge/setup.py` 已加入 `np.get_include()`，通常不需要再手动修。

### 10.3 `test.jpg` 不存在

这是示例输入文件缺失。你可以改成自己的图片路径，或者自行准备一个测试图像。

### 10.4 TensorRT 相关报错

如果你当前只是做 Windows 推理，请先加上：

```powershell
--disable_tensorrt
```

先确保基础推理流程跑通。
