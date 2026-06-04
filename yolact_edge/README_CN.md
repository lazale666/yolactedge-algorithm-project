# YolactEdge 中文 README

## 项目简介

YolactEdge 是一个面向边缘设备的实时实例分割项目，基于 YOLACT 改进，支持：

- 单张图片实例分割
- 视频实例分割
- 摄像头实时推理
- COCO / YouTube VIS 相关训练与评估

本仓库原始说明偏向 Linux / Jetson 环境。本文档面向 Windows 用户，重点说明如何在 Windows 上快速把项目跑起来。

## 适合谁看

如果你符合以下任一情况，这份文档更适合你：

- 第一次在 Windows 上配置 `yolact_edge`
- 想快速完成图片推理
- Google Drive 打不开，不知道该下载哪个模型
- 在 `pycocotools`、`cython_nms`、`cl`、`torch` 版本上遇到报错

更详细的踩坑与排障请看 [配置说明.md](./配置说明.md)。

## 当前推荐的 Windows 配置

建议使用以下组合：

- Python `3.8`
- PyTorch `1.8.1+cu111`
- TorchVision `0.9.1+cu111`
- Visual Studio 2022 C++ 编译工具
- CUDA GPU

不建议直接使用：

- Python `3.14`
- PyTorch `1.6.0+cu101` 搭配 RTX 30 系显卡

## 一、快速开始

### 1. 创建虚拟环境

如果你使用 Miniconda，并且系统没有 `py` 启动器，可直接这样创建：

```powershell
C:\Users\你的用户名\miniconda3\python.exe -m venv .venv38
```

验证：

```powershell
.\.venv38\Scripts\python.exe -V
```

### 2. 加载 VS 编译环境

本项目在 Windows 下需要编译扩展，所以必须保证 `cl` 可用。

示例：

```powershell
& "E:\visual studio\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
cl
```

如果 `cl` 能输出版本信息，说明编译环境已加载成功。

### 3. 安装 PyTorch

推荐：

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

验证：

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

### 4. 安装其他依赖

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -m pip install cython numpy opencv-python pillow matplotlib GitPython termcolor tensorboard colorama
```

### 5. 安装 pycocotools（含 YTVOS）

本项目依赖的 `pycocotools` 推荐使用 `haotian-liu/cocoapi` fork，并在 Windows 下手动去掉 GCC 编译参数后再安装。

详细步骤见 [配置说明.md](./配置说明.md)。

安装成功后可验证：

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -c "from pycocotools.coco import COCO; print('COCO ok')"
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -c "from pycocotools.ytvos import YTVOS; print('YTVOS ok')"
```

### 6. 安装项目

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -m pip install .
```

验证：

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" -c "import yolact_edge; print('yolact_edge ok')"
```

## 二、模型下载

### 1. 推荐首次使用的模型

如果你只想先把图片推理跑通，推荐先下载以下 COCO 模型：

- `yolact_edge_54_800000.pth`
- 或 `yolact_edge_resnet50_54_800000.pth`

其中：

- `yolact_edge_54_800000.pth` 对应 `yolact_edge_config`
- `yolact_edge_resnet50_54_800000.pth` 对应 `yolact_edge_resnet50_config`

### 2. 下载来源

优先参考项目 README 的 `OneDrive mirror`，如果 Google Drive 打不开就直接用镜像。

官方 README：

- https://github.com/WisconsinAIVision/yolact_edge

下载后放到：

```text
weights\
```

## 三、首次推理

### 1. 单图推理

推荐先禁用 TensorRT：

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --image=test.jpg
```

如果使用 `resnet50` 权重：

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" eval.py --disable_tensorrt --trained_model=weights\yolact_edge_resnet50_54_800000.pth --score_threshold=0.3 --image=test.jpg
```

### 2. 保存输出图片

示例：

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --image=input.jpg:output.jpg
```

注意：

Windows 下不要把 `C:\...:D:\...` 这种绝对路径直接传给 `--image`，因为程序内部用 `:` 分隔输入输出，和盘符冲突。

推荐做法：

- 把图片放在项目目录中
- 使用相对路径

### 3. 视频推理

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --video=my_video.mp4
```

### 4. 摄像头推理

```powershell
& "D:\All Program\yolact_edge\.venv38\Scripts\python.exe" eval.py --disable_tensorrt --trained_model=weights\yolact_edge_54_800000.pth --score_threshold=0.3 --video=0
```

## 四、为什么推荐先关闭 TensorRT

官方项目的 TensorRT 配置更偏：

- Linux
- Jetson
- TensorRT 7.x
- torch2trt

Windows 下首次部署时，建议先把目标定为：

- 能加载模型
- 能成功跑一张图
- 能输出分割结果

确认这些都没问题后，再考虑 TensorRT 优化。

因此本文档中的所有首次示例都使用：

```powershell
--disable_tensorrt
```

## 五、常见问题

### 1. `No suitable Python runtime found`

说明没有 `py` 启动器或没有注册 `Python 3.8`。  
直接使用 Miniconda Python 创建 `venv` 即可。

### 2. pip 安装时报 SSL 或 HTTPS 错误

常见原因：

- VPN
- 代理
- 安全软件拦截 HTTPS

建议：

- 关闭 VPN 后重试
- 改用 `pypi.org`
- 或切换镜像源

### 3. `cl` 无法识别

说明 Visual Studio 编译环境没有加载到当前 shell。  
需要先执行 `Launch-VsDevShell.ps1`，或使用 VS 自带的开发终端。

### 4. `pycocotools` 编译时报 `/Wno-cpp`

这是因为 `setup.py` 里存在 GCC 专用参数，Windows 下要手动删掉。  
详见 [配置说明.md](./配置说明.md)。

### 5. `numpy/arrayobject.h` 找不到

这是本仓库 Windows 编译 `cython_nms` 的常见问题。  
当前仓库里的 [setup.py](./setup.py) 已补上 `np.get_include()`。

### 6. 模型加载后报显卡架构不兼容

如果你使用的是 RTX 30 系显卡，不要继续用 `torch 1.6.0+cu101`。  
改为：

- `torch 1.8.1+cu111`
- `torchvision 0.9.1+cu111`

### 7. `test.jpg` 找不到

说明你传给 `--image` 的文件不存在。  
先确认文件路径正确，推荐使用项目目录下的相对路径。

## 六、训练说明

Windows 下训练不是不能做，但比单图推理复杂得多，主要包括：

- 数据集组织
- 预训练 backbone 权重
- COCO / YouTube VIS 标注
- 更严格的 CUDA 与依赖兼容性

如果你的目标只是部署和推理，建议先不要直接进入训练阶段。

训练相关原始说明请参考：

- [README.md](./README.md)
- [INSTALL.md](./INSTALL.md)

## 七、建议的上手顺序

建议按这个顺序操作：

1. 配好 Python 3.8 和 VS C++ 编译环境
2. 安装 `torch 1.8.1+cu111`
3. 装好 `pycocotools`
4. 执行 `pip install .`
5. 下载 `weights\yolact_edge_54_800000.pth`
6. 用 `--disable_tensorrt` 跑一张图
7. 确认输出正常后，再尝试视频或训练

## 八、相关文档

- 原始英文 README: [README.md](./README.md)
- 英文安装说明: [INSTALL.md](./INSTALL.md)
- Windows 实战配置说明: [配置说明.md](./配置说明.md)

