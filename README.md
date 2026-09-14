# ViT 猫狗图片分类

使用 PyTorch 和 torchvision 的 ViT-B/16 模型进行猫狗图片分类。项目包含训练脚本和单张图片预测脚本，运行时会自动选择可用的 CUDA GPU，否则使用 CPU。

**本仓库只包含代码和说明，不包含数据集和模型权重。** 下载代码后，需要安装依赖，并根据用途准备权重或数据集。

## 项目文件

```text
vit-dogs-cats/
├── README.md
├── .gitignore
├── inference.py                  # 单张图片预测
├── train_learn.py                # 基于已有权重继续训练
├── checkpoints/                 # 本地模型权重，不上传 Git
│   ├── vit_catdog_best.pth        # 训练 v2 时加载的 v1 权重
│   ├── vit_catdog_v2_best.pth     # 默认预测权重
│   └── vit_catdog_v2_last.pth     # v2 续训权重
└── data_ready/                  # 本地训练及验证数据，不上传 Git
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── val/
        ├── cats/
        └── dogs/
```

其中 `checkpoints/` 和 `data_ready/` 需要在本地自行准备，克隆仓库不会获得这些文件。

## 安装环境

需要 Python，以及以下依赖：

- `torch`
- `torchvision`
- `Pillow`

建议使用独立的 Python 虚拟环境。在 Linux 服务器上，进入项目目录后执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision Pillow
```

如果使用 NVIDIA GPU，请根据服务器环境选择相匹配的 PyTorch 安装包。可以用下面的命令检查依赖和 CUDA 是否可用：

```bash
python -c "import torch, torchvision, PIL; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('Pillow:', PIL.__version__); print('CUDA available:', torch.cuda.is_available())"
```

项目尚未固定依赖版本，也尚未在全新环境完成安装和运行验证。

## 预测一张图片

预测只需要图片和训练好的模型权重，不需要训练数据集。

先向项目提供者获取 `vit_catdog_v2_best.pth`，放在项目的 `checkpoints/` 目录下。当前仓库未提供权重下载链接。

在项目目录中执行：

```bash
python inference.py /path/to/image.jpg
```

也可以指定其他兼容的权重文件：

```bash
python inference.py /path/to/image.jpg --checkpoint /path/to/model.pth
```

将示例中的路径替换为实际文件路径。程序会输出运行设备、预测类别、置信度，以及各类别的预测概率。

权重必须与项目的 ViT-B/16 模型结构匹配，并包含 `model_state_dict` 和 `class_to_idx`。不能直接用任意 `.pth` 文件替代。图片预处理包括缩放、中心裁剪至 224 × 224 和归一化，输入图片会转换为 RGB。

## 准备训练数据

数据集由使用者自行准备，不随代码发布。将猫狗图片按下面的结构放置：

```text
data_ready/
├── train/
│   ├── cats/
│   │   └── cat_001.jpg
│   └── dogs/
│       └── dog_001.jpg
└── val/
    ├── cats/
    │   └── cat_002.jpg
    └── dogs/
        └── dog_002.jpg
```

代码使用 `ImageFolder` 从子目录名称读取类别。训练集和验证集必须使用一致的类别目录，类别映射也必须与加载的权重一致。训练图片和验证图片应分开，避免使用重复图片。

## 训练与续训

**当前 `train_learn.py` 是基于已有猫狗分类权重继续训练的脚本，不支持在没有项目权重时直接从头训练。** 模型骨干参数被冻结，仅训练最后的分类层。

默认配置为：

```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RESUME_V2 = False
EPOCHS_TO_RUN = 1
```

准备好数据集以及 `checkpoints/vit_catdog_best.pth` 后，在项目目录运行：

```bash
python train_learn.py
```

- `RESUME_V2 = False`：加载 v1 最佳权重，使用新的优化器状态，从 v2 第 1 轮开始训练。
- `RESUME_V2 = True`：加载 `checkpoints/vit_catdog_v2_last.pth`，恢复模型、优化器和训练轮次，继续训练。
- `EPOCHS_TO_RUN`：本次额外训练的轮数，必须至少为 1。

这些配置需要直接修改脚本，目前没有对应的命令行参数。训练过程中，每轮保存 `vit_catdog_v2_last.pth`；验证准确率刷新当前记录时，保存 `vit_catdog_v2_best.pth`。

模型初始化使用 torchvision 的默认预训练权重。如果本地没有缓存，初始化时需要联网下载，之后才加载项目的猫狗分类权重。预测脚本不需要下载这份 torchvision 预训练权重。

当前训练执行部分没有 `if __name__ == "__main__":` 入口保护，且数据加载设置为 `num_workers=4`。在 Windows 或使用 `spawn` 启动多进程的环境下，需要先将训练执行部分封装到 `main()` 并添加入口保护。CPU 可以用于运行，但 ViT 训练较慢，训练时建议使用 GPU；显存不足时可减小 `BATCH_SIZE`。

## Git 上传范围

`.gitignore` 会忽略本地数据目录、数据压缩包、模型权重、训练日志、Python 缓存和虚拟环境。因此正常提交代码时，这些文件不会被一并加入仓库。

模型权重可通过单独的文件分享方式提供给使用者；数据集由使用者自行准备。

提交前可以检查：

```bash
git status --short
git ls-files data_ready cats_and_dogs data_pointer_backup dataset data datasets checkpoints
```

第二条命令应没有输出。注意：`.gitignore` 只阻止加入未跟踪文件，已经被 Git 跟踪的文件仍需单独取消跟踪。
