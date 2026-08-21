# 导入
import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights,vit_b_16
from pathlib import Path
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 配置
DATA_DIR = Path("/data/zyz/Vit_dogs_cats/data_ready")
BATCH_SIZE = 32

# 执行代码
train_transform = transforms.Compose([
	transforms.RandomResizedCrop(224,scale=(0.75,1.0)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485,0.456,0.406],
		std=[0.229,0.224,0.225],
	),
])

val_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485,0.456,0.406],
		std=[0.229,0.224,0.225],
	),
])


train_dataset = datasets.ImageFolder(
	root=DATA_DIR / "train",
	transform=train_transform,
)

val_dataset = datasets.ImageFolder(
	root=DATA_DIR / "val",
	transform=val_transform,
)

train_loader = DataLoader(
	dataset=train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=4,
	pin_memory=True,
)

val_loader = DataLoader(
	dataset=val_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	num_workers=4,
	pin_memory=True,
)

print("类别映射:",train_dataset.class_to_idx)
print("训练图片数量：",len(train_dataset))
print("验证图片数量：",len(val_dataset))
