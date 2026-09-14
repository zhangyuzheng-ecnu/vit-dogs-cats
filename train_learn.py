
# 导入

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights,vit_b_16
from pathlib import Path
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 配置

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR /"data_ready"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device(
	"cuda" if torch.cuda.is_available() else"cpu"
)

# 定义函数

def build_model(num_classes):
	weights = ViT_B_16_Weights.DEFAULT
	model = vit_b_16(weights=weights)

	for parameter in model.parameters():
		parameter.requires_grad = False
	input_features = model.heads.head.in_features
	model.heads.head = nn.Linear(
		in_features=input_features,
		out_features=num_classes,
	)

	return model

def train_one_epoch(model,loader,loss_fn,optimizer,device):
	model.train()

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	for batch_index, (images, labels) in enumerate(loader, start=1):
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()

		logits = model(images)
		loss = loss_fn(logits,labels)

		loss.backward()
		optimizer.step()

		batch_size = labels.size(0)
		total_loss += loss.item() * batch_size
		total_correct += (
			logits.argmax(dim=1) == labels
		).sum().item()
		total_samples += batch_size

		print(
			f"Batch {batch_index}/{len(loader)} | "
			f"loss: {loss.item():.4f}",
			flush=True,
		)

	average_loss = total_loss / total_samples
	accuracy = total_correct / total_samples

	return average_loss, accuracy

@torch.inference_mode()
def evaluate(model,loader,loss_fn,device):
	model.eval()

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	for batch_index, (images,labels) in enumerate(
		loader,
		start=1,
	):
		images = images.to(device)
		labels = labels.to(device)

		logits = model(images)
		loss = loss_fn(logits,labels)

		batch_size = labels.size(0)
		total_loss += loss.item() * batch_size
		total_correct += (
			logits.argmax(dim=1) == labels
		).sum().item()
		total_samples += batch_size

		print(
			f"Val Batch {batch_index}/{len(loader)} |"
			f"loss:{loss.item():.4f}",
			flush=True,
		)

	average_loss = total_loss / total_samples
	accuracy = total_correct / total_samples

	return average_loss, accuracy

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

model = build_model(
	num_classes=len(train_dataset.classes)
)
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
	model.heads.head.parameters(),
	lr=LEARNING_RATE,
	weight_decay=WEIGHT_DECAY
)

trainable_parameters = sum(
	parameter.numel()
	for parameter in model.parameters()
	if parameter.requires_grad
)

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

V1_BEST_CHECKPOINT = CHECKPOINT_DIR / "vit_catdog_best.pth"

V2_LAST_CHECKPOINT = CHECKPOINT_DIR / "vit_catdog_v2_last.pth"
V2_BEST_CHECKPOINT = CHECKPOINT_DIR / "vit_catdog_v2_best.pth"

RESUME_V2 = False
EPOCHS_TO_RUN = 1

if EPOCHS_TO_RUN < 1:
	raise ValueError("EPOCHS_TO_RUN 必须至少为 1")
start_epoch = 1
best_val_accuracy = -1.0

if RESUME_V2:
	if not V2_LAST_CHECKPOINT.exists():
		raise FileNotFoundError(
			f"找不到 v2 续训权重: {V2_LAST_CHECKPOINT}"
		)

	v2_checkpoint = torch.load(
		V2_LAST_CHECKPOINT,
		map_location=DEVICE,
		weights_only=True,
	)

	if v2_checkpoint["class_to_idx"] != train_dataset.class_to_idx:
		raise ValueError(
			"v2 checkpoint 的类别映射与当前数据集不一致"
		)

	model.load_state_dict(
		v2_checkpoint["model_state_dict"]
	)
	optimizer.load_state_dict(
		v2_checkpoint["optimizer_state_dict"]
	)

	start_epoch = int(v2_checkpoint["epoch"]) + 1
	best_val_accuracy = float(
		v2_checkpoint["best_val_accuracy"]
	)

	print(f"已加载 v2 最新权重: {V2_LAST_CHECKPOINT}")
	print(
		f"v2上次已经完成第 "
		f"{v2_checkpoint['epoch']}轮"
	)
	print(f"本次从 v2 第 {start_epoch}轮继续")

else:
	if not V1_BEST_CHECKPOINT.exists():
		raise FileNotFoundError(
			f"找不到 v1 最佳权重：{V1_BEST_CHECKPOINT}"
		)

	v1_checkpoint = torch.load(
		V1_BEST_CHECKPOINT,
		map_location=DEVICE,
		weights_only=True,
	)

	if v1_checkpoint["class_to_idx"] != train_dataset.class_to_idx:
		raise ValueError(
			"v1 checkpoint 的类别映射与当前数据集不一致"
		)

	model.load_state_dict(
		v1_checkpoint["model_state_dict"]
	)

	print(f"已加载 v1 最佳权重: {V1_BEST_CHECKPOINT}")
	print("v2 从第 1 轮开始，优化器使用全新状态")

end_epoch = start_epoch + EPOCHS_TO_RUN -1

for epoch in range(start_epoch, end_epoch + 1):
	run_index = epoch - start_epoch + 1

	print(
		f"\n===== Epoch {epoch} | "
		f"本次进度 {run_index}/{EPOCHS_TO_RUN} ====="
	)
	train_loss, train_accuracy = train_one_epoch(
		model,
		train_loader,
		loss_fn,
		optimizer,
		DEVICE,
	)

	val_loss,val_accuracy = evaluate(
		model,
		val_loader,
		loss_fn,
		DEVICE,
	)

	is_best = val_accuracy > best_val_accuracy

	if is_best:
		best_val_accuracy = val_accuracy

	checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict":optimizer.state_dict(),
		"train_loss":train_loss,
		"train_accuracy":train_accuracy,
		"val_loss":val_loss,
		"val_accuracy":val_accuracy,
		"best_val_accuracy":best_val_accuracy,
		"class_to_idx":train_dataset.class_to_idx,
	}
	##最新权重每轮都会要
	torch.save(checkpoint, V2_LAST_CHECKPOINT)
	print(f"已保存最新权重:{V2_LAST_CHECKPOINT}")
	##只有最好的才留下来
	if is_best:
		torch.save(checkpoint, V2_BEST_CHECKPOINT)
		print(f"已保存最佳权重:{V2_BEST_CHECKPOINT}")

	print(f"训练 loss：{train_loss:.4f}")
	print(f"训练准确率：{train_accuracy:.2%}")
	print(f"验证 loss：{val_loss:.4f}")
	print(f"验证准确率：{val_accuracy:.2%}")
