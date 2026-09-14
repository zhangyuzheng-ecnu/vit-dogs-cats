import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_16


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
	ROOT_DIR / "checkpoints" / "vit_catdog_v2_best.pth"
)

DEVICE = torch.device(
	"cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_TRANSFORM = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485,0.456,0.406],
		std=[0.229,0.224,0.225],
	),
])

def build_model(num_classes):
	model = vit_b_16(weights=None)

	input_features = model.heads.head.in_features
	model.heads.head = nn.Linear(
		in_features=input_features,
		out_features=num_classes,
	)

	return model


def load_model(checkpoint_path, device):
	checkpoint = torch.load(
		checkpoint_path,
		map_location="cpu",
		weights_only=True,
	)

	class_to_idx = checkpoint["class_to_idx"]
	idx_to_class = {
		index: class_name
		for class_name, index in class_to_idx.items()
	}

	model = build_model(
		num_classes=len(class_to_idx)
	)

	model.load_state_dict(
		checkpoint["model_state_dict"],
		strict=True,
	)

	model.load_state_dict(
		checkpoint["model_state_dict"],
		strict=True,
	)

	model = model.to(device)
	model.eval()

	return model, idx_to_class


@torch.inference_mode()
def predict_image(model, image_path, device):
	with Image.open(image_path) as image:
		image = image.convert("RGB")
		image_tensor = IMAGE_TRANSFORM(image)

	image_tensor = image_tensor.unsqueeze(0)
	image_tensor = image_tensor.to(device)

	logits = model(image_tensor)
	probabilities = torch.softmax(
		logits,
		dim=1,
	)[0].cpu()

	predicted_index = probabilities.argmax().item()
	confidence = probabilities[predicted_index].item()

	return predicted_index, confidence,probabilities


def main():
	parser = argparse.ArgumentParser(
		description="使用 VIT 判断图片是猫还是狗"
	)

	parser.add_argument(
		"image",
		type=Path,
		help="需要预测的图片路径",
	)

	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=DEFAULT_CHECKPOINT,
		help="checkpoint 文件路径",
	)

	args = parser.parse_args()

	if not args.image.is_file():
		raise FileNotFoundError(
			f"找不到图片: {args.image}"
		)

	model, idx_to_class = load_model(
		args.checkpoint,
		DEVICE,
	)

	predicted_index, confidence, probabilities = (
		predict_image(
			model,
			args.image,
			DEVICE,
		)
	)

	predicted_class = idx_to_class[predicted_index]

	print(f"运行设备：{DEVICE}")
	print(f"预测类别：{predicted_class}")
	print(f"置信度：{confidence:.2%}")

	print("\n所有类别概率：")
	for index, probability in enumerate(probabilities):
		class_name = idx_to_class[index]
		print(f"{class_name}: {probability.item():.2%}")

if __name__ == "__main__":
	main()
