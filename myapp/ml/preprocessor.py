from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_inference_transform(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns standard inference transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Preprocesses a PIL Image into a normalized tensor with batch dimension (1, 3, H, W)."""
    rgb_image = image.convert("RGB")
    transform = get_inference_transform(target_size)
    tensor = transform(rgb_image).unsqueeze(0)
    return tensor
