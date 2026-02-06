import os
import json
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    img_tensor: (3,H,W) normalized with ImageNet mean/std
    """
    return (img_tensor.detach().cpu() * IMAGENET_STD) + IMAGENET_MEAN


def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert (3,H,W) float tensor in [0,1] to uint8 HxWx3
    """
    img = img_tensor.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def plot_image_grid(images, titles, out_path: str, ncols: int = 4, figsize=(14, 10)):
    """
    images: list of HxWx3 uint8 arrays
    titles: list of strings
    """
    assert len(images) == len(titles)
    n = len(images)
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=figsize)
    for i in range(n):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=200)
    plt.close()
