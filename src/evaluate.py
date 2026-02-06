import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path):
    cfg = load_config(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.OxfordIIITPet(
        root="data",
        split="test",
        target_types="category",
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, len(dataset.classes)
    )
    model.load_state_dict(torch.load(
        os.path.join(cfg["output_dir"], "best_model.pt"),
        map_location=device
    ))
    model.to(device).eval()

    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.numpy())

    f1 = f1_score(targets, preds, average="macro")
    print(f"Test Macro F1: {f1:.4f}")

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    plt.savefig(os.path.join(cfg["output_dir"], "confusion_matrix.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
