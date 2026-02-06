import argparse
import json
import os
import random
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_dataloaders(cfg):
    transform_train = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=transform_train
    )

    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    val_ds.dataset.transform = transform_val

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"]
    )

    return train_loader, val_loader, len(dataset.classes)


def build_model(num_classes):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    f1 = f1_score(targets, preds, average="macro")
    return sum(losses) / len(losses), f1


def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, num_classes = get_dataloaders(cfg)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    best_f1 = 0.0
    metrics = []

    for epoch in range(cfg["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(cfg["output_dir"], "best_model.pt"))

        print(f"Epoch {epoch+1}: val F1={val_f1:.4f}")

    with open(os.path.join(cfg["output_dir"], "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
