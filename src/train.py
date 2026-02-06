import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, save_json
from src.data import get_train_val_loaders
from src.models import build_model


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    if cfg.get("device", "auto") == "cpu":
        return "cpu"
    if cfg.get("device", "auto") == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds, all_targets = [], []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(out.argmax(1).detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())

    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return float(sum(losses) / max(1, len(losses))), float(macro_f1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        losses.append(loss.item())
        all_preds.extend(out.argmax(1).detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())

    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return float(sum(losses) / max(1, len(losses))), float(macro_f1)


def freeze_backbone(model, model_name: str, freeze: bool):
    model_name = model_name.lower()

    # default: freeze everything, then unfreeze head
    for p in model.parameters():
        p.requires_grad = not freeze

    if model_name == "resnet18":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(model_name)


def main(cfg_path: str, model_name: str):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["seed"])

    device = get_device(cfg)

    train_loader, val_loader, num_classes, class_names = get_train_val_loaders(cfg)
    model, _ = build_model(model_name, num_classes)
    model = model.to(device)

    out_dir = os.path.join(cfg["output_dir"], model_name)
    ensure_dir(out_dir)
    save_json({"classes": class_names}, os.path.join(out_dir, "classes.json"))

    criterion = nn.CrossEntropyLoss()

    # phase 1: freeze backbone
    freeze_backbone(model, model_name, freeze=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    history = []
    best_val_f1 = -1.0

    for epoch in range(cfg["num_epochs"]):
        t0 = time.time()

        # unfreeze after freeze_epochs
        if epoch == cfg["freeze_epochs"]:
            freeze_backbone(model, model_name, freeze=False)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["lr"] * 0.3,  # smaller LR for fine-tune
                weight_decay=cfg["weight_decay"],
            )

        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = eval_one_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "seconds": round(time.time() - t0, 2),
            "phase": "frozen" if epoch < cfg["freeze_epochs"] else "finetune",
        }
        history.append(row)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

        print(f"[{model_name}] epoch {epoch+1}/{cfg['num_epochs']} | "
              f"val_f1={val_f1:.4f} | val_loss={val_loss:.4f} | phase={row['phase']}")

    save_json(history, os.path.join(out_dir, "train_metrics.json"))
    save_json({"best_val_macro_f1": best_val_f1}, os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["resnet18", "efficientnet_b0"])
    args = parser.parse_args()
    main(args.config, args.model)
