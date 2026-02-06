import argparse
import os
import yaml
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from src.data import get_test_loader
from src.models import build_model
from src.utils import ensure_dir, save_json, denormalize, tensor_to_uint8, plot_image_grid


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    if cfg.get("device", "auto") == "cpu":
        return "cpu"
    if cfg.get("device", "auto") == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main(cfg_path: str, model_name: str):
    cfg = load_cfg(cfg_path)
    device = get_device(cfg)

    test_ds, test_loader, class_names = get_test_loader(cfg)
    num_classes = len(class_names)

    model, _ = build_model(model_name, num_classes)
    out_dir = os.path.join(cfg["output_dir"], model_name)
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    all_probs, all_preds, all_targets = [], [], []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1).tolist()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds)
        all_targets.extend(y.numpy().tolist())

    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"[{model_name}] Test Macro F1: {macro_f1:.4f}")

    # Save predictions.csv
    df = pd.DataFrame({
        "index": list(range(len(all_preds))),
        "y_true": all_targets,
        "y_pred": all_preds,
        "true_label": [class_names[i] for i in all_targets],
        "pred_label": [class_names[i] for i in all_preds],
        "p_max": [max(p) for p in all_probs],
    })
    ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Confusion matrix plot
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # Classification report (json-ish)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    save_json({"test_macro_f1": macro_f1, "report": report}, os.path.join(out_dir, "test_metrics.json"))

    # Visual examples: correct and wrong
    correct_idx = [i for i in range(len(all_preds)) if all_preds[i] == all_targets[i]]
    wrong_idx   = [i for i in range(len(all_preds)) if all_preds[i] != all_targets[i]]

    n = int(cfg.get("num_visuals", 12))
    correct_idx = correct_idx[:n]
    wrong_idx = wrong_idx[:n]

    def collect_images(idxs):
        images, titles = [], []
        for i in idxs:
            img_t, y = test_ds[i]
            img_u8 = tensor_to_uint8(denormalize(img_t))
            title = f"GT: {class_names[y]}\nPred: {class_names[all_preds[i]]} ({df.loc[i,'p_max']:.2f})"
            images.append(img_u8)
            titles.append(title)
        return images, titles

    imgs, titles = collect_images(correct_idx)
    plot_image_grid(imgs, titles, os.path.join(out_dir, "correct_examples.png"), ncols=4)

    imgs, titles = collect_images(wrong_idx)
    plot_image_grid(imgs, titles, os.path.join(out_dir, "wrong_examples.png"), ncols=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["resnet18", "efficientnet_b0"])
    args = parser.parse_args()
    main(args.config, args.model)
