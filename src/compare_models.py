import argparse
import os
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src.data import get_test_loader
from src.models import build_model
from src.utils import ensure_dir, denormalize, tensor_to_uint8


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    dev = cfg.get("device", "auto")
    if dev == "cpu":
        return "cpu"
    if dev == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class GradCAM:
    """
    Minimal Grad-CAM: forward hook saves activations, full backward hook saves gradients.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self):
        # activations: [B,C,H,W], gradients: [B,C,H,W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)  # [B,H,W]
        cam = torch.relu(cam)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam


def overlay_heatmap(img_u8: np.ndarray, heatmap_2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a heatmap onto an RGB uint8 image using matplotlib colormap (no OpenCV dependency).
    """
    heatmap = heatmap_2d
    heatmap = np.clip(heatmap, 0, 1)

    cmap = plt.get_cmap("jet")
    hm_rgba = cmap(heatmap)  # HxWx4 floats
    hm_rgb = (hm_rgba[..., :3] * 255).astype(np.uint8)

    out = (img_u8.astype(np.float32) * (1 - alpha) + hm_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return out


def load_model_and_cam(cfg, model_name, num_classes, device):
    model, target_layer = build_model(model_name, num_classes)
    ckpt = Path(cfg["output_dir"]) / model_name / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    cam = GradCAM(model, target_layer)
    return model, cam


def pick_disagreements(cfg, pA: pd.DataFrame, pB: pd.DataFrame):
    """
    Choose indices where models disagree.
    Priority: cases where exactly one model is correct (more interesting).
    Deterministic via cfg['seed'].
    """
    n = int(cfg.get("num_disagreements", 12))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    disagree_mask = (pA["y_pred"].values != pB["y_pred"].values)
    idxs = np.where(disagree_mask)[0].tolist()

    one_correct = []
    other = []
    for i in idxs:
        y = int(pA.loc[i, "y_true"])
        a = int(pA.loc[i, "y_pred"])
        b = int(pB.loc[i, "y_pred"])
        if (a == y) != (b == y):
            one_correct.append(i)
        else:
            other.append(i)

    # Shuffle deterministically within groups
    rng.shuffle(one_correct)
    rng.shuffle(other)

    chosen = (one_correct + other)[:n]
    return idxs, chosen


def save_disagreement_panel(out_path: Path, img_u8, overA, overB, title_lines, dpi=140):
    """
    Save 1 disagreement as a 3-panel figure (original + two Grad-CAM overlays).
    """
    plt.figure(figsize=(14, 5))
    for j, (im, t) in enumerate([
        (img_u8, "Original"),
        (overA, "ResNet18 Grad-CAM"),
        (overB, "EfficientNet-B0 Grad-CAM"),
    ]):
        ax = plt.subplot(1, 3, j + 1)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(t, fontsize=11)

    # Big title (GT + preds)
    plt.suptitle("\n".join(title_lines), fontsize=12, y=1.02)
    plt.tight_layout()
    ensure_dir(str(out_path.parent))
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    device = get_device(cfg)

    # Load test dataset (for raw images)
    test_ds, _, class_names = get_test_loader(cfg)
    num_classes = len(class_names)

    out_root = Path(cfg["output_dir"]) / "compare"
    ensure_dir(str(out_root))

    # Load predictions from evaluate.py (must exist)
    pA_path = Path(cfg["output_dir"]) / "resnet18" / "predictions.csv"
    pB_path = Path(cfg["output_dir"]) / "efficientnet_b0" / "predictions.csv"

    if not pA_path.exists() or not pB_path.exists():
        raise FileNotFoundError(
            "Missing predictions.csv. Run evaluate first for both models:\n"
            "  python -m src.evaluate --model resnet18\n"
            "  python -m src.evaluate --model efficientnet_b0"
        )

    pA = pd.read_csv(pA_path).sort_values("index").reset_index(drop=True)
    pB = pd.read_csv(pB_path).sort_values("index").reset_index(drop=True)

    # Choose disagreements
    all_disagree_idxs, chosen = pick_disagreements(cfg, pA, pB)

    # Save disagreements.csv (all disagreements)
    pd.DataFrame({"index": all_disagree_idxs}).to_csv(out_root / "disagreements.csv", index=False)
    print(f"Disagreements: {len(all_disagree_idxs)} / {len(pA)}")
    print(f"Chosen for panels: {len(chosen)}")

    # Load models + cams
    modelA, camA = load_model_and_cam(cfg, "resnet18", num_classes, device)
    modelB, camB = load_model_and_cam(cfg, "efficientnet_b0", num_classes, device)

    # Output folder for individual panels
    panels_dir = out_root / "disagreements"
    ensure_dir(str(panels_dir))

    dpi = int(cfg.get("compare_dpi", 140))

    for k, idx in enumerate(chosen, start=1):
        img_t, y = test_ds[idx]
        x = img_t.unsqueeze(0).to(device)

        # ResNet18
        modelA.zero_grad(set_to_none=True)
        logitsA = modelA(x)
        predA = int(logitsA.argmax(1).item())
        scoreA = logitsA[0, predA]
        scoreA.backward(retain_graph=True)
        hmA = camA.generate().detach().cpu().numpy()[0]

        # EfficientNet
        modelB.zero_grad(set_to_none=True)
        logitsB = modelB(x)
        predB = int(logitsB.argmax(1).item())
        scoreB = logitsB[0, predB]
        scoreB.backward()
        hmB = camB.generate().detach().cpu().numpy()[0]

        img_u8 = tensor_to_uint8(denormalize(img_t))
        overA = overlay_heatmap(img_u8, hmA)
        overB = overlay_heatmap(img_u8, hmB)

        title_lines = [
            f"Disagreement {k}/{len(chosen)} — test index {idx}",
            f"GT: {class_names[y]}",
            f"ResNet18: {class_names[predA]}    |    EfficientNet-B0: {class_names[predB]}",
        ]

        out_path = panels_dir / f"disagree_{k:02d}_idx_{idx}.png"
        save_disagreement_panel(out_path, img_u8, overA, overB, title_lines, dpi=dpi)

    # Optional: build a small contact sheet thumbnail for convenience (safe size)
    # It will be MUCH smaller than the old monolithic PNG.
    thumbs = sorted(panels_dir.glob("*.png"))
    if thumbs:
        contact_path = out_root / "disagreements_contact_sheet.png"
        make_contact_sheet(thumbs, contact_path, cols=2, thumb_width=900)
        print(f"Saved contact sheet: {contact_path}")

    print(f"Saved {len(chosen)} individual panels to: {panels_dir}")


def make_contact_sheet(image_paths, out_path: Path, cols=2, thumb_width=900, padding=10):
    imgs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        # downscale to thumb_width
        if im.width > thumb_width:
            new_h = int(im.height * (thumb_width / im.width))
            im = im.resize((thumb_width, new_h))
        imgs.append(im)

    rows = int(np.ceil(len(imgs) / cols))
    w = cols * thumb_width + (cols + 1) * padding
    row_heights = [0] * rows
    for r in range(rows):
        row_imgs = imgs[r*cols:(r+1)*cols]
        row_heights[r] = max([im.height for im in row_imgs] + [0])

    h = sum(row_heights) + (rows + 1) * padding
    sheet = Image.new("RGB", (w, h), (255, 255, 255))

    y = padding
    idx = 0
    for r in range(rows):
        x = padding
        for c in range(cols):
            if idx >= len(imgs):
                break
            im = imgs[idx]
            sheet.paste(im, (x, y))
            x += thumb_width + padding
            idx += 1
        y += row_heights[r] + padding

    ensure_dir(str(out_path.parent))
    sheet.save(out_path, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
