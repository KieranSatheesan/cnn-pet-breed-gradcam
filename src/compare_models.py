import argparse
import os
import yaml
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from src.data import get_test_loader
from src.models import build_model
from src.utils import ensure_dir, denormalize, tensor_to_uint8


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    if cfg.get("device", "auto") == "cpu":
        return "cpu"
    if cfg.get("device", "auto") == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class GradCAM:
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


def overlay_heatmap(img_u8, heatmap_2d, alpha=0.45):
    h, w = img_u8.shape[:2]
    hm = cv2.resize(heatmap_2d, (w, h))
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    out = (img_u8 * (1 - alpha) + hm_color * alpha).astype(np.uint8)
    return out


def load_model_and_cam(cfg, model_name, num_classes, device):
    model, target_layer = build_model(model_name, num_classes)
    ckpt = os.path.join(cfg["output_dir"], model_name, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    cam = GradCAM(model, target_layer)
    return model, cam


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    device = get_device(cfg)

    test_ds, _, class_names = get_test_loader(cfg)
    num_classes = len(class_names)

    # load predictions
    pA = pd.read_csv(os.path.join(cfg["output_dir"], "resnet18", "predictions.csv"))
    pB = pd.read_csv(os.path.join(cfg["output_dir"], "efficientnet_b0", "predictions.csv"))

    # ensure aligned by index
    pA = pA.sort_values("index").reset_index(drop=True)
    pB = pB.sort_values("index").reset_index(drop=True)

    disagree = pA["y_pred"].values != pB["y_pred"].values
    idxs = np.where(disagree)[0].tolist()

    ensure_dir(os.path.join(cfg["output_dir"], "compare"))
    pd.DataFrame({"index": idxs}).to_csv(os.path.join(cfg["output_dir"], "compare", "disagreements.csv"), index=False)
    print(f"Disagreements: {len(idxs)} / {len(pA)}")

    # choose subset to visualise (prioritise where one is correct)
    chosen = []
    for i in idxs:
        y = int(pA.loc[i, "y_true"])
        a = int(pA.loc[i, "y_pred"])
        b = int(pB.loc[i, "y_pred"])
        if (a == y) != (b == y):  # exactly one correct
            chosen.append(i)
    # if not enough, fill with any disagreements
    for i in idxs:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= int(cfg.get("num_disagreements", 12)):
            break

    modelA, camA = load_model_and_cam(cfg, "resnet18", num_classes, device)
    modelB, camB = load_model_and_cam(cfg, "efficientnet_b0", num_classes, device)

    n = len(chosen)
    ncols = 3
    nrows = n
    plt.figure(figsize=(14, 4 * nrows))

    for r, idx in enumerate(chosen):
        img_t, y = test_ds[idx]
        x = img_t.unsqueeze(0).to(device)

        # forward for each model
        modelA.zero_grad(set_to_none=True)
        logitsA = modelA(x)
        predA = int(logitsA.argmax(1).item())
        scoreA = logitsA[0, predA]
        scoreA.backward(retain_graph=True)
        hmA = camA.generate().detach().cpu().numpy()[0]

        modelB.zero_grad(set_to_none=True)
        logitsB = modelB(x)
        predB = int(logitsB.argmax(1).item())
        scoreB = logitsB[0, predB]
        scoreB.backward()
        hmB = camB.generate().detach().cpu().numpy()[0]

        # original image for display
        img_u8 = tensor_to_uint8(denormalize(img_t))
        overA = overlay_heatmap(img_u8, hmA)
        overB = overlay_heatmap(img_u8, hmB)

        title = f"GT: {class_names[y]}\nResNet18: {class_names[predA]} | EffNetB0: {class_names[predB]}"

        # column 1: original
        ax1 = plt.subplot(nrows, ncols, r * ncols + 1)
        ax1.imshow(img_u8)
        ax1.set_title(title, fontsize=10)
        ax1.axis("off")

        # column 2: resnet cam
        ax2 = plt.subplot(nrows, ncols, r * ncols + 2)
        ax2.imshow(overA)
        ax2.set_title("ResNet18 Grad-CAM", fontsize=10)
        ax2.axis("off")

        # column 3: effnet cam
        ax3 = plt.subplot(nrows, ncols, r * ncols + 3)
        ax3.imshow(overB)
        ax3.set_title("EfficientNet-B0 Grad-CAM", fontsize=10)
        ax3.axis("off")

    out_path = os.path.join(cfg["output_dir"], "compare", "disagreements_gradcam.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
