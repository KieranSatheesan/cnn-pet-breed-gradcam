import argparse
import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / cam.max()
        return cam


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

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, len(dataset.classes)
    )
    model.load_state_dict(torch.load(
        os.path.join(cfg["output_dir"], "best_model.pt"),
        map_location=device
    ))
    model.to(device).eval()

    cam = GradCAM(model, model.features[-1])

    images, labels = zip(*[dataset[i] for i in range(8)])
    batch = torch.stack(images).to(device)
    outputs = model(batch)
    preds = outputs.argmax(1)

    outputs.sum().backward()
    heatmaps = cam.generate().detach().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        heatmap = cv2.resize(heatmaps[i], (img.shape[1], img.shape[0]))
        ax.imshow(img)
        ax.imshow(heatmap, cmap="jet", alpha=0.4)
        ax.axis("off")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    plt.savefig(os.path.join(cfg["output_dir"], "gradcam_examples.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
