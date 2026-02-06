import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, test_tf


def get_train_val_loaders(cfg):
    train_tf, test_tf = get_transforms(cfg["img_size"])

    full = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_tf
    )

    val_size = int(0.15 * len(full))
    train_size = len(full) - val_size

    # deterministic split
    g = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(full, [train_size, val_size], generator=g)

    # switch val transform
    val_ds.dataset.transform = test_tf

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, len(full.classes), full.classes


def get_test_loader(cfg):
    _, test_tf = get_transforms(cfg["img_size"])

    test_ds = datasets.OxfordIIITPet(
        root="data",
        split="test",
        target_types="category",
        download=True,
        transform=test_tf
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return test_ds, test_loader, test_ds.classes
