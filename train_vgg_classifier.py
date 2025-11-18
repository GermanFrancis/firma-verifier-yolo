import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_TRAIN_DIR = "data/vgg/train"
DATA_VAL_DIR   = "data/vgg/val"

OUT_MODEL_PATH = "models/vgg_finetuned_classifier.pt"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FROZEN = 10    # entrenar solo la última capa
EPOCHS_FINETUNE = 10  # luego fine-tuning completo
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-4


def get_dataloaders():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(
            degrees=10,          # rotation_range ~0.1 rad ≈ 10°
            translate=(0.2, 0.2),# width_shift_range, height_shift_range = 0.2
            shear=10             # shear_range ~0.1 rad ≈ 10°
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(DATA_TRAIN_DIR, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(DATA_VAL_DIR,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes):
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    in_features = vgg.classifier[-1].in_features  # 4096
    vgg.classifier[-1] = nn.Linear(in_features, num_classes)

    return vgg


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    os.makedirs(os.path.dirname(OUT_MODEL_PATH), exist_ok=True)

    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Clases detectadas ({num_classes}): {class_names}")

    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[-1].parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FROZEN,
        momentum=0.9,
        weight_decay=1e-4,
    )

    print("FASE 1: entrenando solo la última capa...")
    for epoch in range(EPOCHS_FROZEN):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"[FROZEN] Epoch {epoch+1}/{EPOCHS_FROZEN} "
              f"- Train loss: {train_loss:.4f} acc: {train_acc:.3f} "
              f"- Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR_FINETUNE,
        momentum=0.9,
        weight_decay=1e-4,
    )

    print("FASE 2: fine-tuning de todas las capas...")
    for epoch in range(EPOCHS_FINETUNE):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"[FT] Epoch {epoch+1}/{EPOCHS_FINETUNE} "
              f"- Train loss: {train_loss:.4f} acc: {train_acc:.3f} "
              f"- Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
        },
        OUT_MODEL_PATH,
    )
    print(f"Modelo VGG16 fine-tune guardado en: {OUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
