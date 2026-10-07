from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "flowers_13"
MODEL_DIR = PROJECT_ROOT / "models"

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Data Augmentation
# -------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.RandomPerspective(
        distortion_scale=0.15,
        p=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation'da rastgele değişiklik yapmıyoruz.
validation_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Dataset
# -------------------------

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    DATASET_DIR / "validation",
    transform=validation_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -------------------------
# Class Weights
# -------------------------

# Eğitim verisindeki sınıf dağılımını hesaplıyoruz.
# Az fotoğrafı bulunan sınıfların hatalarına
# loss hesabında daha fazla ağırlık veriyoruz.

class_counts = Counter(train_dataset.targets)

class_weights = []

for class_index in range(
    len(train_dataset.classes)
):
    count = class_counts[class_index]

    weight = len(train_dataset) / (
        len(train_dataset.classes) * count
    )

    class_weights.append(weight)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


# -------------------------
# Model
# -------------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# ResNet18 normalde 1000 ImageNet sınıfı üretir.
# Bizim 13 çiçeğimiz olduğu için
# son katmanı değiştiriyoruz.

model.fc = nn.Linear(
    model.fc.in_features,
    len(train_dataset.classes)
)

model = model.to(DEVICE)


# -------------------------
# Loss & Optimizer
# -------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

best_validation_accuracy = 0.0

MODEL_DIR.mkdir(exist_ok=True)

print(f"\nDevice: {DEVICE}")
print(f"Classes: {len(train_dataset.classes)}")
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")

print("\nClass Weights:")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):
    print(
        f"{class_name:20} "
        f"{weight.item():.4f}"
    )

print()


for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predictions == labels
        ).sum().item()


    train_accuracy = (
        100
        * train_correct
        / train_total
    )


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            validation_total += labels.size(0)

            validation_correct += (
                predictions == labels
            ).sum().item()


    validation_accuracy = (
        100
        * validation_correct
        / validation_total
    )


    # -------------------------
    # Epoch Result
    # -------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: "
        f"{train_loss / len(train_loader):.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}% | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )


    # -------------------------
    # Save Best Model
    # -------------------------

    if (
        validation_accuracy
        > best_validation_accuracy
    ):

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            model.state_dict(),
            MODEL_DIR / "flower_resnet18.pth"
        )

        print(
            f"  -> Best model saved "
            f"({validation_accuracy:.2f}%)"
        )


print("\nTraining completed.")

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.2f}%"
)