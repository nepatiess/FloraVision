from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "flowers_13"
MODEL_PATH = PROJECT_ROOT / "models" / "flower_resnet18.pth"

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Test Transform
# -------------------------

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Test Dataset
# -------------------------

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(test_dataset.classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# -------------------------
# Evaluation
# -------------------------

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# -------------------------
# Results
# -------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(f"\nDevice: {DEVICE}")
print(f"Test images: {len(test_dataset)}")

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4
    )
)


# -------------------------
# Confusion Matrix
# -------------------------

fig, ax = plt.subplots(
    figsize=(13, 11)
)

ConfusionMatrixDisplay.from_predictions(
    all_labels,
    all_predictions,
    display_labels=test_dataset.classes,
    xticks_rotation=45,
    cmap="Blues",
    ax=ax
)

plt.title(
    "FloraVision - ResNet18 Confusion Matrix"
)

plt.tight_layout()

plt.show()