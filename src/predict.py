from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "flowers_13"
MODEL_PATH = PROJECT_ROOT / "models" / "flower_resnet18.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Image Transform
# -------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Classes
# -------------------------

dataset = datasets.ImageFolder(
    DATASET_DIR / "train"
)

class_names = dataset.classes


# -------------------------
# Model
# -------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model = model.to(DEVICE)
model.eval()


# -------------------------
# Prediction
# -------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            3
        )

    print("\nPrediction Results\n")

    for rank, (probability, index) in enumerate(
        zip(
            top_probabilities[0],
            top_indices[0]
        ),
        start=1
    ):

        flower = class_names[index.item()]

        confidence = probability.item() * 100

        print(
            f"{rank}. "
            f"{flower:20} "
            f"{confidence:.2f}%"
        )


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":

    image_path = input(
        "Çiçek fotoğrafının yolunu gir: "
    )

    predict_image(image_path)