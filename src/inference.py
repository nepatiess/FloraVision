from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "flower_resnet18.pth"
TRAIN_DIR = PROJECT_ROOT / "data" / "flowers_13" / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model eğitilirken validation/test tarafında kullandığımız
# görüntü işlemlerinin aynısını kullanıyoruz.
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ImageFolder sınıfları alfabetik sırada oluşturduğu için
# eğitim sırasında kullanılan class sırasını buradan alıyoruz.
class_names = datasets.ImageFolder(TRAIN_DIR).classes


def load_model():
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

    return model


# Model API ayağa kalkarken bir kere yüklenir.
model = load_model()


def predict_image(image: Image.Image):
    image = image.convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):
        results.append({
            "class": class_names[index.item()],
            "confidence": round(probability.item() * 100, 2)
        })

    return results