from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "flowers_13" / "train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=transform
)

# Her çiçek türünden bir örnek bul
examples = {}

for image, label in dataset:
    class_name = dataset.classes[label]

    if class_name not in examples:
        examples[class_name] = image

    if len(examples) == len(dataset.classes):
        break


fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

for ax, (class_name, image) in zip(axes, examples.items()):
    image = image.permute(1, 2, 0)

    ax.imshow(image)
    ax.set_title(class_name)
    ax.axis("off")

# 13 fotoğraf olduğu için kalan iki alanı gizle
for ax in axes[len(examples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()