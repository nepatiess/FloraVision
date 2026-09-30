from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "flowers_13"

# Şimdilik sadece görüntüleri modelin kullanabileceği
# standart boyuta getiriyoruz.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=transform
)

validation_dataset = datasets.ImageFolder(
    DATASET_DIR / "validation",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=transform
)


print("\nSınıflar:")
for index, class_name in enumerate(train_dataset.classes):
    print(f"{index}: {class_name}")


print("\nDataset boyutları:")
print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(validation_dataset)}")
print(f"Test       : {len(test_dataset)}")


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("\nİlk batch:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)