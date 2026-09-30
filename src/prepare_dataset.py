from pathlib import Path
import random
import shutil

from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "jpg"
LABELS_FILE = PROJECT_ROOT / "data" / "imagelabels.mat"

OUTPUT_DIR = PROJECT_ROOT / "data" / "flowers_13"

# Her çalıştırmada aynı train/validation/test dağılımını üretmek için
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


SELECTED_CLASSES = {
    54: "sunflower",
    74: "rose",
    83: "hibiscus",
    78: "lotus",
    8: "bird_of_paradise",
    77: "passion_flower",
    80: "anthurium",
    94: "foxglove",
    25: "grape_hyacinth",
    42: "daffodil",
    44: "poinsettia",
    63: "black_eyed_susan",
    82: "clematis",
}


def load_images_by_class():
    labels = loadmat(LABELS_FILE)["labels"][0]

    images_by_class = {
        class_id: []
        for class_id in SELECTED_CLASSES
    }

    for image_id, class_id in enumerate(labels, start=1):

        if class_id not in SELECTED_CLASSES:
            continue

        image_path = IMAGE_DIR / f"image_{image_id:05d}.jpg"

        images_by_class[class_id].append(image_path)

    return images_by_class


def copy_images(images, split, flower_name):
    destination_dir = OUTPUT_DIR / split / flower_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(
            image_path,
            destination_dir / image_path.name
        )


def prepare_dataset():

    random.seed(RANDOM_SEED)

    # Eski dataset ayrımını temizle
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    images_by_class = load_images_by_class()

    totals = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    print("\nSınıf dağılımları:\n")

    for class_id, images in images_by_class.items():

        flower_name = SELECTED_CLASSES[class_id]

        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        validation_count = int(total * VALIDATION_RATIO)

        train_images = images[:train_count]

        validation_images = images[
            train_count:
            train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]

        copy_images(train_images, "train", flower_name)
        copy_images(validation_images, "validation", flower_name)
        copy_images(test_images, "test", flower_name)

        totals["train"] += len(train_images)
        totals["validation"] += len(validation_images)
        totals["test"] += len(test_images)

        print(
            f"{flower_name:20} "
            f"Train: {len(train_images):3} | "
            f"Validation: {len(validation_images):3} | "
            f"Test: {len(test_images):3}"
        )

    print("\nDataset hazırlandı.\n")

    print(f"Train      : {totals['train']}")
    print(f"Validation : {totals['validation']}")
    print(f"Test       : {totals['test']}")
    print(f"Toplam     : {sum(totals.values())}")


if __name__ == "__main__":
    prepare_dataset()