from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from collections import Counter



# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
MODEL_PATH = PROJECT_ROOT / "models" / "flower_resnet18.pth"
TRAIN_DIR = PROJECT_ROOT / "data" / "flowers_13" / "train"


# --------------------------------------------------
# Settings
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}


# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Classes
# --------------------------------------------------

class_names = sorted([
    folder.name
    for folder in TRAIN_DIR.iterdir()
    if folder.is_dir()
])


# --------------------------------------------------
# Model
# --------------------------------------------------

@st.cache_resource
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


model = load_model()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            3
        )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):
        results.append({
            "class": class_names[index.item()],
            "confidence": probability.item() * 100
        })

    return results


# --------------------------------------------------
# Load Real-World Test Images
# --------------------------------------------------

@st.cache_data
def get_test_images():
    images = []

    for class_folder in sorted(TEST_IMAGES_DIR.iterdir()):
        if not class_folder.is_dir():
            continue

        actual_class = class_folder.name

        for image_path in sorted(class_folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({
                    "path": image_path,
                    "actual_class": actual_class
                })

    return images


test_images = get_test_images()


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

results = []

with st.spinner("Görseller model ile analiz ediliyor..."):
    for item in test_images:
        predictions = predict_image(item["path"])

        predicted_class = predictions[0]["class"]
        confidence = predictions[0]["confidence"]

        results.append({
            "path": item["path"],
            "actual_class": item["actual_class"],
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top3": predictions,
            "correct": predicted_class == item["actual_class"]
        })

# --------------------------------------------------
# Page
# --------------------------------------------------

st.set_page_config(
    page_title="FloraVision",
    page_icon="🌸",
    layout="wide"
)

st.title("FloraVision")
st.subheader("Real-World Flower Classification Test")

st.caption(
    f"Model: ResNet18 | Device: {DEVICE}"
)


# --------------------------------------------------
# Flower Filter
# --------------------------------------------------

flower_options = ["Tüm Çiçekler"] + class_names

selected_flower = st.selectbox(
    "Çiçek türü",
    flower_options
)

if selected_flower == "Tüm Çiçekler":
    flower_results = results
else:
    flower_results = [
        result
        for result in results
        if result["actual_class"] == selected_flower
    ]


# --------------------------------------------------
# Metrics
# --------------------------------------------------

total = len(flower_results)

correct = sum(
    result["correct"]
    for result in flower_results
)

incorrect = total - correct

accuracy = (
    (correct / total) * 100
    if total > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Toplam Görsel", total)
col2.metric("Doğru Tahmin", correct)
col3.metric("Yanlış Tahmin", incorrect)
col4.metric("Accuracy", f"{accuracy:.2f}%")

# --------------------------------------------------
# Class Performance Summary
# --------------------------------------------------

st.subheader("Çiçek Türlerine Göre Sonuçlar")

summary_data = []

for flower in class_names:

    flower_results = [
        result
        for result in results
        if result["actual_class"] == flower
    ]

    flower_total = len(flower_results)

    flower_correct = sum(
        result["correct"]
        for result in flower_results
    )

    flower_incorrect = flower_total - flower_correct

    flower_accuracy = (
        (flower_correct / flower_total) * 100
        if flower_total > 0
        else 0
    )

     # Sadece yanlış tahminleri al
    wrong_predictions = [
        result["predicted_class"]
        for result in flower_results
        if not result["correct"]
    ]

    # En çok hangi çiçekle karıştırılmış?
    if wrong_predictions:
        prediction_counts = Counter(wrong_predictions)

        most_confused_class, confusion_count = (
            prediction_counts.most_common(1)[0]
        )
    else:
        most_confused_class = "-"
        confusion_count = 0

    summary_data.append({
        "Çiçek": flower,
        "Görsel": flower_total,
        "Doğru": flower_correct,
        "Yanlış": flower_incorrect,
        "Accuracy": f"{flower_accuracy:.2f}%",
        "En Çok Karıştırılan": most_confused_class,
        "Karıştırılma Sayısı": confusion_count
    })


st.dataframe(
    summary_data,
    use_container_width=True,
    hide_index=True
)

st.divider()


# --------------------------------------------------
# Result Filter
# --------------------------------------------------

filter_option = st.radio(
    "Sonuçları filtrele",
    [
        "Tümü",
        "Doğru Tahminler",
        "Yanlış Tahminler"
    ],
    horizontal=True
)

if filter_option == "Doğru Tahminler":
    filtered_results = [
        result
        for result in flower_results
        if result["correct"]
    ]

elif filter_option == "Yanlış Tahminler":
    filtered_results = [
        result
        for result in flower_results
        if not result["correct"]
    ]

else:
    filtered_results = flower_results


# --------------------------------------------------
# Result Count
# --------------------------------------------------

st.caption(
    f"{len(filtered_results)} görsel gösteriliyor."
)


# --------------------------------------------------
# Result Gallery
# --------------------------------------------------

columns_per_row = 4

for start in range(
    0,
    len(filtered_results),
    columns_per_row
):

    columns = st.columns(columns_per_row)

    row_results = filtered_results[
        start:start + columns_per_row
    ]

    for column, result in zip(
        columns,
        row_results
    ):

        with column:

            st.image(
                str(result["path"]),
                use_container_width=True
            )

            if result["correct"]:
                st.success("DOĞRU")
            else:
                st.error("YANLIŞ")

            st.write(
                f"**Gerçek:** "
                f"{result['actual_class']}"
            )

            st.write(
                f"**Tahmin:** "
                f"{result['predicted_class']}"
            )

            st.write(
                f"**Confidence:** "
                f"{result['confidence']:.2f}%"
            )

            with st.expander("Top 3 Tahmin"):

                for rank, prediction in enumerate(
                    result["top3"],
                    start=1
                ):

                    st.write(
                        f"{rank}. "
                        f"{prediction['class']} — "
                        f"{prediction['confidence']:.2f}%"
                    )

            st.divider()