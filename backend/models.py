from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS = PROJECT_ROOT / "saved_models"
MODEL_NAMES = ("Custom CNN", "ResNet18", "MobileNetV3")
CHECKPOINT_STEMS = {
    "Custom CNN": "custom_cnn_",
    "ResNet18": "resnet18_",
    "MobileNetV3": "mobilenetv3_",
}
IMAGE_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class BrainCNN(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def create_model(name: str, number_of_classes: int) -> nn.Module:
    if name == "Custom CNN":
        return BrainCNN(number_of_classes)
    if name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
        return model
    if name == "MobileNetV3":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            number_of_classes,
        )
        return model
    raise ValueError(f"Unsupported model name: {name}")


def _report_for_model(
    name: str,
    model_report: dict[str, Any],
    class_names: list[str],
    run_id: str,
    epochs: int,
) -> dict[str, Any]:
    history = {
        key: value
        for key, value in model_report.items()
        if isinstance(value, list)
    }
    confusion_matrices = history.get("test_confusion_matrices", [])
    if not confusion_matrices:
        raise ValueError(f"Test confusion matrix is missing for {name}.")
    matrix = confusion_matrices[-1]
    if len(matrix) != len(class_names) or any(
        len(row) != len(class_names) for row in matrix
    ):
        raise ValueError(f"Test confusion matrix has unexpected dimensions for {name}.")

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(class_names)))
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for index, class_name in enumerate(class_names):
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        true_positives = matrix[index][index]
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / support if support else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "name": name,
        "run_id": run_id,
        "epochs": epochs,
        "history": history,
        "confusion_matrix": matrix,
        "test_report": {
            "accuracy": correct / total if total else 0.0,
            "macro_precision": sum(precisions) / len(precisions),
            "macro_recall": sum(recalls) / len(recalls),
            "macro_f1": sum(f1_scores) / len(f1_scores),
            "support": total,
            "classes": per_class,
        },
    }


def _latest_complete_run() -> tuple[Path, Path]:
    candidates = sorted(SAVED_MODELS.glob("run_*"), reverse=True)
    for run_dir in candidates:
        reports = sorted(run_dir.glob("all_models_*_metrics.json"))
        checkpoints = [
            list(run_dir.glob(f"{stem}*_best.pth")) for stem in CHECKPOINT_STEMS.values()
        ]
        if reports and all(len(matches) == 1 for matches in checkpoints):
            return run_dir, reports[0]
    raise FileNotFoundError(
        "No complete model run found. Expected all three *_best.pth checkpoints "
        "and an all_models_*_metrics.json file in one saved_models/run_* folder."
    )


def load_models_and_reports() -> tuple[dict[str, nn.Module], dict[str, Any]]:
    run_dir, report_path = _latest_complete_run()
    with report_path.open(encoding="utf-8") as report_file:
        saved_report = json.load(report_file)

    class_names = saved_report.get("class_names")
    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Checkpoint class order must be {CLASS_NAMES}; found {class_names}."
        )
    models_report = saved_report.get("models", {})
    loaded_models: dict[str, nn.Module] = {}
    reports: dict[str, Any] = {}

    for name, stem in CHECKPOINT_STEMS.items():
        checkpoint_matches = list(run_dir.glob(f"{stem}*_best.pth"))
        if len(checkpoint_matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one checkpoint for {name} in {run_dir}."
            )
        checkpoint = torch.load(
            checkpoint_matches[0],
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("class_names") != class_names:
            raise ValueError(f"Checkpoint class order does not match for {name}.")
        if checkpoint.get("image_size") != IMAGE_SIZE:
            raise ValueError(f"Checkpoint image size does not match for {name}.")
        if checkpoint.get("model_name") != name:
            raise ValueError(f"Checkpoint name does not match for {name}.")

        model = create_model(name, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(DEVICE)
        model.eval()
        loaded_models[name] = model
        reports[name] = _report_for_model(
            name,
            models_report[name],
            class_names,
            saved_report["run_id"],
            saved_report["epochs"],
        )

    return loaded_models, {
        "run_id": saved_report["run_id"],
        "epochs": saved_report["epochs"],
        "class_names": class_names,
        "device": str(DEVICE),
        "models": [reports[name] for name in MODEL_NAMES],
    }


def predict(
    image: Image.Image,
    model_name: str,
    model: nn.Module,
    class_names: list[str],
) -> dict[str, Any]:
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    prediction_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = probabilities[prediction_index]
    return {
        "model": model_name,
        "predicted_class": class_names[prediction_index],
        "confidence": confidence,
        "uncertainty": 1 - confidence,
        "probabilities": [
            {"class_name": class_names[index], "probability": probability}
            for index, probability in enumerate(probabilities)
        ],
        "run_id": None,
        "disclaimer": (
            "Research demonstration only. This model is not a medical device "
            "and cannot diagnose or rule out a condition."
        ),
    }
