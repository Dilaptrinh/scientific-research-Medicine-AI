from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schemas.prediction import Classification


TYPE_LABELS = ("hop", "vi", "thuoc")


@dataclass(frozen=True)
class MobileNetTypeClassifier:
    """MobileNetV3 adapter for box, blister, and pill crops."""

    model_path: Path
    labels: tuple[str, ...] = TYPE_LABELS

    def __post_init__(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MobileNet type model was not found: {self.model_path}. "
                "It must be trained with the labels: hop, vi, thuoc."
            )

        try:
            import torch
            from torchvision import models
            from torchvision import transforms
        except ImportError as error:
            raise RuntimeError(
                "MobileNet support requires the 'ml' dependencies. "
                "Install them with: pip install -e \".[ml,dev]\""
            ) from error

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features,
            len(self.labels),
        )
        checkpoint = torch.load(self.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        object.__setattr__(self, "_torch", torch)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_transform", transform)

    def classify(self, crop: np.ndarray) -> Classification:
        if crop.size == 0:
            raise ValueError("Cannot classify an empty YOLO crop.")

        tensor = self._transform(crop).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            probabilities = self._torch.softmax(self._model(tensor), dim=1)
            confidence, index = self._torch.max(probabilities, dim=1)

        return Classification(
            label=self.labels[int(index.item())],
            confidence=float(confidence.item()),
        )
