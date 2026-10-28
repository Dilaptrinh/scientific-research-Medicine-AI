from dataclasses import dataclass
from pathlib import Path

import numpy as np


def resolve_checkpoint_path(checkpoint: Path | str | None = None) -> Path | None:
    """Return a user-provided checkpoint or the repo-local MobileNet checkpoint."""
    if checkpoint is not None:
        return Path(checkpoint)

    project_root = Path(__file__).resolve().parents[2]
    for candidate in (
        project_root / "models" / "mobilenet" / "pill_mobilenet_embedding.pt",
        project_root / "models" / "mobilenet" / "pill_mobilenet_embedding.pth",
    ):
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class PillEmbeddingConfig:
    checkpoint_path: Path | None = None
    image_size: int = 224


class MobileNetPillEmbedder:
    """Load the trained MobileNet encoder and create normalized pill vectors."""

    def __init__(self, config: PillEmbeddingConfig) -> None:
        try:
            import torch
            from torchvision import transforms
        except ImportError as error:
            raise RuntimeError(
                "Pill embedding requires the ml dependencies. "
                'Install them with: pip install -e ".[ml]"'
            ) from error

        from torchvision import models

        checkpoint = None
        if config.checkpoint_path is not None:
            if not config.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Embedding checkpoint was not found: {config.checkpoint_path}"
                )
            checkpoint = torch.load(
                config.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        weights = (
            None
            if checkpoint is not None
            else models.MobileNet_V3_Large_Weights.DEFAULT
        )
        backbone = models.mobilenet_v3_large(weights=weights)
        feature_dim = backbone.classifier[0].in_features
        embedding_dim = (
            int(checkpoint["embedding_dim"]) if checkpoint is not None else feature_dim
        )
        model = torch.nn.Module()
        model.features = backbone.features
        model.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        checkpoint_keys = (
            set(checkpoint["model_state_dict"]) if checkpoint is not None else set()
        )
        projection_layers = [
            torch.nn.Linear(feature_dim, 512),
        ]
        if "projection.1.running_mean" in checkpoint_keys:
            projection_layers.append(torch.nn.BatchNorm1d(512))
        projection_layers.extend(
            [
                torch.nn.Hardswish(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(512, embedding_dim),
            ]
        )
        model.projection = torch.nn.Sequential(*projection_layers)

        def forward(images: torch.Tensor) -> torch.Tensor:
            features = model.pool(model.features(images)).flatten(1)
            return torch.nn.functional.normalize(model.projection(features), p=2, dim=1)

        model.forward = forward
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self._torch = torch
        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (
                        int(checkpoint.get("image_size", config.image_size))
                        if checkpoint is not None
                        else config.image_size,
                        int(checkpoint.get("image_size", config.image_size))
                        if checkpoint is not None
                        else config.image_size,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.embedding_dim = embedding_dim

    def embed(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            raise ValueError("Cannot embed an empty pill image.")

        tensor = self._transform(image).unsqueeze(0)
        with self._torch.no_grad():
            embedding = self._model(tensor).cpu().numpy()[0]
        return np.asarray(embedding, dtype=np.float32)
