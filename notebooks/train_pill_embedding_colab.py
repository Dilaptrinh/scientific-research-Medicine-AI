"""Fine-tune MobileNetV3 as a pill-image embedding model in Google Colab.

Expected dataset layout:

    /content/pill_dataset/
      paracetamol_500/
        image_001.jpg
        image_002.jpg
      amoxicillin_500/
        image_001.jpg
        image_002.jpg

The trained checkpoint is an encoder, not a medicine classifier. It is intended
to create vectors that can later be indexed and searched with FAISS.
"""

# In Colab, run this once in a separate cell if needed:
# !pip install -q torch torchvision pillow

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# Change this path after mounting Google Drive, if the dataset is stored there.
DATASET_DIR = Path("/content/pill_dataset")
OUTPUT_PATH = Path("/content/pill_mobilenet_embedding.pt")
IMAGE_SIZE = 224
EMBEDDING_DIM = 256
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MARGIN = 0.2
NUM_WORKERS = 2
SEED = 42

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PillEmbeddingNet(nn.Module):
    """MobileNetV3 encoder that returns L2-normalized image embeddings."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        feature_dim = backbone.classifier[0].in_features
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images)).flatten(1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Use the hardest positive and hardest negative for each anchor."""

    distances = torch.cdist(embeddings, embeddings, p=2)
    same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    not_self = ~torch.eye(
        labels.size(0), dtype=torch.bool, device=labels.device
    )
    positive_mask = same_class & not_self
    negative_mask = ~same_class

    if not positive_mask.any() or not negative_mask.any():
        raise ValueError(
            "Each batch must contain at least two classes and two images "
            "from at least one class."
        )

    positive_distances = distances.masked_fill(~positive_mask, float("-inf"))
    negative_distances = distances.masked_fill(~negative_mask, float("inf"))
    hardest_positive = positive_distances.max(dim=1).values
    hardest_negative = negative_distances.min(dim=1).values

    valid_anchors = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid_anchors.any():
        raise ValueError(
            "No valid triplets in this batch. Increase BATCH_SIZE or add "
            "more images per medicine folder."
        )

    return F.relu(
        hardest_positive[valid_anchors]
        - hardest_negative[valid_anchors]
        + margin
    ).mean()


transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder was not found: {DATASET_DIR}. "
        "Upload a folder containing one subfolder per medicine."
    )

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
if len(dataset.classes) < 2:
    raise ValueError("The dataset must contain at least two medicine folders.")
if len(dataset) < 4:
    raise ValueError("Add more pill images before training the embedding model.")

print(f"Found {len(dataset)} images across {len(dataset.classes)} medicines.")
print("Class mapping:", dataset.class_to_idx)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
if len(loader) == 0:
    raise ValueError(
        "No complete training batch was created. Reduce BATCH_SIZE or add "
        "more images."
    )

model = PillEmbeddingNet().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

best_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        embeddings = model(images)
        loss = batch_hard_triplet_loss(embeddings, labels, MARGIN)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} - triplet loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "embedding_dim": EMBEDDING_DIM,
                "image_size": IMAGE_SIZE,
                "class_to_idx": dataset.class_to_idx,
                "best_train_loss": best_loss,
            },
            OUTPUT_PATH,
        )

print(f"Saved embedding checkpoint to: {OUTPUT_PATH}")
