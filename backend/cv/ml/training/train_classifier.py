"""
Skrypt treningowy Square Classifier (CNN) — klasyfikacja zajętości pól.

JAK UŻYWAĆ (workflow krok po kroku):
-------------------------------------
KROK 1 — Zbierz dane treningowe:
  a) Ustaw szachownicę w pozycji startowej.
  b) Uruchom serwer i skalibruj kamerę (POST /cv/calibrate).
  c) Zainicjuj detektor (POST /cv/game/detector/start).
  d) Rozegraj kilka ruchów. Po każdym:
       POST /cv/ml/collect   ← auto-label 64 pól + augmentacja
  e) Powtórz dla różnych pozycji (środek gry, końcówka itp.)
  f) Sprawdź zebrany dataset:
       GET /cv/ml/dataset/stats   ← powinno być > 500 occupied i > 500 empty

KROK 2 — Wytrenuj model:
  cd backend
  python -m cv.ml.training.train_classifier

KROK 3 — Przetestuj:
  Zrestartuj serwer. GET /cv/health → square_classifier_loaded: true
  GET /cv/snapshot/debug → na obrazie zobaczysz "method=cnn"

WYMAGANIA:
  pip install torch torchvision

CPU Training:  ~5-15 minut dla 2000 próbek
GPU Training:  ~1-3 minuty
"""

import logging
import sys
from pathlib import Path

import yaml

# Upewniamy się że możemy importować cv.ml.square_classifier (wspólna architektura)
_BACKEND = Path(__file__).resolve().parent.parent.parent.parent  # backend/
sys.path.insert(0, str(_BACKEND))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_dataset(dataset_dir: Path, patch_size: int, val_split: float):
    """
    Ładuje patche z dataset/occupied/ i dataset/empty/.

    Zwraca (train_loader, val_loader) — PyTorch DataLoaderów.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split
    import cv2
    import numpy as np

    class PatchDataset(Dataset):
        """
        Prosty Dataset który ładuje JPEG patche z dysku.

        Każdy plik w dataset/occupied/ → label=1
        Każdy plik w dataset/empty/   → label=0
        """
        def __init__(self, occ_dir: Path, emp_dir: Path, size: int):
            self.samples: list[tuple[Path, int]] = []
            if occ_dir.exists():
                self.samples += [(p, 1) for p in occ_dir.glob("*.jpg")]
            if emp_dir.exists():
                self.samples += [(p, 0) for p in emp_dir.glob("*.jpg")]
            self.size = size

            if not self.samples:
                raise RuntimeError(
                    f"Brak próbek w {occ_dir.parent}.\n"
                    "Zbierz dane: POST /cv/ml/collect podczas rozgrywki."
                )

            n_occ = sum(1 for _, l in self.samples if l == 1)
            n_emp = sum(1 for _, l in self.samples if l == 0)
            logger.info("Dataset: %d occupied, %d empty = %d total", n_occ, n_emp, len(self.samples))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Nie można wczytać: {path}")

            # Resize (na wypadek gdyby patch miał inny rozmiar)
            img = cv2.resize(img, (self.size, self.size))

            # Normalizacja [0,1], dodanie wymiaru kanału → (1, H, W)
            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            return tensor, torch.tensor([float(label)])

    ds = PatchDataset(
        dataset_dir / "occupied",
        dataset_dir / "empty",
        patch_size,
    )

    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    logger.info("Split: %d trening / %d walidacja", n_train, n_val)

    return (
        DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0),
        DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
    )


# ---------------------------------------------------------------------------
# Trening
# ---------------------------------------------------------------------------


def train() -> None:
    """Główna pętla treningowa."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("[BŁĄD] PyTorch nie jest zainstalowany.")
        print("  pip install torch torchvision")
        sys.exit(1)

    from cv.ml.square_classifier import _build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Urządzenie: %s", device)

    # Dataset
    train_loader, val_loader = _load_dataset(
        dataset_dir=Path(_BACKEND) / cfg["dataset_dir"],
        patch_size=cfg["patch_size"],
        val_split=cfg["val_split"],
    )

    # Model, loss, optimizer
    model = _build_model().to(device)

    # Wyliczamy pos_weight automatycznie z datasetu — wyrównuje nierówne klasy.
    # Jeśli empty=24000, occupied=9500 → pos_weight ≈ 2.5
    # W każdym batchu próbki occupied będą ważone 2.5x mocniej niż empty.
    n_occ = sum(1 for _, l in train_loader.dataset.dataset.samples if l == 1)
    n_emp = sum(1 for _, l in train_loader.dataset.dataset.samples if l == 0)
    pos_weight_val = n_emp / max(n_occ, 1)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    logger.info("pos_weight: %.3f (empty=%d / occupied=%d)", pos_weight_val, n_emp, n_occ)

    # BCELoss bez redukcji — ręcznie aplikujemy wagi per próbka
    criterion = nn.BCELoss(reduction="none")
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["lr_scheduler_step_size"],
        gamma=cfg["lr_scheduler_factor"],
    )

    weights_out = Path(_BACKEND) / cfg["weights_out"]
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    no_improve = 0

    print("\n" + "=" * 60)
    print("  Square Classifier — trening CNN")
    print("=" * 60)
    print(f"  Epoki:      {cfg['epochs']}")
    print(f"  Batch:      {cfg['batch_size']}")
    print(f"  LR:         {cfg['learning_rate']}")
    print(f"  Patience:   {cfg['patience']}")
    print(f"  Wagi →      {weights_out}")
    print()

    for epoch in range(1, cfg["epochs"] + 1):
        # --- Faza treningu ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)         # (batch, 1)
            # Wagi per próbka: occupied → pos_weight, empty → 1.0
            sample_weights = torch.where(labels == 1, pos_weight, torch.ones_like(labels))
            loss = (criterion(outputs, labels) * sample_weights).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)

            if (batch_idx + 1) % cfg["log_interval"] == 0:
                logger.info(
                    "Epoch %d/%d | batch %d/%d | loss: %.4f",
                    epoch, cfg["epochs"], batch_idx + 1, len(train_loader),
                    loss.item(),
                )

        train_loss /= train_total
        train_acc = train_correct / train_total * 100

        # --- Faza walidacji ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels).mean()
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.1f}% | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.1f}%"
            + (" ← BEST" if val_loss < best_val_loss else "")
        )

        # Early stopping + zapis najlepszych wag
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), str(weights_out))
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"\n[Early stopping] Brak poprawy przez {cfg['patience']} epok.")
                break

    print(f"\n[Sukces] Najlepsze wagi zapisane: {weights_out}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print("\n[Następny krok] Zrestartuj serwer:")
    print("  python -m uvicorn main:app --reload")
    print("  GET /cv/health → square_classifier_loaded: true")
    print("  GET /cv/snapshot/debug → zobaczysz 'method=cnn' na obrazie")


# ---------------------------------------------------------------------------
# Ładowanie konfiguracji i punkt wejścia
# ---------------------------------------------------------------------------


_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    cfg: dict = yaml.safe_load(_f)


if __name__ == "__main__":
    train()
