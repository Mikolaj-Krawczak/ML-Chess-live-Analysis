"""
Square Classifier — inferencja CNN dla klasyfikacji zajętości pól.

Model przyjmuje patch 70×70px (grayscale) i zwraca p(occupied) ∈ [0,1].
Wartość > 0.5 → pole zajęte.

Architektura SquareCNN (zdefiniowana tutaj i w train_classifier.py):
  Conv2d(1→32, 3×3) + BN + ReLU + MaxPool(2×2)   → 32 × 35 × 35
  Conv2d(32→64, 3×3) + BN + ReLU + MaxPool(2×2)  → 64 × 17 × 17
  Conv2d(64→128, 3×3) + BN + ReLU + MaxPool(2×2) → 128 × 8 × 8
  Flatten → 8192
  FC(8192→256) + ReLU + Dropout(0.5)
  FC(256→1) + Sigmoid

Model ładowany jest przy starcie serwera (on_startup w router.py).
Jeśli wagi nie istnieją — board_occupancy.py automatycznie fallback do wariancji.

MODUŁ NIE RZUCA WYJĄTKU jeśli PyTorch nie jest zainstalowany —
graceful degradation na wariancję jest priorytetem stabilności API.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Singleton modelu i sesji PyTorch
_model = None
_device = None
_loaded = False

# Rozmiar wejściowego patcha (musi być taki sam jak w collector.py i train_classifier.py)
PATCH_SIZE = 70


# ---------------------------------------------------------------------------
# Definicja architektury (musi być identyczna z train_classifier.py)
# ---------------------------------------------------------------------------


def _build_model():
    """
    Buduje architekturę SquareCNN.

    Zdefiniowana tu zamiast w osobnym pliku żeby uniknąć dodatkowych importów.
    """
    import torch.nn as nn

    class SquareCNN(nn.Module):
        """
        Lekki CNN do binarnej klasyfikacji pól szachownicy (occupied / empty).

        Wejście:  (batch, 1, 70, 70) — grayscale patch
        Wyjście:  (batch, 1) — p(occupied) po Sigmoid
        """
        def __init__(self):
            super().__init__()

            # Blok 1: 1→32 kanałów, 70×70 → 35×35
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),           # 70 → 35
            )

            # Blok 2: 32→64, 35×35 → 17×17
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),           # 35 → 17
            )

            # Blok 3: 64→128, 17×17 → 8×8
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),           # 17 → 8
            )

            # Klasyfikator FC: 128*8*8=8192 → 256 → 1
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),           # regularyzacja — ważna przy małym datasecie
                nn.Linear(256, 1),
                nn.Sigmoid(),              # p(occupied) ∈ [0,1]
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return self.classifier(x)

    return SquareCNN()


# ---------------------------------------------------------------------------
# Ładowanie modelu
# ---------------------------------------------------------------------------


def load_model() -> bool:
    """
    Ładuje wagi CNN z pliku .pth.

    Zwraca True gdy sukces, False gdy brak wag lub PyTorch niedostępny.
    """
    global _model, _device, _loaded

    from ..config import SQUARE_CLASSIFIER_WEIGHTS
    weights_path = SQUARE_CLASSIFIER_WEIGHTS

    if not weights_path.exists():
        logger.info(
            "Brak wag square classifier: %s — używam fallback wariancja.",
            weights_path,
        )
        return False

    try:
        import torch

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model()
        state = torch.load(str(weights_path), map_location=_device)
        model.load_state_dict(state)
        model.eval()
        model.to(_device)

        _model = model
        _loaded = True
        logger.info(
            "Square classifier załadowany: %s (device: %s)",
            weights_path, _device,
        )
        return True

    except Exception as exc:
        logger.warning("Nie załadowano square classifier: %s", exc)
        _loaded = False
        return False


def is_loaded() -> bool:
    return _loaded


# ---------------------------------------------------------------------------
# Inferencja
# ---------------------------------------------------------------------------


def classify_cells(patches: list[np.ndarray]) -> list[float]:
    """
    Klasyfikuje listę 64 patchy grayscale.

    Parametry:
        patches — lista 64 tablic uint8, każda (H, W), skala szarości

    Zwraca:
        Lista 64 wartości float p(occupied) ∈ [0,1].
        Wartość > 0.5 → pole zajęte.

    Rzuca RuntimeError gdy model nie jest załadowany.
    """
    if not _loaded or _model is None:
        raise RuntimeError(
            "Square classifier nie jest załadowany. "
            "Wytrenuj model (train_classifier.py) i uruchom serwer ponownie."
        )

    import torch

    # Preprocessing: resize → normalize do [0,1] → tensor (N,1,H,W)
    tensors = []
    for patch in patches:
        import cv2
        p = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE)).astype(np.float32) / 255.0
        tensors.append(p)

    # (64, PATCH_SIZE, PATCH_SIZE) → (64, 1, PATCH_SIZE, PATCH_SIZE)
    batch = np.stack(tensors)[:, np.newaxis, :, :]
    tensor = torch.from_numpy(batch).to(_device)

    with torch.no_grad():
        preds = _model(tensor)          # (64, 1)

    return preds.squeeze(1).cpu().numpy().tolist()  # lista 64 floatów
