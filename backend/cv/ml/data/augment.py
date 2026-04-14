"""
Pipeline augmentacji danych dla Square Classifier.

Augmentacja jest kluczowa gdy mamy mało danych (< 2000 próbek) — sztucznie
powiększa dataset i zmusza model do generalizacji na różne warunki.

Stosowane transformacje:
  RandomBrightnessContrast  — symuluje różne oświetlenie (poranne / wieczorne)
  GaussNoise                — szum matrycy kamery CMOS
  GaussianBlur              — ruch kamery / delikatny ruch figury
  HorizontalFlip            — symetria lewo-prawo (pole a1 wygląda jak h1)
  Rotate (±5°)              — drobne przesunięcie kamery / szachownicy
  RandomShadow              — cień ręki gracza na polu

Każda próbka jest augmentowana N_AUGMENTS razy → dataset ×(1 + N_AUGMENTS).

Przykład użycia:
    from cv.ml.data.augment import augment_patch
    patch_gray = np.array(...)  # 70×70 uint8
    variants = augment_patch(patch_gray, n=5)  # lista 5 augmentowanych patchy
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Ile augmentowanych wariantów generować na każdą próbkę podczas collect
N_AUGMENTS = 4


def _build_pipeline():
    """
    Buduje pipeline augmentacji Albumentations.

    Importujemy albumentations lokalnie żeby serwer startował normalnie
    nawet gdy biblioteka nie jest zainstalowana (graceful degradation).
    """
    try:
        import albumentations as A
        return A.Compose([
            # Jasność i kontrast — kluczowe przy zmieniającym się oświetleniu
            A.RandomBrightnessContrast(
                brightness_limit=0.35,   # ±35% jasności
                contrast_limit=0.35,
                p=0.8,
            ),
            # Szum matrycy CMOS kamery
            A.GaussNoise(
                var_limit=(5.0, 30.0),
                p=0.5,
            ),
            # Blur = delikatny ruch / niska ostrość
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.3,
            ),
            # Flip poziomy (pole wygląda tak samo z lewej jak z prawej)
            A.HorizontalFlip(p=0.5),
            # Drobna rotacja (kamera lekko przekręcona)
            A.Rotate(
                limit=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            # Cień = ręka gracza nad planszą
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=4,
                p=0.2,
            ),
        ])
    except ImportError:
        logger.warning("albumentations niedostępne — augmentacja wyłączona.")
        return None


_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _build_pipeline()
    return _PIPELINE


def augment_patch(patch_gray: np.ndarray, n: int = N_AUGMENTS) -> list[np.ndarray]:
    """
    Generuje n augmentowanych wariantów patcha w skali szarości.

    Parametry:
        patch_gray — obraz uint8 (H×W), skala szarości
        n          — liczba wariantów do wygenerowania

    Zwraca:
        Lista n tablic uint8 tego samego rozmiaru co wejście.
        Jeśli albumentations niedostępne — zwraca n kopii oryginału.
    """
    pipeline = _get_pipeline()
    results = []

    for _ in range(n):
        if pipeline is None:
            results.append(patch_gray.copy())
            continue

        # Albumentations wymaga obrazu RGB lub grayscale jako (H, W, 1) lub (H, W)
        augmented = pipeline(image=patch_gray)
        results.append(augmented["image"])

    return results
