"""
Board Detector — detekcja narożników szachownicy przez YOLOv8n.

Pipeline:
  1. YOLOv8n wykrywa bounding box szachownicy (klasa 0: "chessboard")
  2. Z bounding boxa wyznaczamy 4 narożniki via goodFeaturesToTrack
  3. Narożniki przekazujemy do calibration.py → getPerspectiveTransform

Jeśli model nie jest załadowany (brak wag), funkcja detect_board_corners()
zwraca None — calibration.py wtedy używa findChessboardCorners lub manual.

Trening modelu: patrz training/train_board.py
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Singleton modelu — ładowany raz przy starcie
_model = None
_model_loaded = False


def load_model() -> bool:
    """
    Ładuje model YOLOv8n z pliku wag.

    Zwraca True gdy sukces, False gdy brak pliku lub ultralytics niedostępne.
    Nie rzuca wyjątku — graceful degradation do metody fallback.
    """
    global _model, _model_loaded

    # Ścieżka do wag (importujemy config lokalnie żeby uniknąć circular import)
    from ..config import BOARD_DETECTOR_WEIGHTS
    weights_path = BOARD_DETECTOR_WEIGHTS

    if not weights_path.exists():
        logger.info("Brak wag board detectora: %s — fallback do OpenCV.", weights_path)
        return False

    try:
        from ultralytics import YOLO
        _model = YOLO(str(weights_path))
        _model_loaded = True
        logger.info("Board detector załadowany: %s", weights_path)
        return True
    except Exception as exc:
        logger.warning("Nie załadowano board detectora: %s", exc)
        _model_loaded = False
        return False


def is_loaded() -> bool:
    return _model_loaded


def detect_board_corners(frame: np.ndarray) -> np.ndarray | None:
    """
    Wykrywa 4 narożniki szachownicy w klatce BGR.

    Zwraca ndarray shape (4, 2) float32 [TL, TR, BR, BL] lub None gdy błąd.
    """
    if not _model_loaded or _model is None:
        return None

    try:
        results = _model(frame, verbose=False)
    except Exception as exc:
        logger.error("Błąd inferencji YOLO: %s", exc)
        return None

    # Szukamy detekcji z confidence > 0.5
    best_box = None
    best_conf = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

    if best_box is None or best_conf < 0.5:
        logger.debug("Board detector nie znalazł szachownicy (max conf: %.2f).", best_conf)
        return None

    corners = _refine_corners(frame, best_box)
    if corners is None:
        # Fallback: użyj prostokąta bounding boxa
        x1, y1, x2, y2 = best_box
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    logger.debug("Board detector: narożniki wykryte (conf=%.2f).", best_conf)
    return corners


def _refine_corners(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """
    Doprecyzowuje 4 narożniki szachownicy w obszarze bounding boxa.

    Używa goodFeaturesToTrack + logiki wyboru 4 skrajnych punktów.
    Zwraca None gdy nie uda się wybrać dokładnie 4 narożników.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Dodajemy mały margines żeby nie uciąć narożników
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(frame.shape[1], x2 + margin)
    y2 = min(frame.shape[0], y2 + margin)

    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Szukamy mocnych narożników w ROI
    corners = cv2.goodFeaturesToTrack(
        gray_roi,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=30,
    )

    if corners is None or len(corners) < 4:
        return None

    # Przesuwamy z układu ROI do układu całego obrazu
    pts = corners.reshape(-1, 2) + np.array([x1, y1])

    # Wybieramy 4 skrajne punkty: TL (min x+y), TR (max x-y), BR (max x+y), BL (min x-y)
    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmax(pts[:, 0] - pts[:, 1])]
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]

    return np.array([tl, tr, br, bl], dtype=np.float32)
