"""
Centralna konfiguracja modułu CV.

Wszystkie parametry nadpisywalne przez zmienne środowiskowe lub plik .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Kamera
# ---------------------------------------------------------------------------

CAMERA_HOST: str = os.environ.get("CV_CAMERA_HOST", "192.168.0.107")
CAMERA_PORT: int = int(os.environ.get("CV_CAMERA_PORT", "8080"))

# /shot.jpg — pojedyncza klatka JPEG (szybszy polling niż MJPEG)
CAMERA_SNAPSHOT_URL: str = os.environ.get(
    "CV_CAMERA_SNAPSHOT_URL",
    f"http://{CAMERA_HOST}:{CAMERA_PORT}/shot.jpg",
)

# /video — strumień MJPEG dla cv2.VideoCapture
CAMERA_STREAM_URL: str = os.environ.get(
    "CV_CAMERA_STREAM_URL",
    f"http://{CAMERA_HOST}:{CAMERA_PORT}/video",
)

CAMERA_TIMEOUT_S: float = float(os.environ.get("CV_CAMERA_TIMEOUT_S", "5.0"))

# ---------------------------------------------------------------------------
# Kalibracja / warp
# ---------------------------------------------------------------------------

BOARD_SIZE_PX: int = int(os.environ.get("CV_BOARD_SIZE_PX", "800"))
CELL_SIZE_PX: int = BOARD_SIZE_PX // 8

CALIBRATION_PATH: Path = Path(
    os.environ.get(
        "CV_CALIBRATION_PATH",
        str(Path(__file__).parent / "calibration_data.json"),
    )
)

# 7×7 wewnętrznych narożników dla findChessboardCorners (8×8 pól)
CHESSBOARD_INNER_CORNERS: tuple[int, int] = (7, 7)

# ---------------------------------------------------------------------------
# Detekcja zajętości (fallback — wariancja)
# ---------------------------------------------------------------------------

# Statyczny próg wariancji — używany gdy model CNN nie jest załadowany
OCCUPANCY_VARIANCE_THRESHOLD: float = float(
    os.environ.get("CV_OCCUPANCY_VARIANCE_THRESHOLD", "580.0")
)

# Margines wewnętrzny ROI komórki (px) — eliminuje krawędzie/cienie sąsiadów
CELL_MARGIN_PX: int = int(os.environ.get("CV_CELL_MARGIN_PX", "15"))

# Liczba klatek stabilizacji wymagana przez maszynę stanów
OCCUPANCY_STABILITY_FRAMES: int = int(
    os.environ.get("CV_OCCUPANCY_STABILITY_FRAMES", "5")
)

# ---------------------------------------------------------------------------
# Ścieżki modeli ML
# ---------------------------------------------------------------------------

_ML_DIR = Path(__file__).parent / "ml"

# Wagi modelu CNN do klasyfikacji zajętości pól
SQUARE_CLASSIFIER_WEIGHTS: Path = Path(
    os.environ.get(
        "CV_SQUARE_CLASSIFIER_WEIGHTS",
        str(_ML_DIR / "weights" / "square_classifier.pth"),
    )
)

# Wagi modelu YOLOv8n do detekcji szachownicy
BOARD_DETECTOR_WEIGHTS: Path = Path(
    os.environ.get(
        "CV_BOARD_DETECTOR_WEIGHTS",
        str(_ML_DIR / "weights" / "board_detector.pt"),
    )
)

# ---------------------------------------------------------------------------
# Dataset (collector)
# ---------------------------------------------------------------------------

DATASET_DIR: Path = Path(
    os.environ.get(
        "CV_DATASET_DIR",
        str(_ML_DIR / "data" / "dataset"),
    )
)
