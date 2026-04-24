"""
Kalibracja perspektywy szachownicy.

Metody wyznaczania 4 narożników:
  auto   — cv2.findChessboardCorners (wzorzec 7×7) lub YOLOv8 (jeśli załadowany)
  manual — 4 punkty [TL, TR, BR, BL] podane przez użytkownika

Po wyznaczeniu narożników:
  getPerspectiveTransform → macierz homografii H
  warpPerspective → prostokątny obraz BOARD_SIZE_PX × BOARD_SIZE_PX

Kalibracja jest persystowana w JSON i ładowana automatycznie przy starcie.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import (
    BOARD_SIZE_PX,
    CALIBRATION_PATH,
    CHESSBOARD_INNER_CORNERS,
)

logger = logging.getLogger(__name__)

# Singleton — macierz homografii i metadane w pamięci procesu
_homography_matrix: np.ndarray | None = None
_calibration_meta: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _destination_corners(size: int) -> np.ndarray:
    """Docelowe narożniki kwadratu size×size: TL, TR, BR, BL."""
    return np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)


def _corners_from_inner(inner: np.ndarray) -> np.ndarray:
    """
    Ekstrapoluje 4 zewnętrzne narożniki z 7×7 wewnętrznych punktów.

    findChessboardCorners zwraca wewnętrzne przecięcia linii — przesuwa je
    o 1 krok "na zewnątrz" żeby objąć całą planszę.
    """
    cols, rows = CHESSBOARD_INNER_CORNERS
    pts = inner.reshape(rows, cols, 2)

    step_x = float(np.mean(np.diff(pts[:, :, 0], axis=1)))
    step_y = float(np.mean(np.diff(pts[:, :, 1], axis=0)))

    tl = pts[0, 0] + np.array([-step_x, -step_y])
    tr = pts[0, cols - 1] + np.array([step_x, -step_y])
    br = pts[rows - 1, cols - 1] + np.array([step_x, step_y])
    bl = pts[rows - 1, 0] + np.array([-step_x, step_y])

    return np.array([tl, tr, br, bl], dtype=np.float32)


# ---------------------------------------------------------------------------
# Kalibracja auto / manual
# ---------------------------------------------------------------------------


def calibrate_auto(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Wykrywa wzorzec 7×7 wewnętrznych narożników i wyznacza homografię.

    Jeśli załadowany model YOLO — używa go zamiast findChessboardCorners.
    Rzuca RuntimeError gdy wzorzec nie zostanie odnaleziony.
    """
    # Próba użycia modelu YOLO jeśli dostępny
    try:
        from .ml.board_detector import detect_board_corners
        src_corners = detect_board_corners(frame)
        if src_corners is not None:
            logger.info("Board detector (YOLO) wykrył narożniki.")
            warped = _warp(frame, src_corners)
            return warped, src_corners
    except ImportError:
        pass

    # Fallback: findChessboardCorners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_INNER_CORNERS, flags)

    if not found:
        raise RuntimeError(
            f"Nie wykryto wzorca {CHESSBOARD_INNER_CORNERS[0]}x{CHESSBOARD_INNER_CORNERS[1]} "
            "wewnętrznych narożników. Użyj method='manual' lub załaduj model YOLO."
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    src_corners = _corners_from_inner(corners_refined)
    warped = _warp(frame, src_corners)

    logger.info("Auto-kalibracja (findChessboardCorners) zakończona.")
    return warped, src_corners


def calibrate_manual(
    frame: np.ndarray,
    corners: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wyznacza homografię z 4 ręcznie podanych narożników [TL, TR, BR, BL].
    """
    if len(corners) != 4:
        raise ValueError(
            f"Wymagane 4 punkty, podano {len(corners)}. Kolejność: TL, TR, BR, BL."
        )
    src_corners = np.array(corners, dtype=np.float32)
    warped = _warp(frame, src_corners)
    logger.info("Manualna kalibracja zakończona.")
    return warped, src_corners


# ---------------------------------------------------------------------------
# Warp
# ---------------------------------------------------------------------------


def _warp(frame: np.ndarray, src_corners: np.ndarray) -> np.ndarray:
    """Wyprostowuje obraz używając macierzy homografii z src_corners."""
    global _homography_matrix
    dst = _destination_corners(BOARD_SIZE_PX)
    H = cv2.getPerspectiveTransform(src_corners, dst)
    _homography_matrix = H
    return cv2.warpPerspective(frame, H, (BOARD_SIZE_PX, BOARD_SIZE_PX))


def apply_warp(frame: np.ndarray) -> np.ndarray:
    """
    Stosuje zapisaną macierz homografii na nowej klatce.
    Rzuca RuntimeError gdy brak kalibracji.
    """
    if _homography_matrix is None:
        raise RuntimeError("Brak kalibracji. Wywołaj najpierw POST /cv/calibrate.")
    return cv2.warpPerspective(frame, _homography_matrix, (BOARD_SIZE_PX, BOARD_SIZE_PX))


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


def save_calibration(src_corners: np.ndarray, method: str) -> None:
    """Zapisuje macierz homografii i metadane do pliku JSON."""
    global _calibration_meta

    if _homography_matrix is None:
        raise RuntimeError("Brak macierzy homografii — kalibracja nie zakończona.")

    data = {
        "method": method,
        "board_size_px": BOARD_SIZE_PX,
        "source_corners": src_corners.tolist(),
        "homography_matrix": _homography_matrix.tolist(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _calibration_meta = data
    logger.info("Kalibracja zapisana: %s", CALIBRATION_PATH)


def load_calibration() -> bool:
    """
    Ładuje kalibrację z pliku JSON do pamięci.
    Zwraca True gdy sukces, False gdy brak pliku.
    """
    global _homography_matrix, _calibration_meta

    if not CALIBRATION_PATH.exists():
        return False

    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

        # Przelicz homografię dla aktualnego BOARD_SIZE_PX — zapisana H mogła być
        # obliczona dla innego rozmiaru wyjścia (np. przy zmianie 800→640).
        stored_size = data.get("board_size_px", BOARD_SIZE_PX)
        src_corners = np.array(data["source_corners"], dtype=np.float32)
        if stored_size != BOARD_SIZE_PX:
            logger.info(
                "Przeliczam homografię: zapisana dla %dpx, aktualna %dpx.",
                stored_size,
                BOARD_SIZE_PX,
            )
            dst = _destination_corners(BOARD_SIZE_PX)
            _homography_matrix = cv2.getPerspectiveTransform(src_corners, dst)
        else:
            _homography_matrix = np.array(data["homography_matrix"], dtype=np.float64)

        _calibration_meta = data
        logger.info(
            "Kalibracja załadowana (%s, %s)",
            data.get("method"),
            data.get("saved_at"),
        )
        return True
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Uszkodzony plik kalibracji: {exc}") from exc


def get_calibration_status() -> dict:
    """Stan kalibracji z pamięci procesu."""
    if _homography_matrix is None or _calibration_meta is None:
        return {"calibrated": False}
    return {
        "calibrated": True,
        "method": _calibration_meta.get("method"),
        "board_size_px": _calibration_meta.get("board_size_px"),
        "source_corners": _calibration_meta.get("source_corners"),
        "saved_at": _calibration_meta.get("saved_at"),
    }


def reset_calibration() -> None:
    """Czyści kalibrację z pamięci (nie usuwa pliku)."""
    global _homography_matrix, _calibration_meta
    _homography_matrix = None
    _calibration_meta = None
    logger.info("Kalibracja wyczyszczona z pamięci.")
