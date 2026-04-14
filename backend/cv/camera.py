"""
Pobieranie klatek z kamery IP (IPWebcam / dowolny strumień HTTP MJPEG).

fetch_snapshot() — jednorazowe HTTP GET /shot.jpg (używane w endpointach REST)
open_stream()    — cv2.VideoCapture MJPEG (używane w pętli ciągłej detekcji)
"""

import base64
import logging
from contextlib import contextmanager

import cv2
import numpy as np
import requests

from .config import CAMERA_SNAPSHOT_URL, CAMERA_STREAM_URL, CAMERA_TIMEOUT_S

logger = logging.getLogger(__name__)


def fetch_snapshot() -> np.ndarray:
    """
    Pobiera pojedynczą klatkę przez HTTP GET.

    Zwraca obraz BGR jako ndarray.
    Rzuca RuntimeError gdy kamera niedostępna lub odpowiedź nie jest obrazem.
    """
    try:
        resp = requests.get(CAMERA_SNAPSHOT_URL, timeout=CAMERA_TIMEOUT_S, verify=False)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Nie można połączyć się z kamerą ({CAMERA_SNAPSHOT_URL}): {exc}"
        ) from exc

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(
            f"Odpowiedź z kamery nie zawiera obrazu JPEG "
            f"(Content-Type: {resp.headers.get('Content-Type', '?')})."
        )

    logger.debug("Pobrano klatkę %dx%d", frame.shape[1], frame.shape[0])
    return frame


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Koduje obraz BGR do łańcucha base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Nie udało się zakodować obrazu do JPEG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def is_camera_reachable() -> bool:
    """Szybki check dostępności kamery. Nie rzuca wyjątku."""
    try:
        r = requests.get(CAMERA_SNAPSHOT_URL, timeout=CAMERA_TIMEOUT_S, verify=False)
        return r.status_code == 200
    except requests.RequestException:
        return False


@contextmanager
def open_stream():
    """
    Context manager otwierający strumień MJPEG przez cv2.VideoCapture.

    Przykład:
        with open_stream() as cap:
            ok, frame = cap.read()
    """
    cap = cv2.VideoCapture(CAMERA_STREAM_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć strumienia: {CAMERA_STREAM_URL}")
    try:
        logger.info("Strumień MJPEG otwarty: %s", CAMERA_STREAM_URL)
        yield cap
    finally:
        cap.release()
        logger.info("Strumień MJPEG zamknięty.")
