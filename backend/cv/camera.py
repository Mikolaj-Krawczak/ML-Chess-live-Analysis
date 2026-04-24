"""
Pobieranie klatek z kamery IP (IPWebcam / dowolny strumień HTTP MJPEG).

fetch_snapshot()          — jednorazowe HTTP GET /shot.jpg (fallback / endpointy REST)
start_stream_thread()     — uruchamia wątek czytający ciągły strumień MJPEG do bufora
stop_stream_thread()      — zatrzymuje wątek strumienia
get_latest_frame()        — zwraca najnowszą klatkę z bufora (lub None)
fetch_snapshot_fast()     — preferowana w pętli live: bufor → fallback do HTTP GET
open_stream()             — context manager cv2.VideoCapture (stary interfejs)
"""

import base64
import logging
import threading
import time
from contextlib import contextmanager

import cv2
import numpy as np
import requests

from .config import CAMERA_SNAPSHOT_URL, CAMERA_STREAM_URL, CAMERA_TIMEOUT_S

logger = logging.getLogger(__name__)

# Globalna sesja HTTP z keep-alive — wielokrotne połączenia do tej samej kamery
# bez kosztu TCP handshake na każdy request (A2).
_session = requests.Session()
# Nie używaj zmiennych proxy środowiska (częsta przyczyna timeoutów lokalnej kamery).
_session.trust_env = False

# -------------------------------------------------------------------
# Background stream reader (A1) — ciągły MJPEG do bufora w pamięci
# -------------------------------------------------------------------
# Idea: zamiast robić HTTP GET /shot.jpg przy każdym ticku (drogie),
# jeden wątek otwiera strumień /video (MJPEG) i cały czas odświeża
# globalną zmienną _latest_frame. Pętla live czyta stąd (zero I/O).

_stream_thread: threading.Thread | None = None
_stream_stop_event = threading.Event()
_latest_frame: np.ndarray | None = None
_latest_frame_lock = threading.Lock()
_latest_frame_ts: float = 0.0


def _stream_worker() -> None:
    """Pętla wątku: odczytuje MJPEG i aktualizuje bufor."""
    global _latest_frame, _latest_frame_ts

    while not _stream_stop_event.is_set():
        try:
            cap = cv2.VideoCapture(CAMERA_STREAM_URL)
            if not cap.isOpened():
                logger.warning(
                    "Stream worker: nie można otworzyć %s — retry za 2s",
                    CAMERA_STREAM_URL,
                )
                _stream_stop_event.wait(2.0)
                continue

            logger.info("Stream worker: strumień otwarty %s", CAMERA_STREAM_URL)

            while not _stream_stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    logger.warning("Stream worker: read() się nie powiodło — reconnect")
                    break
                with _latest_frame_lock:
                    _latest_frame = frame
                    _latest_frame_ts = time.monotonic()

            cap.release()
        except Exception as exc:  # pragma: no cover
            logger.exception("Stream worker wyjątek: %s", exc)
            _stream_stop_event.wait(2.0)

    logger.info("Stream worker: zakończony.")


def start_stream_thread() -> None:
    """Uruchamia wątek backgroundowy czytający MJPEG. Idempotentne."""
    global _stream_thread

    if _stream_thread is not None and _stream_thread.is_alive():
        return

    _stream_stop_event.clear()
    _stream_thread = threading.Thread(
        target=_stream_worker,
        name="camera-stream-reader",
        daemon=True,
    )
    _stream_thread.start()
    logger.info("Camera stream thread wystartował.")


def stop_stream_thread(join_timeout: float = 2.0) -> None:
    """Zatrzymuje wątek strumienia. Bezpieczne do wywołania wielokrotnie."""
    global _stream_thread

    _stream_stop_event.set()
    if _stream_thread is not None and _stream_thread.is_alive():
        _stream_thread.join(timeout=join_timeout)
    _stream_thread = None


def get_latest_frame(max_age_s: float = 2.0) -> np.ndarray | None:
    """
    Zwraca kopię ostatniej klatki z bufora jeśli nie starsza niż max_age_s.

    Zwraca None gdy bufor pusty lub klatka zbyt stara (stream padł).
    """
    with _latest_frame_lock:
        if _latest_frame is None:
            return None
        age = time.monotonic() - _latest_frame_ts
        if age > max_age_s:
            return None
        return _latest_frame.copy()


def fetch_snapshot() -> np.ndarray:
    """
    Pobiera pojedynczą klatkę przez HTTP GET /shot.jpg.

    Używana w endpointach snapshotów (kalibracja, debug).
    Dla pętli live preferuj fetch_snapshot_fast().
    """
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = _session.get(
                CAMERA_SNAPSHOT_URL, timeout=CAMERA_TIMEOUT_S, verify=False
            )
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.2)
                continue
            raise RuntimeError(
                f"Nie można połączyć się z kamerą ({CAMERA_SNAPSHOT_URL}): {exc}"
            ) from exc
    else:  # pragma: no cover
        raise RuntimeError(
            f"Nie można połączyć się z kamerą ({CAMERA_SNAPSHOT_URL}): {last_exc}"
        )

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(
            f"Odpowiedź z kamery nie zawiera obrazu JPEG "
            f"(Content-Type: {resp.headers.get('Content-Type', '?')})."
        )

    logger.debug("Pobrano klatkę %dx%d", frame.shape[1], frame.shape[0])
    return frame


def fetch_snapshot_fast() -> np.ndarray:
    """
    Szybka wersja dla pętli live: najpierw bufor z wątku strumienia,
    fallback do HTTP GET jeśli bufor pusty / nieaktywny.
    """
    frame = get_latest_frame(max_age_s=2.0)
    if frame is not None:
        return frame
    # Fallback — albo stream jeszcze się nie uruchomił, albo padł
    return fetch_snapshot()


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Koduje obraz BGR do surowych bajtów JPEG."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Nie udało się zakodować obrazu do JPEG.")
    return buf.tobytes()


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Koduje obraz BGR do łańcucha base64 JPEG."""
    return base64.b64encode(frame_to_jpeg_bytes(frame, quality)).decode("utf-8")


def is_camera_reachable() -> bool:
    """Szybki check dostępności kamery. Nie rzuca wyjątku."""
    try:
        r = _session.get(
            CAMERA_SNAPSHOT_URL, timeout=CAMERA_TIMEOUT_S, verify=False
        )
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
