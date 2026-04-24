"""
Wątek tła do ciągłej detekcji ruchów szachowych.

Zamiast HTTP-pollingu z frontendu (POST /detector/tick co 500ms backend
przez CNN), wątek sam przetwarza klatki z bufora MJPEG ze stałą częstotliwością.
Frontend potrzebuje tylko lekkiego GET /game/state — bez kosztownych tick-ów.

Architektura:
  DetectorWorker._loop():
    1. fetch_snapshot_fast()  → zero I/O, klatka z bufora MJPEG
    2. apply_warp()           → perspektywiczne wyprostowanie (640×640)
    3. detector.process_frame() → CNN batch-64 + maszyna stanów
    4. sleep(pozostały czas)  → throttle do target_fps

Wynik każdej klatki jest cache'owany w WorkerStatus (thread-safe lock).
Endpoint /detector/tick czyta ten cache — odpowiada natychmiast (<1ms).
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkerStatus:
    """Aktualny stan wątku tła — snapshot do odczytu przez router."""

    running: bool = False
    last_frame_ts: float = 0.0
    last_detector_state: str = "IDLE"
    last_move_uci: Optional[str] = None
    last_reason: str = ""
    frames_processed: int = 0
    errors: int = 0
    avg_frame_ms: float = 0.0


class DetectorWorker:
    """
    Singleton wątku detekcji — jeden wątek na całą sesję gry.

    Użycie:
        worker.start(detector, target_fps=4.0)
        # ... gra ...
        worker.stop()
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status = WorkerStatus()
        self._detector = None
        self._frame_interval: float = 0.25

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def start(self, detector, target_fps: float = 4.0) -> None:
        """Uruchamia wątek tła. Zatrzymuje poprzedni jeśli był aktywny."""
        self.stop()
        self._detector = detector
        self._frame_interval = 1.0 / max(0.5, target_fps)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="detector-worker",
            daemon=True,
        )
        self._thread.start()
        with self._status_lock:
            self._status = WorkerStatus(running=True)
        logger.info(
            "DetectorWorker uruchomiony (target %.1f fps, interval %.0fms).",
            target_fps,
            self._frame_interval * 1000,
        )

    def stop(self, join_timeout: float = 2.0) -> None:
        """Zatrzymuje wątek tła. Bezpieczne do wywołania wielokrotnie."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
        self._thread = None
        with self._status_lock:
            self._status.running = False
        logger.info("DetectorWorker zatrzymany.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> WorkerStatus:
        """Zwraca kopię aktualnego statusu (thread-safe)."""
        with self._status_lock:
            s = self._status
            return WorkerStatus(
                running=s.running,
                last_frame_ts=s.last_frame_ts,
                last_detector_state=s.last_detector_state,
                last_move_uci=s.last_move_uci,
                last_reason=s.last_reason,
                frames_processed=s.frames_processed,
                errors=s.errors,
                avg_frame_ms=s.avg_frame_ms,
            )

    # ------------------------------------------------------------------
    # Główna pętla wątku
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Pętla wątku tła — przetwarza klatki aż do stop()."""
        from . import camera, calibration

        alpha = 0.1
        avg_ms: float = 0.0

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            try:
                frame = camera.fetch_snapshot_fast()
                warped = calibration.apply_warp(frame)
                result = self._detector.process_frame(warped)

                elapsed_ms = (time.monotonic() - t0) * 1000
                avg_ms = alpha * elapsed_ms + (1 - alpha) * avg_ms if avg_ms else elapsed_ms

                with self._status_lock:
                    self._status.last_frame_ts = t0
                    self._status.last_detector_state = result.state
                    self._status.last_reason = result.reason
                    self._status.frames_processed += 1
                    self._status.avg_frame_ms = round(avg_ms, 1)
                    if result.move_detected:
                        self._status.last_move_uci = result.move_uci
                        logger.info(
                            "DetectorWorker: wykryto ruch %s (%s)",
                            result.move_uci,
                            result.reason,
                        )

            except RuntimeError as exc:
                # Oczekiwane błędy: brak kalibracji, kamera niedostępna
                with self._status_lock:
                    self._status.errors += 1
                logger.debug("DetectorWorker: %s", exc)
            except Exception as exc:
                with self._status_lock:
                    self._status.errors += 1
                logger.warning("DetectorWorker wyjątek: %s", exc)

            elapsed_total = time.monotonic() - t0
            sleep_s = max(0.0, self._frame_interval - elapsed_total)
            # Przerywalny sleep — reaguje natychmiast na stop()
            self._stop_event.wait(sleep_s)

        logger.info("DetectorWorker: pętla zakończona.")
