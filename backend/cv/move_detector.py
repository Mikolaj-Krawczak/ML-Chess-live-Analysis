"""
Maszyna stanów detekcji ruchu szachowego (Etap 4).

Stany:
  IDLE         — plansza spoczywa; zapisany snapshot "before"
  IN_MOVE      — wykryto zakłócenie (ręka gracza lub figura w powietrzu)
  STABLE_AFTER — maska ustabilizowała się po ruchu; wyznaczamy delta

Stabilizacja: maska occupied musi być identyczna przez OCCUPANCY_STABILITY_FRAMES
kolejnych klatek żeby przejść IN_MOVE → STABLE_AFTER.

Użycie:
    detector = MoveDetector()
    detector.start(warped_frame)
    while True:
        warped = calibration.apply_warp(camera.fetch_snapshot())
        result = detector.process_frame(warped)
        if result.move_detected:
            send_to_stockfish(result.fen_after)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from . import board_occupancy, game_state, move_inference
from .config import OCCUPANCY_STABILITY_FRAMES

logger = logging.getLogger(__name__)


class DetectorState(Enum):
    IDLE = auto()
    IN_MOVE = auto()
    STABLE_AFTER = auto()


@dataclass
class FrameResult:
    state: str
    move_detected: bool = False
    move_uci: Optional[str] = None
    fen_after: Optional[str] = None
    reason: str = ""
    occupied_now: list[str] = field(default_factory=list)


class MoveDetector:
    """Singleton detekcji ruchów — jedna instancja na sesję gry."""

    def __init__(self) -> None:
        self._state = DetectorState.IDLE
        self._before: set[str] = set()
        self._before_image: Optional[np.ndarray] = None  # obraz klatki 'before'
        self._candidate: set[str] = set()
        self._stable_count: int = 0

    def start(self, warped: Optional[np.ndarray] = None) -> None:
        """Inicjalizuje detektor i opcjonalnie ustawia snapshot 'before'."""
        self._state = DetectorState.IDLE
        self._stable_count = 0
        self._candidate = set()
        if warped is not None:
            self._before = board_occupancy.get_occupied_squares(warped)
            self._before_image = warped.copy()
            logger.info("Detektor uruchomiony. Before: %d pól.", len(self._before))
        else:
            self._before = set()
            self._before_image = None

    def reset_to_idle(self, warped: np.ndarray) -> None:
        """Wraca do IDLE i aktualizuje 'before' z bieżącej klatki."""
        self._before = board_occupancy.get_occupied_squares(warped)
        self._before_image = warped.copy()
        self._state = DetectorState.IDLE
        self._stable_count = 0
        self._candidate = set()

    def process_frame(self, warped: np.ndarray) -> FrameResult:
        """Przetwarza jedną klatkę i aktualizuje maszynę stanów."""
        current = board_occupancy.get_occupied_squares(warped)

        if not self._before:
            self._before = current
            self._before_image = warped.copy()
            return FrameResult(state="IDLE", reason="Inicjalizacja before.", occupied_now=list(current))

        if self._state == DetectorState.IDLE:
            return self._handle_idle(current)
        if self._state == DetectorState.IN_MOVE:
            return self._handle_in_move(current, warped)
        if self._state == DetectorState.STABLE_AFTER:
            return self._handle_stable_after(current, warped)

        return FrameResult(state=self._state.name, occupied_now=list(current))

    def _handle_idle(self, current: set[str]) -> FrameResult:
        if current == self._before:
            return FrameResult(state="IDLE", occupied_now=list(current))

        self._state = DetectorState.IN_MOVE
        self._candidate = current
        self._stable_count = 1
        logger.debug("Wykryto zmianę planszy → IN_MOVE.")
        return FrameResult(state="IN_MOVE", reason="Wykryto zmianę.", occupied_now=list(current))

    def _handle_in_move(self, current: set[str], warped: np.ndarray) -> FrameResult:
        if current == self._candidate:
            self._stable_count += 1
        else:
            self._candidate = current
            self._stable_count = 1
            return FrameResult(
                state="IN_MOVE",
                reason=f"Niestabilne ({self._stable_count}/{OCCUPANCY_STABILITY_FRAMES}).",
                occupied_now=list(current),
            )

        if self._stable_count < OCCUPANCY_STABILITY_FRAMES:
            return FrameResult(
                state="IN_MOVE",
                reason=f"Stabilizacja {self._stable_count}/{OCCUPANCY_STABILITY_FRAMES}.",
                occupied_now=list(current),
            )

        # Dodatkowe potwierdzenie stabilnego stanu na kolejnej klatce:
        # chroni przed false-positive gdy chwilowo wykryje się pole pośrednie.
        self._state = DetectorState.STABLE_AFTER
        return FrameResult(
            state="STABLE_AFTER",
            reason="Kandydat stabilny — oczekiwanie na potwierdzenie kolejnej klatki.",
            occupied_now=list(current),
        )

    def _handle_stable_after(self, current: set[str], warped: np.ndarray) -> FrameResult:
        """
        Dodatkowa bramka antyszumowa: finalizuj ruch tylko gdy kolejna klatka
        potwierdza dokładnie ten sam układ pól zajętych.
        """
        if current != self._candidate:
            self._state = DetectorState.IN_MOVE
            self._candidate = current
            self._stable_count = 1
            return FrameResult(
                state="IN_MOVE",
                reason="Potwierdzenie nieudane — wracam do stabilizacji.",
                occupied_now=list(current),
            )
        return self._finalize_move(current, warped)

    def _finalize_move(self, after: set[str], after_image: np.ndarray) -> FrameResult:
        """Plansza ustabilizowała się — inferujemy ruch i aktualizujemy Board."""
        board = game_state.get_board_copy()
        move_uci, reason = move_inference.infer_move(
            self._before, after, board,
            before_image=self._before_image,
            after_image=after_image,
        )

        self._state = DetectorState.IDLE
        self._stable_count = 0

        if move_uci is None:
            # Nie aktualizujemy _before — zachowujemy poprzedni dobry stan.
            # Dzięki temu zakłócenie (cień, szum CNN) nie psuje kolejnych detekcji.
            logger.warning("Nie rozpoznano ruchu: %s — _before bez zmian.", reason)
            return FrameResult(state="IDLE", move_detected=False,
                               reason=reason, occupied_now=list(after))

        try:
            game_state.push(move_uci)
        except ValueError as exc:
            # Ruch UCI wykryty, ale nielegalny — też nie aktualizujemy _before.
            logger.error("Push ruchu %s nie powiódł się: %s", move_uci, exc)
            return FrameResult(state="IDLE", move_detected=False,
                               reason=str(exc), occupied_now=list(after))

        # Tylko przy zatwierdzonym, legalnym ruchu aktualizujemy bazę porównawczą.
        self._before = after
        self._before_image = after_image.copy()

        fen = game_state.get_fen()
        logger.info("Ruch zatwierdzony: %s | FEN: %s", move_uci, fen)
        return FrameResult(
            state="IDLE", move_detected=True,
            move_uci=move_uci, fen_after=fen,
            reason=reason, occupied_now=list(after),
        )

    @property
    def state_name(self) -> str:
        return self._state.name

    @property
    def before_snapshot(self) -> set[str]:
        return set(self._before)
