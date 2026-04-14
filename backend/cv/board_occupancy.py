"""
Detekcja zajętości 64 pól szachownicy.

Dwie strategie (automatyczny wybór):
  CNN  — square_classifier.py (PyTorch), ładowany gdy dostępne wagi
  Fallback — wariancja pikseli po preprocessingu (bez modelu ML)

Preprocessing wspólny dla obu strategii:
  BGR → grayscale → GaussianBlur(5×5)
  ROI z marginesem 15px (eliminuje krawędzie/cienie sąsiadów)

Mapowanie indeksów:
  wiersz 0 (góra) = rząd 8  →  a8..h8
  wiersz 7 (dół)  = rząd 1  →  a1..h1
  kolumna 0 (lewo) = linia a
  kolumna 7 (prawa) = linia h
"""

import logging
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from .config import (
    BOARD_SIZE_PX,
    CELL_MARGIN_PX,
    CELL_SIZE_PX,
    OCCUPANCY_VARIANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class CellAnalysis:
    """Wynik analizy pojedynczego pola."""

    row: int
    col: int
    square_name: str
    chess_square: int
    score: float          # wariancja lub p(occupied) z CNN
    occupied: bool
    method: str           # "variance" lub "cnn"
    threshold: float


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _preprocess(warped: np.ndarray) -> np.ndarray:
    """BGR → grayscale → GaussianBlur(5×5)."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _extract_cell(proc: np.ndarray, row: int, col: int) -> np.ndarray:
    """Wycina ROI komórki z marginesem wewnętrznym."""
    x1 = col * CELL_SIZE_PX + CELL_MARGIN_PX
    y1 = row * CELL_SIZE_PX + CELL_MARGIN_PX
    x2 = (col + 1) * CELL_SIZE_PX - CELL_MARGIN_PX
    y2 = (row + 1) * CELL_SIZE_PX - CELL_MARGIN_PX
    return proc[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Analiza planszy
# ---------------------------------------------------------------------------


def analyze_board(warped: np.ndarray) -> list[CellAnalysis]:
    """
    Analizuje wyprostowany obraz planszy (800×800px).

    Próbuje użyć modelu CNN — jeśli niedostępny, używa wariancji.
    Zwraca listę 64 CellAnalysis w kolejności row-major (a8→h1).
    """
    if warped.shape[:2] != (BOARD_SIZE_PX, BOARD_SIZE_PX):
        raise ValueError(
            f"Oczekiwano {BOARD_SIZE_PX}x{BOARD_SIZE_PX}px, "
            f"otrzymano {warped.shape[1]}x{warped.shape[0]}px."
        )

    proc = _preprocess(warped)

    # Spróbuj CNN
    try:
        from .ml.square_classifier import classify_cells
        patches = [_extract_cell(proc, r, c) for r in range(8) for c in range(8)]
        scores = classify_cells(patches)
        threshold = 0.5
        method = "cnn"
    except (ImportError, RuntimeError):
        scores = [float(np.var(_extract_cell(proc, r, c))) for r in range(8) for c in range(8)]
        threshold = OCCUPANCY_VARIANCE_THRESHOLD
        method = "variance"

    results: list[CellAnalysis] = []
    for idx, (row, col) in enumerate((r, c) for r in range(8) for c in range(8)):
        sq = chess.square(col, 7 - row)
        results.append(CellAnalysis(
            row=row,
            col=col,
            square_name=chess.square_name(sq),
            chess_square=sq,
            score=round(scores[idx], 4),
            occupied=scores[idx] > threshold,
            method=method,
            threshold=threshold,
        ))

    occupied_n = sum(1 for c in results if c.occupied)
    logger.debug("%d/64 pól zajętych (metoda: %s, próg: %.3f)", occupied_n, method, threshold)
    return results


def get_occupied_squares(warped: np.ndarray) -> set[str]:
    """Zwraca zbiór nazw zajętych pól (np. {'e2', 'e4'})."""
    return {c.square_name for c in analyze_board(warped) if c.occupied}


def occupancy_mask(warped: np.ndarray) -> list[bool]:
    """Płaska lista 64 bool (indeks = chess.square, a1=0..h8=63)."""
    analysis = analyze_board(warped)
    mask = [False] * 64
    for cell in analysis:
        mask[cell.chess_square] = cell.occupied
    return mask


# ---------------------------------------------------------------------------
# Wizualizacja debug
# ---------------------------------------------------------------------------


def draw_debug_grid(warped: np.ndarray, analysis: list[CellAnalysis]) -> np.ndarray:
    """
    Rysuje siatkę 8×8 na wyprostowanym obrazie.

    Zielony = zajęte, czerwony = puste. Wyświetla score i nazwę pola.
    """
    overlay = warped.copy()
    output = warped.copy()

    threshold = analysis[0].threshold if analysis else OCCUPANCY_VARIANCE_THRESHOLD
    method = analysis[0].method if analysis else "?"

    for cell in analysis:
        x1, y1 = cell.col * CELL_SIZE_PX, cell.row * CELL_SIZE_PX
        x2, y2 = x1 + CELL_SIZE_PX, y1 + CELL_SIZE_PX
        color = (0, 180, 0) if cell.occupied else (0, 0, 180)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(output, (x1, y1), (x2, y2),
                      (0, 255, 0) if cell.occupied else (0, 0, 255), 2)

    cv2.addWeighted(overlay, 0.28, output, 0.72, 0, output)

    for cell in analysis:
        cx = cell.col * CELL_SIZE_PX + CELL_SIZE_PX // 2
        cy = cell.row * CELL_SIZE_PX + CELL_SIZE_PX // 2
        cv2.putText(output, cell.square_name, (cx - 16, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"{cell.score:.2f}" if method == "cnn" else f"{cell.score:.0f}"
        cv2.putText(output, label, (cx - 16, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(output, f"method={method} thr={threshold:.2f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return output
