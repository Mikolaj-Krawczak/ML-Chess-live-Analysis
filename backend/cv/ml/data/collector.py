"""
Auto-labeled dataset collector dla Square Classifier.

KLUCZOWY MODUŁ — zamiast ręcznego labelowania 1000+ zdjęć, korzystamy z faktu
że podczas gry wiemy DOKŁADNIE które pola są zajęte (python-chess Board).

JAK TO DZIAŁA:
--------------
1. Gracz ustawia pozycję lub rozgrywa partię.
2. Po każdym zatwierdzonym ruchu (move_detector) wywołujemy collect_from_frame().
3. Funkcja pobiera aktualny FEN z game_state i wie który square jest zajęty.
4. Warpy klatkę, ekstrahuje 64 patche 70×70px.
5. Zapisuje każdy patch do odpowiedniego folderu: dataset/occupied/ lub dataset/empty/.
6. Opcjonalnie augmentuje (×4) zwiększając dataset bez dodatkowych zdjęć.

EFEKT:
------
20 pozycji × 64 pola × (1 oryginał + 4 augmentacje) = 6400 próbek
~3200 occupied + ~3200 empty → więcej niż wystarczy do treningu CNN.

STRUKTURA DATASETU:
-------------------
backend/cv/ml/data/dataset/
├── occupied/
│   ├── pos001_a1_orig.jpg
│   ├── pos001_a1_aug0.jpg
│   ├── pos001_a1_aug1.jpg
│   ...
└── empty/
    ├── pos001_a3_orig.jpg
    ...

UŻYCIE:
-------
# Przez API (podczas gry):
POST /cv/ml/collect         → zapisuje 64 patche z aktualnej klatki

# Bezpośrednio z kodu (skrypt CLI):
from cv.ml.data.collector import collect_from_frame
n_occ, n_emp, fen = collect_from_frame(warped_frame, board.fen())
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import chess
import cv2
import numpy as np

from ...config import BOARD_SIZE_PX, CELL_SIZE_PX, CELL_MARGIN_PX, DATASET_DIR
from .augment import augment_patch, N_AUGMENTS

logger = logging.getLogger(__name__)

# Rozmiar patcha zapisywanego do datasetu (musi pasować do wejścia CNN)
PATCH_SIZE = 70

# Czy augmentować podczas zbierania (True = dataset ×(1+N_AUGMENTS))
AUGMENT_ON_COLLECT = True


# ---------------------------------------------------------------------------
# Publiczne API
# ---------------------------------------------------------------------------


def collect_from_frame(
    warped: np.ndarray,
    fen: str,
    augment: bool = AUGMENT_ON_COLLECT,
) -> tuple[int, int, str]:
    """
    Ekstrahuje 64 patche z wyprostowanego obrazu i zapisuje do datasetu.

    Na podstawie FEN wiemy dokładnie które pola są zajęte — zero ręcznego labelowania.

    Parametry:
        warped  — obraz BGR 800×800px po warpPerspective
        fen     — aktualny FEN pozycji (z game_state.get_fen())
        augment — czy generować augmentowane warianty

    Zwraca:
        (n_occupied_saved, n_empty_saved, fen_used)
    """
    if warped.shape[:2] != (BOARD_SIZE_PX, BOARD_SIZE_PX):
        raise ValueError(
            f"Oczekiwano {BOARD_SIZE_PX}×{BOARD_SIZE_PX}px, "
            f"otrzymano {warped.shape[1]}×{warped.shape[0]}px."
        )

    # Parsujemy FEN żeby wiedzieć które pola są zajęte
    occupied_squares = _fen_to_occupied_set(fen)

    # Unikalny prefix dla tej sesji zbierania (timestamp)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

    # Preprocessing: grayscale + blur (taki sam jak w board_occupancy.py)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(gray, (5, 5), 0)

    occ_dir, emp_dir = _ensure_dirs()
    n_occ = 0
    n_emp = 0

    for row in range(8):
        for col in range(8):
            # Wyznaczamy nazwę pola (np. "e4")
            sq = chess.square(col, 7 - row)
            sq_name = chess.square_name(sq)

            # Wycinamy patch z marginesem — taki sam sposób jak w board_occupancy.py
            x1 = col * CELL_SIZE_PX + CELL_MARGIN_PX
            y1 = row * CELL_SIZE_PX + CELL_MARGIN_PX
            x2 = (col + 1) * CELL_SIZE_PX - CELL_MARGIN_PX
            y2 = (row + 1) * CELL_SIZE_PX - CELL_MARGIN_PX
            patch = proc[y1:y2, x1:x2]

            # Skalujemy do PATCH_SIZE × PATCH_SIZE
            patch_resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))

            is_occupied = sq_name in occupied_squares
            target_dir = occ_dir if is_occupied else emp_dir
            prefix = f"{ts}_{sq_name}"

            # Zapis oryginału
            _save_patch(patch_resized, target_dir, f"{prefix}_orig")

            # Augmentowane warianty
            if augment:
                for i, aug_patch in enumerate(augment_patch(patch_resized)):
                    _save_patch(aug_patch, target_dir, f"{prefix}_aug{i}")

            if is_occupied:
                n_occ += 1 + (N_AUGMENTS if augment else 0)
            else:
                n_emp += 1 + (N_AUGMENTS if augment else 0)

    logger.info(
        "Zebrano: %d occupied, %d empty (FEN: %s...)",
        n_occ, n_emp, fen[:30],
    )
    return n_occ, n_emp, fen


def collect_from_frame_with_batch(
    warped: np.ndarray,
    fen: str,
    augment: bool = AUGMENT_ON_COLLECT,
) -> tuple[int, int, str, str]:
    """
    Jak collect_from_frame(), ale dodatkowo zwraca batch_id (prefix timestamp).
    """
    if warped.shape[:2] != (BOARD_SIZE_PX, BOARD_SIZE_PX):
        raise ValueError(
            f"Oczekiwano {BOARD_SIZE_PX}×{BOARD_SIZE_PX}px, "
            f"otrzymano {warped.shape[1]}×{warped.shape[0]}px."
        )

    occupied_squares = _fen_to_occupied_set(fen)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(gray, (5, 5), 0)

    occ_dir, emp_dir = _ensure_dirs()
    n_occ = 0
    n_emp = 0

    for row in range(8):
        for col in range(8):
            sq = chess.square(col, 7 - row)
            sq_name = chess.square_name(sq)

            x1 = col * CELL_SIZE_PX + CELL_MARGIN_PX
            y1 = row * CELL_SIZE_PX + CELL_MARGIN_PX
            x2 = (col + 1) * CELL_SIZE_PX - CELL_MARGIN_PX
            y2 = (row + 1) * CELL_SIZE_PX - CELL_MARGIN_PX
            patch = proc[y1:y2, x1:x2]
            patch_resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))

            is_occupied = sq_name in occupied_squares
            target_dir = occ_dir if is_occupied else emp_dir
            prefix = f"{ts}_{sq_name}"

            _save_patch(patch_resized, target_dir, f"{prefix}_orig")
            if augment:
                for i, aug_patch in enumerate(augment_patch(patch_resized)):
                    _save_patch(aug_patch, target_dir, f"{prefix}_aug{i}")

            if is_occupied:
                n_occ += 1 + (N_AUGMENTS if augment else 0)
            else:
                n_emp += 1 + (N_AUGMENTS if augment else 0)

    logger.info(
        "Zebrano batch=%s: %d occupied, %d empty (FEN: %s...)",
        ts, n_occ, n_emp, fen[:30],
    )
    return n_occ, n_emp, fen, ts


def delete_batch(batch_id: str) -> dict:
    """
    Usuwa wszystkie pliki datasetu należące do wskazanego batcha.
    """
    if not re.fullmatch(r"\d{8}_\d{6}_\d{3}", batch_id):
        raise ValueError(f"Nieprawidłowy batch_id: {batch_id}")

    occ_dir = DATASET_DIR / "occupied"
    emp_dir = DATASET_DIR / "empty"

    def _delete_matching(directory: Path) -> int:
        if not directory.exists():
            return 0
        deleted = 0
        for path in directory.glob(f"{batch_id}_*.jpg"):
            path.unlink(missing_ok=True)
            deleted += 1
        return deleted

    occ_deleted = _delete_matching(occ_dir)
    emp_deleted = _delete_matching(emp_dir)

    logger.info(
        "Usunieto batch=%s: occupied=%d, empty=%d",
        batch_id, occ_deleted, emp_deleted,
    )
    return {
        "batch_id": batch_id,
        "occupied_deleted": occ_deleted,
        "empty_deleted": emp_deleted,
        "total_deleted": occ_deleted + emp_deleted,
    }


def get_dataset_stats() -> dict:
    """
    Zwraca statystyki zebranego datasetu.

    Używane przez GET /cv/ml/dataset/stats.
    """
    occ_dir = DATASET_DIR / "occupied"
    emp_dir = DATASET_DIR / "empty"

    occ_count = len(list(occ_dir.glob("*.jpg"))) if occ_dir.exists() else 0
    emp_count = len(list(emp_dir.glob("*.jpg"))) if emp_dir.exists() else 0

    return {
        "occupied_count": occ_count,
        "empty_count": emp_count,
        "total": occ_count + emp_count,
        "dataset_dir": str(DATASET_DIR),
    }


# ---------------------------------------------------------------------------
# Helpers prywatne
# ---------------------------------------------------------------------------


def _fen_to_occupied_set(fen: str) -> set[str]:
    """
    Parsuje FEN i zwraca zbiór nazw zajętych pól.

    FEN format: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Używamy python-chess żeby nie duplikować logiki parsowania.
    """
    try:
        board = chess.Board(fen)
        occupied = set()
        for sq in chess.SQUARES:
            if board.piece_at(sq) is not None:
                occupied.add(chess.square_name(sq))
        return occupied
    except ValueError as exc:
        raise ValueError(f"Nieprawidłowy FEN: {fen!r}: {exc}") from exc


def _ensure_dirs() -> tuple[Path, Path]:
    """Tworzy foldery dataset/occupied/ i dataset/empty/ jeśli nie istnieją."""
    occ_dir = DATASET_DIR / "occupied"
    emp_dir = DATASET_DIR / "empty"
    occ_dir.mkdir(parents=True, exist_ok=True)
    emp_dir.mkdir(parents=True, exist_ok=True)
    return occ_dir, emp_dir


def _save_patch(patch: np.ndarray, directory: Path, name: str) -> None:
    """Zapisuje patch jako JPEG do katalogu."""
    path = directory / f"{name}.jpg"
    cv2.imwrite(str(path), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
