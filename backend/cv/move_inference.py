"""
Wnioskowanie ruchu UCI z delty masek zajętości.

Wejście:  before: set[str], after: set[str], board: chess.Board
Wyjście:  (move_uci | None, reason: str)

Obsługiwane wzorce:
  Standardowy ruch / bicie    → 1 zniknięcie + 1 pojawienie
  Bicie na pole zajęte        → 1 zniknięcie + 0 pojawień (cel był zajęty przed i po)
  Roszada                     → 2 zniknięcia + 2 pojawienia (wzorce e1/e8)
  En passant                  → 2 zniknięcia + 1 pojawienie
  Promocja                    → jak standardowy, z automatycznym hetmanem
"""

import logging
from typing import Optional

import chess
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PROMOTION = chess.QUEEN


def infer_move(
    before: set[str],
    after: set[str],
    board: chess.Board,
    before_image: Optional[np.ndarray] = None,
    after_image: Optional[np.ndarray] = None,
) -> tuple[str | None, str]:
    """
    Wyznacza ruch UCI z porównania masek zajętości.

    Zwraca (move_uci, reason). move_uci=None gdy brak jednoznacznego kandydata.
    """
    disappeared = before - after
    appeared = after - before

    logger.debug("Delta: zniknęło=%s pojawiło=%s", disappeared, appeared)

    if not disappeared and not appeared:
        return None, "Brak zmian między klatkami."

    # Roszada (2+2)
    if len(disappeared) == 2 and len(appeared) == 2:
        uci = _try_castling(disappeared, appeared, board)
        if uci:
            return uci, f"Roszada: {uci}"

    # En passant (2 znikają, 1 pojawia)
    if len(disappeared) == 2 and len(appeared) == 1:
        uci = _try_en_passant(disappeared, appeared, board)
        if uci:
            return uci, f"En passant: {uci}"

    # Standardowy ruch / bicie (1+1)
    if len(disappeared) == 1 and len(appeared) == 1:
        src = next(iter(disappeared))
        dst = next(iter(appeared))
        return _validate_simple(src, dst, board)

    # Bicie na pole, które było już zajęte (1 znika, 0 pojawia)
    # CNN widzi cel jako "nadal zajęte" — delta nie pokazuje pojawienia się nowego pola.
    if len(disappeared) == 1 and len(appeared) == 0:
        uci = _try_capture_occupied_dst(
            disappeared, before, after, board, before_image, after_image
        )
        if uci:
            return uci, f"Bicie (cel zajęty): {uci}"

    return None, (
        f"Nierozpoznany wzorzec: zniknęło={sorted(disappeared)}, "
        f"pojawiło={sorted(appeared)}. Prawdopodobnie zakłócenie lub szum."
    )


def _validate_simple(src: str, dst: str, board: chess.Board) -> tuple[str | None, str]:
    src_sq = chess.parse_square(src)
    dst_sq = chess.parse_square(dst)
    piece = board.piece_at(src_sq)

    is_promotion = (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(dst_sq) in (0, 7)
    )

    move = chess.Move(src_sq, dst_sq, promotion=DEFAULT_PROMOTION if is_promotion else None)

    if move in board.legal_moves:
        return move.uci(), f"Ruch: {src}->{dst}"

    return None, f"Ruch {src}->{dst} ({move.uci()}) jest nielegalny. FEN: {board.fen()}"


def _try_castling(
    disappeared: set[str],
    appeared: set[str],
    board: chess.Board,
) -> str | None:
    patterns = {
        (frozenset({"e1", "h1"}), frozenset({"g1", "f1"})): "e1g1",
        (frozenset({"e1", "a1"}), frozenset({"c1", "d1"})): "e1c1",
        (frozenset({"e8", "h8"}), frozenset({"g8", "f8"})): "e8g8",
        (frozenset({"e8", "a8"}), frozenset({"c8", "d8"})): "e8c8",
    }
    uci = patterns.get((frozenset(disappeared), frozenset(appeared)))
    if uci is None:
        return None
    move = chess.Move.from_uci(uci)
    return uci if move in board.legal_moves else None


def _try_en_passant(
    disappeared: set[str],
    appeared: set[str],
    board: chess.Board,
) -> str | None:
    dst_name = next(iter(appeared))
    dst_sq = chess.parse_square(dst_name)
    for src_name in disappeared:
        src_sq = chess.parse_square(src_name)
        move = chess.Move(src_sq, dst_sq)
        piece = board.piece_at(src_sq)
        if move in board.legal_moves and piece and piece.piece_type == chess.PAWN:
            return move.uci()
    return None


def _try_capture_occupied_dst(
    disappeared: set[str],
    before: set[str],
    after: set[str],
    board: chess.Board,
    before_image: Optional[np.ndarray] = None,
    after_image: Optional[np.ndarray] = None,
) -> str | None:
    """
    Obsługuje bicie, gdy pole docelowe było zajęte zarówno przed jak i po ruchu.
    CNN raportuje cel jako 'nadal zajęty' — znikneło tylko pole źródłowe.

    Gdy bicia legalnego są dwa lub więcej (np. hetman może bić na c6 lub f7),
    używamy różnicy pikselowej między before_image a after_image, żeby wybrać
    pole które WIZUALNIE się zmieniło (bita figura → bijąca figura zmienia kolor).
    """
    src = next(iter(disappeared))
    src_sq = chess.parse_square(src)
    piece = board.piece_at(src_sq)

    # Kandydaci: pola zajęte w obu stanach (były zajęte i nadal są)
    candidates = (before & after) - {src}

    # Filtrujemy tylko legalne bicia
    legal_captures: list[chess.Move] = []
    for dst_name in candidates:
        dst_sq = chess.parse_square(dst_name)
        is_promotion = (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(dst_sq) in (0, 7)
        )
        move = chess.Move(
            src_sq, dst_sq, promotion=DEFAULT_PROMOTION if is_promotion else None
        )
        if move in board.legal_moves:
            legal_captures.append(move)

    if not legal_captures:
        logger.debug("Brak legalnych bić z %s na pola: %s", src, sorted(candidates))
        return None

    logger.info("Znaleziono %d legalnych bić z %s: %s", len(legal_captures), src, [m.uci() for m in legal_captures])

    # Jednoznaczny wynik — nie potrzeba pixel-diff
    if len(legal_captures) == 1:
        m = legal_captures[0]
        logger.info("Bicie (jednoznaczne): %s", m.uci())
        return m.uci()

    # Niejednoznaczność — kilka legalnych bić na "ciągle zajęte" pola.
    # Używamy różnicy pikselowej: pole, na którym figura się ZMIENIŁA,
    # będzie miało wyższy MAE niż pole z nienaruszonym pionkiem.
    if before_image is not None and after_image is not None:
        best_move = _disambiguate_by_pixel_diff(
            legal_captures, before_image, after_image
        )
        if best_move is not None:
            logger.info(
                "Bicie (pixel-diff disambiguacja spośród %d kandydatów): %s",
                len(legal_captures),
                best_move.uci(),
            )
            return best_move.uci()
        else:
            logger.warning(
                "Pixel-diff nie wybrał kandydata z %d bić: %s. Fallback do pierwszego.",
                len(legal_captures),
                [m.uci() for m in legal_captures],
            )

    # Fallback gdy brak obrazów — zwróć pierwsze legalne bicie (stare zachowanie)
    m = legal_captures[0]
    logger.warning(
        "Brak obrazów do disambiguacji; zwracam pierwsze legalne bicie: %s", m.uci()
    )
    return m.uci()


def _square_roi(sq_name: str, image: np.ndarray) -> np.ndarray:
    """Wycina ROI pola z obrazu BOARD_SIZE_PX × BOARD_SIZE_PX."""
    from .config import BOARD_SIZE_PX, CELL_MARGIN_PX

    cell_px = BOARD_SIZE_PX // 8
    sq = chess.parse_square(sq_name)
    col = chess.square_file(sq)          # 0=a … 7=h
    row = 7 - chess.square_rank(sq)      # rank 8 → row 0, rank 1 → row 7

    x1 = col * cell_px + CELL_MARGIN_PX
    y1 = row * cell_px + CELL_MARGIN_PX
    x2 = (col + 1) * cell_px - CELL_MARGIN_PX
    y2 = (row + 1) * cell_px - CELL_MARGIN_PX
    return image[y1:y2, x1:x2]


def _disambiguate_by_pixel_diff(
    candidates: list[chess.Move],
    before_image: np.ndarray,
    after_image: np.ndarray,
) -> chess.Move | None:
    """
    Zwraca ruch, którego pole docelowe ma największą średnią różnicę pikselową
    między before_image a after_image.

    Pole gdzie figura ZMIENIŁA SIĘ (bicie) wykaże dużo wyższe MAE niż pole
    które pozostało niezmienione (ten sam pionek siedzi spokojnie).
    """
    best_move: chess.Move | None = None
    best_diff: float = -1.0
    min_threshold = 3.0  # minimalna różnica pikselowa (0-255 skala) - obniżono z 5.0

    logger.info(
        "Pixel-diff disambiguacja %d kandydatów: %s na obrazie %s",
        len(candidates),
        [m.uci() for m in candidates],
        before_image.shape,
    )

    for move in candidates:
        dst_name = chess.square_name(move.to_square)
        try:
            roi_before = _square_roi(dst_name, before_image).astype(np.float32)
            roi_after = _square_roi(dst_name, after_image).astype(np.float32)
            if roi_before.size == 0 or roi_after.size == 0:
                continue
            diff = float(np.mean(np.abs(roi_after - roi_before)))
        except Exception as exc:
            logger.debug("Pixel-diff błąd dla %s: %s", dst_name, exc)
            continue

        logger.info("Pixel-diff %s: %.2f", dst_name, diff)
        if diff > best_diff:
            best_diff = diff
            best_move = move

    # Zwróć tylko gdy różnica jest wystarczająco duża
    if best_diff >= min_threshold:
        logger.info("Wybrano %s z diff=%.2f (próg: %.1f)", best_move.uci() if best_move else None, best_diff, min_threshold)
        return best_move
    else:
        logger.warning("Wszystkie diff <%.1f, brak wyraźnego zwycięzcy. Najwyższy: %.2f", min_threshold, best_diff)
        return None
