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

import chess

logger = logging.getLogger(__name__)

DEFAULT_PROMOTION = chess.QUEEN


def infer_move(
    before: set[str],
    after: set[str],
    board: chess.Board,
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
        uci = _try_capture_occupied_dst(disappeared, before, after, board)
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
) -> str | None:
    """
    Obsługuje bicie, gdy pole docelowe było zajęte zarówno przed jak i po ruchu.
    CNN raportuje cel jako 'nadal zajęty' — znikneło tylko pole źródłowe.
    Szukamy legalnego bicia spośród pól wspólnych dla before i after.
    """
    src = next(iter(disappeared))
    src_sq = chess.parse_square(src)
    piece = board.piece_at(src_sq)

    # Kandydaci: pola zajęte w obu stanach (były zajęte i nadal są)
    candidates = (before & after) - {src}

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
            logger.debug("Bicie na zajęte pole: %s->%s", src, dst_name)
            return move.uci()

    return None
