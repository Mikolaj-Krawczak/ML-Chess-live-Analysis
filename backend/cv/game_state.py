"""
Stan gry szachowej — singleton python-chess.Board.

Jedyne źródło prawdy o pozycji. Thread-safe (Lock).
Integruje się z move_detector (push ruchów) i /evaluate (FEN → Stockfish).
"""

import logging
import threading
import json
from pathlib import Path

import chess

logger = logging.getLogger(__name__)

STARTING_FEN = chess.STARTING_FEN

_lock = threading.RLock()
_board: chess.Board = chess.Board()
_history: list[str] = []
_STATE_PATH = Path(__file__).parent / "game_state_data.json"


def _save_state_to_disk() -> None:
    """Zapisuje aktualny stan gry na dysk (FEN + historia)."""
    with _lock:
        payload = {
            "fen": _board.fen(),
            "history": list(_history),
        }
    _STATE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_state_from_disk() -> None:
    """Ładuje ostatni stan gry z dysku, jeśli plik istnieje i jest poprawny."""
    global _board, _history
    if not _STATE_PATH.exists():
        return

    try:
        payload = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        fen = payload.get("fen")
        history = payload.get("history", [])
        if not isinstance(fen, str):
            raise ValueError("Brak poprawnego pola fen.")
        if not isinstance(history, list):
            raise ValueError("Brak poprawnego pola history.")

        board = chess.Board(fen)
        with _lock:
            _board = board
            _history = [str(m) for m in history]
        logger.info("Zaladowano stan gry z pliku: %s", _STATE_PATH)
    except Exception as exc:
        logger.warning("Nie udalo sie zaladowac stanu gry: %s", exc)


def _castling_diagnostic(board: chess.Board, move: chess.Move) -> str:
    """
    Zwraca szczegółową przyczynę odrzucenia roszady.
    """
    piece = board.piece_at(move.from_square)
    side = "white" if board.turn == chess.WHITE else "black"
    is_kingside = chess.square_file(move.to_square) == 6

    if piece is None or piece.piece_type != chess.KING:
        return "Roszada wymaga ruchu królem z pola startowego."

    if side == "white":
        if move.from_square != chess.E1:
            return "Biały król musi stać na e1."
        rook_sq = chess.H1 if is_kingside else chess.A1
        empty_squares = [chess.F1, chess.G1] if is_kingside else [chess.D1, chess.C1, chess.B1]
        king_path = [chess.E1, chess.F1, chess.G1] if is_kingside else [chess.E1, chess.D1, chess.C1]
        right_label = "K" if is_kingside else "Q"
    else:
        if move.from_square != chess.E8:
            return "Czarny król musi stać na e8."
        rook_sq = chess.H8 if is_kingside else chess.A8
        empty_squares = [chess.F8, chess.G8] if is_kingside else [chess.D8, chess.C8, chess.B8]
        king_path = [chess.E8, chess.F8, chess.G8] if is_kingside else [chess.E8, chess.D8, chess.C8]
        right_label = "k" if is_kingside else "q"

    rook_piece = board.piece_at(rook_sq)
    if rook_piece is None or rook_piece.piece_type != chess.ROOK or rook_piece.color != board.turn:
        return f"Brakuje właściwej wieży na {chess.square_name(rook_sq)}."

    if not board.has_castling_rights(board.turn):
        return "Brak praw roszady dla tej strony."
    if right_label not in board.castling_xfen():
        return f"Brak prawa roszady {'krótkiej' if is_kingside else 'długiej'} ({right_label})."

    blocked = [sq for sq in empty_squares if board.piece_at(sq) is not None]
    if blocked:
        blocked_names = ", ".join(chess.square_name(sq) for sq in blocked)
        return f"Pola między królem i wieżą nie są puste: {blocked_names}."

    for sq in king_path:
        if board.is_attacked_by(not board.turn, sq):
            return f"Król przechodzi przez pole atakowane: {chess.square_name(sq)}."

    return "Roszada nie spełnia warunków pozycyjnych."


def _normalize_auto_promotion_uci(board: chess.Board, move_uci: str) -> str:
    """
    Auto-promocja do hetmana:
    jeśli ruch ma format e7e8 / e2e1 i jest ruchem piona na ostatnią linię,
    dopisujemy sufiks "q".
    """
    if len(move_uci) != 4:
        return move_uci

    try:
        from_sq = chess.parse_square(move_uci[:2])
        to_sq = chess.parse_square(move_uci[2:4])
    except ValueError:
        return move_uci

    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.PAWN:
        return move_uci

    to_rank = chess.square_rank(to_sq)
    if to_rank in (0, 7):
        return f"{move_uci}q"
    return move_uci


def reset(fen: str = STARTING_FEN) -> None:
    """Resetuje planszę do podanego FEN i czyści historię."""
    global _board, _history
    with _lock:
        _board = chess.Board(fen)
        _history = []
    _save_state_to_disk()
    logger.info("Stan gry zresetowany. FEN: %s", fen)


def push(move_uci: str) -> chess.Move:
    """
    Wykonuje legalny ruch UCI na planszy.
    Rzuca ValueError gdy ruch niepoprawny lub nielegalny.
    """
    global _history
    with _lock:
        move_uci_normalized = _normalize_auto_promotion_uci(_board, move_uci)

        try:
            move = chess.Move.from_uci(move_uci_normalized)
        except ValueError as exc:
            raise ValueError(f"Niepoprawne UCI '{move_uci}': {exc}") from exc

        if move not in _board.legal_moves:
            sample = [m.uci() for m in list(_board.legal_moves)[:8]]
            is_castling_attempt = (
                _board.piece_at(move.from_square) is not None
                and _board.piece_at(move.from_square).piece_type == chess.KING
                and abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) == 2
            )
            if is_castling_attempt:
                reason = _castling_diagnostic(_board, move)
                raise ValueError(
                    f"Ruch {move_uci_normalized} jest nielegalny. Powód roszady: {reason}"
                )
            raise ValueError(
                f"Ruch {move_uci_normalized} jest nielegalny. Przykłady legalnych: {sample}"
            )

        _board.push(move)
        _history.append(move_uci_normalized)
        _save_state_to_disk()
        logger.info("Ruch: %s | FEN: %s", move_uci_normalized, _board.fen())
        return move


def undo_last_move() -> str:
    """
    Cofa ostatni ruch i zwraca UCI cofniętego ruchu.
    """
    global _history
    with _lock:
        if not _board.move_stack:
            raise ValueError("Brak ruchów do cofnięcia.")

        undone = _board.pop().uci()
        if _history:
            _history.pop()
        _save_state_to_disk()
        logger.info("Cofnieto ruch: %s | FEN: %s", undone, _board.fen())
        return undone


def get_fen() -> str:
    with _lock:
        return _board.fen()


def get_board_copy() -> chess.Board:
    with _lock:
        return _board.copy()


def get_history() -> list[str]:
    with _lock:
        return list(_history)


def get_turn() -> str:
    with _lock:
        return "white" if _board.turn == chess.WHITE else "black"


def is_game_over() -> bool:
    with _lock:
        return _board.is_game_over()


def get_legal_moves_uci() -> list[str]:
    with _lock:
        return [m.uci() for m in _board.legal_moves]


def get_status() -> dict:
    """Pełny status — używany przez GET /cv/game/state."""
    with _lock:
        b = _board.copy()
        h = list(_history)
    return {
        "fen": b.fen(),
        "turn": "white" if b.turn == chess.WHITE else "black",
        "move_number": b.fullmove_number,
        "halfmove_clock": b.halfmove_clock,
        "is_check": b.is_check(),
        "is_checkmate": b.is_checkmate(),
        "is_stalemate": b.is_stalemate(),
        "is_game_over": b.is_game_over(),
        "history": h,
        "history_length": len(h),
    }


_load_state_from_disk()
