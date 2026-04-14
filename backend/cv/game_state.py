"""
Stan gry szachowej — singleton python-chess.Board.

Jedyne źródło prawdy o pozycji. Thread-safe (Lock).
Integruje się z move_detector (push ruchów) i /evaluate (FEN → Stockfish).
"""

import logging
import threading

import chess

logger = logging.getLogger(__name__)

STARTING_FEN = chess.STARTING_FEN

_lock = threading.Lock()
_board: chess.Board = chess.Board()
_history: list[str] = []


def reset(fen: str = STARTING_FEN) -> None:
    """Resetuje planszę do podanego FEN i czyści historię."""
    global _board, _history
    with _lock:
        _board = chess.Board(fen)
        _history = []
    logger.info("Stan gry zresetowany. FEN: %s", fen)


def push(move_uci: str) -> chess.Move:
    """
    Wykonuje legalny ruch UCI na planszy.
    Rzuca ValueError gdy ruch niepoprawny lub nielegalny.
    """
    global _history
    with _lock:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError as exc:
            raise ValueError(f"Niepoprawne UCI '{move_uci}': {exc}") from exc

        if move not in _board.legal_moves:
            sample = [m.uci() for m in list(_board.legal_moves)[:8]]
            raise ValueError(
                f"Ruch {move_uci} jest nielegalny. Przykłady legalnych: {sample}"
            )

        _board.push(move)
        _history.append(move_uci)
        logger.info("Ruch: %s | FEN: %s", move_uci, _board.fen())
        return move


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
