"""
API FastAPI do oceny pozycji szachowej przez silnik UCI (Stockfish).

Endpoint POST /evaluate przyjmuje FEN i parametry analizy; zwraca ocenę w pionkach
lub informację o macie, najlepszy ruch i linię PV. Przed importem python-chess
na Windows ustawiana jest polityka pętli zdarzeń Proactor (wymagana do subprocess).
"""

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

# Na Windows domyślny SelectorEventLoop nie obsługuje subprocess — Stockfish przez UCI tego wymaga.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import chess
import chess.engine
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# --- Aplikacja HTTP ---

app = FastAPI(title="Chess Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Router CV (Computer Vision) ---
# Import po definicji app żeby uniknąć circular import

from cv.router import router as cv_router, on_startup, on_shutdown  # noqa: E402

app.include_router(cv_router)


@app.on_event("startup")
def _startup():
    """Ładuje kalibrację szachownicy i modele ML przy starcie serwera."""
    on_startup()


@app.on_event("shutdown")
def _shutdown():
    """Zatrzymuje wątki backgroundowe (stream kamery, silnik szachowy)."""
    on_shutdown()
    _close_engine()

# --- Ścieżka do Stockfisha: .env (STOCKFISH_PATH) lub domyślna lokalizacja w repozytorium ---

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
_DEFAULT_STOCKFISH = _REPO_ROOT / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", str(_DEFAULT_STOCKFISH))

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A7: Persistent Stockfish engine
# ---------------------------------------------------------------------------
# Tworzenie procesu Stockfisha (popen_uci) zajmuje ~0.5-1.5s. Przy trybie live
# analiza po każdym ruchu oznaczałaby nowy proces za każdym razem.
# Trzymamy jedną instancję na cały czas życia aplikacji, z blokadą wątkową
# (sesja UCI jest stanowa — tylko jedno analyse() naraz).

_engine_lock = threading.Lock()
_engine: chess.engine.SimpleEngine | None = None


def _get_engine() -> chess.engine.SimpleEngine:
    """Zwraca singleton silnika, tworząc go przy pierwszym użyciu."""
    global _engine
    if _engine is None:
        if not os.path.exists(STOCKFISH_PATH):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Stockfish hasn't been found at {STOCKFISH_PATH}. "
                    ),
            )
        _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        _log.info("Stockfish engine has been initialized from: %s", STOCKFISH_PATH)
    return _engine


def _close_engine() -> None:
    """Zamyka silnik — wywoływane przy shutdown aplikacji."""
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception as exc:  # pragma: no cover
            _log.warning("Błąd przy zamykaniu silnika: %s", exc)
        _engine = None


def _configure_engine_for_request(
    engine: chess.engine.SimpleEngine, req: "FENRequest"
) -> None:
    """
    Ustawia opcje UCI zgodnie z trybem siły z żądania.
    Reset do pełnej siły jest kluczowy, bo persistent engine zachowuje
    stan UCI między żądaniami.
    """
    if req.elo_limit is not None:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": req.elo_limit})
    elif req.skill_level is not None:
        engine.configure(
            {"UCI_LimitStrength": False, "Skill Level": req.skill_level}
        )
    else:
        engine.configure({"UCI_LimitStrength": False, "Skill Level": 20})


def _clamp_int(value: int, low: int, high: int) -> int:
    """Ogranicza liczbę całkowitą do zamkniętego przedziału [low, high]."""
    return max(low, min(high, value))


# --- Modele żądania i odpowiedzi (Pydantic) ---


class FENRequest(BaseModel):
    """Wejście analizy: pozycja FEN oraz limity silnika."""

    fen: str
    depth: int = 18
    skill_level: int | None = None  # 0–20, None = pełna siła (gdy brak limitu Elo)
    elo_limit: int | None = None    # 1320–3190, UCI_LimitStrength; pierwszeństwo nad skill_level

    @field_validator("fen")
    @classmethod
    def fen_strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("depth")
    @classmethod
    def depth_bounds(cls, v: int) -> int:
        return _clamp_int(v, 1, 40)

    @field_validator("skill_level")
    @classmethod
    def skill_bounds(cls, v: int | None) -> int | None:
        if v is None:
            return None
        return _clamp_int(v, 0, 20)

    @field_validator("elo_limit")
    @classmethod
    def elo_bounds(cls, v: int | None) -> int | None:
        if v is None:
            return None
        return _clamp_int(v, 1320, 3190)


class EvalResponse(BaseModel):
    """Wynik analizy: ocena z perspektywy białych, typ, PV, głębokość, strona ruchu."""

    score: float  # pionki: dodatnie = przewaga białych
    score_type: str  # "cp" lub "mate"
    mate_in: int | None
    best_move: str | None
    pv: list[str] # principal variation — pełna linia ruchów UC
    depth: int
    turn: str  # "white" | "black" — czyja kolej na szachownicy
    is_valid: bool


# --- Pomocnicze: PV, strona ruchu, mapowanie wyniku silnika na odpowiedź API ---


def _extract_pv(info: chess.engine.InfoDict) -> list[str]:
    """Zwraca principal variation jako listę ruchów w notacji UCI."""
    raw = info.get("pv")
    if not raw:
        return []
    return [m.uci() if isinstance(m, chess.Move) else str(m) for m in raw]


def _turn_label(board: chess.Board) -> str:
    """Etykieta strony mającej ruch (dla pola turn w JSON)."""
    return "white" if board.turn == chess.WHITE else "black"


def _eval_response_from_engine(
    board: chess.Board,
    info: chess.engine.InfoDict,
    requested_depth: int,
) -> EvalResponse:
    """
    Buduje EvalResponse z obiektu info zwróconego przez engine.analyse().
    Perspektywa oceny: zawsze białe (score_obj.white()).
    """
    pov = info.get("score")
    if pov is None:
        raise HTTPException(
            status_code=500,
            detail="Stockfish nie zwrócił oceny (brak pola score).",
        )

    score_obj = pov.white()
    pv_line = _extract_pv(info)
    best_move = pv_line[0] if pv_line else None
    actual_depth = info.get("depth", requested_depth)
    turn = _turn_label(board)

    if score_obj.is_mate():
        mate_val = score_obj.mate()
        # W trybie mata zwracamy stałą „skalę” ±100; szczegół w mate_in
        score_display = 100.0 if mate_val > 0 else -100.0
        return EvalResponse(
            score=score_display,
            score_type="mate",
            mate_in=mate_val,
            best_move=best_move,
            pv=pv_line,
            depth=actual_depth,
            turn=turn,
            is_valid=True,
        )

    cp = score_obj.score()
    if cp is None:
        cp = score_obj.score(mate_score=32000) or 0
    score_pawns = round(cp / 100, 2)

    return EvalResponse(
        score=score_pawns,
        score_type="cp",
        mate_in=None,
        best_move=best_move,
        pv=pv_line,
        depth=actual_depth,
        turn=turn,
        is_valid=True,
    )


# --- Endpointy ---


@app.get("/health")
def health():
    """Prosty ping + informacja, skąd ładowany jest Stockfish."""
    return {"status": "ok", "stockfish": STOCKFISH_PATH}


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: FENRequest):
    """
    Uruchamia analizę pozycji używając persistent engine (A7).
    Blokada zapewnia, że tylko jedno żądanie analizy leci do silnika naraz.
    """
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowy FEN: {e}")

    with _engine_lock:
        try:
            engine = _get_engine()
            _configure_engine_for_request(engine, req)
            info = engine.analyse(board, chess.engine.Limit(depth=req.depth))
        except chess.engine.EngineTerminatedError:
            # Proces silnika padł — zrestartuj i spróbuj ponownie raz
            _log.warning("Stockfish padł — restart i retry.")
            _close_engine()
            try:
                engine = _get_engine()
                _configure_engine_for_request(engine, req)
                info = engine.analyse(board, chess.engine.Limit(depth=req.depth))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Błąd Stockfisha po restarcie ({type(e).__name__}): {e}",
                )
        except HTTPException:
            raise
        except Exception as e:
            msg = str(e).strip() or repr(e)
            raise HTTPException(
                status_code=500,
                detail=f"Błąd Stockfisha ({type(e).__name__}): {msg}",
            )

    return _eval_response_from_engine(board, info, req.depth)
