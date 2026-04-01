import asyncio
import os
import sys
from pathlib import Path

# Na Windows SelectorEventLoop nie implementuje subprocess_exec (używane przez python-chess).
# Uvicorn / inne biblioteki mogą ustawić WindowsSelectorEventLoopPolicy → NotImplementedError przy Stockfish.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import chess
import chess.engine
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

app = FastAPI(title="Chess Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Katalog główny repozytorium (ML-Chess/) — domyślna lokalizacja binarki z oficjalnej paczki Windows x64 AVX2
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_STOCKFISH_EXE = _REPO_ROOT / "stockfish" / "stockfish-windows-x86-64-avx2.exe"

# Nadpisanie: .env → STOCKFISH_PATH=... albo set STOCKFISH_PATH=... (PowerShell: $env:STOCKFISH_PATH="...")
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", str(_DEFAULT_STOCKFISH_EXE))


class FENRequest(BaseModel):
    fen: str
    depth: int = 18

    @field_validator("fen")
    @classmethod
    def fen_strip(cls, v: str) -> str:
        return v.strip()


class EvalResponse(BaseModel):
    score: float          # w pionkach, np. +1.3 dla białych, -2.1 dla czarnych
    score_type: str       # "cp" (centypiony) lub "mate"
    mate_in: int | None   # liczba ruchów do mata, None jeśli nie ma
    best_move: str | None # np. "e2e4"
    turn: str             # "white" lub "black"
    is_valid: bool


def _best_move_uci(info: chess.engine.InfoDict) -> str | None:
    """
    Pierwszy ruch z linii PV.
    Uwaga: info.get('pv', [None])[0] jest błędne — jeśli klucz 'pv' istnieje, ale ma wartość None,
    .get zwraca None (nie domyślną listę), co daje TypeError przy [0].
    """
    raw = info.get("pv")
    if not raw:
        return None
    first = raw[0]
    if isinstance(first, chess.Move):
        return first.uci()
    return str(first)


@app.get("/health")
def health():
    return {"status": "ok", "stockfish": STOCKFISH_PATH}


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: FENRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowy FEN: {e}")

    if not os.path.exists(STOCKFISH_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Stockfish nie znaleziony pod: {STOCKFISH_PATH}. "
                   f"Ustaw zmienną środowiskową STOCKFISH_PATH.",
        )

    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=req.depth))
    except Exception as e:
        msg = str(e).strip() or repr(e)
        raise HTTPException(
            status_code=500,
            detail=f"Błąd Stockfisha ({type(e).__name__}): {msg}",
        )

    pov = info.get("score")
    if pov is None:
        raise HTTPException(
            status_code=500,
            detail="Stockfish nie zwrócił oceny (brak pola score).",
        )

    score_obj = pov.white()
    best_move = _best_move_uci(info)

    if score_obj.is_mate():
        mate_val = score_obj.mate()
        # +M3 = białe dają mata w 3, -M3 = czarne dają mata w 3
        score_cp = 100.0 if mate_val > 0 else -100.0  # cap dla termometru
        return EvalResponse(
            score=score_cp,
            score_type="mate",
            mate_in=mate_val,
            best_move=best_move,
            turn="white" if board.turn == chess.WHITE else "black",
            is_valid=True,
        )

    cp = score_obj.score()
    if cp is None:
        cp = score_obj.score(mate_score=32000) or 0
    # Normalizuj do przedziału [-10, +10] pionków dla czytelności
    score_pawns = round(cp / 100, 2)

    return EvalResponse(
        score=score_pawns,
        score_type="cp",
        mate_in=None,
        best_move=best_move,
        turn="white" if board.turn == chess.WHITE else "black",
        is_valid=True,
    )
