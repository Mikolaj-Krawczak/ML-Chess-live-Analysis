import chess
import chess.engine
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


class EvalResponse(BaseModel):
    score: float          # w pionkach, np. +1.3 dla białych, -2.1 dla czarnych
    score_type: str       # "cp" (centypiony) lub "mate"
    mate_in: int | None   # liczba ruchów do mata, None jeśli nie ma
    best_move: str | None # np. "e2e4"
    turn: str             # "white" lub "black"
    is_valid: bool


@app.get("/health")
def health():
    return {"status": "ok", "stockfish": STOCKFISH_PATH}


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: FENRequest):
    # Walidacja FEN
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowy FEN: {e}")

    if not os.path.exists(STOCKFISH_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Stockfish nie znaleziony pod: {STOCKFISH_PATH}. "
                   f"Ustaw zmienną środowiskową STOCKFISH_PATH."
        )

    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=req.depth))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd Stockfisha: {e}")

    score_obj = info["score"].white()  # zawsze z perspektywy białych
    best_move = info.get("pv", [None])[0]

    if score_obj.is_mate():
        mate_val = score_obj.mate()
        # +M3 = białe dają mata w 3, -M3 = czarne dają mata w 3
        score_cp = 100.0 if mate_val > 0 else -100.0  # cap dla termometru
        return EvalResponse(
            score=score_cp,
            score_type="mate",
            mate_in=mate_val,
            best_move=str(best_move) if best_move else None,
            turn="white" if board.turn == chess.WHITE else "black",
            is_valid=True,
        )

    cp = score_obj.score()
    # Normalizuj do przedziału [-10, +10] pionków dla czytelności
    score_pawns = round(cp / 100, 2)

    return EvalResponse(
        score=score_pawns,
        score_type="cp",
        mate_in=None,
        best_move=str(best_move) if best_move else None,
        turn="white" if board.turn == chess.WHITE else "black",
        is_valid=True,
    )