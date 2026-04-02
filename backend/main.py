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


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class FENRequest(BaseModel):
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
    score: float          # w pionkach, np. +1.3 dla białych, -2.1 dla czarnych
    score_type: str       # "cp" (centypiony) lub "mate"
    mate_in: int | None   # liczba ruchów do mata, None jeśli nie ma
    best_move: str | None # np. "e2e4"
    pv: list[str]         # principal variation — pełna linia ruchów UCI
    depth: int            # faktyczna głębokość analizy zwrócona przez silnik
    turn: str             # "white" lub "black"
    is_valid: bool


def _extract_pv(info: chess.engine.InfoDict) -> list[str]:
    """Wyciąga linię PV z InfoDict jako listę stringów UCI."""
    raw = info.get("pv")
    if not raw:
        return []
    return [m.uci() if isinstance(m, chess.Move) else str(m) for m in raw]


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
            if req.elo_limit is not None:
                engine.configure({
                    "UCI_LimitStrength": True,
                    "UCI_Elo": req.elo_limit,
                })
            elif req.skill_level is not None:
                engine.configure({"Skill Level": req.skill_level})

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
    pv_line = _extract_pv(info)
    best_move = pv_line[0] if pv_line else None
    actual_depth = info.get("depth", req.depth)

    if score_obj.is_mate():
        mate_val = score_obj.mate()
        score_cp = 100.0 if mate_val > 0 else -100.0
        return EvalResponse(
            score=score_cp,
            score_type="mate",
            mate_in=mate_val,
            best_move=best_move,
            pv=pv_line,
            depth=actual_depth,
            turn="white" if board.turn == chess.WHITE else "black",
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
        turn="white" if board.turn == chess.WHITE else "black",
        is_valid=True,
    )
