import asyncio
import os
import sys
from pathlib import Path


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import chess
import chess.engine
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

app = FastAPI(title="ML CHESS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
_DEFAULT_STOCKFISH = _REPO_ROOT / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", str(_DEFAULT_STOCKFISH))


def _clamp_int(value: int, low: int, high: int) -> int:
    
    return max(low, min(high, value))




class FENRequest(BaseModel):    

    fen: str
    depth: int = 18
    skill_level: int | None = None  # 0–20
    elo_limit: int | None = None    # 1320–3190, UCI_LimitStrength;

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
    
    score: float  
    score_type: str  
    mate_in: int | None
    best_move: str | None
    pv: list[str] 
    depth: int
    turn: str  
    is_valid: bool

def _extract_pv(info: chess.engine.InfoDict) -> list[str]:
    raw = info.get("pv")
    if not raw:
        return []
    return [m.uci() if isinstance(m, chess.Move) else str(m) for m in raw]


def _turn_label(board: chess.Board) -> str:
    return "white" if board.turn == chess.WHITE else "black"


def _eval_response_from_engine(
    board: chess.Board,
    info: chess.engine.InfoDict,
    requested_depth: int,
) -> EvalResponse:

    
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


@app.get("/health")
def health():
    """Prosty ping + informacja, skąd ładowany jest Stockfish."""
    return {"status": "ok", "stockfish": STOCKFISH_PATH}


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: FENRequest):
    """
    Uruchamia analizę pozycji: konfiguruje silnik (Elo / Skill Level), analyse(depth).
    """
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowy FEN: {e}")

    if not os.path.exists(STOCKFISH_PATH):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Stockfish nie znaleziony pod: {STOCKFISH_PATH}. "
                "Ustaw zmienną środowiskową STOCKFISH_PATH."
            ),
        )

    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            if req.elo_limit is not None:
                engine.configure(
                    {"UCI_LimitStrength": True, "UCI_Elo": req.elo_limit}
                )
            elif req.skill_level is not None:
                engine.configure({"Skill Level": req.skill_level})

            info = engine.analyse(board, chess.engine.Limit(depth=req.depth))
    except Exception as e:
        msg = str(e).strip() or repr(e)
        raise HTTPException(
            status_code=500,
            detail=f"Błąd Stockfisha ({type(e).__name__}): {msg}",
        )

    return _eval_response_from_engine(board, info, req.depth)
