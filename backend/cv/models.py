"""
Pydantic modele żądań i odpowiedzi dla endpointów /cv/*.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class SnapshotResponse(BaseModel):
    ok: bool
    message: str
    image_b64: str | None = None
    width: int | None = None
    height: int | None = None


# ---------------------------------------------------------------------------
# Kalibracja
# ---------------------------------------------------------------------------


class CalibrateRequest(BaseModel):
    """
    method='auto'   → findChessboardCorners 7×7 lub YOLO (jeśli załadowany)
    method='manual' → 4 punkty [TL, TR, BR, BL] w pikselach klatki kamery
    """
    method: Literal["auto", "manual"] = "auto"
    corners: list[list[float]] | None = Field(
        default=None,
        description="4 punkty [TL, TR, BR, BL] w pikselach — tylko dla method='manual'",
    )


class CalibrationStatus(BaseModel):
    calibrated: bool
    method: Literal["auto", "manual"] | None = None
    board_size_px: int | None = None
    source_corners: list[list[float]] | None = None
    saved_at: str | None = None


class CalibrateResponse(BaseModel):
    ok: bool
    message: str
    calibration: CalibrationStatus | None = None
    warped_preview_b64: str | None = None


# ---------------------------------------------------------------------------
# Zajętość pól
# ---------------------------------------------------------------------------


class CellInfo(BaseModel):
    square: str
    occupied: bool
    score: float          # wariancja lub p(occupied) z CNN
    method: str           # "variance" lub "cnn"


class OccupancyResponse(BaseModel):
    ok: bool
    message: str
    occupied_squares: list[str]
    empty_squares: list[str]
    occupied_count: int
    cells: list[CellInfo]
    debug_image_b64: str | None = None
    threshold_used: float
    occupancy_method: str   # "variance" lub "cnn"


# ---------------------------------------------------------------------------
# Stan gry
# ---------------------------------------------------------------------------


class GameStateResponse(BaseModel):
    fen: str
    turn: str
    move_number: int
    halfmove_clock: int
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    is_game_over: bool
    history: list[str]
    history_length: int


class GameResetRequest(BaseModel):
    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class ManualMoveRequest(BaseModel):
    move_uci: str


class MoveResultResponse(BaseModel):
    ok: bool
    move_uci: str | None = None
    fen_after: str | None = None
    reason: str = ""
    detector_state: str = ""


# ---------------------------------------------------------------------------
# Diagnostyka
# ---------------------------------------------------------------------------


class CVHealthResponse(BaseModel):
    camera_reachable: bool
    camera_url: str
    calibrated: bool
    board_size_px: int
    square_classifier_loaded: bool
    board_detector_loaded: bool
    message: str


# ---------------------------------------------------------------------------
# ML — dataset
# ---------------------------------------------------------------------------


class DatasetStatsResponse(BaseModel):
    occupied_count: int
    empty_count: int
    total: int
    dataset_dir: str


class CollectResponse(BaseModel):
    ok: bool
    message: str
    occupied_saved: int
    empty_saved: int
    fen_used: str


class MoveCollectRequest(BaseModel):
    move_uci: str


class MoveCollectResponse(BaseModel):
    ok: bool
    move_uci: str
    fen_after: str
    occupied_saved: int
    empty_saved: int
    message: str


class UndoMoveCollectResponse(BaseModel):
    ok: bool
    undone_move_uci: str
    fen_after_undo: str
    occupied_deleted: int
    empty_deleted: int
    message: str
