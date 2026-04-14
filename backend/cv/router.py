"""
FastAPI router /cv/* — wszystkie endpointy Computer Vision.

Etap 1 — kamera i kalibracja:
  GET  /cv/health
  GET  /cv/snapshot
  GET  /cv/snapshot/warped
  GET  /cv/snapshot/debug
  POST /cv/calibrate
  GET  /cv/calibration
  DELETE /cv/calibration

Etap 1 — detekcja zajętości:
  GET  /cv/occupancy

Etap 1 — gra:
  GET  /cv/game/state
  POST /cv/game/reset
  POST /cv/game/move
  POST /cv/game/detector/start
  POST /cv/game/detector/tick

Integracja ze Stockfishem:
  GET  /cv/evaluate-current

ML — zbieranie danych (Etap 3):
  POST /cv/ml/collect
  GET  /cv/ml/dataset/stats
"""

import logging

from fastapi import APIRouter, HTTPException

from . import board_occupancy, calibration, camera, game_state, move_detector
from .config import BOARD_SIZE_PX, CAMERA_SNAPSHOT_URL, OCCUPANCY_VARIANCE_THRESHOLD
from .models import (
    CalibrationStatus,
    CalibrateRequest,
    CalibrateResponse,
    CellInfo,
    CollectResponse,
    CVHealthResponse,
    DatasetStatsResponse,
    GameResetRequest,
    GameStateResponse,
    ManualMoveRequest,
    MoveResultResponse,
    OccupancyResponse,
    SnapshotResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cv", tags=["Computer Vision"])

# Singleton detektora ruchów
_detector = move_detector.MoveDetector()


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------


def on_startup() -> None:
    """Wywoływana przy starcie aplikacji — ładuje kalibrację i modele ML."""
    try:
        loaded = calibration.load_calibration()
        if loaded:
            logger.info("Kalibracja załadowana przy starcie.")
    except RuntimeError as exc:
        logger.warning("Nie załadowano kalibracji: %s", exc)

    _load_ml_models()


def _load_ml_models() -> None:
    """Ładuje modele ML jeśli dostępne wagi (nie rzuca wyjątku)."""
    try:
        from .ml.square_classifier import load_model as load_sq
        load_sq()
        logger.info("Square classifier załadowany.")
    except Exception as exc:
        logger.info("Square classifier niedostępny: %s", exc)

    try:
        from .ml.board_detector import load_model as load_bd
        load_bd()
        logger.info("Board detector załadowany.")
    except Exception as exc:
        logger.info("Board detector niedostępny: %s", exc)


# ---------------------------------------------------------------------------
# GET /cv/health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=CVHealthResponse)
def cv_health():
    """Status kamery, kalibracji i załadowanych modeli ML."""
    reachable = camera.is_camera_reachable()
    cal = calibration.get_calibration_status()

    sq_loaded = False
    bd_loaded = False
    try:
        from .ml.square_classifier import is_loaded as sq_is_loaded
        sq_loaded = sq_is_loaded()
    except ImportError:
        pass
    try:
        from .ml.board_detector import is_loaded as bd_is_loaded
        bd_loaded = bd_is_loaded()
    except ImportError:
        pass

    parts = []
    if not reachable:
        parts.append("kamera NIEOSIAGALNA")
    if not cal["calibrated"]:
        parts.append("brak kalibracji")
    if not sq_loaded:
        parts.append("sq_classifier: fallback wariancja")
    if not bd_loaded:
        parts.append("board_detector: fallback manual")

    message = "OK" if not parts else " | ".join(parts)

    return CVHealthResponse(
        camera_reachable=reachable,
        camera_url=CAMERA_SNAPSHOT_URL,
        calibrated=cal["calibrated"],
        board_size_px=BOARD_SIZE_PX,
        square_classifier_loaded=sq_loaded,
        board_detector_loaded=bd_loaded,
        message=message,
    )


# ---------------------------------------------------------------------------
# GET /cv/snapshot
# ---------------------------------------------------------------------------


@router.get("/snapshot", response_model=SnapshotResponse)
def get_snapshot():
    """Surowa klatka z kamery (base64 JPEG)."""
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    h, w = frame.shape[:2]
    return SnapshotResponse(ok=True, message=f"Klatka {w}x{h}.",
                            image_b64=camera.frame_to_base64(frame), width=w, height=h)


# ---------------------------------------------------------------------------
# GET /cv/snapshot/warped
# ---------------------------------------------------------------------------


@router.get("/snapshot/warped", response_model=SnapshotResponse)
def get_snapshot_warped():
    """Klatka po perspektive warp (wymaga kalibracji)."""
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))
    return SnapshotResponse(ok=True, message=f"Warped {BOARD_SIZE_PX}x{BOARD_SIZE_PX}px.",
                            image_b64=camera.frame_to_base64(warped),
                            width=BOARD_SIZE_PX, height=BOARD_SIZE_PX)


# ---------------------------------------------------------------------------
# GET /cv/snapshot/debug
# ---------------------------------------------------------------------------


@router.get("/snapshot/debug", response_model=SnapshotResponse)
def get_snapshot_debug():
    """Klatka z siatką 8x8, etykietami pól i wynikami klasyfikacji."""
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    analysis = board_occupancy.analyze_board(warped)
    debug_img = board_occupancy.draw_debug_grid(warped, analysis)
    occupied_n = sum(1 for c in analysis if c.occupied)
    method = analysis[0].method if analysis else "?"

    return SnapshotResponse(
        ok=True,
        message=f"Debug [{method}]: {occupied_n}/64 zajętych.",
        image_b64=camera.frame_to_base64(debug_img),
        width=BOARD_SIZE_PX, height=BOARD_SIZE_PX,
    )


# ---------------------------------------------------------------------------
# POST /cv/calibrate
# ---------------------------------------------------------------------------


@router.post("/calibrate", response_model=CalibrateResponse)
def run_calibrate(req: CalibrateRequest):
    """
    Kalibracja perspektywy szachownicy.

    method='auto'   — findChessboardCorners lub YOLO (jeśli załadowany)
    method='manual' — corners: [[x,y]×4] w kolejności TL, TR, BR, BL
    """
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))

    try:
        if req.method == "auto":
            warped, src = calibration.calibrate_auto(frame)
        else:
            if not req.corners:
                raise HTTPException(422, detail="Brak corners dla method='manual'.")
            warped, src = calibration.calibrate_manual(frame, req.corners)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, detail=str(exc))

    try:
        calibration.save_calibration(src, method=req.method)
    except RuntimeError as exc:
        logger.error("Zapis kalibracji nie powiódł się: %s", exc)

    status = CalibrationStatus(**calibration.get_calibration_status())
    return CalibrateResponse(
        ok=True,
        message=f"Kalibracja ({req.method}) zakończona. Warped {BOARD_SIZE_PX}x{BOARD_SIZE_PX}px.",
        calibration=status,
        warped_preview_b64=camera.frame_to_base64(warped),
    )


# ---------------------------------------------------------------------------
# GET /cv/calibration
# ---------------------------------------------------------------------------


@router.get("/calibration", response_model=CalibrationStatus)
def get_calibration():
    return CalibrationStatus(**calibration.get_calibration_status())


# ---------------------------------------------------------------------------
# DELETE /cv/calibration
# ---------------------------------------------------------------------------


@router.delete("/calibration")
def clear_calibration():
    calibration.reset_calibration()
    return {"ok": True, "message": "Kalibracja usunięta z pamięci (plik na dysku bez zmian)."}


# ---------------------------------------------------------------------------
# GET /cv/occupancy
# ---------------------------------------------------------------------------


@router.get("/occupancy", response_model=OccupancyResponse)
def get_occupancy(debug: bool = False):
    """
    Klasyfikacja zajętości 64 pól z live klatki.

    debug=true — dołącza obraz z siatką debug (base64 JPEG).
    """
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    analysis = board_occupancy.analyze_board(warped)
    occupied = [c.square_name for c in analysis if c.occupied]
    empty = [c.square_name for c in analysis if not c.occupied]
    method = analysis[0].method if analysis else "variance"
    threshold = analysis[0].threshold if analysis else OCCUPANCY_VARIANCE_THRESHOLD

    debug_b64 = None
    if debug:
        debug_img = board_occupancy.draw_debug_grid(warped, analysis)
        debug_b64 = camera.frame_to_base64(debug_img)

    cells = [CellInfo(square=c.square_name, occupied=c.occupied,
                      score=c.score, method=c.method) for c in analysis]

    return OccupancyResponse(
        ok=True,
        message=f"{len(occupied)}/64 pól zajętych [{method}].",
        occupied_squares=occupied,
        empty_squares=empty,
        occupied_count=len(occupied),
        cells=cells,
        debug_image_b64=debug_b64,
        threshold_used=threshold,
        occupancy_method=method,
    )


# ---------------------------------------------------------------------------
# GET /cv/game/state
# ---------------------------------------------------------------------------


@router.get("/game/state", response_model=GameStateResponse)
def get_game_state():
    return GameStateResponse(**game_state.get_status())


# ---------------------------------------------------------------------------
# POST /cv/game/reset
# ---------------------------------------------------------------------------


@router.post("/game/reset")
def reset_game(req: GameResetRequest):
    """Resetuje grę do podanego FEN. Po resecie wywołaj /game/detector/start."""
    try:
        game_state.reset(req.fen)
    except ValueError as exc:
        raise HTTPException(422, detail=f"Nieprawidłowy FEN: {exc}")
    return {"ok": True, "fen": req.fen, "message": "Gra zresetowana."}


# ---------------------------------------------------------------------------
# POST /cv/game/move
# ---------------------------------------------------------------------------


@router.post("/game/move", response_model=MoveResultResponse)
def manual_move(req: ManualMoveRequest):
    """Ręcznie wykonuje ruch UCI (do testów / korekty pomyłek)."""
    try:
        game_state.push(req.move_uci)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    return MoveResultResponse(ok=True, move_uci=req.move_uci,
                              fen_after=game_state.get_fen(),
                              reason="Ruch ręczny.",
                              detector_state=_detector.state_name)


# ---------------------------------------------------------------------------
# POST /cv/game/detector/start
# ---------------------------------------------------------------------------


@router.post("/game/detector/start")
def detector_start():
    """
    Inicjalizuje detektor — pobiera snapshot 'before' z aktualnej klatki.
    Wywołuj po kalibracji i po każdym resecie gry.
    """
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    _detector.start(warped)
    return {
        "ok": True,
        "message": f"Detektor uruchomiony. Snapshot before: {len(_detector.before_snapshot)} pol.",
        "before_occupied": sorted(_detector.before_snapshot),
        "detector_state": _detector.state_name,
    }


# ---------------------------------------------------------------------------
# POST /cv/game/detector/tick
# ---------------------------------------------------------------------------


@router.post("/game/detector/tick", response_model=MoveResultResponse)
def detector_tick():
    """
    Jedna iteracja detekcji ruchu.

    Wywołuj cyklicznie (np. co 500ms). Gdy move_detected=True → ruch zatwierdzony,
    FEN zaktualizowany, gotowy do /evaluate-current.
    """
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    result = _detector.process_frame(warped)
    return MoveResultResponse(
        ok=True,
        move_uci=result.move_uci,
        fen_after=result.fen_after,
        reason=result.reason,
        detector_state=result.state,
    )


# ---------------------------------------------------------------------------
# GET /cv/evaluate-current
# ---------------------------------------------------------------------------


@router.get("/evaluate-current")
def evaluate_current():
    """Przekazuje aktualny FEN z game_state do silnika Stockfish."""
    current_fen = game_state.get_fen()
    from main import FENRequest, evaluate  # noqa: PLC0415
    try:
        req = FENRequest(fen=current_fen, depth=18)
        result = evaluate(req)
        return {"fen": current_fen, "source": "game_state", "evaluation": result}
    except Exception as exc:
        raise HTTPException(500, detail=f"Blad Stockfisha: {exc}")


# ---------------------------------------------------------------------------
# POST /cv/ml/collect
# ---------------------------------------------------------------------------


@router.post("/ml/collect", response_model=CollectResponse)
def ml_collect():
    """
    Pobiera klatkę, warps, ekstrahuje 64 patche i auto-labeluje na podstawie
    aktualnego FEN z game_state. Zapisuje do datasetu.

    Używaj po każdym zatwierdzonym ruchu żeby budować dataset treningowy.
    """
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    from .ml.data.collector import collect_from_frame  # noqa: PLC0415
    try:
        occ, emp, fen_used = collect_from_frame(warped, game_state.get_fen())
    except Exception as exc:
        raise HTTPException(500, detail=f"Blad collectora: {exc}")

    return CollectResponse(
        ok=True,
        message=f"Zapisano {occ} occupied, {emp} empty.",
        occupied_saved=occ,
        empty_saved=emp,
        fen_used=fen_used,
    )


# ---------------------------------------------------------------------------
# GET /cv/ml-dataset-stats
# ---------------------------------------------------------------------------


@router.get("/ml/dataset/stats", response_model=DatasetStatsResponse)
def ml_dataset_stats():
    """Statystyki zebranego datasetu (ile próbek occupied / empty)."""
    from .ml.data.collector import get_dataset_stats  # noqa: PLC0415
    stats = get_dataset_stats()
    return DatasetStatsResponse(**stats)
