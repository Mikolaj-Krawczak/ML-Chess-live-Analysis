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
  POST /cv/game/move-collect
  POST /cv/game/move-collect/undo
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
from fastapi.responses import HTMLResponse, Response

from . import board_occupancy, calibration, camera, detector_worker, game_state, move_detector
from .config import BOARD_SIZE_PX, CAMERA_SNAPSHOT_URL, OCCUPANCY_VARIANCE_THRESHOLD
from .models import (
    CalibrationStatus,
    CalibrateRequest,
    CalibrateResponse,
    CellInfo,
    CollectResponse,
    CVHealthResponse,
    DatasetStatsResponse,
    EditMoveRequest,
    EditMoveResponse,
    GameResetRequest,
    GameStateResponse,
    ManualMoveRequest,
    MoveCollectRequest,
    MoveCollectResponse,
    MoveResultResponse,
    OccupancyResponse,
    SnapshotResponse,
    UndoMoveCollectResponse,
    ValidatePositionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cv", tags=["Computer Vision"])

# Singleton detektora ruchów i wątku tła
_detector = move_detector.MoveDetector()
_worker = detector_worker.DetectorWorker()
_move_collect_history: list[dict] = []


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

    # A1: wątek backgroundowy ciągle czyta MJPEG — eliminuje HTTP GET per tick
    try:
        camera.start_stream_thread()
    except Exception as exc:
        logger.warning("Nie udało się uruchomić wątku strumienia kamery: %s", exc)


def on_shutdown() -> None:
    """Graceful shutdown — zatrzymuje wątki backgroundowe."""
    try:
        _worker.stop()
    except Exception as exc:  # pragma: no cover
        logger.warning("Błąd przy zatrzymywaniu DetectorWorker: %s", exc)
    try:
        camera.stop_stream_thread()
    except Exception as exc:  # pragma: no cover
        logger.warning("Błąd przy zatrzymywaniu wątku kamery: %s", exc)


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
    sq_backend = "n/a"
    try:
        from .ml.square_classifier import is_loaded as sq_is_loaded
        from .ml.square_classifier import get_backend_mode as sq_backend_mode
        sq_loaded = sq_is_loaded()
        if sq_loaded:
            sq_backend = sq_backend_mode()
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
    if sq_loaded:
        parts.append(f"sq_backend={sq_backend}")

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
# GET /cv/snapshot.jpg  — surowy obraz bezpośrednio w przeglądarce
# ---------------------------------------------------------------------------


@router.get("/snapshot.jpg", response_class=Response)
def get_snapshot_jpg():
    """Surowa klatka z kamery jako JPEG — można otworzyć bezpośrednio w przeglądarce."""
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    return Response(content=camera.frame_to_jpeg_bytes(frame), media_type="image/jpeg")


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
# GET /cv/snapshot/warped.jpg  — wyprostowany obraz bezpośrednio w przeglądarce
# ---------------------------------------------------------------------------


@router.get("/snapshot/warped.jpg", response_class=Response)
def get_snapshot_warped_jpg():
    """Klatka po warp jako JPEG — można otworzyć bezpośrednio w przeglądarce."""
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))
    return Response(content=camera.frame_to_jpeg_bytes(warped), media_type="image/jpeg")


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
# GET /cv/snapshot/debug.jpg  — debug z siatką bezpośrednio w przeglądarce
# ---------------------------------------------------------------------------


@router.get("/snapshot/debug.jpg", response_class=Response)
def get_snapshot_debug_jpg():
    """Klatka z siatką 8×8 jako JPEG — można otworzyć bezpośrednio w przeglądarce."""
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
    return Response(content=camera.frame_to_jpeg_bytes(debug_img), media_type="image/jpeg")


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
# GET /cv/calibrate/ui  — interaktywna kalibracja w przeglądarce
# ---------------------------------------------------------------------------

_CALIBRATION_UI_HTML = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Kalibracja szachownicy</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#111;color:#eee;font-family:monospace;height:100vh;display:flex;flex-direction:column}
  #toolbar{display:flex;align-items:center;gap:8px;padding:8px 12px;background:#1e1e1e;border-bottom:1px solid #333;flex-shrink:0}
  button{padding:5px 14px;border:none;border-radius:4px;cursor:pointer;font:13px monospace;font-weight:bold}
  #btn-refresh{background:#2563eb;color:#fff}
  #btn-reset{background:#7c3aed;color:#fff}
  #btn-calibrate{background:#16a34a;color:#fff;display:none}
  #btn-warped{background:#d97706;color:#fff;display:none}
  button:hover{filter:brightness(1.15)}
  #status{flex:1;font-size:13px;padding:0 8px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
  #main{flex:1;display:flex;overflow:hidden}
  #canvas-wrap{flex:1;position:relative;overflow:hidden;cursor:crosshair}
  canvas{display:block;width:100%;height:100%;object-fit:contain}
  #preview-wrap{width:0;overflow:hidden;transition:width .2s;background:#0a0a0a;border-left:1px solid #333;display:flex;flex-direction:column;align-items:center;justify-content:center}
  #preview-wrap.visible{width:420px}
  #preview-label{font-size:11px;color:#888;padding:6px;text-align:center}
  #preview-img{max-width:400px;max-height:calc(100vh - 80px);border:1px solid #333}
  #coords-list{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.7);padding:8px;border-radius:6px;font-size:12px;line-height:1.8}
  .msg-ok{color:#4ade80} .msg-err{color:#f87171} .msg-info{color:#facc15}
</style>
</head>
<body>
<div id="toolbar">
  <button id="btn-refresh" title="F">&#8635; Odśwież klatkę</button>
  <button id="btn-reset" title="R">&#10006; Reset punktów</button>
  <button id="btn-calibrate">&#10003; Kalibruj</button>
  <button id="btn-warped">&#9654; Pokaż warped</button>
  <span id="status">Ładowanie obrazu...</span>
</div>
<div id="main">
  <div id="canvas-wrap">
    <canvas id="cvs"></canvas>
    <div id="coords-list"></div>
  </div>
  <div id="preview-wrap">
    <div id="preview-label">Podgląd po kalibracji</div>
    <img id="preview-img" src="" alt="warped">
  </div>
</div>

<script>
const LABELS = ['TL lewy-górny','TR prawy-górny','BR prawy-dolny','BL lewy-dolny'];
const COLORS = ['#22c55e','#facc15','#ef4444','#60a5fa'];

const cvs = document.getElementById('cvs');
const ctx = cvs.getContext('2d');
let img = new Image();
let pts = [];       // [{x,y}] — współrzędne w pikselach oryginalnego obrazu
let imgW = 0, imgH = 0;

// ---- Ładowanie obrazu ----
async function loadFrame() {
  setStatus('Pobieranie klatki z kamery...', 'info');

  // Pobierz status kamery żeby mieć URL w razie błędu
  let cameraUrl = '';
  try {
    const h = await fetch('/cv/health');
    const hd = await h.json();
    cameraUrl = hd.camera_url || '';
    if (!hd.camera_reachable) {
      setStatus(
        `&#9888; Kamera niedostępna: <b>${cameraUrl}</b><br>` +
        `Sprawdź czy aplikacja IPWebcam działa na telefonie i czy jesteś w tej samej sieci WiFi. ` +
        `<a href="/cv/health" target="_blank" style="color:#60a5fa">health</a>`,
        'err'
      );
      return;
    }
  } catch (_) {}

  img = new Image();
  img.onload = () => {
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    pts = [];
    resizeCanvas();
    draw();
    setStatus(
      `Kamera ${imgW}×${imgH}px — kliknij punkt 1/4: <b>${LABELS[0]}</b>`,
      'info'
    );
    updateUI();
  };
  img.onerror = () => setStatus(
    `&#9888; Nie można pobrać obrazu z kamery` +
    (cameraUrl ? ` (<b>${cameraUrl}</b>)` : '') +
    ` — sprawdź IPWebcam i IP telefonu.`,
    'err'
  );
  img.src = '/cv/snapshot.jpg?t=' + Date.now();
}

// ---- Canvas resize ----
function resizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  cvs.width  = wrap.clientWidth;
  cvs.height = wrap.clientHeight;
}

// ---- Transformacja współrzędnych ----
// Obraz jest rysowany letterbox (object-fit: contain) — obliczamy offset i skalę
function imgRect() {
  const cw = cvs.width, ch = cvs.height;
  const scale = Math.min(cw / imgW, ch / imgH);
  const w = imgW * scale, h = imgH * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h, scale };
}

function canvasToOrig(cx, cy) {
  const r = imgRect();
  return {
    x: Math.round((cx - r.x) / r.scale),
    y: Math.round((cy - r.y) / r.scale),
  };
}

function origToCanvas(ox, oy) {
  const r = imgRect();
  return { x: r.x + ox * r.scale, y: r.y + oy * r.scale };
}

// ---- Rysowanie ----
function draw() {
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  if (!imgW) return;

  const r = imgRect();
  ctx.drawImage(img, r.x, r.y, r.w, r.h);

  // Linie między punktami
  if (pts.length >= 2) {
    ctx.beginPath();
    const p0 = origToCanvas(pts[0].x, pts[0].y);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < pts.length; i++) {
      const pi = origToCanvas(pts[i].x, pts[i].y);
      ctx.lineTo(pi.x, pi.y);
    }
    if (pts.length === 4) {
      ctx.lineTo(p0.x, p0.y);
    }
    ctx.strokeStyle = 'rgba(255,255,100,0.85)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Punkty
  pts.forEach((pt, i) => {
    const {x, y} = origToCanvas(pt.x, pt.y);
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[i];
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Numer
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i + 1, x, y);

    // Etykieta
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = COLORS[i];
    ctx.fillText(LABELS[i], x + 14, y - 4);
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillText(`(${pt.x}, ${pt.y})`, x + 14, y + 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText(`(${pt.x}, ${pt.y})`, x + 14, y + 10);
  });
}

// ---- Klik na canvas ----
cvs.addEventListener('click', (e) => {
  if (pts.length >= 4) return;
  const rect = cvs.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (cvs.width / rect.width);
  const cy = (e.clientY - rect.top)  * (cvs.height / rect.height);
  const {x, y} = canvasToOrig(cx, cy);
  if (x < 0 || y < 0 || x >= imgW || y >= imgH) return;
  pts.push({x, y});
  draw();
  updateUI();
  updateCoordsList();
  if (pts.length < 4) {
    setStatus(
      `Punkt ${pts.length}/4 dodany — kliknij punkt ${pts.length+1}/4: <b>${LABELS[pts.length]}</b>`,
      'info'
    );
  } else {
    setStatus('4 punkty zaznaczone — kliknij <b>Kalibruj</b> lub R żeby zacząć od nowa.', 'ok');
  }
});

// ---- Lista współrzędnych ----
function updateCoordsList() {
  const el = document.getElementById('coords-list');
  if (!pts.length) { el.innerHTML = ''; return; }
  el.innerHTML = pts.map((p, i) =>
    `<span class="dot" style="background:${COLORS[i]}"></span>${LABELS[i].split(' ')[0]}: ${p.x}, ${p.y}`
  ).join('<br>');
}

// ---- Przyciski ----
function updateUI() {
  const ready = pts.length === 4;
  document.getElementById('btn-calibrate').style.display = ready ? '' : 'none';
}

document.getElementById('btn-refresh').onclick = loadFrame;
document.getElementById('btn-reset').onclick = () => {
  pts = [];
  draw();
  updateCoordsList();
  updateUI();
  document.getElementById('preview-wrap').classList.remove('visible');
  document.getElementById('btn-warped').style.display = 'none';
  setStatus(`Reset — kliknij punkt 1/4: <b>${LABELS[0]}</b>`, 'info');
};

document.getElementById('btn-calibrate').onclick = async () => {
  const corners = pts.map(p => [p.x, p.y]);
  setStatus('Wysyłanie kalibracji...', 'info');
  try {
    const resp = await fetch('/cv/calibrate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({method: 'manual', corners}),
    });
    const data = await resp.json();
    if (resp.ok) {
      setStatus('&#10003; ' + (data.message || 'Kalibracja zakończona!'), 'ok');
      document.getElementById('btn-warped').style.display = '';
      showWarpedPreview();
    } else {
      setStatus('Błąd: ' + (data.detail || resp.statusText), 'err');
    }
  } catch (err) {
    setStatus('Błąd połączenia z backendem: ' + err, 'err');
  }
};

document.getElementById('btn-warped').onclick = showWarpedPreview;

function showWarpedPreview() {
  const wrap = document.getElementById('preview-wrap');
  const imgEl = document.getElementById('preview-img');
  imgEl.src = '/cv/snapshot/warped.jpg?t=' + Date.now();
  wrap.classList.add('visible');
}

// ---- Status ----
function setStatus(msg, type) {
  const el = document.getElementById('status');
  el.className = type === 'ok' ? 'msg-ok' : type === 'err' ? 'msg-err' : 'msg-info';
  el.innerHTML = msg;
}

// ---- Klawiatura ----
document.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') document.getElementById('btn-reset').click();
  if (e.key === 'f' || e.key === 'F') document.getElementById('btn-refresh').click();
  if ((e.key === 'Enter' || e.key === 's' || e.key === 'S') && pts.length === 4)
    document.getElementById('btn-calibrate').click();
});

// ---- Resize ----
window.addEventListener('resize', () => { resizeCanvas(); draw(); });

// ---- Start ----
loadFrame();
</script>
</body>
</html>"""


@router.get("/calibrate/ui", response_class=HTMLResponse)
def calibration_ui():
    """Interaktywna kalibracja perspektywy w przeglądarce — kliknij 4 narożniki planszy."""
    return HTMLResponse(content=_CALIBRATION_UI_HTML)


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
    global _move_collect_history
    _worker.stop()
    try:
        game_state.reset(req.fen)
    except ValueError as exc:
        raise HTTPException(422, detail=f"Nieprawidłowy FEN: {exc}")
    _move_collect_history = []
    return {"ok": True, "fen": req.fen, "message": "Gra zresetowana. Wywołaj /game/detector/start żeby wznowić detekcję."}


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
# POST /cv/game/move-collect
# ---------------------------------------------------------------------------


@router.post("/game/move-collect", response_model=MoveCollectResponse)
def move_collect(req: MoveCollectRequest):
    """
    Pipeline do budowy datasetu:
    1) waliduje i wykonuje ruch UCI na game_state,
    2) natychmiast zbiera próbki przez /ml/collect logikę,
    3) zwraca FEN po ruchu oraz liczbę zapisanych patchy.
    """
    try:
        game_state.push(req.move_uci)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))

    try:
        try:
            frame = camera.fetch_snapshot()
        except RuntimeError as exc:
            raise HTTPException(503, detail=str(exc))
        try:
            warped = calibration.apply_warp(frame)
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc))

        from .ml.data.collector import collect_from_frame_with_batch  # noqa: PLC0415
        current_fen = game_state.get_fen()
        try:
            occ, emp, fen_used, batch_id = collect_from_frame_with_batch(warped, current_fen)
        except Exception as exc:
            raise HTTPException(500, detail=f"Blad collectora: {exc}")
    except HTTPException:
        # Pipeline atomowy: gdy collect się nie powiedzie, cofamy wykonany ruch.
        try:
            game_state.undo_last_move()
        except ValueError:
            logger.warning("Nie udalo sie cofnac ruchu po nieudanym move-collect.")
        raise

    _move_collect_history.append(
        {
            "move_uci": req.move_uci,
            "batch_id": batch_id,
            "occupied_saved": occ,
            "empty_saved": emp,
        }
    )

    return MoveCollectResponse(
        ok=True,
        move_uci=req.move_uci,
        fen_after=fen_used,
        occupied_saved=occ,
        empty_saved=emp,
        message=f"Ruch zaakceptowany. Zapisano {occ} occupied i {emp} empty.",
    )


# ---------------------------------------------------------------------------
# POST /cv/game/move-collect/undo
# ---------------------------------------------------------------------------


@router.post("/game/move-collect/undo", response_model=UndoMoveCollectResponse)
def undo_move_collect():
    """
    Cofa ostatni ruch wykonany przez pipeline move-collect i usuwa jego batch danych.
    """
    if not _move_collect_history:
        raise HTTPException(409, detail="Brak ruchu move-collect do cofnięcia.")

    last = _move_collect_history[-1]

    try:
        undone = game_state.undo_last_move()
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc))

    if undone != last["move_uci"]:
        raise HTTPException(
            409,
            detail=(
                "Niespójność historii: ostatni ruch na planszy różni się od move-collect. "
                "Zrób reset gry i rozpocznij nową sesję datasetu."
            ),
        )

    from .ml.data.collector import delete_batch  # noqa: PLC0415
    try:
        deleted = delete_batch(last["batch_id"])
    except Exception as exc:
        raise HTTPException(500, detail=f"Blad usuwania batcha: {exc}")

    _move_collect_history.pop()
    return UndoMoveCollectResponse(
        ok=True,
        undone_move_uci=undone,
        fen_after_undo=game_state.get_fen(),
        occupied_deleted=deleted["occupied_deleted"],
        empty_deleted=deleted["empty_deleted"],
        message=(
            f"Cofnieto ruch {undone}. "
            f"Usunieto occupied={deleted['occupied_deleted']}, empty={deleted['empty_deleted']}."
        ),
    )


# ---------------------------------------------------------------------------
# POST /cv/game/detector/start
# ---------------------------------------------------------------------------


@router.post("/game/detector/start")
def detector_start():
    """
    Inicjalizuje detektor i uruchamia wątek tła (DetectorWorker).

    Wywołuj po kalibracji i po każdym resecie gry. Wątek tła przetwarza
    klatki z kamery z częstotliwością ~4fps — frontend nie musi już triggerować
    ciężkich tick-ów z inferencją CNN.
    """
    try:
        frame = camera.fetch_snapshot_fast()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    _detector.start(warped)
    _worker.start(_detector, target_fps=4.0)

    return {
        "ok": True,
        "message": (
            f"Detektor uruchomiony (background worker aktywny). "
            f"Snapshot before: {len(_detector.before_snapshot)} pol."
        ),
        "before_occupied": sorted(_detector.before_snapshot),
        "detector_state": _detector.state_name,
        "worker_running": _worker.is_running,
    }


# ---------------------------------------------------------------------------
# POST /cv/game/detector/tick
# ---------------------------------------------------------------------------


@router.post("/game/detector/tick", response_model=MoveResultResponse)
def detector_tick():
    """
    Zwraca aktualny stan detektora.

    Gdy DetectorWorker jest aktywny (normalny tryb): odpowiada natychmiast (<1ms)
    z cached wynikiem ostatniej klatki — bez inferencji CNN w tym wątku.

    Gdy worker nieaktywny (tryb fallback/legacy): przetwarza jedną klatkę na żądanie
    (stare zachowanie — wolniejsze, ale zachowuje kompatybilność wsteczną).
    """
    if _worker.is_running:
        status = _worker.get_status()
        return MoveResultResponse(
            ok=True,
            move_uci=None,
            fen_after=None,
            reason=status.last_reason,
            detector_state=status.last_detector_state,
        )

    # Fallback: worker nie uruchomiony — stary flow (np. detector/start nie wywołany)
    try:
        frame = camera.fetch_snapshot_fast()
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
# POST /cv/game/detector/stop
# ---------------------------------------------------------------------------


@router.post("/game/detector/stop")
def detector_stop():
    """Zatrzymuje wątek tła DetectorWorker. Gra pozostaje niezmieniona."""
    _worker.stop()
    return {
        "ok": True,
        "message": "DetectorWorker zatrzymany.",
        "detector_state": _detector.state_name,
    }


# ---------------------------------------------------------------------------
# GET /cv/game/detector/status
# ---------------------------------------------------------------------------


@router.get("/game/detector/status")
def detector_status():
    """Zwraca diagnostykę wątku tła: fps, liczba klatek, błędy."""
    status = _worker.get_status()
    return {
        "worker_running": _worker.is_running,
        "detector_state": _detector.state_name,
        "frames_processed": status.frames_processed,
        "errors": status.errors,
        "avg_frame_ms": status.avg_frame_ms,
        "last_move_uci": status.last_move_uci,
    }


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
# POST /cv/game/move/edit
# ---------------------------------------------------------------------------


@router.post("/game/move/edit", response_model=EditMoveResponse)
def edit_last_move(req: EditMoveRequest):
    """
    Edytuje ostatni ruch - cofa go i wykonuje nowy ruch.
    Gracz musi następnie ustawić figury zgodnie z nową pozycją i wywołać walidację.
    """
    try:
        old_move, new_fen = game_state.edit_last_move(req.new_move_uci)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    
    return EditMoveResponse(
        ok=True,
        old_move_uci=old_move,
        new_move_uci=req.new_move_uci,
        fen_after_edit=new_fen,
        message=f"Ruch edytowany: {old_move} → {req.new_move_uci}. Ustaw figury zgodnie z nową pozycją.",
        requires_validation=True
    )


# ---------------------------------------------------------------------------
# POST /cv/game/validate-position
# ---------------------------------------------------------------------------


@router.post("/game/validate-position", response_model=ValidatePositionResponse)
def validate_position():
    """
    Sprawdza czy pozycja na szachownicy (kamera) odpowiada aktualnemu FEN w game_state.
    Używane po edycji ruchu - gracz ustawia figury i sprawdza czy są właściwie.
    """
    try:
        frame = camera.fetch_snapshot()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc))
    try:
        warped = calibration.apply_warp(frame)
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))

    # Pobierz aktualny FEN i oczekiwaną pozycję
    current_fen = game_state.get_fen()
    
    # Analizuj obecną zajętość pól
    analysis = board_occupancy.analyze_board(warped)
    actual_occupied = [c.square_name for c in analysis if c.occupied]
    
    # Wyciągnij oczekiwaną zajętość z FEN
    expected_occupied = _extract_occupied_squares_from_fen(current_fen)
    
    # Porównaj
    missing_pieces = [sq for sq in expected_occupied if sq not in actual_occupied]
    extra_pieces = [sq for sq in actual_occupied if sq not in expected_occupied]
    
    position_matches = len(missing_pieces) == 0 and len(extra_pieces) == 0
    
    if position_matches:
        message = "Pozycja na szachownicy odpowiada oczekiwanej. Gra może być kontynuowana."
    else:
        message = f"Pozycja nie odpowiada oczekiwanej. Brakujące figury: {missing_pieces}, Nadmiarowe: {extra_pieces}"
    
    return ValidatePositionResponse(
        ok=True,
        position_matches=position_matches,
        expected_occupied=expected_occupied,
        actual_occupied=actual_occupied,
        missing_pieces=missing_pieces,
        extra_pieces=extra_pieces,
        message=message
    )


def _extract_occupied_squares_from_fen(fen: str) -> list[str]:
    """
    Wyciąga listę zajętych pól z FEN string.
    """
    board_part = fen.split(' ')[0]  # Tylko część szachownicy
    occupied_squares = []
    
    rank = 8  # Zaczynamy od 8 linii (a8-h8)
    file = 0  # Kolumny a=0, b=1, ..., h=7
    
    for char in board_part:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)  # Przesuń o liczbę pustych pól
        else:
            # To jest figura
            square_name = chr(ord('a') + file) + str(rank)
            occupied_squares.append(square_name)
            file += 1
    
    return occupied_squares


# ---------------------------------------------------------------------------
# GET /cv/ml/dataset/ui
# ---------------------------------------------------------------------------

_DATASET_UI_HTML = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Collector</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;padding:20px;background:#101317;color:#e5e7eb;font-family:monospace}
  .wrap{max-width:1120px;margin:0 auto;display:grid;grid-template-columns:1fr 380px;gap:16px}
  .card{background:#171b21;border:1px solid #2a3240;border-radius:8px;padding:14px}
  h1{margin:0 0 10px;font-size:20px}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  input{flex:1;min-width:200px;background:#0f1318;border:1px solid #374151;color:#e5e7eb;padding:9px 10px;border-radius:6px}
  button{border:none;border-radius:6px;padding:9px 12px;cursor:pointer;font-weight:700}
  #btn-submit{background:#22c55e;color:#052e16}
  #btn-refresh{background:#3b82f6;color:#eff6ff}
  #btn-reset{background:#f59e0b;color:#111827}
  #btn-undo{background:#ef4444;color:#fff}
  #status{margin-top:10px;min-height:24px}
  .ok{color:#86efac}.err{color:#fca5a5}.info{color:#93c5fd}
  .mono{font-family:Consolas,monospace;word-break:break-all;font-size:13px}
  img{width:100%;height:auto;border-radius:6px;border:1px solid #374151;background:#000}
  .hint{margin-top:10px;color:#9ca3af;font-size:12px;line-height:1.5}
</style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Move -> Validate -> Collect</h1>
      <div class="row">
        <input id="move" placeholder="Wpisz ruch UCI, np. e2e4" autocomplete="off" />
        <button id="btn-submit">Wyślij ruch</button>
        <button id="btn-refresh">Odśwież obraz</button>
        <button id="btn-reset">Reset do startu</button>
        <button id="btn-undo">Cofnij ostatni ruch</button>
      </div>
      <div id="status" class="info">Gotowe. Wpisz ruch UCI i Enter.</div>
      <div class="hint">
        Pipeline działa automatycznie: poprawny ruch wywołuje collect od razu.<br>
        Po sukcesie wypisywany jest aktualny FEN do szybkiej kontroli ustawienia deski.
      </div>
    </section>
    <aside class="card">
      <div><strong>Live podgląd warped</strong></div>
      <img id="warped" src="/cv/snapshot/warped.jpg" alt="warped board preview" />
      <div style="margin-top:10px"><strong>FEN po ruchu</strong></div>
      <div id="fen" class="mono">-</div>
      <div style="margin-top:10px"><strong>Kolej ruchu</strong></div>
      <div id="turn" class="mono">-</div>
      <div style="margin-top:10px"><strong>Ostatni ruch</strong></div>
      <div id="last-move" class="mono">-</div>
      <div style="margin-top:10px"><strong>Zapisane próbki</strong></div>
      <div id="counts" class="mono">occupied: -, empty: -</div>
    </aside>
  </div>

<script>
const moveInput = document.getElementById('move');
const statusEl = document.getElementById('status');
const fenEl = document.getElementById('fen');
const turnEl = document.getElementById('turn');
const lastMoveEl = document.getElementById('last-move');
const countsEl = document.getElementById('counts');
const warpedEl = document.getElementById('warped');
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.className = type;
}

function refreshWarped() {
  warpedEl.src = '/cv/snapshot/warped.jpg?t=' + Date.now();
}

async function refreshGameState() {
  try {
    const resp = await fetch('/cv/game/state');
    if (!resp.ok) return;
    const data = await resp.json();
    fenEl.textContent = data.fen || '-';
    turnEl.textContent = data.turn || '-';
    const history = Array.isArray(data.history) ? data.history : [];
    lastMoveEl.textContent = history.length ? history[history.length - 1] : '-';
  } catch (_) {}
}

async function resetToStart() {
  setStatus('Reset pozycji startowej...', 'info');
  try {
    const resp = await fetch('/cv/game/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({fen: START_FEN}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      setStatus('Blad resetu: ' + (data.detail || resp.statusText), 'err');
      return;
    }
    await refreshGameState();
    setStatus('OK: reset do pozycji startowej (white to move).', 'ok');
  } catch (err) {
    setStatus('Blad polaczenia z backendem: ' + err, 'err');
  }
}

async function sendMove() {
  const move = moveInput.value.trim().toLowerCase();
  if (!move) {
    setStatus('Podaj ruch UCI.', 'err');
    return;
  }

  setStatus('Walidacja ruchu i collect w toku...', 'info');
  try {
    const resp = await fetch('/cv/game/move-collect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({move_uci: move}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      setStatus('Blad: ' + (data.detail || resp.statusText), 'err');
      await refreshGameState();
      return;
    }

    fenEl.textContent = data.fen_after || '-';
    await refreshGameState();
    countsEl.textContent = `occupied: ${data.occupied_saved}, empty: ${data.empty_saved}`;
    setStatus('OK: ruch przyjety i collect wykonany.', 'ok');
    moveInput.value = '';
    refreshWarped();
  } catch (err) {
    setStatus('Blad polaczenia z backendem: ' + err, 'err');
  }
}

async function undoLastMove() {
  setStatus('Cofanie ostatniego ruchu i usuwanie batcha...', 'info');
  try {
    const resp = await fetch('/cv/game/move-collect/undo', { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      setStatus('Blad cofania: ' + (data.detail || resp.statusText), 'err');
      await refreshGameState();
      return;
    }
    countsEl.textContent = `occupied: -${data.occupied_deleted}, empty: -${data.empty_deleted}`;
    await refreshGameState();
    refreshWarped();
    setStatus('OK: ruch cofniety, batch usuniety.', 'ok');
  } catch (err) {
    setStatus('Blad polaczenia z backendem: ' + err, 'err');
  }
}

document.getElementById('btn-submit').addEventListener('click', sendMove);
document.getElementById('btn-refresh').addEventListener('click', refreshWarped);
document.getElementById('btn-reset').addEventListener('click', resetToStart);
document.getElementById('btn-undo').addEventListener('click', undoLastMove);
moveInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMove();
});
refreshWarped();
refreshGameState();
</script>
</body>
</html>"""


@router.get("/ml/dataset/ui", response_class=HTMLResponse)
def ml_dataset_ui():
    """Prosty panel do zbierania datasetu przez pipeline move->collect."""
    return HTMLResponse(content=_DATASET_UI_HTML)


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
