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
from fastapi.responses import HTMLResponse, Response

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
