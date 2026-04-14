# Chess Vision — Architektura Systemu

> Wersja: **v0.3.0-ml** | Branch: `from-above`  
> Stack: Python 3.11+, FastAPI, OpenCV, PyTorch, python-chess, YOLOv8

---

## Spis treści
1. [Czym jest ten system](#1-czym-jest-ten-system)
2. [Jak działa — przepływ danych](#2-jak-działa-przepływ-danych)
3. [Struktura plików](#3-struktura-plików)
4. [Moduły — szczegółowy opis](#4-moduły-szczegółowy-opis)
5. [Modele ML](#5-modele-ml)
6. [API Reference — wszystkie endpointy](#6-api-reference)
7. [Jak uruchomić](#7-jak-uruchomić)
8. [Workflow treningu własnego modelu](#8-workflow-treningu)
9. [Co dalej — możliwe rozszerzenia](#9-co-dalej)

---

## 1. Czym jest ten system

System rozpoznaje pozycje szachowe z kamery zamontowanej **pionowo nad szachownicą**.  
Zamiast samodzielnie analizować każde zdjęcie od zera, system:

1. **Kalibruje** perspektywę (prostuje zniekształcenia kamery)  
2. **Wykrywa** szachownicę (YOLO lub ręczna kalibracja)  
3. **Klasyfikuje** czy każde z 64 pól jest zajęte (CNN lub wariancja pikseli)  
4. **Wnioskuje** jaki ruch został wykonany (przed/po porównanie)  
5. **Zarządza** stanem gry (python-chess) i integruje z Stockfishem  

---

## 2. Jak działa — przepływ danych

```
Kamera IP (192.168.0.107:8080)
        │
        │  MJPEG/JPEG snapshot
        ▼
┌─────────────────────────────┐
│       camera.py             │  pobiera klatkę BGR (numpy array)
│  fetch_snapshot()           │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│      calibration.py         │  homografia: warpPerspective 800×800px
│  apply_warp(frame)          │  4 punkty ręczne lub wykryte przez YOLO
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│    board_occupancy.py       │  dzieli 800×800 na 64 komórki 100×100
│  analyze_board(warped)      │  każda komórka → score (CNN lub std/mean)
│                             │  score > threshold → pole zajęte
└────────────┬────────────────┘
             │  bool[64] + score[64]
             ▼
┌─────────────────────────────┐
│     move_detector.py        │  maszyna stanów: IDLE → IN_MOVE → STABLE
│  process_frame(occupancy)   │  czeka aż plansza się ustabilizuje
│                             │  (ręka gracza nie zakłóca detekcji)
└────────────┬────────────────┘
             │  (before_mask, after_mask)
             ▼
┌─────────────────────────────┐
│    move_inference.py        │  porównuje before vs after
│  infer_move(before, after)  │  dedukcja ruchu: e2e4, O-O, en passant...
│                             │  walidacja przez python-chess legal_moves
└────────────┬────────────────┘
             │  UCI string np. "e2e4"
             ▼
┌─────────────────────────────┐
│      game_state.py          │  chess.Board singleton (thread-safe Lock)
│  push(move_uci)             │  aktualizacja FEN, historia, tura
└────────────┬────────────────┘
             │  FEN string
             ▼
┌─────────────────────────────┐
│   Stockfish (main.py)       │  ocena pozycji centypionkami
│  /cv/evaluate-current       │  najlepszy ruch UCI
└─────────────────────────────┘
```

---

## 3. Struktura plików

```
backend/
├── main.py                         ← FastAPI app, integracja z cv_router
├── requirements.txt                ← wszystkie zależności
├── .env                            ← CAMERA_HOST, STOCKFISH_PATH itp.
├── ARCHITECTURE.md                 ← ten plik
│
└── cv/                             ← główny moduł Computer Vision
    ├── __init__.py
    ├── config.py                   ← JEDNA centralna konfiguracja
    ├── models.py                   ← Pydantic request/response modele
    ├── camera.py                   ← HTTP klient do kamery IP
    ├── calibration.py              ← homografia + zapis/odczyt kalibracji
    ├── board_occupancy.py          ← analiza 64 pól (CNN lub wariancja)
    ├── game_state.py               ← chess.Board singleton (thread-safe)
    ├── move_inference.py           ← dedukcja ruchu z before/after masek
    ├── move_detector.py            ← maszyna stanów detekcji ruchu
    ├── router.py                   ← wszystkie /cv/* endpointy FastAPI
    │
    └── ml/                         ← moduł ML (modele + dane + trening)
        ├── __init__.py
        ├── board_detector.py       ← inference YOLOv8 dla detekcji planszy
        ├── square_classifier.py    ← inference CNN dla pól (occupied/empty)
        │
        ├── data/
        │   ├── __init__.py
        │   ├── collector.py        ← auto-label podczas gry + zapis patchy
        │   ├── augment.py          ← pipeline augmentacji (albumentations)
        │   └── dataset/            ← zbierane dane (gitignore *.jpg)
        │       ├── occupied/
        │       └── empty/
        │
        ├── training/
        │   ├── __init__.py
        │   ├── config.yaml         ← hyperparametry treningu CNN
        │   ├── train_board.py      ← fine-tuning YOLOv8n (ultralytics)
        │   └── train_classifier.py ← trening CNN (PyTorch)
        │
        └── weights/                ← wagi modeli (gitignore *.pt/*.pth)
            ├── board_detector.pt   ← wagi YOLO (generowane przez train_board.py)
            └── square_classifier.pth ← wagi CNN (generowane przez train_classifier.py)
```

---

## 4. Moduły — szczegółowy opis

### `config.py` — Centralna konfiguracja
Jeden plik, zero hardkodowanych wartości w logice.

| Stała | Wartość | Do czego |
|-------|---------|---------|
| `CAMERA_HOST` | `192.168.0.107` | IP webcam |
| `BOARD_SIZE_PX` | `800` | rozmiar warped obrazu w px |
| `CELL_SIZE_PX` | `100` | 800/8 = 100px na pole |
| `CELL_MARGIN_PX` | `15` | margines wewnątrz pola (unika krawędzi) |
| `OCCUPANCY_VARIANCE_THRESHOLD` | `580.0` | fallback próg wariancji |
| `OCCUPANCY_STABILITY_FRAMES` | `3` | ile klatek stabilności do zatwierdzenia ruchu |
| `CALIBRATION_PATH` | `cv/calibration.json` | zapis homografii |
| `BOARD_DETECTOR_WEIGHTS` | `cv/ml/weights/board_detector.pt` | wagi YOLO |
| `SQUARE_CLASSIFIER_WEIGHTS` | `cv/ml/weights/square_classifier.pth` | wagi CNN |
| `DATASET_DIR` | `cv/ml/data/dataset` | folder z próbkami |

---

### `camera.py` — Klient kamery IP

Cztery funkcje publiczne:

```python
fetch_snapshot() → np.ndarray          # pobiera jeden frame BGR
frame_to_base64(frame) → str           # konwertuje do base64 JPEG dla API
is_camera_reachable() → bool           # health check (HEAD request)
open_stream() → ContextManager         # stream MJPEG (generator klatek)
```

**Dlaczego IP webcam a nie USB?**  
Kamera telefonu przez DroidCam/IP Webcam daje wyższą rozdzielczość i jest łatwiej
repozytojonowana nad szachownicą niż webcam USB.

---

### `calibration.py` — Perspektywa

**Problem**: Kamera pod kątem zniekształca szachownicę (trapez zamiast kwadratu).  
**Rozwiązanie**: Homografia — transformacja perspektywiczna 4-punktów na kwadrat 800×800px.

```python
calibrate_manual(corners: list[list[float]]) → dict
# corners = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] — rogi szachownicy w pikselach
# kolejność: lewy-górny, prawy-górny, prawy-dolny, lewy-dolny

apply_warp(frame) → np.ndarray         # aplikuje homografię na klatkę
save_calibration(data) → None          # zapisuje JSON na dysk
load_calibration() → None              # ładuje przy starcie serwera
get_calibration_status() → dict        # czy skalibrowane, kiedy, skąd rogi
```

**Singleton**: `_homography_matrix` trzymany w pamięci. Przy restarcie serwera
`on_startup()` w `router.py` wczytuje `calibration.json` jeśli istnieje.

---

### `board_occupancy.py` — Analiza pól

To **serce systemu** — decyduje które z 64 pól jest zajęte.

**Pipeline per pole:**
```
warped 800×800
    → podziel na 64 komórki 100×100
    → każda komórka: obetnij margines 15px (70×70px "core")
    → preprocessing: cv2.GaussianBlur(5,5)   [usuwa szum kamery]
    → JEŚLI CNN załadowane:
        → CNN predict(patch) → p(occupied) ∈ [0,1]
        → p > 0.5 → zajęte
    → FALLBACK (brak CNN):
        → score = np.std(cell) / np.mean(cell)  [coefficient of variation]
        → score > OCCUPANCY_VARIANCE_THRESHOLD/1000 → zajęte
```

**Dlaczego coefficient of variation (std/mean) a nie zwykła wariancja?**  
Pole z figurką ma więcej tekstury (kontrast czerń/biel figury) niezależnie od oświetlenia.
Wariancja zależy od absolutnej jasności — białe puste pole ma wysoką wariancję przez
szorstką fakturę, coefficient of variation to normalizuje.

**Debug endpoint**: `GET /cv/snapshot/debug` rysuje siatkę z wynikami na obrazie.

---

### `game_state.py` — Stan gry

Singleton `chess.Board` z `threading.Lock`.  
Dlaczego Lock? FastAPI uruchamia handlery asynchronicznie — bez locka można
zapis do Board w środku innego zapisu.

```python
reset()                         # nowa partia (pozycja startowa)
push(move_uci: str) → bool      # wykonaj ruch e2e4, O-O itp.
get_fen() → str                 # aktualny FEN
get_legal_moves_uci() → list    # dostępne ruchy aktualnej strony
get_status() → dict             # kto gra, czy szach-mat, FEN
```

---

### `move_inference.py` — Dedukcja ruchu

Porównuje dwa zestawy zajętych pól (before, after) i dedukuje UCI.

**Obsługiwane przypadki:**
- **Zwykły ruch**: jedno pole opuszczone + jedno zajęte nowe
- **Bicie**: jedno pole opuszczone + pole które było zajęte przez wroga
- **Roszada**: Król e1→g1 AND Wieża h1→f1 (wykrywane razem)
- **En passant**: Pion bije po przejściu — opuszcza kolumnę, bita figura znika
- **Promocja**: Pion na 8 rzędzie → domyślnie Hetman (=q)

Wynik jest walidowany przez `board.is_legal(move)` z python-chess.

---

### `move_detector.py` — Maszyna stanów

Problem: Ręka gracza nad planszą powoduje fałszywe wykrycia (zasłania pola).

```
IDLE ──────────────── zmiana occupancy ────────────► IN_MOVE
  ▲                                                      │
  │                                              plansza stabilna
  │                                            przez N klatek (N=3)
  │                                                      │
  └──────────────── ruch zatwierdzony ◄──────────── STABLE_AFTER
                   wywołuje infer_move()
```

**Klucz**: `OCCUPANCY_STABILITY_FRAMES = 3` — po zabraniu ręki system czeka
3 identyczne klatki zanim zatwierdzi ruch. To filtruje drgania.

---

## 5. Modele ML

### Model 1: Board Detector (YOLOv8n)

| Atrybut | Wartość |
|---------|---------|
| Architektura | YOLOv8 nano (najszybszy wariant) |
| Zadanie | Object detection — bounding box szachownicy |
| Wejście | Frame BGR dowolnej rozdzielczości |
| Wyjście | Bounding box (x1,y1,x2,y2) + 4 rogi z goodFeaturesToTrack |
| Wagi startowe | `yolov8n.pt` (pretrained COCO, pobierane automatycznie) |
| Twój plik | `cv/ml/weights/board_detector.pt` |
| Trening | `python -m cv.ml.training.train_board` |
| Dataset | ~100-150 zdjęć + annotacje YOLO (1 klasa: chessboard) |

**Kiedy YOLO zastępuje ręczną kalibrację?**  
Gdy wagi `board_detector.pt` istnieją, `calibration.calibrate_auto()` wywołuje
`board_detector.detect_board_corners()` zamiast `findChessboardCorners`.
To eliminuje potrzebę ręcznego wpisywania koordynatów.

---

### Model 2: Square Classifier (SquareCNN)

| Atrybut | Wartość |
|---------|---------|
| Architektura | 3× Conv+BN+ReLU+MaxPool → FC(256) → Sigmoid |
| Zadanie | Klasyfikacja binarna: occupied (1) / empty (0) |
| Wejście | Patch 70×70px (grayscale), normalizowany [0,1] |
| Wyjście | p(occupied) ∈ [0,1] |
| Wagi | `cv/ml/weights/square_classifier.pth` |
| Trening | `python -m cv.ml.training.train_classifier` |
| Dataset | Auto-zbierany przez `/cv/ml/collect` podczas gry |

**Architektura szczegółowo:**
```
Wejście: (batch, 1, 70, 70)

Block 1: Conv2d(1→32, 3×3, pad=1) + BN + ReLU + MaxPool(2)  → (batch, 32, 35, 35)
Block 2: Conv2d(32→64, 3×3, pad=1) + BN + ReLU + MaxPool(2) → (batch, 64, 17, 17)
Block 3: Conv2d(64→128, 3×3, pad=1) + BN + ReLU + MaxPool(2)→ (batch, 128, 8, 8)

Flatten: 128*8*8 = 8192
FC(8192→256) + ReLU + Dropout(0.5)
FC(256→1) + Sigmoid

Wyjście: (batch, 1) — wartość ∈ [0,1]
```

**Dlaczego Dropout(0.5)?**  
Dataset jest mały (~2000-6000 próbek). Dropout regularyzuje — wymusza sieć
do nie polegania na konkretnych neuronach, co poprawia generalizację.

---

## 6. API Reference

Wszystkie endpointy mają prefiks `/cv`. Interaktywna dokumentacja: http://localhost:8000/docs

### Health & Snapshot

| Metoda | URL | Opis |
|--------|-----|------|
| `GET` | `/cv/health` | Status kamery, kalibracji, modeli ML |
| `GET` | `/cv/snapshot` | Surowa klatka z kamery (base64 JPEG) |
| `GET` | `/cv/snapshot/warped` | Klatka po warpPerspective (prostowana) |
| `GET` | `/cv/snapshot/debug` | Klatka z siatką pól + scoring każdego pola |

### Kalibracja

| Metoda | URL | Body | Opis |
|--------|-----|------|------|
| `POST` | `/cv/calibrate` | `{"corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}` | Ręczna kalibracja 4 rogów |
| `GET` | `/cv/calibration` | — | Sprawdź status kalibracji |
| `DELETE` | `/cv/calibration` | — | Resetuj kalibrację |

**Kolejność rogów:** lewy-górny → prawy-górny → prawy-dolny → lewy-dolny

### Analiza planszy

| Metoda | URL | Opis |
|--------|-----|------|
| `GET` | `/cv/occupancy` | 64-elementowa lista: które pola zajęte (bool + score) |

### Gra

| Metoda | URL | Body | Opis |
|--------|-----|------|------|
| `GET` | `/cv/game/state` | — | FEN, tura, historia, status gry |
| `POST` | `/cv/game/reset` | `{"fen": "..."}` (opcjonalnie) | Nowa partia |
| `POST` | `/cv/game/move` | `{"move_uci": "e2e4"}` | Ręczne wpisanie ruchu |
| `POST` | `/cv/game/detector/start` | — | Uruchamia detektor ruchu (zapamiętuje before) |
| `POST` | `/cv/game/detector/tick` | — | Jedna klatka detekcji (wywołuj co ~300ms) |
| `POST` | `/cv/evaluate-current` | — | Stockfish ocena aktualnej pozycji |

### ML — zbieranie danych

| Metoda | URL | Opis |
|--------|-----|------|
| `POST` | `/cv/ml/collect` | Zbiera 64 patche z aktualnej klatki + auto-label z FEN |
| `GET` | `/cv/ml/dataset/stats` | Ile próbek zebrano (occupied/empty/total) |

---

## 7. Jak uruchomić

### Wymagania
```bash
# Python 3.11+
python --version

# Zainstaluj zależności
cd backend
pip install -r requirements.txt

# Opcjonalnie (do treningu modeli — duże biblioteki)
pip install torch torchvision ultralytics albumentations
```

### Start serwera
```powershell
cd D:\ML-Chess\backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Weryfikacja
```bash
curl http://localhost:8000/cv/health
# Oczekiwany output:
# {
#   "camera_reachable": true/false,
#   "calibrated": true/false,
#   "board_detector_loaded": true/false,   # tylko jeśli wagi istnieją
#   "square_classifier_loaded": true/false  # tylko jeśli wagi istnieją
# }
```

---

## 8. Workflow treningu

### Faza 1 — Board Detector (YOLOv8)

> Celem jest eliminacja ręcznej kalibracji. Po treningu kamera automatycznie
> wykrywa szachownicę.

**Krok 1**: Zbierz ~100 zdjęć szachownicy w różnych warunkach.  
**Krok 2**: Annotuj w [Roboflow](https://roboflow.com) (format YOLO, 1 klasa: `chessboard`).  
**Krok 3**: Pobierz i rozpakuj do:
```
backend/cv/ml/data/board_dataset/
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```
**Krok 4**: Uruchom trening:
```bash
cd backend
python -m cv.ml.training.train_board
```
**Krok 5**: Wagi `board_detector.pt` pojawiają się w `cv/ml/weights/`.  
**Krok 6**: Restart serwera → `GET /cv/health` → `board_detector_loaded: true`

---

### Faza 2 — Square Classifier (CNN)

> To twój własny model! Rozpoznaje czy pole jest zajęte lepiej niż wariancja.

**Krok 1**: Zainstaluj szachownicę pod kamerą. Skalibruj raz ręcznie.

**Krok 2**: Zbierz dane podczas gry:
```bash
# Ustaw pozycję startową
POST /cv/game/reset

# Zacznij zbierać (wywołuj po KAŻDYM ruchu)
POST /cv/ml/collect   ← zapisuje 64 patche × 5 wariantów = 320 plików

# Sprawdź postęp
GET /cv/ml/dataset/stats
# Cel: co najmniej 500 occupied + 500 empty
```

**Krok 3**: Wytrenuj:
```bash
cd backend
python -m cv.ml.training.train_classifier
# Czas: ~5 min CPU, ~1 min GPU
# Output: cv/ml/weights/square_classifier.pth
```

**Krok 4**: Restart serwera → CNN aktywne → `GET /cv/snapshot/debug` pokaże `method=cnn`

---

### Monitorowanie treningu

Podczas `train_classifier.py` widzisz w terminalu:
```
Epoch   1/25 | Train loss: 0.6234 acc: 61.2% | Val loss: 0.5891 acc: 68.4%
Epoch   2/25 | Train loss: 0.4123 acc: 78.1% | Val loss: 0.3456 acc: 83.2%
...
Epoch  15/25 | Train loss: 0.0234 acc: 98.7% | Val loss: 0.0312 acc: 97.8% ← BEST
[Early stopping] Brak poprawy przez 7 epok.
```

**Dobry wynik:** val_acc > 95%, val_loss < 0.05  
**Jeśli val_acc < 80%**: zbierz więcej danych (minimum 1000 próbek każdej klasy).

---

## 9. Co dalej — możliwe rozszerzenia

| Feature | Złożoność | Opis |
|---------|-----------|------|
| **Rozpoznawanie figur** | ⭐⭐⭐ | CNN multiclass (13 klas: puste + 6 biel + 6 czerń) — Wymaga większego datasetu (~5000 zdjęć) |
| **Detekcja z klonu FEN** | ⭐⭐ | Porównanie FEN z poprzedniego i aktualnego → szybszy ruch inference |
| **Automatyczna kalibracja** | ⭐⭐ | Po wytrenowaniu Board Detector — zero ręcznej pracy |
| **Frontend live stream** | ⭐⭐⭐ | WebSocket z live kamerą + overlay wyników |
| **Multi-kamera** | ⭐⭐⭐⭐ | Dwie kamery z różnych kątów → eliminacja ślepych pól |
| **Rozpoznawanie czasu** | ⭐ | OCR na zegarze szachowym (python-chess wspiera) |
| **ONNX export** | ⭐⭐ | Konwersja modeli do ONNX → szybszy CPU inference (bez PyTorch runtime) |

---

*Ostatnia aktualizacja: Etap 4 — wszystkie pliki ML napisane, serwer działa.*
