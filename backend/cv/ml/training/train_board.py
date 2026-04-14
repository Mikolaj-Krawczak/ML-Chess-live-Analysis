"""
Skrypt do trenowania modelu detekcji szachownicy (YOLOv8n).

JAK UŻYWAĆ:
-----------
1. Zbierz ~100-150 zdjęć szachownicy z kamery (różne oświetlenie, kąty).
2. Otwórz LabelImg lub Roboflow i annotuj bounding boxy.
   - LabelImg (lokalnie):  pip install labelImg && labelImg
   - Roboflow (online):    https://roboflow.com  (darmowe konto)
   Format annotacji: YOLO (.txt) — jedna linia: "0 cx cy w h" (wartości znormalizowane 0-1)
3. Ułóż dane w strukturze:
       backend/cv/ml/data/board_dataset/
       ├── images/
       │   ├── train/   ← 80% zdjęć
       │   └── val/     ← 20% zdjęć
       └── labels/
           ├── train/   ← pliki .txt odpowiadające zdjęciom
           └── val/
4. Uruchom ten skrypt:
       cd backend
       python -m cv.ml.training.train_board

Wynik: wagi zapisane w cv/ml/weights/board_detector.pt

PARAMETRY:
----------
Możesz dostosować zmienne na początku skryptu (EPOCHS, IMG_SIZE, BATCH) do
swoich możliwości sprzętowych. CPU: zmniejsz BATCH do 4, EPOCHS do 30.
"""

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Parametry treningu — dostosuj do swojego sprzętu
# ---------------------------------------------------------------------------

EPOCHS      = 50      # liczba epok; więcej = lepszy model, dłuższy trening
IMG_SIZE    = 640     # rozmiar wejścia sieci (640 = standard YOLOv8)
BATCH       = 4      # batch size; zmniejsz do 4-6 jeśli brakuje RAM/VRAM
PATIENCE    = 15      # early stopping — zatrzymaj jeśli brak poprawy przez N epok
PRETRAINED  = "yolov8n.pt"  # punkt startowy: nano model z Ultralytics Hub

# Ścieżki
_SCRIPT_DIR = Path(__file__).parent
_ML_DIR     = _SCRIPT_DIR.parent
_DATA_DIR   = _ML_DIR / "data" / "board_dataset"
_WEIGHTS_OUT = _ML_DIR / "weights" / "board_detector.pt"

# Plik konfiguracyjny YOLO dataset (generowany automatycznie)
_YAML_PATH  = _DATA_DIR / "dataset.yaml"


def _check_dataset() -> None:
    """Weryfikuje że struktura datasetu jest poprawna przed treningiem."""
    required = [
        _DATA_DIR / "images" / "train",
        _DATA_DIR / "images" / "val",
        _DATA_DIR / "labels" / "train",
        _DATA_DIR / "labels" / "val",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Brak wymaganych folderów datasetu:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nZobacz instrukcję na początku pliku train_board.py"
        )

    train_imgs = list((_DATA_DIR / "images" / "train").glob("*.jpg")) + \
                 list((_DATA_DIR / "images" / "train").glob("*.png"))
    val_imgs   = list((_DATA_DIR / "images" / "val").glob("*.jpg")) + \
                 list((_DATA_DIR / "images" / "val").glob("*.png"))

    print(f"[Dataset] Trening: {len(train_imgs)} zdjęć | Walidacja: {len(val_imgs)} zdjęć")

    if len(train_imgs) < 20:
        raise ValueError(
            f"Za mało zdjęć treningowych ({len(train_imgs)}). Minimalne zalecane: 80."
        )


def _write_yaml() -> None:
    """Generuje plik dataset.yaml wymagany przez ultralytics."""
    yaml_content = f"""# Konfiguracja datasetu YOLOv8 — detekcja szachownicy
# Wygenerowany automatycznie przez train_board.py

path: {_DATA_DIR.resolve()}
train: images/train
val:   images/val

nc: 1          # liczba klas
names:
  0: chessboard
"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _YAML_PATH.write_text(yaml_content, encoding="utf-8")
    print(f"[Config] Zapisano: {_YAML_PATH}")


def train() -> None:
    """Główna funkcja treningowa."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics nie jest zainstalowane. Uruchom:\n"
            "  pip install ultralytics"
        )

    print("=" * 60)
    print("  Chess Board Detector — trening YOLOv8n")
    print("=" * 60)

    _check_dataset()
    _write_yaml()

    print(f"\n[Model] Ładowanie punktu startowego: {PRETRAINED}")
    model = YOLO(PRETRAINED)

    print(f"\n[Trening] Parametry:")
    print(f"  Epoki:      {EPOCHS}")
    print(f"  Batch size: {BATCH}")
    print(f"  Img size:   {IMG_SIZE}px")
    print(f"  Patience:   {PATIENCE} (early stopping)")
    print()

    # Uruchamiamy trening Ultralytics
    # Wyniki (metryki, wykresy, wagi) zapisywane są do runs/detect/train*/
    results = model.train(
        data=str(_YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        name="chess_board_detector",
        exist_ok=True,
        # Augmentacje wbudowane w YOLOv8 (hsv, flip, mosaic itp.)
        degrees=5.0,     # drobna rotacja — kamera może być lekko skręcona
        flipud=0.0,      # nie odwracamy góra-dół (widok z kamery stały)
        fliplr=0.5,      # flip lewo-prawo OK (szachownica symetryczna)
        mosaic=0.5,      # mozaika: 4 zdjęcia w 1 — pomaga przy małym datasecie
        hsv_h=0.02,      # mała zmiana hue
        hsv_s=0.4,       # zmiana saturacji (różne oświetlenie)
        hsv_v=0.4,       # zmiana jasności (różne pory dnia)
    )

    # Kopiujemy najlepsze wagi do docelowej lokalizacji
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        _WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, _WEIGHTS_OUT)
        print(f"\n[Sukces] Wagi zapisane: {_WEIGHTS_OUT}")
        print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    else:
        print(f"\n[UWAGA] Nie znaleziono wag w {best_weights}. Sprawdź logi treningu.")

    print("\n[Następny krok] Uruchom backend i przetestuj:")
    print("  python -m uvicorn main:app --reload")
    print("  GET http://localhost:8000/cv/health  → board_detector_loaded: true")


if __name__ == "__main__":
    train()
