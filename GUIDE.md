# Chess Vision — Kompletny Przewodnik Użytkownika

> Przeczytaj ten plik **od góry do dołu** i wykonuj zadania po kolei.  
> Każda sekcja to osobna faza. Nie pomijaj żadnej — każda kolejna zależy od poprzedniej.

---

## Spis faz

| # | Faza | Co osiągasz |
|---|------|-------------|
| 0 | [Przygotowanie środowiska](#faza-0--przygotowanie-środowiska) | Działający serwer backend |
| 1 | [Ustawienie kamery](#faza-1--ustawienie-kamery) | Kamera widzi szachownicę dobrze |
| 2 | [Kalibracja ręczna](#faza-2--kalibracja-ręczna-perspektywy) | Szachownica jest prostokątem w obrazie |
| 3 | [Zbieranie danych do CNN](#faza-3--zbieranie-danych-treningowych-do-cnn) | Dataset occupied/empty na dysku |
| 4 | [Trening Square Classifier](#faza-4--trening-square-classifier-cnn) | Model CNN rozpoznaje zajęte pola |
| 5 | [Labelowanie dla YOLO](#faza-5--labelowanie-szachownicy-dla-yolo-opcjonalne) | Automatyczna kalibracja bez ręcznych rogów |
| 6 | [Trening Board Detector](#faza-6--trening-board-detector-yolo-opcjonalne) | YOLO wykrywa szachownicę automatycznie |
| 7 | [Weryfikacja całości](#faza-7--weryfikacja-całego-systemu) | Ruch jest wykrywany poprawnie |
| 8 | [Frontend](#faza-8--uruchomienie-frontendu) | Widzisz wynik w przeglądarce |

---

## Faza 0 — Przygotowanie środowiska

### Zadanie 0.1 — Sprawdź Python i wirtualne środowisko

- [ ] Otwórz terminal PowerShell w katalogu projektu:
  ```powershell
  cd D:\ML-Chess
  ```
- [ ] Sprawdź wersję Pythona (wymagane 3.11+):
  ```powershell
  python --version
  ```
- [ ] Jeśli nie masz `.venv`, utwórz je:
  ```powershell
  cd backend
  python -m venv .venv
  ```
- [ ] Aktywuj środowisko wirtualne:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  > Jeśli pojawi się błąd z polityką wykonywania skryptów, uruchom:
  > `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### Zadanie 0.2 — Zainstaluj zależności

- [ ] Zainstaluj wszystkie pakiety bazowe:
  ```powershell
  pip install -r requirements.txt
  ```
- [ ] Zainstaluj pakiety ML (duże pobieranie ~3 GB — raz):
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install ultralytics albumentations
  ```
  > `--index-url cpu` oznacza build bez CUDA — mniejszy plik, wystarcza do klasyfikacji.  
  > Jeśli masz NVIDIA GPU, pomiń flagę `--index-url` i pobierze się wersja z CUDA.

### Zadanie 0.3 — Uruchom serwer backend

- [ ] Uruchom FastAPI:
  ```powershell
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  ```
- [ ] Zaczekaj aż zobaczysz w terminalu:
  ```
  INFO:     Application startup complete.
  ```
- [ ] Otwórz w przeglądarce: http://localhost:8000/docs
  - Powinieneś zobaczyć Swagger UI ze wszystkimi endpointami `/cv/*`

### Zadanie 0.4 — Zweryfikuj połączenie z kamerą

- [ ] W przeglądarce otwórz: http://localhost:8000/cv/health
- [ ] Sprawdź odpowiedź JSON:
  ```json
  {
    "camera_reachable": true,     ← musi być true
    "calibrated": false,          ← na tym etapie false to OK
    "board_detector_loaded": false,
    "square_classifier_loaded": false
  }
  ```
- [ ] Jeśli `camera_reachable: false`:
  - Sprawdź czy kamera IP (telefon) jest uruchomiona: http://192.168.0.107:8080
  - Upewnij się że telefon i komputer są w tej samej sieci Wi-Fi
  - Zmień adres IP w `backend/cv/config.py` → stała `CAMERA_HOST`

---

## Faza 1 — Ustawienie kamery

### Zadanie 1.1 — Fizyczne ustawienie kamery

- [ ] Zamontuj kamerę **pionowo nad szachownicą**, co najmniej 40-50 cm nad nią
- [ ] Kamera powinna patrzeć **prostopadle w dół** — szachownica powinna wyglądać jak kwadrat, nie jak trapez
- [ ] Upewnij się że:
  - Cała szachownica jest widoczna (bez obciętych rogów)
  - Oświetlenie jest równomierne — brak silnych cieni jednego rogu
  - Szachownica jest ustawiona na płaskiej powierzchni
  - Kamera nie drgnie — przykręć lub połóż stabilnie

### Zadanie 1.2 — Sprawdź podgląd kamery

- [ ] Otwórz w przeglądarce podgląd kamery: http://192.168.0.107:8080
- [ ] Zrób zdjęcie testowe przez API:
  ```
  GET http://localhost:8000/cv/snapshot
  ```
  Zwróci JSON z polem `image_b64` — skopiuj wartość i wklej na https://base64.guru/converter/decode/image
- [ ] Na zdjęciu sprawdź:
  - Wszystkie 4 rogi szachownicy są widoczne
  - Pola są czytelne (nie zamazane, nie prześwietlone)
  - Figurki (jeśli są) są wyraźne

### Zadanie 1.3 — Ustal orientację

- [ ] Zdecyduj które narożniki są którymi rogami szachownicy:
  - **Lewy-górny** = pole A8 (biała lewa strona, czarna prawa — lub odwrotnie, jak ustawisz)
  - Nie ma znaczenia która strona jest biała, ale zapamiętaj układ
- [ ] Zapisz sobie na kartce orientację: "białe po lewej, czarne po prawej" lub odwrotnie
  > To ważne przy interpreacji FEN — pole A1 to zawsze lewy-dolny narożnik z perspektywy białych.

---

## Faza 2 — Kalibracja ręczna perspektywy

> **Cel**: System przetransformuje obraz z kamery tak, żeby szachownica była idealnym
> kwadratem 800×800 px. To eliminuje zniekształcenia perspektywiczne.

### Zadanie 2.1 — Pobierz aktualne zdjęcie i znajdź koordynaty rogów

- [ ] Otwórz podgląd kamery lub użyj Swagger UI: `GET /cv/snapshot`
- [ ] Zapisz zdjęcie i otwórz je w **programie z podglądem koordynatów pikseli**:
  - **Windows**: otwórz w Paint → najedź myszką na każdy róg → odczytaj koordynaty w pasku stanu (dół ekranu)
  - **Alternatywa**: GIMP, IrfanView, lub strona https://www.imgonline.com.ua/get-pixel-color.php
- [ ] Znajdź pikselowe koordynaty **4 wewnętrznych rogów szachownicy** (gdzie linie się przecinają):

  ```
  Kolejność rogów (WAŻNE — nie pomyl):
  
  1. Lewy-GÓRNY  (A8 z perspektywy białych)
  2. Prawy-GÓRNY (H8)
  3. Prawy-DOLNY (H1)
  4. Lewy-DOLNY  (A1)
  ```
  
  > **Wskazówka**: Zaznacz **sam wewnętrzny narożnik** pola, nie środek pola, nie krawędź ramki.

### Zadanie 2.2 — Wyślij żądanie kalibracji

- [ ] Otwórz Swagger UI: http://localhost:8000/docs
- [ ] Znajdź endpoint `POST /cv/calibrate`
- [ ] Kliknij "Try it out" i wpisz koordynaty (przykład z poprzedniej sesji):
  ```json
  {
    "corners": [
      [916, 129],
      [1998, 148],
      [1971, 1251],
      [879, 1226]
    ]
  }
  ```
  > Zastąp liczby swoimi koordynatami z Zadania 2.1

- [ ] Kliknij Execute i sprawdź odpowiedź:
  ```json
  {
    "ok": true,
    "message": "Kalibracja zapisana",
    "source": "manual"
  }
  ```

### Zadanie 2.3 — Zweryfikuj kalibrację wizualnie

- [ ] Wywołaj: `GET /cv/snapshot/warped`
- [ ] Zdekoduj base64 i obejrzyj zdjęcie
- [ ] Sprawdź:
  - [ ] Szachownica wypełnia **cały obraz** od krawędzi do krawędzi
  - [ ] Linie siatki są **równoległe** (nie skrzywione)
  - [ ] Pola są **kwadratami**, nie prostokątami
  - [ ] Orientacja jest poprawna (A1 w lewym-dolnym rogu jeśli tak ustawiłeś kamerę)

- [ ] Jeśli obraz wygląda źle (skrzywiony, odwrócony):
  - Zmień kolejność rogów — spróbuj zacząć od innego narożnika
  - Najpopularniejszy błąd: zamienione lewy-górny z prawym-górnym

### Zadanie 2.4 — Test wstępnej detekcji zajętości

- [ ] Ustaw szachownicę w **pozycji startowej** (wszystkie figury na swoich miejscach)
- [ ] Wywołaj: `GET /cv/occupancy`
- [ ] Sprawdź wynik:
  ```json
  {
    "occupied_count": 32,   ← musi być DOKŁADNIE 32
    "empty_count": 32,
    ...
  }
  ```
- [ ] Jeśli liczba jest inna niż 32:
  - Sprawdź `GET /cv/snapshot/debug` — zobaczysz obraz z siatką pokazującą wyniki per pole
  - Jeśli wiele pustych pól jest oznaczonych jako zajęte: otwórz `backend/cv/config.py`
    i zwiększ `OCCUPANCY_VARIANCE_THRESHOLD` (spróbuj 700, 800, 1000)
  - Jeśli zajęte pola są oznaczane jako puste: zmniejsz próg
  - Zrestartuj serwer po każdej zmianie konfiguracji
  - **Cel na tym etapie**: 32/64 przy pełnej planszy. Nie musi być idealne — CNN to poprawi.

---

## Faza 3 — Zbieranie danych treningowych do CNN

> **Cel**: Zebrać dataset zdjęć pól szachownicy z automatycznym oznaczaniem (labeling).
> Nie musisz ręcznie oznaczać żadnego zdjęcia — system wie z FEN które pole jest zajęte.

### Zadanie 3.1 — Zrozum jak działa auto-labeling

Mechanizm jest prosty:
- Strona `POST /cv/ml/collect` robi zdjęcie, warps je, wycina 64 patche 70×70px
- Sprawdza aktualny FEN (stan gry) i wie dokładnie które pola są zajęte
- Patch z zajętego pola → zapisuje do `cv/ml/data/dataset/occupied/`
- Patch z pustego pola → zapisuje do `cv/ml/data/dataset/empty/`
- Każdy patch jest augmentowany ×4 (jasność, szum, blur, obrót)
- Jedno wywołanie `collect` = 64 × 5 wariantów = **320 plików JPEG**

### Zadanie 3.2 — Zaplanuj sesje zbierania

Cel: minimum **500 próbek occupied** i **500 próbek empty** (im więcej tym lepiej).

Oblicz ile sesji potrzebujesz:
- 1 wywołanie `/cv/ml/collect` = 32 occupied + 32 empty × 5 augmentacji = 160 + 160
- **Minimum 4 wywołania** = 640+640 (wystarczy do treningu)
- **Zalecane 10+ wywołań** z różnych pozycji = lepsze uogólnienie modelu

Zbieraj z różnorodnych pozycji:
```
✓ Pozycja startowa (wszystkie figury)
✓ Środek partii (mieszane pola zajęte/puste)
✓ Końcówka (mało figur — więcej pustych pól)
✓ Różne pory dnia (inne oświetlenie)
✓ Z figurkami w różnych miejscach planszy
```

### Zadanie 3.3 — Sesja zbierania danych

Wykonaj te kroki dla każdej pozycji szachowej:

- [ ] **Krok A** — Ustaw pozycję fizyczną: rozstaw figury na planszy
- [ ] **Krok B** — Zsynchronizuj stan gry: wywołaj `POST /cv/game/reset` żeby wirtualna plansza była czysta
  
  > Jeśli grasz partię krok po kroku:
  > ```
  > POST /cv/game/reset              ← reset do pozycji startowej
  > POST /cv/game/move {"move_uci": "e2e4"}   ← wpisz ruch który właśnie wykonałeś
  > POST /cv/game/move {"move_uci": "e7e5"}   ← i kolejne
  > ```
  > FEN musi odzwierciedlać DOKŁADNIE to co jest na fizycznej planszy.

- [ ] **Krok C** — Zbierz dane z tej pozycji:
  ```
  POST /cv/ml/collect
  ```
  Powtórz ten krok 2-3 razy z tej samej pozycji (różne klatki, drobne różnice oświetlenia).

- [ ] **Krok D** — Zmień pozycję i wróć do Kroku A

### Zadanie 3.4 — Monitoruj postęp zbierania

- [ ] Regularnie sprawdzaj ile masz próbek:
  ```
  GET /cv/ml/dataset/stats
  ```
  Przykładowy wynik:
  ```json
  {
    "occupied_count": 480,
    "empty_count": 480,
    "total": 960,
    "dataset_dir": "D:\\ML-Chess\\backend\\cv\\ml\\data\\dataset"
  }
  ```
- [ ] Kontynuuj zbieranie aż osiągniesz minimum:
  - `occupied_count` > 500
  - `empty_count` > 500
  - Najlepiej > 1000 każdego

### Zadanie 3.5 — Sprawdź zebrane pliki

- [ ] Otwórz folder `D:\ML-Chess\backend\cv\ml\data\dataset\occupied\`
- [ ] Kliknij na kilka plików JPEG i sprawdź wizualnie:
  - Powinieneś widzieć fragment pola z figurką na nim
  - Nie powinieneś widzieć całego pola — tylko środek (margines 15px obcięty)
- [ ] Otwórz folder `dataset\empty\`:
  - Powinieneś widzieć puste pola szachownicy (tylko wzór)

---

## Faza 4 — Trening Square Classifier (CNN)

> **Cel**: Wytrenować własny model CNN który rozpozna zajęte/puste pola z wysoką dokładnością.

### Zadanie 4.1 — Zrozum architekturę modelu

Model `SquareCNN` to prosta sieć konwolucyjna:
```
Wejście: patch 70×70px (grayscale)
     ↓
Conv(32 filtry, 3×3) + BatchNorm + ReLU + MaxPool  → 35×35
Conv(64 filtry, 3×3) + BatchNorm + ReLU + MaxPool  → 17×17
Conv(128 filtry, 3×3) + BatchNorm + ReLU + MaxPool → 8×8
     ↓
Flatten(8192) → FC(256) + Dropout(0.5) → FC(1) + Sigmoid
     ↓
Wyjście: liczba 0.0–1.0 (prawdopodobieństwo że pole jest zajęte)
```

Próg decyzji: > 0.5 → zajęte, ≤ 0.5 → puste.

### Zadanie 4.2 — Opcjonalnie dostosuj hyperparametry

- [ ] Otwórz `backend/cv/ml/training/config.yaml`
- [ ] Dostosuj jeśli potrzebujesz:
  ```yaml
  epochs: 25         # zwiększ do 40 jeśli masz dużo danych
  batch_size: 32     # zmniejsz do 16 jeśli brakuje RAM
  learning_rate: 0.001   # zmniejsz do 0.0003 jeśli loss oscyluje i nie maleje
  patience: 7        # early stopping — ile epok bez poprawy zanim zatrzyma
  ```
- [ ] Przy małym datasecie (< 1000 próbek): zwiększ `patience` do 10

### Zadanie 4.3 — Uruchom trening

- [ ] Otwórz **nowe okno terminala** (nie zatrzymuj serwera w obecnym):
  ```powershell
  cd D:\ML-Chess\backend
  .\.venv\Scripts\Activate.ps1
  python -m cv.ml.training.train_classifier
  ```
- [ ] Obserwuj output w terminalu. Każda linia to jedna epoka:
  ```
  Epoch   1/25 | Train loss: 0.6234 acc: 61.2% | Val loss: 0.5891 acc: 68.4%
  Epoch   2/25 | Train loss: 0.4123 acc: 78.1% | Val loss: 0.3456 acc: 83.2%
  Epoch   3/25 | Train loss: 0.2341 acc: 89.3% | Val loss: 0.2123 acc: 91.1% ← BEST
  ...
  Epoch  15/25 | Train loss: 0.0156 acc: 99.2% | Val loss: 0.0234 acc: 98.7% ← BEST
  [Early stopping] Brak poprawy przez 7 epok.
  
  [Sukces] Wagi zapisane: cv/ml/weights/square_classifier.pth
  ```

### Zadanie 4.4 — Interpretacja wyników treningu

| Wynik | Co to znaczy | Co zrobić |
|-------|-------------|-----------|
| `val_acc > 95%` | Doskonały model | Gotowe — przejdź dalej |
| `val_acc 85-95%` | Dobry model | Możesz użyć, zbierz więcej danych w przyszłości |
| `val_acc < 85%` | Słaby model | Zbierz więcej danych (min. 2000 próbek każdej klasy) |
| Loss oscyluje, nie spada | Problem z lr | Zmniejsz `learning_rate` do 0.0003 w config.yaml |
| `val_loss` rośnie od epoki 5 | Overfitting | Dataset za mały — zbierz więcej lub zwiększ augmentację |

### Zadanie 4.5 — Zaktywuj model w serwerze

- [ ] Wróć do terminala z serwerem i **zatrzymaj go** (Ctrl+C)
- [ ] Uruchom serwer ponownie:
  ```powershell
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  ```
- [ ] Sprawdź że model się załadował:
  ```
  GET http://localhost:8000/cv/health
  ```
  ```json
  {
    "square_classifier_loaded": true   ← musi być true
  }
  ```
- [ ] Sprawdź że CNN jest używane:
  ```
  GET /cv/snapshot/debug
  ```
  Na obrazie zobaczysz napis `method=cnn` przy każdym polu (zamiast `method=variance`).

### Zadanie 4.6 — Przetestuj CNN z pozycją startową

- [ ] Ustaw szachownicę w pozycji startowej
- [ ] Wywołaj: `GET /cv/occupancy`
- [ ] Wynik powinien być **dokładnie 32/64**
- [ ] Przesuń kilka figur — wywołaj occupancy ponownie
- [ ] Sprawdź że model poprawnie wykrywa zmiany

---

## Faza 5 — Labelowanie szachownicy dla YOLO (opcjonalne)

> **Ta faza jest opcjonalna.** Bez niej kalibracja jest ręczna (podajesz 4 punkty).
> Z YOLO kamera automatycznie wykrywa szachownicę — zero wpisywania koordynatów.

### Zadanie 5.1 — Zbierz zdjęcia do datasetu YOLO

- [ ] Zrób **co najmniej 80-120 zdjęć** szachownicy przez endpoint:
  ```
  GET /cv/snapshot
  ```
- [ ] Zapisz zdjęcia jako pliki JPEG na dysku
- [ ] Zadbaj o różnorodność:
  - Różne poziomy oświetlenia (mocne, słabe, boczne)
  - Różne kąty kamery (lekko skrzywiona, prosto)
  - Z figurkami i bez
  - Różne pory dnia
  - Różne otoczenie (inne tło za szachownicą)

### Zadanie 5.2 — Instalacja narzędzia do annotacji: LabelImg

- [ ] Zainstaluj LabelImg:
  ```powershell
  pip install labelImg
  ```
- [ ] Uruchom:
  ```powershell
  labelImg
  ```

### Zadanie 5.3 — Konfiguracja LabelImg

- [ ] Po uruchomieniu LabelImg:
  - Kliknij **"Open Dir"** → wskaż folder ze zdjęciami
  - Kliknij **"Change Save Dir"** → wskaż gdzie zapisywać annotacje (np. ten sam folder)
  - W górnym menu zaznacz format zapisu: **"YOLO"** (nie PascalVOC!)
    > Opcja: `View → Auto Save mode` — ułatwia pracę

### Zadanie 5.4 — Proces labelowania (dla każdego zdjęcia)

> Czas na jedno zdjęcie: ~15-30 sekund. 100 zdjęć = ~30-50 minut.

Dla każdego zdjęcia w LabelImg:

- [ ] **Krok A** — Naciśnij `W` żeby wejść w tryb rysowania bounding boxa
- [ ] **Krok B** — Narysuj prostokąt **wokół całej szachownicy**:
  - Kliknij i przeciągnij od lewego-górnego rogu szachownicy do prawego-dolnego
  - Box powinien obejmować całą szachownicę, ale jak najmniej marginesu poza nią
- [ ] **Krok C** — W okienku które się pojawi wpisz nazwę klasy: `chessboard`
  - Przy pierwszym razem wpisujesz ręcznie, potem wybierasz z listy
- [ ] **Krok D** — Naciśnij `Ctrl+S` żeby zapisać (lub włącz Auto Save)
- [ ] **Krok E** — Naciśnij `D` żeby przejść do następnego zdjęcia
- [ ] Powtarzaj Kroki A-E dla wszystkich zdjęć

### Zadanie 5.5 — Weryfikacja annotacji

Po zakończeniu labelowania:

- [ ] W folderze ze zdjęciami powinny pojawić się pliki `.txt` o tych samych nazwach co zdjęcia
- [ ] Otwórz przykładowy plik `.txt` — powinien wyglądać tak:
  ```
  0 0.512 0.498 0.743 0.821
  ```
  Format: `klasa cx cy w h` (wszystkie wartości 0-1, znormalizowane)
- [ ] Sprawdź kilka annotacji wizualnie w LabelImg — box powinien ciasno otaczać szachownicę

### Zadanie 5.6 — Przygotuj strukturę datasetu YOLO

- [ ] Utwórz strukturę folderów:
  ```powershell
  mkdir backend\cv\ml\data\board_dataset\images\train
  mkdir backend\cv\ml\data\board_dataset\images\val
  mkdir backend\cv\ml\data\board_dataset\labels\train
  mkdir backend\cv\ml\data\board_dataset\labels\val
  ```
- [ ] Podziel zdjęcia: 80% do `train`, 20% do `val`
  - Przy 100 zdjęciach: 80 do train, 20 do val
  - Wybierz losowo (np. co 5. zdjęcie do val)
- [ ] Przekopiuj zdjęcia do `images/train/` i `images/val/`
- [ ] Przekopiuj odpowiadające pliki `.txt` do `labels/train/` i `labels/val/`
- [ ] Każde zdjęcie `abc.jpg` musi mieć plik `abc.txt` w tym samym folderze (tylko w labels/)

---

## Faza 6 — Trening Board Detector (YOLO) — opcjonalne

### Zadanie 6.1 — Uruchom trening

- [ ] W nowym oknie terminala:
  ```powershell
  cd D:\ML-Chess\backend
  .\.venv\Scripts\Activate.ps1
  python -m cv.ml.training.train_board
  ```
- [ ] Trening trwa ~15-30 minut na CPU, ~5 min na GPU
- [ ] Obserwuj output:
  ```
  [Dataset] Trening: 80 zdjęć | Walidacja: 20 zdjęć
  [Config] Zapisano: ...board_dataset/dataset.yaml
  [Model] Ładowanie punktu startowego: yolov8n.pt
  
  Epoch 1/50   ←── postęp treningu
  ...
  [Sukces] Wagi zapisane: cv/ml/weights/board_detector.pt
    mAP50: 0.987
  ```

### Zadanie 6.2 — Aktywuj w serwerze

- [ ] Zrestartuj serwer (Ctrl+C + ponowne uruchomienie)
- [ ] Sprawdź: `GET /cv/health` → `board_detector_loaded: true`
- [ ] Teraz zamiast ręcznych rogów możesz użyć:
  ```
  POST /cv/calibrate
  {"mode": "auto"}
  ```
  YOLO automatycznie wykryje szachownicę i ustawi kalibrację.

---

## Faza 7 — Weryfikacja całego systemu

> Kompletny test end-to-end: od klatki kamery do wykrytego ruchu.

### Zadanie 7.1 — Sprawdź status wszystkich komponentów

- [ ] Wywołaj: `GET /cv/health`
- [ ] Oczekiwany wynik (po przejściu wszystkich faz):
  ```json
  {
    "camera_reachable": true,
    "calibrated": true,
    "board_detector_loaded": true,    ← jeśli robiłeś fazy 5-6
    "square_classifier_loaded": true  ← po fazie 4
  }
  ```

### Zadanie 7.2 — Test detekcji ruchu krok po kroku

- [ ] **Krok 1** — Ustaw pozycję startową na szachownicy
- [ ] **Krok 2** — Zresetuj stan gry:
  ```
  POST /cv/game/reset
  ```
- [ ] **Krok 3** — Sprawdź że API widzi 32 zajęte pola:
  ```
  GET /cv/occupancy
  → occupied_count: 32
  ```
- [ ] **Krok 4** — Uruchom detektor ruchu (zapamiętuje aktualny stan jako "przed"):
  ```
  POST /cv/game/detector/start
  ```
- [ ] **Krok 5** — Wykonaj fizyczny ruch na szachownicy (np. e2→e4)
- [ ] **Krok 6** — Wywołaj tick detekcji kilka razy co ~0.5 sekundy:
  ```
  POST /cv/game/detector/tick
  POST /cv/game/detector/tick
  POST /cv/game/detector/tick
  ```
  Po 3 stabilnych klatkach system powinien zwrócić:
  ```json
  {
    "status": "move_detected",
    "move_uci": "e2e4",
    "fen_after": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
  }
  ```
- [ ] **Krok 7** — Sprawdź aktualny stan gry:
  ```
  GET /cv/game/state
  ```
- [ ] **Krok 8** — Opcjonalnie: poproś Stockfish o ocenę:
  ```
  POST /cv/evaluate-current
  ```

### Zadanie 7.3 — Przetestuj kilka ruchów z rzędu

- [ ] Wykonaj sekwencję 5-10 ruchów, po każdym wywołując:
  1. `POST /cv/game/detector/start` (zapamiętaj "przed")
  2. Wykonaj fizyczny ruch
  3. `POST /cv/game/detector/tick` × 3-5
- [ ] Sprawdź `GET /cv/game/state` — historia powinna zawierać wszystkie ruchy
- [ ] Sprawdź FEN po każdym ruchu — powinien zgadzać się z pozycją na planszy

### Zadanie 7.4 — Diagnostyka błędów detekcji

Jeśli system wykrywa błędny ruch:

| Objaw | Przyczyna | Rozwiązanie |
|-------|-----------|-------------|
| Detektor mówi "brak ruchu" mimo wykonanego ruchu | Za mało stabilizacji | Zwiększ `OCCUPANCY_STABILITY_FRAMES` do 5 w config.py |
| Błędny ruch UCI (np. `a1b1` zamiast `e2e4`) | Zła orientacja szachownicy | Sprawdź kalibrację — pole A1 musi być lewy-dolny |
| `occupied_count` = 33 lub 31 w pozycji startowej | CNN nie jest pewny granicznych pól | Zbierz więcej danych dla problematycznych pól |
| Ruch wykrywany gdy nikt nie rusza | Zmieniające się oświetlenie | Zwiększ `OCCUPANCY_STABILITY_FRAMES` do 7 |

---

## Faza 8 — Uruchomienie frontendu

### Zadanie 8.1 — Sprawdź czy frontend istnieje

- [ ] Sprawdź folder frontend:
  ```powershell
  ls D:\ML-Chess\frontend
  ```
- [ ] Sprawdź czy jest `package.json`:
  ```powershell
  cat D:\ML-Chess\frontend\package.json
  ```

### Zadanie 8.2 — Zainstaluj zależności frontendu

- [ ] Przejdź do folderu frontend:
  ```powershell
  cd D:\ML-Chess\frontend
  ```
- [ ] Zainstaluj pakiety Node.js:
  ```powershell
  npm install
  ```
  > Jeśli nie masz Node.js: pobierz z https://nodejs.org (wersja LTS)

### Zadanie 8.3 — Uruchom frontend

- [ ] Upewnij się że backend działa (terminal 1)
- [ ] W nowym terminalu (terminal 2):
  ```powershell
  cd D:\ML-Chess\frontend
  npm run dev
  ```
- [ ] Poczekaj aż zobaczysz:
  ```
  VITE v5.x.x  ready in XXX ms
  ➜  Local:   http://localhost:5173/
  ```
- [ ] Otwórz w przeglądarce: http://localhost:5173

### Zadanie 8.4 — Weryfikacja połączenia frontend-backend

- [ ] Na stronie frontendowej sprawdź:
  - [ ] Brak błędów CORS w konsoli przeglądarki (F12 → Console)
  - [ ] API calls wychodzą na `localhost:8000` i wracają z kodem 200
  - [ ] Jeśli są błędy CORS: sprawdź w `backend/main.py` czy `allow_origins` zawiera `http://localhost:5173`

### Zadanie 8.5 — Test pełnego workflow przez UI

- [ ] Przez interfejs UI:
  - [ ] Wywołaj kalibrację
  - [ ] Wyświetl podgląd kamery (debug snapshot)
  - [ ] Resetuj grę do pozycji startowej
  - [ ] Wykonaj ruch i sprawdź czy UI go wykrywa
  - [ ] Sprawdź czy Stockfish podaje ocenę

---

## Szybkie komendy — ściągawka

```powershell
# Start środowiska i serwera
cd D:\ML-Chess\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start frontendu (osobne okno)
cd D:\ML-Chess\frontend
npm run dev

# Trening CNN (osobne okno, po zebraniu danych)
cd D:\ML-Chess\backend
.\.venv\Scripts\Activate.ps1
python -m cv.ml.training.train_classifier

# Trening YOLO (osobne okno, po labelowaniu)
cd D:\ML-Chess\backend
.\.venv\Scripts\Activate.ps1
python -m cv.ml.training.train_board
```

```bash
# Kalibracja
POST /cv/calibrate          {"corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}

# Zbieranie danych (wywołuj po każdym ruchu)
POST /cv/ml/collect

# Statystyki datasetu
GET  /cv/ml/dataset/stats

# Detekcja ruchu
POST /cv/game/reset
POST /cv/game/detector/start
POST /cv/game/detector/tick   ← wywołuj co ~0.5s

# Diagnostyka
GET  /cv/health
GET  /cv/occupancy
GET  /cv/snapshot/debug
GET  /cv/game/state
```

---

## Pliki które możesz edytować — co i kiedy

| Plik | Kiedy edytować | Co zmieniać |
|------|---------------|-------------|
| `cv/config.py` | Złe wyniki occupancy | `OCCUPANCY_VARIANCE_THRESHOLD`, `CELL_MARGIN_PX` |
| `cv/config.py` | Fałszywe detekcje ruchów | `OCCUPANCY_STABILITY_FRAMES` (zwiększ do 5-7) |
| `cv/ml/training/config.yaml` | Przed treningiem CNN | `epochs`, `batch_size`, `learning_rate`, `patience` |
| `.env` | Zmiana IP kamery | `CV_CAMERA_HOST=192.168.x.x` |
| `main.py` | Problemy CORS | `allow_origins` — dodaj adres frontendu |

---

## Częste problemy i rozwiązania

### Problem: `camera_reachable: false`
```
1. Otwórz http://192.168.0.107:8080 w przeglądarce
2. Jeśli nie ładuje: sprawdź czy aplikacja na telefonie działa
3. Sprawdź IP telefonu: Ustawienia → WiFi → szczegóły
4. Zmień CV_CAMERA_HOST w backend/cv/config.py
5. Zrestartuj serwer
```

### Problem: `occupied_count` jest różne od 32 przy pozycji startowej
```
1. GET /cv/snapshot/debug — pobierz obraz z siatką
2. Znajdź pola które są błędnie oznaczone
3. Jeśli puste pola = zajęte: zwiększ OCCUPANCY_VARIANCE_THRESHOLD o 100
4. Jeśli zajęte pola = puste: zmniejsz threshold o 100
5. Zrestartuj serwer i testuj ponownie
6. Docelowo: po treningu CNN ten problem zniknie
```

### Problem: trening CNN — `val_acc` utknęło poniżej 85%
```
1. Zbierz więcej danych: minimum 2000 próbek każdej klasy
2. Sprawdź czy dane są poprawne — przejrzyj kilka plików w dataset/occupied/
3. Sprawdź czy FEN był poprawny gdy zbierałeś dane
4. Zmniejsz learning_rate w config.yaml do 0.0003
5. Zwiększ epochs do 40
```

### Problem: ruch nie jest wykrywany (detector/tick zwraca "waiting")
```
1. Sprawdź czy kalibracja jest poprawna (GET /cv/snapshot/warped)
2. Sprawdź occupancy przed i po ruchu (GET /cv/occupancy × 2)
3. Jeśli occupancy się zmienia: zmniejsz OCCUPANCY_STABILITY_FRAMES do 3
4. Jeśli occupancy się NIE zmienia: problem z detekcją pola — zbierz więcej danych
```

### Problem: błędy CORS w przeglądarce
```
1. Otwórz backend/main.py
2. Znajdź CORSMiddleware
3. Dodaj do allow_origins adres frontendu:
   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"]
4. Zrestartuj serwer
```

---

*Dokument wygenerowany dla wersji: Chess Vision v0.3.0-ml | branch: from-above*
