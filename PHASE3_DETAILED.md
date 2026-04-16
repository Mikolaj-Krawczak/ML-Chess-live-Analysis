# Faza 3 — Szczegółowe wyjaśnienie: Zbieranie danych, Model CNN i Detekcja ruchów

> Ten dokument wyjaśnia **dokładnie** jak działa cały pipeline:
> od fizycznej szachownicy → przez kamerę → do wygenerowanego FEN.
> Napisany prostym językiem, krok po kroku.

---

## Spis treści

1. [Wielki obraz — co robi ten program](#1-wielki-obraz)
2. [Co robi model CNN (i czego NIE robi)](#2-co-robi-model-cnn)
3. [Skąd system zna FEN w trakcie zbierania danych](#3-skąd-system-zna-fen)
4. [Jak działa sprawdzanie zajętości pól (bez CNN)](#4-jak-działa-sprawdzanie-zajętości-bez-cnn)
5. [Dokładne flow zbierania danych — krok po kroku](#5-flow-zbierania-danych)
6. [Augmentacja — skąd te 320 obrazków](#6-augmentacja)
7. [Jak działa detekcja ruchów na żywo](#7-detekcja-ruchów-na-żywo)
8. [Jak generowany jest FEN z pozycji](#8-generowanie-fen)
9. [Najlepsza strategia zbierania datasetu](#9-strategia-zbierania)
10. [Instrukcja krok po kroku — co robić](#10-instrukcja-krok-po-kroku)

---

## 1. Wielki obraz

```
Fizyczna szachownica + kamera IP
        │
        ▼
   Klatka JPEG z kamery (np. 1920×1080)
        │
        ▼
   Warp perspektywy (kalibracja 4 rogów)
        │
        ▼
   Prostokątny obraz 800×800px (szachownica od krawędzi do krawędzi)
        │
        ▼
   Podział na 64 pól (każde 100×100px, po obcięciu marginesu ~70×70px)
        │
        ▼
   CNN klasyfikuje KAŻDE pole: "zajęte" (jest figura) vs "puste" (brak figury)
        │
        ▼
   Porównanie z poprzednim stanem → wykrycie KTÓRE pole się zmieniło
        │
        ▼
   Wnioskowanie ruchu UCI (np. "e2e4") z delty zmian
        │
        ▼
   Aktualizacja wewnętrznego FEN przez python-chess
        │
        ▼
   Wizualizacja + analiza Stockfish
```

### Kluczowa zasada

Model CNN **NIE rozpoznaje figur** (nie wie czy to król, hetman, pionek).
Model CNN odpowiada TYLKO na jedno pytanie per pole:

> **"Czy na tym polu COKOLWIEK stoi, czy jest puste?"**

Odpowiedź: liczba 0.0–1.0. Powyżej 0.5 = zajęte. Poniżej = puste.

Skąd więc system wie JAKA to figura? Z **FEN-u**, który jest śledzony
od samego początku gry przez bibliotekę `python-chess`.

---

## 2. Co robi model CNN

### Architektura

```
Wejście: patch 70×70px w skali szarości (1 kanał)
    ↓
Conv2D(32 filtry, 3×3) + BatchNorm + ReLU + MaxPool → 35×35
Conv2D(64 filtry, 3×3) + BatchNorm + ReLU + MaxPool → 17×17
Conv2D(128 filtry, 3×3) + BatchNorm + ReLU + MaxPool → 8×8
    ↓
Flatten → 8192 neuronów
    ↓
FC(256) + ReLU + Dropout(0.5)
    ↓
FC(1) + Sigmoid
    ↓
Wyjście: p(zajęte) ∈ [0.0, 1.0]
```

### Co model widzi

Model dostaje **mały wycinek** jednego pola (70×70px, czarno-biały).
Na tym wycinku widzi albo:
- gładką teksturę pola (= puste)
- kształt figury na tle pola (= zajęte)

### Czego model NIE robi

- NIE rozpoznaje **typu** figury (król vs pionek vs hetman)
- NIE rozpoznaje **koloru** figury (biała vs czarna)
- NIE generuje FEN
- NIE śledzi ruchów

Jest to **binarny klasyfikator obrazów**: `occupied` vs `empty`.

---

## 3. Skąd system zna FEN

To jest kluczowe pytanie. Odpowiedź:

### Podczas zbierania danych (Faza 3):

FEN jest **ręcznie utrzymywany przez Ciebie** za pomocą API:

```
POST /cv/game/reset          → ustawia FEN startowy (lub dowolny)
POST /cv/game/move {"move_uci": "e2e4"}  → aktualizuje FEN o ten ruch
```

Wewnętrznie system używa biblioteki `python-chess`, która:
- Zna zasady szachów (legalne ruchy, roszada, en passant, promocja)
- Utrzymuje pełny stan planszy w pamięci
- Na żądanie generuje FEN string

**System NIE patrzy na kamerę żeby ustalić FEN podczas zbierania danych!**
FEN pochodzi WYŁĄCZNIE z Twoich komend `game/reset` i `game/move`.

### Podczas gry na żywo (Faza 7):

FEN jest **automatycznie aktualizowany** przez detektor ruchów:
1. System pamięta poprzedni stan zajętości pól (maska "before")
2. Co pół sekundy robi nowe zdjęcie i sprawdza aktualny stan ("after")
3. Porównuje before vs after → wyznacza ruch UCI
4. Pushuje ruch do `python-chess` → nowy FEN

Ale żeby detektor działał, potrzebuje wytrenowanego CNN!
A żeby wytrenować CNN, potrzebujesz danych z Fazy 3.

### Łańcuch zależności:

```
Faza 3: Ty mówisz FEN (game/reset + game/move) → zbierasz dane → trenujesz CNN
Faza 7: CNN sprawdza pola → detektor porównuje → python-chess generuje FEN
```

---

## 4. Jak działa sprawdzanie zajętości bez CNN

Zanim wytrenujesz CNN, system używa **fallback-u opartego na wariancji pikseli**.

### Jak to działa:

1. Bierze wycinek pola (70×70px w skali szarości)
2. Oblicza **wariancję** (rozrzut wartości pikseli):
   - Puste pole = dość jednolita tekstura → **niska wariancja** (np. 200-400)
   - Pole z figurą = figura tworzy kontrastowe krawędzie → **wysoka wariancja** (np. 600-1500)
3. Porównuje z progiem `OCCUPANCY_VARIANCE_THRESHOLD` (domyślnie 580)
4. Wariancja > 580 → "zajęte"; wariancja ≤ 580 → "puste"

### Czemu to nie jest idealne:

- Cienie mogą zwiększyć wariancję pustego pola → fałszywy "zajęte"
- Ciemna figura na ciemnym polu → niska wariancja → fałszywy "puste"
- Oświetlenie się zmienia → próg przestaje pasować

Dlatego potrzebujesz CNN — jest DUŻO lepszy niż zwykły próg wariancji.

**WAŻNE**: Ten fallback wariancji jest używany RÓWNIEŻ podczas zbierania danych
w Fazie 3, ale **nie do labelowania** (etykiety bierze z FEN-u), tylko do
endpointu `/cv/occupancy` jeśli chcesz podejrzeć co system "widzi".

---

## 5. Flow zbierania danych — krok po kroku

### Scenariusz: Rozgrywasz partię i zbierasz dane po każdym ruchu

```
KROK 1: Ustawiasz fizyczną szachownicę w pozycji startowej

KROK 2: POST /cv/game/reset
         → System ustawia wewnętrzny FEN na:
           "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
         → python-chess wie: 32 figury na swoich miejscach

KROK 3: POST /cv/ml/collect  (OPCJONALNIE — zbierasz dane z pozycji startowej)
         → System robi:
           a) Pobiera klatkę JPEG z kamery IP
           b) Warp perspektywy → obraz 800×800px
           c) Konwertuje na grayscale + GaussianBlur
           d) Wycina 64 patche (po jednym na pole) z marginesem 15px
           e) Skaluje każdy patch do 70×70px
           f) Czyta aktualny FEN z game_state → buduje zbiór zajętych pól
              np. {"a1","b1","c1","d1","e1","f1","g1","h1",  ← białe figury
                   "a2","b2","c2","d2","e2","f2","g2","h2",  ← białe pionki
                   "a7","b7","c7","d7","e7","f7","g7","h7",  ← czarne pionki
                   "a8","b8","c8","d8","e8","f8","g8","h8"}  ← czarne figury
           g) Patch z pola "e1" (jest w zbiorze) → ZAPISUJE do dataset/occupied/
              Patch z pola "e4" (nie ma w zbiorze) → ZAPISUJE do dataset/empty/
           h) Każdy patch × 4 augmentacje → 5 plików per pole
         → Wynik: 32×5 = 160 occupied + 32×5 = 160 empty = 320 plików JPEG

KROK 4: Wykonujesz FIZYCZNIE ruch e2→e4 na planszy

KROK 5: POST /cv/game/move {"move_uci": "e2e4"}
         → python-chess aktualizuje FEN:
           "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
         → Teraz e2 jest puste, e4 jest zajęte

KROK 6: POST /cv/ml/collect
         → Dokładnie to samo co KROK 3, ale z NOWYM FEN-em
         → Teraz system wie że e2 = empty, e4 = occupied
         → 320 nowych plików JPEG, tym razem z innym rozkładem figur

KROK 7: Przeciwnik (lub Ty) wykonuje FIZYCZNIE ruch e7→e5

KROK 8: POST /cv/game/move {"move_uci": "e7e5"}
         → FEN: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"

KROK 9: POST /cv/ml/collect
         → 320 nowych plików z kolejną pozycją

... i tak dalej po każdym ruchu.
```

### Co DOKŁADNIE dzieje się w `collect`:

```python
# 1. Kamera → klatka
frame = camera.fetch_snapshot()        # JPEG z http://192.168.x.x:8080/shot.jpg

# 2. Kalibracja → prostokąt
warped = calibration.apply_warp(frame)  # 800×800px

# 3. Preprocessing
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # kolorowy → szary
proc = cv2.GaussianBlur(gray, (5, 5), 0)         # delikatne wygładzenie

# 4. FEN → zbiór zajętych pól
fen = game_state.get_fen()
board = chess.Board(fen)
occupied = set()
for sq in chess.SQUARES:        # 0..63
    if board.piece_at(sq) is not None:
        occupied.add(chess.square_name(sq))  # np. "e4"

# 5. Dla każdego z 64 pól:
for row in range(8):          # wiersz 0 = rząd 8 (góra obrazu)
    for col in range(8):      # kolumna 0 = linia a (lewa strona)
        # Wycinanie patcha z marginesem
        x1 = col * 100 + 15   # CELL_SIZE_PX=100, CELL_MARGIN_PX=15
        y1 = row * 100 + 15
        x2 = (col + 1) * 100 - 15
        y2 = (row + 1) * 100 - 15
        patch = proc[y1:y2, x1:x2]        # ~70×70px

        # Resize do dokładnie 70×70
        patch_resized = cv2.resize(patch, (70, 70))

        # Labelowanie na podstawie FEN (nie na podstawie CV!)
        sq_name = chess.square_name(chess.square(col, 7 - row))
        if sq_name in occupied:
            save_to("dataset/occupied/", patch_resized)
        else:
            save_to("dataset/empty/", patch_resized)

        # Augmentacja: 4 warianty
        for i, augmented in enumerate(augment_patch(patch_resized)):
            save_to(same_folder, augmented)
```

---

## 6. Augmentacja — skąd te 320 obrazków

Każde pole generuje **5 plików** (1 oryginał + 4 augmentowane warianty):

```
pos_20260415_143022_e4_orig.jpg   ← oryginał
pos_20260415_143022_e4_aug0.jpg   ← wariant 1
pos_20260415_143022_e4_aug1.jpg   ← wariant 2
pos_20260415_143022_e4_aug2.jpg   ← wariant 3
pos_20260415_143022_e4_aug3.jpg   ← wariant 4
```

### Co robią augmentacje:

| Transformacja | Co symuluje | Szansa |
|---|---|---|
| RandomBrightnessContrast ±35% | Zmiana oświetlenia (rano/wieczorem) | 80% |
| GaussNoise | Szum matrycy kamery telefonu | 50% |
| GaussianBlur | Lekkie rozmazanie (ruch kamery) | 30% |
| HorizontalFlip | Symetria (pole a1 wygląda jak h1) | 50% |
| Rotate ±5° | Kamera lekko przekręcona | 40% |
| RandomShadow | Cień ręki gracza na polu | 20% |

### Matematyka:

```
1 collect = 64 pola × 5 plików = 320 JPEG
- 32 pola zajęte × 5 = 160 plików w occupied/
- 32 pola puste × 5  = 160 plików w empty/

10 collectów = 3200 plików łącznie (1600 occupied + 1600 empty)
```

---

## 7. Detekcja ruchów na żywo

To jest Faza 7, ale wyjaśniam tutaj bo pytałeś jak program śledzi grę.

### Maszyna stanów (3 stany):

```
┌─────────┐                        ┌───────────┐
│  IDLE   │── maska się zmieniła ──│  IN_MOVE  │
│ (czeka) │                        │ (ręka na  │
│         │◄── ruch zatwierdzony ──│  planszy) │
└─────────┘                        └───────────┘
                                         │
                                   maska stabilna
                                   przez N klatek
                                         │
                                         ▼
                                   ┌─────────────┐
                                   │ STABLE_AFTER│
                                   │ (wylicz ruch)│
                                   └─────────────┘
```

### Jak to działa w praktyce:

```
Czas 0.0s: Detektor w stanie IDLE
           Before = {a1,b1,c1,d1,e1,f1,g1,h1, a2,b2,...,h2,
                     a7,b7,...,h7, a8,b8,...,h8}  ← 32 pola

Czas 0.5s: tick → klatka z kamery → CNN → current = {... te same 32 ...}
           current == before → "bez zmian" → IDLE

Czas 1.0s: Gracz chwyta pionka na e2...
           tick → CNN → current = {31 pól} (e2 zniknęło, ręka zasłania)
           current != before → przejście do IN_MOVE

Czas 1.5s: Gracz stawia pionka na e4...
           tick → CNN → current = {31 pól} (ręka jeszcze na planszy)
           niestabilne, nadal IN_MOVE

Czas 2.0s: Gracz zabrał rękę.
           tick → CNN → current = {a1,b1,...,d2,f2,...,h2, e4, a7,...}
           e2 zniknęło, e4 pojawiło się → candidate = current
           stable_count = 1

Czas 2.5s: tick → CNN → ta sama maska → stable_count = 2
Czas 3.0s: tick → ta sama → stable_count = 3
Czas 3.5s: tick → ta sama → stable_count = 4
Czas 4.0s: tick → ta sama → stable_count = 5 → STABILNE!

           Teraz system oblicza deltę:
             disappeared = before - after = {"e2"}
             appeared    = after - before = {"e4"}

           1 zniknięcie + 1 pojawienie = standardowy ruch
           Sprawdza: na e2 stał pionek (wie z FEN)
           Ruch e2→e4 jest legalny? → TAK

           python-chess.push("e2e4")
           Nowy FEN: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

           Before = current (nowy stan bazowy)
           Stan = IDLE (czeka na następny ruch)
```

### Specjalne przypadki detekcji:

| Wzorzec | Ruch |
|---|---|
| 1 zniknęło + 1 pojawiło się | Zwykły ruch lub bicie |
| 2 zniknęły + 2 pojawiły się | Roszada (e1→g1 + h1→f1) |
| 2 zniknęły + 1 pojawiło się | En passant (pionek bije w przelocie) |
| Pionek na 8. rzędzie | Promocja (automatycznie na hetmana) |

---

## 8. Generowanie FEN

### Kto generuje FEN?

**Biblioteka `python-chess`** — NIE model CNN, NIE kamera.

```python
import chess
board = chess.Board()              # pozycja startowa
board.push(chess.Move.from_uci("e2e4"))
print(board.fen())
# → "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

### Cały łańcuch:

```
CNN mówi: "pola {a1,b1,...,e4,...} są zajęte, reszta pusta"
     ↓
Porównanie z poprzednim stanem: "e2 zniknęło, e4 pojawiło"
     ↓
move_inference: "to jest ruch e2e4"
     ↓
python-chess: board.push("e2e4") → generuje nowy FEN
     ↓
FEN jest dostępny przez GET /cv/game/state
```

### Dlaczego to działa?

Bo **zaczynamy od znanego FEN** (pozycja startowa lub dowolna)
i **każdy ruch jest weryfikowany** przez python-chess.

System wie DOKŁADNIE jaka figura stoi na jakim polu, bo:
1. Znamy punkt startowy (FEN startowy)
2. Każdy ruch jest legalny (python-chess waliduje)
3. Python-chess śledzi pełny stan gry (roszada, en passant, promocja)

CNN nie musi rozróżniać figur — wystarczy że wie "tu coś stoi, tu jest pusto".
Reszta wynika z logiki szachowej.

---

## 9. Najlepsza strategia zbierania datasetu

### Opcja A: Rozegranie jednej długiej partii (ZALECANE)

```
Zalety:
  ✓ Naturalny rozkład pozycji (otwarcie, środek gry, końcówka)
  ✓ Mało pracy — grasz normalnie
  ✓ Różne konfiguracje pól (mało figur, dużo figur, bicia)
  ✓ Dużo danych za jednym razem

Plan:
  1. POST /cv/game/reset
  2. POST /cv/ml/collect  ← pozycja startowa (32 occupied, 32 empty)
  3. Ruch fizyczny e2→e4
  4. POST /cv/game/move {"move_uci": "e2e4"}
  5. POST /cv/ml/collect  ← po ruchu (31 occupied na starych + 1 nowe)
  6. Ruch fizyczny e7→e5
  7. POST /cv/game/move {"move_uci": "e7e5"}
  8. POST /cv/ml/collect
  ... powtarzaj po każdym ruchu ...

50 ruchów = 50 collectów × 320 plików = 16 000 plików!
To jest DUŻO więcej niż potrzebne minimum (1000).
```

### Opcja B: Kilka krótszych partii

```
Zalety:
  ✓ Różne otwarcia = różne rozkłady figur
  ✓ Możesz zmienić oświetlenie między partiami

Plan:
  Partia 1: 20 ruchów → 20 collectów → 6400 plików
  (zmień oświetlenie)
  Partia 2: 15 ruchów → 15 collectów → 4800 plików
  (przesuń kamerę minimalnie)
  Partia 3: 10 ruchów → 10 collectów → 3200 plików
```

### Opcja C: Ustawianie ręczne pozycji (szybkie, ale mniej naturalne)

```
Możesz ustawiać dowolne pozycje i podawać FEN:

POST /cv/game/reset {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"}
POST /cv/ml/collect

Ale musisz znać FEN dokładnie, co jest trudne dla niestandardowych pozycji.
```

### Moja rekomendacja:

**Rozegraj 2-3 pełne partie (każda po 30-50 ruchów).**
Robisz `collect` po KAŻDYM ruchu.

```
Partia 1 (30 ruchów):  30 × 320 = 9 600 plików
Partia 2 (40 ruchów):  40 × 320 = 12 800 plików
                        RAZEM: ~22 400 plików
                        ~11 200 occupied + ~11 200 empty

To jest OBFITY dataset. Val accuracy > 95% prawie gwarantowana.
```

---

## 10. Instrukcja krok po kroku — co robić

### PRZED ROZPOCZĘCIEM

- [ ] Serwer backend działa (`python -m uvicorn main:app ...`)
- [ ] Kamera IP działa (sprawdź `GET /cv/health` → `camera_reachable: true`)
- [ ] Kalibracja wykonana (sprawdź `GET /cv/health` → `calibrated: true`)
- [ ] Szachownica ustawiona w pozycji startowej na stole pod kamerą

### PARTIA ZBIERAJĄCA DANE

```
Krok 1: Reset gry
─────────────────
POST /cv/game/reset
(bez body = domyślny FEN startowy)

→ Odpowiedź: {"ok": true, "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}


Krok 2: Collect z pozycji startowej
────────────────────────────────────
POST /cv/ml/collect

→ Odpowiedź: {"occupied_saved": 160, "empty_saved": 160, "fen_used": "rnbqkbnr/..."}
  (32 pól × 5 wariantów = 160 occupied, 32 pustych × 5 = 160 empty)


Krok 3: Wykonaj FIZYCZNY ruch na planszy
─────────────────────────────────────────
Bierzesz pionka z e2 i stawiasz na e4.
(Rękami na fizycznej szachownicy pod kamerą.)


Krok 4: Powiedz systemowi co zrobiłeś
──────────────────────────────────────
POST /cv/game/move
Body: {"move_uci": "e2e4"}

→ System aktualizuje wewnętrzny FEN
  Teraz wie że e2 = puste, e4 = biały pionek

WAŻNE: Format UCI = "skąd" + "dokąd", bez spacji, bez myślnika
  e2e4 ← pionek z e2 na e4
  g1f3 ← skoczek z g1 na f3
  e1g1 ← roszada krótka (król z e1 na g1)
  a7a8q ← promocja na hetmana (dodaj literę na końcu)


Krok 5: Collect po ruchu
────────────────────────
POST /cv/ml/collect

→ System robi zdjęcie i wie z FEN który pola są zajęte
  (teraz e2 ląduje w empty/, e4 w occupied/)


Krok 6: Ruch czarnych
──────────────────────
Fizycznie przesuń czarnego pionka z e7 na e5.

POST /cv/game/move
Body: {"move_uci": "e7e5"}


Krok 7: Collect
───────────────
POST /cv/ml/collect


Krok 8: Powtarzaj kroki 3-7 dla każdego ruchu
──────────────────────────────────────────────
Grasz jak w normalnych szachach, tylko po KAŻDYM ruchu:
  a) POST /cv/game/move {"move_uci": "RUCH"}
  b) POST /cv/ml/collect
```

### MONITOROWANIE POSTĘPU

Po każdych kilku ruchach sprawdź ile masz danych:

```
GET /cv/ml/dataset/stats

→ {"occupied_count": 1500, "empty_count": 1500, "total": 3000}
```

### PO ZAKOŃCZENIU PARTII

Chcesz zagrać kolejną? Ustaw figury na miejsca i:

```
POST /cv/game/reset
POST /cv/ml/collect
... i znowu od kroku 3.
```

### PODSUMOWANIE SEKWENCJI

```
┌─────────────────────────────────────────────┐
│ POST /cv/game/reset                         │  ← raz na partię
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │  POST /cv/ml/collect                    │ │  ← opcjonalnie z poz. startowej
│ │                                         │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │  1. Fizyczny ruch na planszy     │  │ │  ← powtarzaj
│ │  │  2. POST /cv/game/move {uci}     │  │ │     aż koniec
│ │  │  3. POST /cv/ml/collect          │  │ │     partii
│ │  └───────────────────────────────────┘  │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ GET /cv/ml/dataset/stats                    │  ← sprawdź ile danych
└─────────────────────────────────────────────┘
```

---

## Odpowiedzi na konkretne pytania

### "Skąd się bierze FEN po ruchu e4?"

Z biblioteki `python-chess`. Kiedy wywołujesz `POST /cv/game/move {"move_uci": "e2e4"}`,
system robi `chess.Board.push(Move.from_uci("e2e4"))`. Python-chess zna zasady szachów
i sam generuje nowy FEN.

### "Na jakiej podstawie sprawdza czy pole zajęte czy puste?"

**Przy zbieraniu danych (collect):** NIE sprawdza wizualnie! Bierze FEN z game_state
i na tej podstawie wie które pola są zajęte. To jest cały geniusz auto-labelingu.

**Przy detekcji na żywo (tick):** CNN (lub fallback wariancji) analizuje obraz pola.

### "Czy to działa na podstawie zwykłego CV? Thresholdu?"

Fallback (bez CNN) = tak, używa wariancji pikseli z progiem 580.
Z CNN = sieć neuronowa podejmuje decyzję (próg 0.5 na wyjściu sigmoid).

### "Jakie znaczenie mają ruchy zapisane przez game/move?"

**Kluczowe!** Każdy `game/move` aktualizuje FEN wewnątrz game_state.
Kiedy potem robisz `collect`, system czyta ten FEN żeby wiedzieć
które pola oznaczyć jako occupied a które jako empty.

Bez prawidłowej sekwencji `game/move` → etykiety w datasecie byłyby BŁĘDNE
→ model by się nauczył śmieci.

### "Czy kamerka ma sprawdzać co pół sekundy?"

TAK — ale w Fazie 7 (detekcja na żywo), nie w Fazie 3 (zbieranie danych).
W Fazie 3 Ty ręcznie wywołujesz `collect` po każdym ruchu.
W Fazie 7 frontend (lub skrypt) woła `POST /cv/game/detector/tick` co ~500ms.

### "Czy lepiej rozegrać jedną długą grę?"

TAK — to najlepsza strategia. Jedna partia 50 ruchów = 50 collectów = 16 000 plików.
Dwie partie = ~30 000+ plików. Model będzie miał dużo danych i wysoki accuracy.

---

## Diagram całego systemu

```
╔══════════════════════════════════════════════════════════════╗
║                    FAZA 3: ZBIERANIE DANYCH                 ║
║                                                              ║
║  [TY]                      [SYSTEM]                          ║
║                                                              ║
║  Ustawiasz szachownicę     POST /cv/game/reset               ║
║  w pozycji startowej       → FEN = startowy                  ║
║                                                              ║
║  Przesuwasz pionka e2→e4   POST /cv/game/move {e2e4}         ║
║                            → FEN zaktualizowany              ║
║                                                              ║
║  (nic nie robisz)          POST /cv/ml/collect                ║
║                            → Kamera → zdjęcie                ║
║                            → Warp → 800×800px                ║
║                            → 64 patche × 5 = 320 JPEG        ║
║                            → Etykiety z FEN (nie z kamery!)   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                    FAZA 7: GRA NA ŻYWO                       ║
║                                                              ║
║  [TY]                      [SYSTEM]                          ║
║                                                              ║
║  POST /cv/game/reset       FEN = startowy                    ║
║  POST /detector/start      Snapshot "before" (CNN → 32 pola) ║
║                                                              ║
║  Przesuwasz pionka e2→e4   (nic — czeka na tick)             ║
║                                                              ║
║  (co 0.5s automatycznie)   POST /detector/tick                ║
║                            → CNN → current = {31 pól...}     ║
║                            → current != before → IN_MOVE     ║
║                                                              ║
║  Zdejmujesz rękę           POST /detector/tick (× kilka)     ║
║                            → CNN → current stabilne 5 klatek ║
║                            → Delta: e2 zniknęło, e4 się poj.║
║                            → move_inference → "e2e4"         ║
║                            → python-chess → nowy FEN         ║
║                            → Stockfish → ocena pozycji       ║
║                                                              ║
║                            GET /cv/game/state                 ║
║                            → {"fen": "...", "history": [...]} ║
╚══════════════════════════════════════════════════════════════╝
```

---

*Dokument wygenerowany: 15.04.2026 | Projekt: Chess Vision ML*
