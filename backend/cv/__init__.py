"""
Pakiet CV — Computer Vision dla detekcji ruchów szachowych z kamery.

Moduły:
  config            — konfiguracja (URL kamery, ścieżki modeli, progi)
  models            — Pydantic modele żądań / odpowiedzi
  camera            — pobieranie klatek z kamery IP
  calibration       — detekcja szachownicy + perspektive warp
  board_occupancy   — klasyfikacja zajętości 64 pól
  move_detector     — maszyna stanów IDLE/IN_MOVE/STABLE
  move_inference    — delta before/after → ruch UCI
  game_state        — python-chess Board (singleton)
  router            — FastAPI router /cv/*
  ml                — modele ML (board detector, square classifier, trening)
"""
