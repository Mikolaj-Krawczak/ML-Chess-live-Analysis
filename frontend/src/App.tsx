import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, ChangeEvent } from "react";
import { Chess } from "chess.js";
import "./App.css";
import BoardPanel from "./BoardPanel";

// System dźwięków szachowych
class ChessAudio {
  private static instance: ChessAudio;
  private sounds: Record<string, HTMLAudioElement> = {};
  
  constructor() {
    // Preload wszystkich dźwięków
    this.sounds.move = new Audio('/sounds/move.mp3');
    this.sounds.capture = new Audio('/sounds/capture.mp3');
    this.sounds.castle = new Audio('/sounds/castle.mp3');
    this.sounds.promote = new Audio('/sounds/promote.mp3');
    this.sounds.check = new Audio('/sounds/move-check.mp3');
    
    // Ustaw głośność i optymalizacje dla szybkiego odtwarzania
    Object.values(this.sounds).forEach(audio => {
      audio.volume = 0.3;
      audio.preload = 'auto';
      // Wymuś natychmiastowe odtwarzanie bez buforowania
      audio.load();
    });
  }
  
  static getInstance(): ChessAudio {
    if (!ChessAudio.instance) {
      ChessAudio.instance = new ChessAudio();
    }
    return ChessAudio.instance;
  }
  
  play(soundType: 'move' | 'capture' | 'castle' | 'promote' | 'check'): void {
    const audio = this.sounds[soundType];
    if (audio) {
      // Zatrzymaj poprzedni dźwięk tego typu jeśli gra
      audio.pause();
      audio.currentTime = 0;
      
      // Natychmiastowe odtwarzanie
      const playPromise = audio.play();
      if (playPromise) {
        playPromise.catch(err => console.warn('Nie można odtworzyć dźwięku:', err));
      }
    }
  }
}

interface MoveAnalysis {
  isCapture: boolean;
  isCastle: boolean;
  isPromotion: boolean;
  isCheck: boolean;
}

// Analizuje rodzaj ruchu na podstawie zmian FEN i stanu gry
function analyzeMoveType(
  prevFen: string,
  newFen: string,
  gameState: GameStateResponse | null
): MoveAnalysis | null {
  if (!prevFen || !newFen || prevFen === newFen) return null;
  
  const prevPieces = prevFen.split(' ')[0].replace(/[^a-zA-Z]/g, '');
  const newPieces = newFen.split(' ')[0].replace(/[^a-zA-Z]/g, '');
  
  const isCapture = prevPieces.length > newPieces.length;
  const isCheck = gameState?.is_check || false;
  
  // Wykryj roszadę (król przesunął się o 2 pola)
  const isCastle = /[Kk]/.test(prevFen) && /[Kk]/.test(newFen) && 
    Math.abs(prevFen.indexOf('K') - newFen.indexOf('K')) === 2;
  
  // Wykryj promocję (pojawił się nowy hetman/wieża/goniec/koń)
  const prevCount = (prevPieces.match(/[QRBNqrbn]/g) || []).length;
  const newCount = (newPieces.match(/[QRBNqrbn]/g) || []).length;
  const isPromotion = newCount > prevCount;
  
  return { isCapture, isCastle, isPromotion, isCheck };
}

// Odtwarza odpowiedni dźwięk na podstawie analizy ruchu
function playMoveSound(moveAnalysis: MoveAnalysis, gameState: GameStateResponse | null) {
  const chessAudio = ChessAudio.getInstance();
  
  // Wybierz odpowiedni dźwięk wg priorytetu
  if (gameState?.is_checkmate) {
    // Mat = brak dźwięku lub specjalny (nie mamy pliku)
  } else if (moveAnalysis.isCheck) {
    chessAudio.play('check');
  } else if (moveAnalysis.isCastle) {
    chessAudio.play('castle');
  } else if (moveAnalysis.isPromotion) {
    chessAudio.play('promote');
  } else if (moveAnalysis.isCapture) {
    chessAudio.play('capture');
  } else {
    chessAudio.play('move');
  }
}

// Wartości materiałowe figur
const PIECE_VALUES: Record<string, number> = {
  'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
  'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
};

// Unicode figury szachowe
const PIECE_SYMBOLS: Record<string, string> = {
  'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
  'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
};

interface MaterialInfo {
  whiteCaptured: string[];
  blackCaptured: string[];
  whiteMaterial: number;
  blackMaterial: number;
  materialAdvantage: number; // + = white advantage, - = black advantage
}

function buildCaptureRows(captured: string[]): string[][] {
  const byType: Record<string, string[]> = {
    p: [],
    n: [],
    b: [],
    r: [],
    q: [],
  };

  captured.forEach((piece) => {
    const key = piece.toLowerCase();
    if (key in byType) {
      byType[key].push(piece);
    }
  });

  // Stały układ wg ważności figur (od najcenniejszych do najmniej cennych),
  // niezależnie od czasu zbicia.
  const rows: string[][] = [];
  if (byType.q.length > 0) rows.push(byType.q);
  if (byType.r.length > 0) rows.push(byType.r);
  if (byType.b.length > 0) rows.push(byType.b);
  if (byType.n.length > 0) rows.push(byType.n);
  if (byType.p.length > 0) rows.push(byType.p);

  return rows;
}

// Analizuje materiał z FEN vs pozycja startowa
function analyzeMaterial(fen: string): MaterialInfo {
  const startingPieces = "rnbqkbnrppppppppPPPPPPPPRNBQKBNR";
  const currentPieces = fen.split(' ')[0].replace(/[^a-zA-Z]/g, '');
  
  // Zlicz figury w pozycji startowej vs obecnej
  const startCount: Record<string, number> = {};
  const currentCount: Record<string, number> = {};
  
  for (const piece of startingPieces) {
    startCount[piece] = (startCount[piece] || 0) + 1;
  }
  
  for (const piece of currentPieces) {
    currentCount[piece] = (currentCount[piece] || 0) + 1;
  }
  
  const whiteCaptured: string[] = [];
  const blackCaptured: string[] = [];
  let whiteMaterial = 0;
  let blackMaterial = 0;
  
  // Sprawdź jakie figury zostały zbite
  for (const piece in startCount) {
    const missing = startCount[piece] - (currentCount[piece] || 0);
    const isWhitePiece = piece === piece.toUpperCase();
    
    // Figury zbite przez białych (brakuje czarnych figur)
    if (!isWhitePiece && missing > 0) {
      for (let i = 0; i < missing; i++) {
        whiteCaptured.push(piece);
      }
    }
    
    // Figury zbite przez czarnych (brakuje białych figur)
    if (isWhitePiece && missing > 0) {
      for (let i = 0; i < missing; i++) {
        blackCaptured.push(piece);
      }
    }
  }
  
  // Policz obecny materiał każdej strony
  for (const piece of currentPieces) {
    const value = PIECE_VALUES[piece] || 0;
    if (piece === piece.toUpperCase()) {
      whiteMaterial += value;
    } else {
      blackMaterial += value;
    }
  }
  
  return {
    whiteCaptured,
    blackCaptured,
    whiteMaterial,
    blackMaterial,
    materialAdvantage: whiteMaterial - blackMaterial,
  };
}

interface MaterialDisplayProps {
  fen: string;
  boardOrientation: "white" | "black";
  boardSizePx: number;
}

function MaterialDisplay({ fen, boardOrientation, boardSizePx }: MaterialDisplayProps) {
  const material = analyzeMaterial(fen);

  const capturesByWhite = material.whiteCaptured; // Czarne figury zbite przez białego
  const capturesByBlack = material.blackCaptured; // Białe figury zbite przez czarnego

  const topPlayerIsWhite = boardOrientation === "black";
  const topCaptured = topPlayerIsWhite ? capturesByWhite : capturesByBlack;
  const bottomCaptured = topPlayerIsWhite ? capturesByBlack : capturesByWhite;
  const topRows = buildCaptureRows(topCaptured);
  const bottomRows = buildCaptureRows(bottomCaptured);

  const topAdvantage =
    (topPlayerIsWhite && material.materialAdvantage > 0) ||
    (!topPlayerIsWhite && material.materialAdvantage < 0)
      ? `+${Math.abs(material.materialAdvantage)}`
      : "";

  const bottomAdvantage =
    (!topPlayerIsWhite && material.materialAdvantage > 0) ||
    (topPlayerIsWhite && material.materialAdvantage < 0)
      ? `+${Math.abs(material.materialAdvantage)}`
      : "";

  return (
    <div className="material-display" style={{ height: `${boardSizePx}px` }}>
      {/* Materiał gracza z górnej strony planszy */}
      <div className="material-section material-section--top">
        <div className="captured-pieces">
          {topRows.map((row, rowIndex) => (
            <div key={`top-row-${rowIndex}`} className="captured-row">
              {row.map((piece, index) => (
                <span
                  key={`${piece}-${rowIndex}-${index}-top`}
                  className={`captured-piece ${
                    piece === piece.toUpperCase() ? "captured-piece--white" : "captured-piece--black"
                  }`}
                >
                  {PIECE_SYMBOLS[piece]}
                </span>
              ))}
            </div>
          ))}
        </div>
        {topAdvantage && <span className="material-advantage">{topAdvantage}</span>}
      </div>

      {/* Materiał gracza z dolnej strony planszy */}
      <div className="material-section material-section--bottom">
        {bottomAdvantage && <span className="material-advantage">{bottomAdvantage}</span>}
        <div className="captured-pieces">
          {bottomRows.map((row, rowIndex) => (
            <div key={`bottom-row-${rowIndex}`} className="captured-row">
              {row.map((piece, index) => (
                <span
                  key={`${piece}-${rowIndex}-${index}-bottom`}
                  className={`captured-piece ${
                    piece === piece.toUpperCase() ? "captured-piece--white" : "captured-piece--black"
                  }`}
                >
                  {PIECE_SYMBOLS[piece]}
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Stan gry zwracany przez GET /cv/game/state */
interface GameStateResponse {
  fen: string;
  turn: "white" | "black";
  move_number: number;
  halfmove_clock: number;
  is_check: boolean;
  is_checkmate: boolean;
  is_stalemate: boolean;
  is_game_over: boolean;
  history: string[];
  history_length: number;
}

interface OccupancyResponse {
  ok: boolean;
  message: string;
  occupied_squares: string[];
  empty_squares: string[];
  occupied_count: number;
  cells: Array<{
    square: string;
    occupied: boolean;
    score: number;
    method: string;
  }>;
  debug_image_b64?: string;
  threshold_used: number;
  occupancy_method: string;
}

const API = "http://localhost:8000";

const STARTING_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const DEPTH_MIN = 6;
const DEPTH_MAX = 24;
const ELO_MIN = 1320;
const ELO_MAX = 3190;
const DEFAULT_DEPTH = 18;
const DEFAULT_ELO = 1500;
const DEFAULT_SKILL = 10;

// A6: odstęp między kolejnymi tickami (łańcuchowo — dopiero po zakończeniu poprzedniego)
const LIVE_POLL_INTERVAL_MS = 500;

// A8: niższa głębokość dla auto-analizy po ruchu z kamery
// Manualne "Analizuj" nadal korzysta z ustawienia użytkownika (do 24)
const LIVE_EVAL_DEPTH = 12;

type StrengthMode = "full" | "elo" | "skill";
type ViewTab = "live" | "analysis";

/** Odpowiedź z backendu FastAPI /evaluate (EvalResponse) */
export interface EvalResponse {
  score: number;
  score_type: "cp" | "mate";
  mate_in: number | null;
  best_move: string | null;
  pv: string[];
  depth: number;
  turn: "white" | "black";
  is_valid: boolean;
}

// Klamp wartości do przedziału
const clamp = (v: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, v));

// Zamień ocenę na etykietę tekstową z uwzględnieniem orientacji planszy
function formatScore(
  score: number,
  scoreType: "cp" | "mate",
  mateIn: number | null,
  boardOrientation: "white" | "black"
): string {
  if (scoreType === "mate") {
    // Dla mata opieramy znak o mateIn (+ = białe wygrywają, - = czarne wygrywają).
    // To jest bardziej niezawodne niż score, które może być niespójne przy M0.
    const signedMate = mateIn != null && mateIn > 0 ? 1 : -1;
    const orientedSign = boardOrientation === "black" ? -signedMate : signedMate;
    const mateDistance = Math.abs(mateIn ?? 0);
    if (mateDistance === 0) {
      return orientedSign > 0 ? "+M0" : "-M0";
    }
    return orientedSign > 0 ? `+M${mateDistance}` : `-M${mateDistance}`;
  }
  
  let displayScore = score;
  // Odwróć znak wyniku jeśli plansza jest od strony czarnych
  if (boardOrientation === "black") {
    displayScore = -score;
  }
  
  return displayScore >= 0 ? `+${displayScore.toFixed(2)}` : displayScore.toFixed(2);
}

interface ThermometerProps {
  score: number;
  scoreType: "cp" | "mate";
  mateIn: number | null;
  loading: boolean;
  /** Zgodnie z widokiem szachownicy: od czarnych — czarna strefa termometru na dole. */
  boardOrientation: "white" | "black";
  boardSizePx: number;
}

// Termometr: 0% = czarne, 100% = białe (skala ok. ±10 pionów)
function Thermometer({
  score,
  scoreType,
  mateIn,
  loading,
  boardOrientation,
  boardSizePx,
}: ThermometerProps) {
  const CAP = 10;
  const rawPercent =
    scoreType === "mate"
      // Dla mata opieramy kolor paska o mateIn (nie score):
      // + mateIn = białe wygrywają (pasek biały 100%)
      // - mateIn = czarne wygrywają (pasek czarny 0%)
      ? mateIn != null && mateIn > 0
        ? 100
        : 0
      : 50 + (clamp(score, -CAP, CAP) / CAP) * 50;

  const whitePercent = clamp(rawPercent, 2, 98);
  const blackPercent = 100 - whitePercent;

  const label = formatScore(score, scoreType, mateIn, boardOrientation);
  const advantage =
    score > 0.2 ? "white" : score < -0.2 ? "black" : "equal";

  const fromBlack =
    boardOrientation === "black" ? " thermo-wrap--from-black" : "";

  return (
    <div className={`thermo-wrap${fromBlack}`}>
      <div
        className="thermo-bar"
        aria-label="Ocena pozycji"
        style={
          {
            "--thermo-black-pct": `${blackPercent}%`,
            "--thermo-white-pct": `${whitePercent}%`,
            height: `${boardSizePx}px`,
          } as CSSProperties
        }
      >
        <div
          className="thermo-black"
          style={{
            transition: loading
              ? "none"
              : "height 0.65s cubic-bezier(0.34,1.56,0.64,1), width 0.65s cubic-bezier(0.34,1.56,0.64,1)",
          }}
        />
        <div
          className="thermo-white"
          style={{
            transition: loading
              ? "none"
              : "height 0.65s cubic-bezier(0.34,1.56,0.64,1), width 0.65s cubic-bezier(0.34,1.56,0.64,1)",
          }}
        />
        <div
          className={`thermo-score-overlay thermo-score-overlay--${advantage} ${loading ? "pulse" : ""}`}
          aria-live="polite"
        >
          {loading ? "…" : label}
        </div>
      </div>
    </div>
  );
}

/** Wyciąga komunikat błędu z odpowiedzi FastAPI (detail: string | tablica walidacji) */
function parseApiErrorPayload(data: unknown): string {
  if (typeof data !== "object" || data === null) return "Błąd serwera";
  const d = data as { detail?: unknown };
  if (typeof d.detail === "string") return d.detail;
  if (Array.isArray(d.detail)) {
    return d.detail
      .map((x) => {
        if (typeof x === "object" && x !== null && "msg" in x) {
          return String((x as { msg: string }).msg);
        }
        return String(x);
      })
      .join("; ");
  }
  return "Błąd serwera";
}

function buildEvaluatePayload(
  fen: string,
  depth: number,
  strengthMode: StrengthMode,
  eloLimit: number,
  skillLevel: number
): Record<string, string | number> {
  const body: Record<string, string | number> = {
    fen: fen.trim(),
    depth,
  };
  if (strengthMode === "elo") {
    body.elo_limit = eloLimit;
  } else if (strengthMode === "skill") {
    body.skill_level = skillLevel;
  }
  return body;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ViewTab>("live");
  const [fen, setFen] = useState(STARTING_FEN);
  const [analysisFen, setAnalysisFen] = useState(STARTING_FEN);
  const [result, setResult] = useState<EvalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBestMove, setShowBestMove] = useState(true);
  const [boardOrientation, setBoardOrientation] = useState<"white" | "black">("white");
  const [boardWidth, setBoardWidth] = useState(400);
  const boardColRef = useRef<HTMLDivElement>(null);
  const [depth, setDepth] = useState(DEFAULT_DEPTH);
  const [strengthMode, setStrengthMode] = useState<StrengthMode>("full");
  const [eloLimit, setEloLimit] = useState(DEFAULT_ELO);
  const [skillLevel, setSkillLevel] = useState(DEFAULT_SKILL);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const currentBoardSize = Math.max(200, Math.floor(boardWidth * 0.6));

  // Obsługa klawisza Escape w trybie fullscreen
  useEffect(() => {
    const handleEscape = (e: Event) => {
      const keyEvent = e as globalThis.KeyboardEvent;
      if (keyEvent.key === "Escape" && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    
    if (isFullscreen) {
      document.addEventListener("keydown", handleEscape);
      return () => document.removeEventListener("keydown", handleEscape);
    }
  }, [isFullscreen]);

  // --- Stan live gry z kamery ---
  const [gameState, setGameState] = useState<GameStateResponse | null>(null);
  const [detectorRunning, setDetectorRunning] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [occupancyError, setOccupancyError] = useState<string | null>(null);
  // A5: łańcuchowy polling przez setTimeout — timeout aktualnie zaplanowany
  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Flaga anulowania — przerywa pętlę gdy użytkownik kliknie Stop
  const pollCancelledRef = useRef(false);
  // Zapamiętana długość historii — do wykrywania nowego ruchu i auto-analizy
  const prevHistoryLenRef = useRef(0);
  // Poprzedni FEN — do analizy rodzaju ruchu i odtwarzania dźwięków
  const prevFenRef = useRef<string>(STARTING_FEN);
  const prevVirtualFenRef = useRef<string>(STARTING_FEN);

  // Ref do bieżących ustawień silnika — pozwala wywołać analizę ze świeżymi
  // wartościami bez dodawania ich do deps useEffect nasłuchującego na FEN
  const engineSettingsRef = useRef({ depth, strengthMode, eloLimit, skillLevel });
  useEffect(() => {
    engineSettingsRef.current = { depth, strengthMode, eloLimit, skillLevel };
  }, [depth, strengthMode, eloLimit, skillLevel]);

  // Automatycznie zatrzymaj detektor po zakończeniu partii (np. mat).
  useEffect(() => {
    if (!detectorRunning || !gameState) return;
    if (gameState.is_checkmate || gameState.is_game_over) {
      setDetectorRunning(false);
    }
  }, [detectorRunning, gameState]);

  useEffect(() => {
    if (activeTab !== "live" && detectorRunning) {
      setDetectorRunning(false);
    }
    // Reset termometru przy przejściu między zakładkami
    setResult(null);
    setError(null);
  }, [activeTab, detectorRunning]);

  const evaluate = useCallback(async (fenOverride?: string) => {
    const fenToUse = fenOverride ?? analysisFen;
    if (!fenToUse.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { depth: d, strengthMode: sm, eloLimit: el, skillLevel: sl } =
        engineSettingsRef.current;
      const res = await fetch(`${API}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          buildEvaluatePayload(fenToUse, d, sm, el, sl)
        ),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        throw new Error(parseApiErrorPayload(data));
      }
      setResult(data as EvalResponse);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Nieznany błąd");
    } finally {
      setLoading(false);
    }
  }, [analysisFen]);

  // Pobiera aktualny stan gry z backendu i synchronizuje FEN planszy
  const fetchGameState = useCallback(async () => {
    try {
      const res = await fetch(`${API}/cv/game/state`);
      if (!res.ok) return;
      const data = (await res.json()) as GameStateResponse;
      setGameState(data);
      setFen(data.fen);
      setLiveError(null);
      
      // Auto-analiza Stockfisha gdy wykryto nowy ruch — niższy depth dla szybkiej reakcji (A8)
      if (data.history_length > prevHistoryLenRef.current) {
        prevHistoryLenRef.current = data.history_length;
        
        // Analiza ruchu i odtwarzanie dźwięku dla live camera
        const moveAnalysis = analyzeMoveType(prevFenRef.current, data.fen, data);
        if (moveAnalysis) {
          playMoveSound(moveAnalysis, data);
        }
        
        // Zapisz obecny FEN jako poprzedni
        prevFenRef.current = data.fen;
        
        void evaluateWithFen(data.fen, LIVE_EVAL_DEPTH);
      }
    } catch {
      setLiveError("Brak połączenia z backendem CV.");
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Jedna iteracja pollingowa: tick detektora → odczyt stanu gry
  const tickAndSync = useCallback(async () => {
    try {
      // Tick przetwarza klatkę z kamery i wykrywa ruch (jeśli nastąpił)
      await fetch(`${API}/cv/game/detector/tick`, { method: "POST" });
    } catch {
      // Błąd ticki nie blokuje odczytu stanu — logujemy cicho
    }
    // Zawsze odśwież stan po ticku (nowy FEN, historia, kolej)
    await fetchGameState();
  }, [fetchGameState]);

  // Sprawdza occupancy przed startem gry
  const validateOccupancy = async (): Promise<boolean> => {
    try {
      const res = await fetch(`${API}/cv/occupancy`);
      if (!res.ok) {
        throw new Error("Nie można sprawdzić occupancy planszy.");
      }
      const data = (await res.json()) as OccupancyResponse;
      
      if (data.occupied_count !== 32) {
        const expectedSquares = [
          'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', // białe figury
          'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', // białe pionki
          'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7', // czarne pionki
          'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'  // czarne figury
        ];
        
        const wronglyEmpty = expectedSquares.filter(sq => !data.occupied_squares.includes(sq));
        const wronglyOccupied = data.occupied_squares.filter(sq => !expectedSquares.includes(sq));
        
        let errorMsg = `Wykryto ${data.occupied_count} figur zamiast 32. `;
        if (wronglyEmpty.length > 0) {
          errorMsg += `Brak figur na: ${wronglyEmpty.join(', ')}. `;
        }
        if (wronglyOccupied.length > 0) {
          errorMsg += `Nieprawidłowo wykryte figury na: ${wronglyOccupied.join(', ')}.`;
        }
        
        setOccupancyError(errorMsg);
        return false;
      }
      
      setOccupancyError(null);
      return true;
    } catch (e: unknown) {
      setOccupancyError(e instanceof Error ? e.message : "Błąd walidacji planszy.");
      return false;
    }
  };

  // Uruchamia detektor (inicjalizuje snapshot 'before' z kamery)
  const handleStart = async () => {
    setLiveLoading(true);
    setLiveError(null);
    setOccupancyError(null);
    
    // Najpierw sprawdź occupancy
    const occupancyValid = await validateOccupancy();
    if (!occupancyValid) {
      setLiveLoading(false);
      return;
    }
    
    try {
      const res = await fetch(`${API}/cv/game/detector/start`, { method: "POST" });
      if (!res.ok) {
        const data: unknown = await res.json();
        throw new Error(parseApiErrorPayload(data));
      }
      setDetectorRunning(true);
      await fetchGameState();
    } catch (e: unknown) {
      setLiveError(e instanceof Error ? e.message : "Błąd uruchamiania detektora.");
    } finally {
      setLiveLoading(false);
    }
  };

  // Zatrzymuje polling — nie woła backendu, tylko zatrzymuje interwał
  const handleStop = () => {
    setDetectorRunning(false);
  };

  // Resetuje stan gry po stronie backendu i odświeża planszę
  const handleReset = async () => {
    setLiveLoading(true);
    setLiveError(null);
    try {
      const res = await fetch(`${API}/cv/game/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: STARTING_FEN }),
      });
      if (!res.ok) {
        const data: unknown = await res.json();
        throw new Error(parseApiErrorPayload(data));
      }
      prevHistoryLenRef.current = 0;
      prevFenRef.current = STARTING_FEN; // Reset FEN referencyjnego
      setResult(null);
      setError(null);
      await fetchGameState();
    } catch (e: unknown) {
      setLiveError(e instanceof Error ? e.message : "Błąd resetowania gry.");
    } finally {
      setLiveLoading(false);
    }
  };

  // Wewnętrzna wersja evaluate przyjmująca FEN jako parametr (nie z state).
  // depthOverride pozwala wymusić niższą głębokość dla auto-analizy live (A8).
  const evaluateWithFen = useCallback(
    async (fenStr: string, depthOverride?: number) => {
      if (!fenStr.trim()) return;
      setLoading(true);
      setError(null);
      try {
        const { depth: d, strengthMode: sm, eloLimit: el, skillLevel: sl } =
          engineSettingsRef.current;
        const effectiveDepth = depthOverride ?? d;
        const res = await fetch(`${API}/evaluate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            buildEvaluatePayload(fenStr, effectiveDepth, sm, el, sl)
          ),
        });
        const data: unknown = await res.json();
        if (!res.ok) throw new Error(parseApiErrorPayload(data));
        setResult(data as EvalResponse);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Nieznany błąd");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  // A5 + A6: łańcuchowy polling — kolejny tick dopiero gdy poprzedni się skończy.
  // Eliminuje backlog requestów na backendzie i odstęp 500 ms działa zawsze
  // jako minimalna przerwa, a nie sztywny rytm.
  useEffect(() => {
    if (!detectorRunning) {
      pollCancelledRef.current = true;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
      return;
    }

    pollCancelledRef.current = false;

    const loop = async () => {
      if (pollCancelledRef.current) return;
      await tickAndSync();
      if (pollCancelledRef.current) return;
      pollTimeoutRef.current = setTimeout(() => {
        void loop();
      }, LIVE_POLL_INTERVAL_MS);
    };

    void fetchGameState(); // natychmiastowy odczyt stanu po starcie
    void loop();

    return () => {
      pollCancelledRef.current = true;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
    };
  }, [detectorRunning, fetchGameState, tickAndSync]);

  // Etykieta statusu gry do wyświetlenia w tabeli
  const gameStatusLabel = (() => {
    if (!gameState) return "—";
    if (gameState.is_checkmate) return "Mat!";
    if (gameState.is_stalemate) return "Pat!";
    if (gameState.is_game_over) return "Koniec gry";
    if (gameState.is_check) return "Szach!";
    return "Trwa gra";
  })();

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !loading) void evaluate();
  };

  // Auto-analiza po wklejeniu FEN — debounce 600 ms, żeby nie strzelać
  // zapytania przy każdym znaku wpisywanym ręcznie
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleFenChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setAnalysisFen(val);
    
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      void evaluate(val);
    }, 600);
  };

  const handleAnalysisPieceDrop = useCallback(
    ({
      sourceSquare,
      targetSquare,
    }: {
      sourceSquare: string;
      targetSquare: string | null;
      piece: unknown;
    }): boolean => {
      if (activeTab !== "analysis") return false;
      if (!targetSquare) return false;

      try {
        const game = new Chess(analysisFen);
        // Sprawdź czy ruch to promocja piona:
        // 1. Figura na polu źródłowym to pion
        // 2. Pole docelowe to linia promocji (1 dla czarnych, 8 dla białych)
        const sourcePiece = game.get(sourceSquare as Parameters<typeof game.get>[0]);
        const isPawn = sourcePiece && sourcePiece.type === "p";
        const targetRank = parseInt(targetSquare[1]);
        
        // Biały pion na linię 8, czarny pion na linię 1
        const isPromotion = isPawn && 
          ((sourcePiece.color === "w" && targetRank === 8) || 
           (sourcePiece.color === "b" && targetRank === 1));

        const moveResult = game.move({
          from: sourceSquare,
          to: targetSquare,
          ...(isPromotion ? { promotion: "q" } : {}),
        });

        if (!moveResult) {
          const targetPiece = game.get(targetSquare as Parameters<typeof game.get>[0]);
          const targetOccupied = targetPiece != null;
          const hint =
            isPawn && targetSquare[0] === sourceSquare[0] && targetOccupied
              ? "Pion nie może bić na wprost. Jeśli b8 jest zajęte, zbijasz tylko na skos (a8/c8)."
              : "Sprawdź czy to właściwa kolej i czy ruch jest legalny.";
          setError(`Nielegalny ruch ${sourceSquare} -> ${targetSquare}. ${hint}`);
          return false;
        }

        const nextFen = game.fen();
        setAnalysisFen(nextFen);
        setError(null);
        void evaluate(nextFen);
        return true;
      } catch {
        return false;
      }
    },
    [activeTab, analysisFen, evaluate]
  );

  // Debounced sound effect - unikaj wielokrotnych dźwięków przy szybkich zmianach
  const soundDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Effect do synchronizacji dźwięków z wirtualną deską
  useEffect(() => {
    if (fen !== prevVirtualFenRef.current && fen.trim() && prevVirtualFenRef.current.trim()) {
      // Wyczyść poprzedni timeout
      if (soundDebounceRef.current) {
        clearTimeout(soundDebounceRef.current);
      }
      
      // Krótkie opóźnienie żeby zsynchronizować z animacją React Chessboard (280ms)
      soundDebounceRef.current = setTimeout(() => {
        const moveAnalysis = analyzeMoveType(prevVirtualFenRef.current, fen, gameState);
        if (moveAnalysis) {
          playMoveSound(moveAnalysis, gameState);
        }
      }, 280); // Dokładnie dopasowane do animationDurationInMs z BoardPanel
    }
    prevVirtualFenRef.current = fen;
  }, [fen, gameState]);

  useEffect(() => {
    const el = boardColRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setBoardWidth(Math.floor(w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const boardFen = activeTab === "analysis" ? analysisFen : fen;
  const fullscreenBoardSize = Math.min(
    window.innerHeight - 80,
    window.innerWidth - 80
  );

  return (
    <>
    <main className={`app app--${activeTab}`}>
      <header className="header">
        <h1>
          Chess <span className="title-live">Live</span>{" "}
          <span className="title-analysis">Analysis</span>
        </h1>
        <p>
          Analiza pozycji · <strong>Stockfish 18</strong> · silnik UCI · MVP v0.1
        </p>
        <div className="view-tabs" role="tablist" aria-label="Tryb widoku">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "live"}
            className={`view-tab${activeTab === "live" ? " view-tab--active" : ""}`}
            onClick={() => setActiveTab("live")}
          >
            Live
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "analysis"}
            className={`view-tab${activeTab === "analysis" ? " view-tab--active" : ""}`}
            onClick={() => setActiveTab("analysis")}
          >
            Analiza statycznej pozycji
          </button>
        </div>
      </header>

      <div className="board-col" ref={boardColRef}>
        <div className="board-with-thermo">
          <BoardPanel
            fen={boardFen}
            bestMove={result?.best_move ?? null}
            showBestMove={showBestMove}
            boardWidth={currentBoardSize}
            boardOrientation={boardOrientation}
            allowDragging={activeTab === "analysis"}
            onPieceDrop={handleAnalysisPieceDrop}
          />
          {result ? (
            <Thermometer
              score={result.score}
              scoreType={result.score_type}
              mateIn={result.mate_in}
              loading={loading}
              boardOrientation={boardOrientation}
              boardSizePx={currentBoardSize}
            />
          ) : (
            <div
              className={`thermo-wrap${boardOrientation === "black" ? " thermo-wrap--from-black" : ""}`}
            >
              <div
                className="thermo-bar thermo-bar--placeholder"
                style={
                  {
                    "--thermo-black-pct": "50%",
                    "--thermo-white-pct": "50%",
                    height: `${currentBoardSize}px`,
                  } as CSSProperties
                }
              >
                <div className="thermo-black" />
                <div className="thermo-white" />
              </div>
            </div>
          )}
          {/* Materiał obok termometru */}
          <MaterialDisplay 
            fen={boardFen} 
            boardOrientation={boardOrientation}
            boardSizePx={currentBoardSize}
          />
        </div>
      </div>

      {activeTab === "live" && (
      <div className="live-col">
        <div className="live-section">
          <div className="live-section-header">
            <span className="live-section-title">Live Camera</span>
            <span className={`live-badge${detectorRunning ? " live-badge--active" : ""}`}>
              {detectorRunning ? "● LIVE" : "○ IDLE"}
            </span>
          </div>

          {/* Tabela stanu gry */}
          <table className="game-state-table">
            <tbody>
              <tr>
                <td className="gst-label">FEN</td>
                <td className="gst-val gst-val--fen">{gameState?.fen ?? STARTING_FEN}</td>
              </tr>
              <tr>
                <td className="gst-label">Kolej</td>
                <td className="gst-val">
                  {gameState
                    ? gameState.turn === "white"
                      ? "♟ Białe"
                      : "♙ Czarne"
                    : "—"}
                </td>
              </tr>
              <tr>
                <td className="gst-label">Ruch #</td>
                <td className="gst-val">{gameState?.move_number ?? "—"}</td>
              </tr>
              <tr>
                <td className="gst-label">Status</td>
                <td
                  className={`gst-val${
                    gameState?.is_check || gameState?.is_game_over
                      ? " gst-val--alert"
                      : ""
                  }`}
                >
                  {gameStatusLabel}
                </td>
              </tr>
            </tbody>
          </table>

          {/* Historia ostatnich 10 ruchów */}
          {gameState && gameState.history.length > 0 && (
            <div className="move-history-wrap">
              <span className="control-label">Ostatnie ruchy</span>
              <div className="move-history-list">
                {gameState.history.slice(-10).map((move, i) => (
                  <span key={i} className="history-move-chip">
                    {move}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Przyciski START / STOP / RESET */}
          <div className="live-btn-row">
            <button
              type="button"
              className={`live-btn live-btn--start${detectorRunning ? " live-btn--stop" : ""}`}
              onClick={() => void (detectorRunning ? handleStop() : handleStart())}
              disabled={liveLoading}
            >
              {liveLoading
                ? "…"
                : detectorRunning
                ? "⏹ Stop"
                : "▶ Start"}
            </button>
            <button
              type="button"
              className="live-btn live-btn--reset"
              onClick={() => void handleReset()}
              disabled={liveLoading}
            >
              ↺ Reset
            </button>
          </div>

          {liveError && <div className="error-msg">⚠ {liveError}</div>}
          {occupancyError && <div className="error-msg occupancy-error"> {occupancyError}</div>}
        </div>

        {/* Przyciski kontroli planszy pod Live Camera */}
        <div className="board-controls">
          <button
            type="button"
            className={`arrow-toggle${showBestMove ? " arrow-toggle--active" : ""}`}
            onClick={() => setShowBestMove((v) => !v)}
            aria-pressed={showBestMove}
          >
            {showBestMove ? "⟵ Ukryj strzałkę" : "⟶ Pokaż najlepszy ruch"}
          </button>
          <button
            type="button"
            className="flip-board-btn"
            onClick={() => setBoardOrientation((o) => (o === "white" ? "black" : "white"))}
            title="Obróć szachownicę o 180°"
          >
            ⟲ Obróć
          </button>
          <button
            type="button"
            className="fullscreen-btn"
            onClick={() => setIsFullscreen(true)}
            title="Pełny ekran"
          >
            ⛶ Pełny ekran
          </button>
        </div>

        {/* Info pod przyciskami: kolej i najlepszy ruch */}
        <div className="board-info">
          {result ? (
            <>
              <div className="board-info-card">
                <div className="bic-label">Kolej</div>
                <div className="bic-val bic-val--turn">
                  {result.turn === "white" ? (
                    <>
                      <span className="bic-piece bic-piece--plate-light" aria-hidden>
                        ♟
                      </span>
                      <span>BIAŁE</span>
                    </>
                  ) : (
                    <>
                      <span className="bic-piece bic-piece--plate-dark" aria-hidden>
                        ♙
                      </span>
                      <span>CZARNE</span>
                    </>
                  )}
                </div>
              </div>
              <div className="board-info-card">
                <div className="bic-label">Najlepszy ruch</div>
                <div className="bic-val bic-val--move">
                  {result.best_move ? (
                    <span className="move-arrow">
                      {result.best_move.slice(0, 2)} → {result.best_move.slice(2, 4)}
                      {result.best_move.length > 4 && result.best_move.slice(4)}
                    </span>
                  ) : (
                    <span className="move-none">—</span>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="board-info-placeholder">
              Wykonaj analizę, aby zobaczyć szczegóły pozycji.
            </div>
          )}
        </div>
      </div>
      )}

      {activeTab === "analysis" && (
      <div className="input-panel">
        <div className="section-divider" />

        {/* ===== SEKCJA STOCKFISH ===== */}
        <label className="input-label" htmlFor="fen-input">
          Pozycja FEN
        </label>
        <textarea
          id="fen-input"
          value={analysisFen}
          onChange={handleFenChange}
          onKeyDown={handleKey}
          placeholder="Wklej notację FEN…"
          spellCheck={false}
        />

        <div className="analysis-controls">
          <div className="control-group">
            <div className="control-head">
              <label className="control-label" htmlFor="depth-range">
                Głębokość analizy (depth)
              </label>
              <span className="control-value">{depth}</span>
            </div>
            <input
              id="depth-range"
              type="range"
              min={DEPTH_MIN}
              max={DEPTH_MAX}
              value={depth}
              onChange={(e) => setDepth(Number(e.target.value))}
            />
            <p className="control-hint">
              Wyższa wartość = dokładniej, ale wolniej (typowo 10–20).
            </p>
          </div>

          <div className="control-group">
            <span className="control-label" id="strength-mode-label">
              Siła silnika
            </span>
            <div
              className="strength-toggle"
              role="group"
              aria-labelledby="strength-mode-label"
            >
              <button
                type="button"
                className={`strength-toggle__btn${strengthMode === "full" ? " strength-toggle__btn--active" : ""}`}
                onClick={() => setStrengthMode("full")}
                aria-pressed={strengthMode === "full"}
              >
                Pełna siła
              </button>
              <button
                type="button"
                className={`strength-toggle__btn${strengthMode === "elo" ? " strength-toggle__btn--active" : ""}`}
                onClick={() => setStrengthMode("elo")}
                aria-pressed={strengthMode === "elo"}
              >
                Limit Elo
              </button>
              <button
                type="button"
                className={`strength-toggle__btn${strengthMode === "skill" ? " strength-toggle__btn--active" : ""}`}
                onClick={() => setStrengthMode("skill")}
                aria-pressed={strengthMode === "skill"}
              >
                Skill 0–20
              </button>
            </div>
            <p className="control-hint">
              Elo: UCI 1320–3190 · Skill: skala Stockfish, 20 = max.
            </p>
          </div>

          {strengthMode === "elo" && (
            <div className="control-group">
              <div className="control-head">
                <label className="control-label" htmlFor="elo-range">
                  Docelowe Elo
                </label>
                <span className="control-value">{eloLimit}</span>
              </div>
              <input
                id="elo-range"
                type="range"
                min={ELO_MIN}
                max={ELO_MAX}
                step={10}
                value={eloLimit}
                onChange={(e) => setEloLimit(Number(e.target.value))}
              />
              <p className="control-hint">
                Symulacja gracza o podanym rankingu (Stockfish UCI).
              </p>
            </div>
          )}

          {strengthMode === "skill" && (
            <div className="control-group">
              <div className="control-head">
                <label className="control-label" htmlFor="skill-range">
                  Skill Level
                </label>
                <span className="control-value">{skillLevel}</span>
              </div>
              <input
                id="skill-range"
                type="range"
                min={0}
                max={20}
                value={skillLevel}
                onChange={(e) => setSkillLevel(Number(e.target.value))}
              />
              <p className="control-hint">
                0 = bardzo słaby, 20 = pełna siła silnika (~3800).
              </p>
            </div>
          )}
        </div>

        <div className="btn-row">
          <button
            type="button"
            className="eval-btn"
            onClick={() => void evaluate()}
            disabled={loading || !fen.trim()}
          >
            {loading ? "Analizuję…" : "Analizuj"}
          </button>
          <button
            type="button"
            className="reset-btn"
            onClick={() => {
              setResult(null);
              setError(null);
              setAnalysisFen(STARTING_FEN);
              setDepth(DEFAULT_DEPTH);
              setStrengthMode("full");
              setEloLimit(DEFAULT_ELO);
              setSkillLevel(DEFAULT_SKILL);
            }}
            title="Wyczyść wyniki analizy i przywróć domyślne ustawienia silnika"
          >
            Wyczyść
          </button>
        </div>
        {error && <div className="error-msg">⚠ {error}</div>}
      </div>
      )}

    </main>
    
    {/* Fullscreen overlay */}
    {isFullscreen && (
      <div 
        className="fullscreen-overlay"
        onClick={(e) => {
          if (e.target === e.currentTarget) {
            setIsFullscreen(false);
          }
        }}
      >
        <div className="fullscreen-board-container">
          <div className="fullscreen-board-with-thermo">
            <BoardPanel
              fen={boardFen}
              bestMove={result?.best_move ?? null}
              showBestMove={showBestMove}
              boardWidth={fullscreenBoardSize}
              boardOrientation={boardOrientation}
              allowDragging={activeTab === "analysis"}
              onPieceDrop={handleAnalysisPieceDrop}
            />
            {result ? (
              <Thermometer
                score={result.score}
                scoreType={result.score_type}
                mateIn={result.mate_in}
                loading={loading}
                boardOrientation={boardOrientation}
                boardSizePx={fullscreenBoardSize}
              />
            ) : (
              <div
                className={`thermo-wrap${boardOrientation === "black" ? " thermo-wrap--from-black" : ""}`}
              >
                <div
                  className="thermo-bar thermo-bar--placeholder"
                  style={
                    {
                      "--thermo-black-pct": "50%",
                      "--thermo-white-pct": "50%",
                      height: `${fullscreenBoardSize}px`,
                    } as CSSProperties
                  }
                >
                  <div className="thermo-black" />
                  <div className="thermo-white" />
                </div>
              </div>
            )}
            <MaterialDisplay 
              fen={boardFen} 
              boardOrientation={boardOrientation}
              boardSizePx={fullscreenBoardSize}
            />
          </div>
        </div>
      </div>
    )}
    </>
  );
}
