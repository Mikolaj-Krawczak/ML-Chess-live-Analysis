import { useCallback, useState } from "react";
import type { KeyboardEvent } from "react";
import "./App.css";

const API = "http://localhost:8000";

const STARTING_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/** Odpowiedź z backendu FastAPI /evaluate (EvalResponse) */
export interface EvalResponse {
  score: number;
  score_type: "cp" | "mate";
  mate_in: number | null;
  best_move: string | null;
  turn: "white" | "black";
  is_valid: boolean;
}

// Klamp wartości do przedziału
const clamp = (v: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, v));

// Zamień ocenę na etykietę tekstową
function formatScore(
  score: number,
  scoreType: "cp" | "mate",
  mateIn: number | null
): string {
  if (scoreType === "mate") {
    if (mateIn == null) return "M?";
    return mateIn > 0 ? `+M${mateIn}` : `-M${Math.abs(mateIn)}`;
  }
  return score >= 0 ? `+${score.toFixed(2)}` : score.toFixed(2);
}

interface ThermometerProps {
  score: number;
  scoreType: "cp" | "mate";
  mateIn: number | null;
  loading: boolean;
}

// Termometr: 0% = czarne, 100% = białe (skala ok. ±10 pionów)
function Thermometer({
  score,
  scoreType,
  mateIn,
  loading,
}: ThermometerProps) {
  const CAP = 10;
  const rawPercent =
    scoreType === "mate"
      ? mateIn != null && mateIn > 0
        ? 100
        : 0
      : 50 + (clamp(score, -CAP, CAP) / CAP) * 50;

  const whitePercent = clamp(rawPercent, 2, 98);
  const blackPercent = 100 - whitePercent;

  const label = formatScore(score, scoreType, mateIn);
  const advantage =
    score > 0.2 ? "white" : score < -0.2 ? "black" : "equal";

  return (
    <div className="thermo-wrap">
      <div className="thermo-labels">
        <span className="label-black">♟ Czarne</span>
        <span className="label-white">♙ Białe</span>
      </div>

      <div className="thermo-bar" aria-label="Ocena pozycji">
        <div
          className="thermo-black"
          style={{
            height: `${blackPercent}%`,
            transition: loading
              ? "none"
              : "height 0.6s cubic-bezier(0.34,1.56,0.64,1)",
          }}
        />
        <div
          className="thermo-white"
          style={{
            height: `${whitePercent}%`,
            transition: loading
              ? "none"
              : "height 0.6s cubic-bezier(0.34,1.56,0.64,1)",
          }}
        />
        <div
          className="thermo-divider"
          style={{ bottom: `${whitePercent}%` }}
        />
      </div>

      <div
        className={`score-badge score-${advantage} ${loading ? "pulse" : ""}`}
      >
        {loading ? "…" : label}
      </div>
    </div>
  );
}

interface BestMoveDisplayProps {
  move: string | null;
}

function BestMoveDisplay({ move }: BestMoveDisplayProps) {
  if (!move) return null;
  const from = move.slice(0, 2);
  const to = move.slice(2, 4);
  return (
    <div className="best-move">
      <span className="move-label">Najlepszy ruch</span>
      <span className="move-arrow">
        {from} → {to}
      </span>
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

export default function App() {
  const [fen, setFen] = useState(STARTING_FEN);
  const [result, setResult] = useState<EvalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const evaluate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: fen.trim(), depth: 18 }),
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
  }, [fen]);

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !loading) void evaluate();
  };

  return (
    <div className="app">
      <header className="header">
        <h1>
          Chess <span>Vision</span>
        </h1>
        <p>Analiza pozycji · Stockfish Engine · MVP v0.1</p>
      </header>

      <div className="input-panel">
        <label className="input-label" htmlFor="fen-input">
          Pozycja FEN
        </label>
        <textarea
          id="fen-input"
          value={fen}
          onChange={(e) => setFen(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Wklej notację FEN…"
          spellCheck={false}
        />
        <div className="btn-row">
          <button
            type="button"
            className="eval-btn"
            onClick={() => void evaluate()}
            disabled={loading || !fen.trim()}
          >
            {loading ? "Analizuję…" : "Analizuj ↵"}
          </button>
          <button
            type="button"
            className="reset-btn"
            onClick={() => {
              setFen(STARTING_FEN);
              setResult(null);
              setError(null);
            }}
          >
            Reset
          </button>
        </div>
        {error && <div className="error-msg">⚠ {error}</div>}
      </div>

      {result ? (
        <Thermometer
          score={result.score}
          scoreType={result.score_type}
          mateIn={result.mate_in}
          loading={loading}
        />
      ) : (
        <div className="thermo-wrap">
          <div className="thermo-labels">
            <span className="label-black">♟ Czarne</span>
            <span className="label-white">♙ Białe</span>
          </div>
          <div className="thermo-bar" style={{ opacity: 0.2 }}>
            <div className="thermo-black" style={{ height: "50%" }} />
            <div className="thermo-white" style={{ height: "50%" }} />
          </div>
        </div>
      )}

      <div className="info-cards">
        {result ? (
          <>
            <div className="info-card">
              <div className="ic-label">Kolej</div>
              <div className="ic-val">
                {result.turn === "white" ? "♙ Białe" : "♟ Czarne"}
              </div>
            </div>
            <BestMoveDisplay move={result.best_move} />
          </>
        ) : (
          <div className="empty-state">
            Wklej FEN i kliknij <strong>Analizuj</strong> lub naciśnij{" "}
            <strong>Enter</strong>.
            <br />
            <br />
            Przykład startowej pozycji jest już wpisany.
            <br />
            <br />
            FEN możesz skopiować z Lichess, Chess.com lub wygenerować ręcznie.
          </div>
        )}
      </div>
    </div>
  );
}
