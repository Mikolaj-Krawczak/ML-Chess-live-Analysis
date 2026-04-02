import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, ChangeEvent } from "react";
import "./App.css";
import BoardPanel from "./BoardPanel";

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

type StrengthMode = "full" | "elo" | "skill";

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
        <span className="thermo-label thermo-label--black">
          <span className="thermo-piece" aria-hidden>
            ♙
          </span>
          <span className="thermo-label-text">CZARNE</span>
        </span>
        <span className="thermo-label thermo-label--white">
          <span className="thermo-piece" aria-hidden>
            ♟
          </span>
          <span className="thermo-label-text">BIAŁE</span>
        </span>
      </div>

      <div
        className="thermo-bar"
        aria-label="Ocena pozycji"
        style={
          {
            "--thermo-black-pct": `${blackPercent}%`,
            "--thermo-white-pct": `${whitePercent}%`,
            "--thermo-split-bottom": `${whitePercent}%`,
            "--thermo-split-left": `${100 - whitePercent}%`,
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
        <div className="thermo-divider" />
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
  const [fen, setFen] = useState(STARTING_FEN);
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

  // Ref do bieżących ustawień silnika — pozwala wywołać analizę ze świeżymi
  // wartościami bez dodawania ich do deps useEffect nasłuchującego na FEN
  const engineSettingsRef = useRef({ depth, strengthMode, eloLimit, skillLevel });
  useEffect(() => {
    engineSettingsRef.current = { depth, strengthMode, eloLimit, skillLevel };
  }, [depth, strengthMode, eloLimit, skillLevel]);

  const evaluate = useCallback(async (fenOverride?: string) => {
    const fenToUse = fenOverride ?? fen;
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
  }, [fen]);

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !loading) void evaluate();
  };

  // Auto-analiza po wklejeniu FEN — debounce 600 ms, żeby nie strzelać
  // zapytania przy każdym znaku wpisywanym ręcznie
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleFenChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setFen(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      void evaluate(val);
    }, 600);
  };

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

  return (
    <main className="app">
      <header className="header">
        <h1>
          Chess <span className="title-live">Live</span>{" "}
          <span className="title-analysis">Analysis</span>
        </h1>
        <p>
          Analiza pozycji · <strong>Stockfish 18</strong> · silnik UCI · MVP v0.1
        </p>
      </header>

      <div className="board-col" ref={boardColRef}>
        <BoardPanel
          fen={fen}
          bestMove={result?.best_move ?? null}
          showBestMove={showBestMove}
          boardWidth={boardWidth}
          boardOrientation={boardOrientation}
        />
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
        </div>

        {/* Info pod szachownicą: kolej i najlepszy ruch */}
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

      <div className="input-panel">
        <label className="input-label" htmlFor="fen-input">
          Pozycja FEN
        </label>
        <textarea
          id="fen-input"
          value={fen}
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
              setFen(STARTING_FEN);
              setResult(null);
              setError(null);
              setDepth(DEFAULT_DEPTH);
              setStrengthMode("full");
              setEloLimit(DEFAULT_ELO);
              setSkillLevel(DEFAULT_SKILL);
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
            <span className="thermo-label thermo-label--black">
              <span className="thermo-piece" aria-hidden>
                ♙
              </span>
            <span className="thermo-label-text">CZARNE</span>
          </span>
          <span className="thermo-label thermo-label--white">
            <span className="thermo-piece" aria-hidden>
              ♟
            </span>
            <span className="thermo-label-text">BIAŁE</span>
            </span>
          </div>
          <div
            className="thermo-bar thermo-bar--placeholder"
            style={
              {
                "--thermo-black-pct": "50%",
                "--thermo-white-pct": "50%",
                "--thermo-split-bottom": "50%",
                "--thermo-split-left": "50%",
              } as CSSProperties
            }
          >
            <div className="thermo-black" />
            <div className="thermo-white" />
          </div>
        </div>
      )}

      <div className="info-cards">
        {!result && (
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
    </main>
  );
}
