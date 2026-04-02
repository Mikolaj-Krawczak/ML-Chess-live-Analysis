import { useMemo } from "react";
import { Chessboard } from "react-chessboard";
import type { Arrow } from "react-chessboard/dist/types";
import { Chess } from "chess.js";

interface BoardPanelProps {
  fen: string;
  bestMove: string | null;
  showBestMove: boolean;
  boardWidth: number;
  boardOrientation: "white" | "black";
}

/**
 * Kontrolowany widok szachownicy.
 * Źródło pozycji (FEN) jest zarządzane wyżej — tutaj tylko renderujemy.
 * Jutro zamiast textarea podepniesz output z modelu CV — ten komponent nie wymaga zmian.
 */
export default function BoardPanel({
  fen,
  bestMove,
  showBestMove,
  boardWidth,
  boardOrientation,
}: BoardPanelProps) {
  const isValidFen = useMemo(() => {
    try {
      new Chess(fen);
      return true;
    } catch {
      return false;
    }
  }, [fen]);

  const arrows: Arrow[] = useMemo(() => {
    if (!showBestMove || !bestMove || bestMove.length < 4) return [];
    const from = bestMove.slice(0, 2);
    const to = bestMove.slice(2, 4);
    return [{ startSquare: from, endSquare: to, color: "#111111" }];
  }, [showBestMove, bestMove]);

  if (!isValidFen) {
    return (
      <div className="board-panel board-panel--invalid">
        <p className="board-invalid-msg">Nieprawidłowa notacja FEN</p>
      </div>
    );
  }

  return (
    <div className="board-panel">
      <Chessboard
        options={{
          id: "main-board",
          position: fen,
          boardOrientation,
          showNotation: true,
          allowDragging: false,
          allowDrawingArrows: false,
          arrows,
          arrowOptions: {
            color: "#111111",
            secondaryColor: "rgba(79, 168, 120, 0.65)",
            tertiaryColor: "rgba(232, 168, 124, 0.65)",
            arrowLengthReducerDenominator: 3.2,
            sameTargetArrowLengthReducerDenominator: 1.8,
            arrowWidthDenominator: 12,
            activeArrowWidthMultiplier: 1.2,
            opacity: 1,
            activeOpacity: 1,
            arrowStartOffset: 0.3,
          },
          animationDurationInMs: 280,
          darkSquareStyle: {
            backgroundColor: "#6d8b74",
          },
          lightSquareStyle: {
            backgroundColor: "#e8dcc8",
          },
          boardStyle: {
            borderRadius: "8px",
            overflow: "hidden",
            boxShadow: "0 8px 32px rgba(0,0,0,0.45), inset 0 0 0 2px rgba(212,175,88,0.22)",
            width: `${boardWidth}px`,
            height: `${boardWidth}px`,
          },
        }}
      />
    </div>
  );
}
