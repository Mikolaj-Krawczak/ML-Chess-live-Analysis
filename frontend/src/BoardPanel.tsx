import { useMemo } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";

type Arrow = {
  startSquare: string;
  endSquare: string;
  color: string;
};

interface BoardPanelProps {
  fen: string;
  bestMove: string | null;
  showBestMove: boolean;
  boardWidth: number;
  boardOrientation: "white" | "black";
  allowDragging?: boolean;
  onPieceDrop?: (args: {
    sourceSquare: string;
    targetSquare: string | null;
    piece: unknown;
  }) => boolean;
}

/**
 * Controlled chessboard view.
 * The FEN position source is managed above — here we only render.
 * Once ready, this will receive input from the CV model — no changes needed.
 */
export default function BoardPanel({
  fen,
  bestMove,
  showBestMove,
  boardWidth,
  boardOrientation,
  allowDragging = false,
  onPieceDrop,
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
        <p className="board-invalid-msg">Invalid FEN notation</p>
      </div>
    );
  }

  return (
    <div
      className="board-panel"
      style={{
        width: `${boardWidth}px`,
        height: `${boardWidth}px`,
      }}
    >
      <Chessboard
        options={{
          id: "main-board",
          position: fen,
          boardOrientation,
          showNotation: true,
          allowDragging,
          allowDrawingArrows: false,
          onPieceDrop,
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
