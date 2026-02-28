import { useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import axios from 'axios'
import commentatorImg from "../assets/cat-commentator.jpeg";
import "../styles/chess-commentator.css";


export default function ChessCommentator() {
    const [game, setGame] = useState(new Chess());
    const [currentComment, setCurrentComment] = useState<string>("Welcome to the match!");
    const [moveIndex, setMoveIndex] = useState(0);

        
    function onDrop({ sourceSquare, targetSquare }: { sourceSquare: string; targetSquare: string | null }) {
        if (!targetSquare) return false;
        const gameCopy = new Chess(game.fen());
        const move = gameCopy.move({ from: sourceSquare, to: targetSquare, promotion: "q" });
        if (!move) return false;

        commentary(game.fen(), move.from + move.to, gameCopy.fen()).then((comment) => {
            setCurrentComment(comment);
        });

        setGame(gameCopy);
        const next = moveIndex + 1;
        setMoveIndex(next);
        return true;
    }

    const commentary = async (fen_before: string, move: string, fen_after: string) => {
        const response = await axios.post("http://localhost:5001/commentary", {
            fen_before,
            move,
            fen_after
        })

        return response.data.comment
    }

    return (
        <div className="page">

        {/* Left column: photo + commentary */}
        <div className="leftCol">
            <div className="photoBox">
            <img src={commentatorImg} alt="Commentator" />
            </div>
            <div className="commentBox">
            <span className="commentLabel">🎙 LIVE COMMENTARY</span>
            <p className="commentText">{currentComment}</p>
            </div>
        </div>

        {/* Right column: chessboard centered */}
        <div className="rightCol">
            <div style={{ width: "800px", height: "800px", position: "relative"}}>
                <div style={{ width: "800px", height: "800px", opacity: game.isGameOver() ? 0.5 : 1 }}>
                    <Chessboard
                        options={{
                            position: game.fen(),
                            onPieceDrop: onDrop,
                            darkSquareStyle: { backgroundColor: "#b58863" },
                            lightSquareStyle: { backgroundColor: "#f0d9b5" },
                    }}
                    />
                
                </div>
                    
                {game.isGameOver() && 
                    <div className="overlay">
                        <p className="overlayTitle">
                            {game.isCheckmate()
                            ? (game.turn() === "w" ? "Black Wins!" : "White Wins!")
                            : "Draw!"}
                        </p>
                        <button className="overlayButton" onClick={() => {
                            setGame(new Chess());
                            setCurrentComment("Welcome to the match!");
                            setMoveIndex(0);
                        }}>
                            Play Again
                        </button>
                    </div>
                }
                
            </div>
        </div>

        </div>
    );
}