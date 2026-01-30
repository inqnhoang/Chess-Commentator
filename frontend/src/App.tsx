import { Chessboard } from './Chessboard';

function App() {
  return <Chessboard options={{ 
    position: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',
    boardOrientation: 'white'
  }} />;
}

export default App;