const SIZE = 4;
const START_TILES = 2;
const WIN_VALUE = 2048;
const BEST_SCORE_KEY = "pretask1-2048-best-score";

const boardElement = document.getElementById("board");
const scoreElement = document.getElementById("score");
const bestScoreElement = document.getElementById("best-score");
const messageElement = document.getElementById("message");
const newGameButton = document.getElementById("new-game");

const tileColors = {
  0: { bg: "rgba(238, 228, 218, 0.35)", color: "transparent" },
  2: { bg: "#eee4da", color: "#776e65" },
  4: { bg: "#ede0c8", color: "#776e65" },
  8: { bg: "#f2b179", color: "#f9f6f2" },
  16: { bg: "#f59563", color: "#f9f6f2" },
  32: { bg: "#f67c5f", color: "#f9f6f2" },
  64: { bg: "#f65e3b", color: "#f9f6f2" },
  128: { bg: "#edcf72", color: "#f9f6f2" },
  256: { bg: "#edcc61", color: "#f9f6f2" },
  512: { bg: "#edc850", color: "#f9f6f2" },
  1024: { bg: "#edc53f", color: "#f9f6f2" },
  2048: { bg: "#edc22e", color: "#f9f6f2" },
};

let board = [];
let score = 0;
let bestScore = Number(localStorage.getItem(BEST_SCORE_KEY)) || 0;
let gameOver = false;
let hasWon = false;

bestScoreElement.textContent = bestScore;

function createEmptyBoard() {
  return Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
}

function startGame() {
  board = createEmptyBoard();
  score = 0;
  gameOver = false;
  hasWon = false;
  messageElement.textContent = "开始游戏吧！";
  addRandomTile();
  addRandomTile();
  renderBoard();
  updateScore();
}

function getEmptyCells() {
  const cells = [];
  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      if (board[row][col] === 0) {
        cells.push({ row, col });
      }
    }
  }
  return cells;
}

function addRandomTile() {
  const emptyCells = getEmptyCells();
  if (!emptyCells.length) return;
  const { row, col } = emptyCells[Math.floor(Math.random() * emptyCells.length)];
  board[row][col] = Math.random() < 0.9 ? 2 : 4;
}

function updateScore() {
  scoreElement.textContent = score;
  if (score > bestScore) {
    bestScore = score;
    localStorage.setItem(BEST_SCORE_KEY, String(bestScore));
  }
  bestScoreElement.textContent = bestScore;
}

function getCellStyle(value) {
  if (tileColors[value]) {
    return tileColors[value];
  }

  return { bg: "#3c3a32", color: "#f9f6f2" };
}

function renderBoard() {
  boardElement.innerHTML = "";

  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      const value = board[row][col];
      const cell = document.createElement("div");
      const { bg, color } = getCellStyle(value);
      cell.className = `cell ${value === 0 ? "cell--empty" : "cell--active"}`;
      cell.style.backgroundColor = bg;
      cell.style.color = color;
      if (value >= 1024) {
        cell.style.fontSize = "clamp(18px, 4vw, 28px)";
      }
      cell.textContent = value === 0 ? "" : String(value);
      boardElement.appendChild(cell);
    }
  }
}

function slideAndMerge(line) {
  const compacted = line.filter((value) => value !== 0);
  const merged = [];

  for (let index = 0; index < compacted.length; index += 1) {
    const current = compacted[index];
    const next = compacted[index + 1];

    if (current !== undefined && current === next) {
      const newValue = current * 2;
      merged.push(newValue);
      score += newValue;
      index += 1;
      if (newValue === WIN_VALUE && !hasWon) {
        hasWon = true;
        messageElement.textContent = "恭喜你合成了 2048，可以继续挑战更高分！";
      }
    } else {
      merged.push(current);
    }
  }

  while (merged.length < SIZE) {
    merged.push(0);
  }

  return merged;
}

function reverseRows(matrix) {
  return matrix.map((row) => [...row].reverse());
}

function transpose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
}

function boardsEqual(first, second) {
  return JSON.stringify(first) === JSON.stringify(second);
}

function moveLeft() {
  board = board.map((row) => slideAndMerge(row));
}

function moveRight() {
  board = reverseRows(board);
  moveLeft();
  board = reverseRows(board);
}

function moveUp() {
  board = transpose(board);
  moveLeft();
  board = transpose(board);
}

function moveDown() {
  board = transpose(board);
  moveRight();
  board = transpose(board);
}

function hasAvailableMoves() {
  if (getEmptyCells().length > 0) return true;

  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      const current = board[row][col];
      const right = board[row][col + 1];
      const down = board[row + 1]?.[col];
      if (current === right || current === down) {
        return true;
      }
    }
  }

  return false;
}

function handleMove(direction) {
  if (gameOver) return;

  const snapshot = board.map((row) => [...row]);

  switch (direction) {
    case "ArrowLeft":
      moveLeft();
      break;
    case "ArrowRight":
      moveRight();
      break;
    case "ArrowUp":
      moveUp();
      break;
    case "ArrowDown":
      moveDown();
      break;
    default:
      return;
  }

  if (boardsEqual(snapshot, board)) {
    return;
  }

  addRandomTile();
  updateScore();
  renderBoard();

  if (!hasAvailableMoves()) {
    gameOver = true;
    messageElement.textContent = `游戏结束，最终得分：${score}`;
  } else if (!hasWon) {
    messageElement.textContent = "继续合并，冲击 2048！";
  }
}

function normalizeKey(key) {
  const map = {
    w: "ArrowUp",
    a: "ArrowLeft",
    s: "ArrowDown",
    d: "ArrowRight",
    W: "ArrowUp",
    A: "ArrowLeft",
    S: "ArrowDown",
    D: "ArrowRight",
  };
  return map[key] || key;
}

document.addEventListener("keydown", (event) => {
  const direction = normalizeKey(event.key);
  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(direction)) {
    event.preventDefault();
    handleMove(direction);
  }
});

newGameButton.addEventListener("click", startGame);

startGame();
