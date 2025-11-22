<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=2c3e50&height=300&section=header&text=Checkmate%20AI&fontSize=90&animation=fadeIn&fontAlignY=38&desc=Intelligent%20Chess%20Engine%20%7C%20Minimax%20%2B%20Alpha-Beta&descAlignY=51&descAlign=62" alt="Checkmate AI Banner" />

  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-Minimax-red?style=for-the-badge" alt="Algorithm">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <br>
</div>

---

## ♟️ Overview

**Checkmate AI** is a lightweight yet powerful chess engine designed to evaluate positions, search optimal moves, and play competitive-level chess using modern AI techniques.

Built for developers, students, and chess enthusiasts, this engine focuses on code clarity without sacrificing performance. It uses a classical search-driven architecture, making it the perfect base for building your own bots or learning game theory.

---

## 🚀 Features

* 🧠 **Smart Search:** Implements Minimax algorithm with Alpha–Beta pruning for efficient decision making.
* ⚖️ **Heuristic Evaluation:** Considers material, mobility, king safety, and pawn structure.
* ⚡ **Fast Move Generation:** Optimized legality checks and board state management.
* 🔍 **Depth-Control:** Adjustable search depth for difficulty balancing.
* 🔌 **Modular Design:** Easy to integrate into GUIs, web apps, or existing bots.

---

## 🧠 How It Works

Checkmate AI follows a classical 4-step pipeline to determine the best move:

```mermaid
graph TD
    A[Current Board State] --> B(Generate Legal Moves)
    B --> C{Minimax Search}
    C -->|Recursive Depth| D[Evaluate Positions]
    D -->|Heuristics| E[Score: Material + Position]
    E --> C
    C --> F[Alpha-Beta Pruning]
    F --> G[Return Best Move]
````

1.  **Move Generation:** The engine calculates all possible legal moves from the current FEN string or board state.
2.  **Search:** It explores the game tree using Minimax.
3.  **Pruning:** Alpha-Beta pruning cuts off branches that are statistically worse, saving computation time.
4.  **Evaluation:** Leaf nodes are scored based on:
      * *Material:* (Queen=9, Rook=5, etc.)
      * *Position:* Control of the center.
      * *Safety:* King exposure.

-----

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 🐍 |
| **Core Logic** | Minimax + Alpha-Beta Pruning |
| **Data Structure** | 8x8 Matrix / FEN Strings |
| **Libraries** | NumPy (Optional), Standard Math Lib |

-----

## 📦 Installation

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/pratham-ctrl/checkmate-ai.git](https://github.com/pratham-ctrl/checkmate-ai.git)
    cd checkmate-ai
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

-----

## ▶️ Usage

### Option 1: Python Script

You can import the engine into your own projects easily.

```python
from checkmate_ai import ChessEngine

# Initialize the engine
engine = ChessEngine()

# Get the best move for the starting position (depth 3)
best_move = engine.get_best_move(fen="startpos", depth=3)

print(f"Engine recommends: {best_move}")
```

### Option 2: Command Line Interface (CLI)

Run the engine directly from the terminal to test positions.

```bash
python main.py
```

-----

## 🖼️ Demo

\<div align="center"\>
\<pre\>
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
⬜ ⬛ ⬜ ⬛ ⬜ ⬛ ⬜ ⬛
⬛ ⬜ ⬛ ⬜ ⬛ ⬜ ⬛ ⬜
⬜ ⬛ ⬜ ⬛ ⬜ ⬛ ⬜ ⬛
⬛ ⬜ ⬛ ⬜ ⬛ ⬜ ⬛ ⬜
♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖

> Engine calculating move...

\</pre\>
\</div\>

-----

## 📌 Roadmap

  - [ ] **Iterative Deepening:** For better time management.
  - [ ] **Transposition Table:** To cache previously evaluated positions.
  - [ ] **Opening Book:** Integrate standard opening lines (ECO).
  - [ ] **NN-Evaluation:** Add a neural network mode for positional scoring.
  - [ ] **GUI:** Build a web-based or desktop interface.

-----

## 🤝 Contributing

Contributions are welcome\! Whether it's optimizing the search function or adding a GUI.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

\<div align="center"\>
\<small\>Made with ❤️ by \<a href="//github.com/pratham-ctrl"</a\>\</small\>
\</div\>

```
