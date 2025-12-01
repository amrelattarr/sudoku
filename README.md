Sudoku Solver Project (Backtracking + Cultural Algorithm)
=====================================================

Files:
- sudoku.py            : Backtracking solver with step recording.
- cultural_solver.py   : Cultural Algorithm (CA) implementation — population + belief space.
- gui.py               : Tkinter GUI to enter puzzles, choose algorithm and solve (9x9).

Quick Start (local):
1. Ensure you have Python 3.8+ installed.
2. Unzip or copy the project folder to your machine.
3. Run: python gui.py
   (A Tkinter window will open for entering puzzles or loading .json puzzles.)

Notes & Design:
- Backtracking solver is deterministic and will always find a solution for valid puzzles (unique or not).
- Cultural Algorithm is a heuristic evolutionary approach. It tries to evolve full candidate boards guided by a belief space derived from top individuals.
  CA may be slower and is not guaranteed to find a perfect solution quickly, but it demonstrates the CA design and provides comparative metrics.
- The GUI includes 'Step (BT)' which visually shows placements and backtracks performed by the backtracking algorithm.

Customization:
- You may tweak population size and max iterations in gui.py when using Cultural Algorithm.
- The code is organized for clarity and educational use; you can extend the Cultural solver with better operators (block-aware crossover, targeted repair, etc.).

How it meets your requirements:
- Two algorithms implemented: Backtracking and Cultural Algorithm (population + belief space).
- GUI included for perfect user experience (enter puzzles, save/load, step visualization, performance metrics).
- Outputs performance metrics and visual explanations for backtracking.

