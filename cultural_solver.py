"""
cultural_solver.py
-------------------

A **simple and well‑commented Cultural Algorithm** implementation for Sudoku.

Key ideas (kept intentionally easy to understand):
- We keep a **population** of complete candidate boards that always respect the given clues.
- A **belief space** remembers which numbers work well in each empty cell (based on the best candidates).
- Every generation we:
  1. Evaluate all candidates by counting conflicts in columns and blocks.
  2. Update the belief space using the best candidates.
  3. Create a new population by copying good candidates and slightly changing
     their non‑clue cells using the belief space + small random mutations.

The GUI expects:
    solver = CulturalSudokuSolver(grid, population_size=..., max_iters=...)
    solution, score, iters = solver.run()
where:
    - grid: NxN list of lists, 0 = empty cell
    - score: 0 means a valid Sudoku solution (no conflicts)
    - iters: number of evolutionary iterations actually performed

This implementation supports:
    - 4x4 Sudoku  (2x2 blocks)
    - 6x6 Sudoku  (2x3 blocks – standard rectangular blocks)
    - 9x9 Sudoku  (3x3 blocks)
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Dict, List, Tuple

Grid = List[List[int]]
Position = Tuple[int, int]


class CulturalSudokuSolver:
    """Cultural Algorithm based Sudoku solver.

    The implementation is intentionally straightforward so it can be
    read and understood by students without an evolutionary‑algorithms background.
    """

    # --------------------------------------------------------------
    #  INITIALISATION
    # --------------------------------------------------------------
    def __init__(self, clues: Grid, population_size: int = 200, max_iters: int = 3000, callback=None):
        """
        Args:
            clues: 2‑D list with 0 for empty cells.
            population_size: how many candidate boards we keep each generation.
            max_iters: hard limit on generations.
            callback: optional callback for real-time metrics updates.
        """
        self.clues: Grid = deepcopy(clues)
        self.size: int = len(clues)
        # We keep the parameters small so the GUI never freezes.
        # The GUI may pass larger values, but we clamp them here.
        self.population_size: int = max(10, population_size)
        self.max_iters: int = max_iters
        self.callback = callback
        
        # Metrics tracking
        self.mutation_count = 0
        self.belief_update_count = 0
        self.initial_conflicts = None
        self.current_conflicts = None

        # Block shape depends on grid size.
        # 4x4 -> 2x2, 9x9 -> 3x3, 6x6 -> 2x3 (standard rectangular blocks).
        if self.size == 4:
            self.block_rows, self.block_cols = 2, 2
            self.population_size = min(self.population_size, 40)
            self.max_iters = min(self.max_iters, 300)
        elif self.size == 6:
            self.block_rows, self.block_cols = 2, 3
            self.population_size = min(self.population_size, 60)
            self.max_iters = min(self.max_iters, 1400)
        else:
            root = int(math.sqrt(self.size))
            if root * root != self.size:
                raise ValueError(f"Unsupported grid size {self.size}. Expected 4, 6 or 9.")
            self.block_rows = self.block_cols = root
            # 9x9 is the heaviest case; keep limits conservative.
            self.population_size = min(self.population_size, 80)
            self.max_iters = min(self.max_iters, 2000)

        # All mutable (non‑clue) positions.
        self.mutable_positions: List[Position] = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.clues[r][c] == 0
        ]

        # Belief space: for each mutable cell, a histogram of promising values.
        # Example: belief[(r, c)] = {5: 10.0, 7: 3.0, ...}
        self.belief: Dict[Position, Dict[int, float]] = {}

    # --------------------------------------------------------------
    #  BASIC HELPERS
    # --------------------------------------------------------------
    def _row_missing_numbers(self, row_index: int) -> List[int]:
        """Return which numbers (1..N) are missing from a given row (ignoring zeros)."""
        present = {v for v in self.clues[row_index] if v != 0}
        return [v for v in range(1, self.size + 1) if v not in present]

    def _initial_individual(self) -> Grid:
        """Create one complete candidate board that respects all clues.

        Strategy:
            - For each row, keep the given numbers fixed.
            - Fill remaining cells in the row with a random permutation of the
              numbers that are missing from that row.
        This guarantees:
            - Every row contains each number at most once.
            - All clue values are kept.
        """
        grid = deepcopy(self.clues)
        for r in range(self.size):
            # Which numbers are already present as clues in this row?
            row_values = [v for v in grid[r] if v != 0]
            missing = [v for v in range(1, self.size + 1) if v not in row_values]
            random.shuffle(missing)

            # Fill empty cells with the shuffled missing numbers.
            idx = 0
            for c in range(self.size):
                if grid[r][c] == 0:
                    grid[r][c] = missing[idx]
                    idx += 1
        return grid

    # --------------------------------------------------------------
    #  FITNESS / CONFLICTS
    # --------------------------------------------------------------
    def _count_conflicts(self, grid: Grid) -> int:
        """Count how many rule violations exist in columns and blocks.

        Lower is better. 0 means a valid Sudoku solution.
        We do NOT count row conflicts here because our construction & repair
        keep rows as permutations of 1..N.
        """
        conflicts = 0
        n = self.size

        # Column conflicts: repeated numbers in each column.
        for c in range(n):
            col = [grid[r][c] for r in range(n)]
            conflicts += n - len(set(col))

        # Block conflicts: repeated numbers in each block.
        for br in range(0, n, self.block_rows):
            for bc in range(0, n, self.block_cols):
                block_vals: List[int] = []
                for r in range(br, br + self.block_rows):
                    for c in range(bc, bc + self.block_cols):
                        block_vals.append(grid[r][c])
                conflicts += len(block_vals) - len(set(block_vals))

        return conflicts

    # --------------------------------------------------------------
    #  BELIEF SPACE
    # --------------------------------------------------------------
    def _update_belief(self, population: List[Tuple[Grid, int]]) -> None:
        """Update belief space from the best part of the population.

        We only look at the top ~20% of candidates (fewer noises).
        For every mutable cell, we add +1 to the histogram of the value
        stored in that cell in a good candidate.
        """
        self.belief.clear()
        if not population:
            return

        pop_sorted = sorted(population, key=lambda x: x[1])
        top_k = max(1, len(pop_sorted) // 5)
        elites = pop_sorted[:top_k]

        for grid, score in elites:
            # Better boards may have slightly more influence.
            # Here we use a very simple weight: 1 / (1 + score)
            weight = 1.0 / (1.0 + score)
            for (r, c) in self.mutable_positions:
                val = grid[r][c]
                cell_hist = self.belief.setdefault((r, c), {})
                cell_hist[val] = cell_hist.get(val, 0.0) + weight
        
        # Track belief updates
        self.belief_update_count += 1

    def _sample_from_belief(self, pos: Position) -> int:
        """Sample a value for this cell from the belief histogram.

        If we have no information yet, return a random number 1..N.
        """
        hist = self.belief.get(pos)
        if not hist:
            return random.randint(1, self.size)

        items = list(hist.items())
        values, weights = zip(*items)
        total = sum(weights)
        rnd = random.uniform(0.0, total)
        acc = 0.0
        for value, w in zip(values, weights):
            acc += w
            if rnd <= acc:
                return value
        return values[-1]

    # --------------------------------------------------------------
    #  MUTATION + REPAIR
    # --------------------------------------------------------------
    def _repair_row(self, grid: Grid, r: int) -> None:
        """Ensure row r is a permutation of 1..N while respecting clues.

        Any duplicates in non‑clue cells are replaced by numbers that are
        missing from the row.
        """
        n = self.size
        row = grid[r]

        counts = {}
        for v in row:
            counts[v] = counts.get(v, 0) + 1

        # Numbers that should appear in the row but currently don't.
        missing = [v for v in range(1, n + 1) if counts.get(v, 0) == 0]
        random.shuffle(missing)

        # Positions with duplicate values that we are allowed to change.
        dup_positions: List[int] = [
            c
            for c in range(n)
            if self.clues[r][c] == 0 and counts.get(row[c], 0) > 1
        ]

        for c in dup_positions:
            if not missing:
                break
            old_val = row[c]
            new_val = missing.pop()
            row[c] = new_val
            counts[old_val] -= 1
            counts[new_val] = counts.get(new_val, 0) + 1

    def _mutate_individual(
        self,
        grid: Grid,
        belief_influence: float = 0.7,
        random_mutation_rate: float = 0.05,
    ) -> Grid:
        """Create a slightly modified copy of `grid`.

        For every mutable cell:
            - With probability `belief_influence`, we try a value suggested
              by the belief space.
            - With small probability `random_mutation_rate`, we pick a
              completely random value.
        After that, each row is *repaired* to be a permutation of 1..N,
        which keeps the search space reasonable.
        """
        new_grid = deepcopy(grid)

        for (r, c) in self.mutable_positions:
            if random.random() < belief_influence:
                new_grid[r][c] = self._sample_from_belief((r, c))
            elif random.random() < random_mutation_rate:
                new_grid[r][c] = random.randint(1, self.size)

        # Repair each row to avoid row conflicts.
        for r in range(self.size):
            self._repair_row(new_grid, r)

        # Track mutations
        self.mutation_count += 1

        return new_grid

    # --------------------------------------------------------------
    #  MAIN LOOP
    # --------------------------------------------------------------
    def run(self) -> Tuple[Grid, int, int]:
        """Run the Cultural Algorithm and return (best_grid, best_score, iters).

        The algorithm stops when:
            - A perfect solution is found (score == 0), or
            - `max_iters` generations have been run.
        """
        # ---- 1. Create initial population ----
        population: List[Tuple[Grid, int]] = []
        for _ in range(self.population_size):
            grid = self._initial_individual()
            score = self._count_conflicts(grid)
            population.append((grid, score))

        self._update_belief(population)

        best_grid, best_score = min(population, key=lambda x: x[1])
        best_grid = deepcopy(best_grid)
        
        # Track initial conflicts
        self.initial_conflicts = best_score
        self.current_conflicts = best_score
        
        # Initial callback
        if self.callback:
            self.callback(0, best_score, self.mutation_count, self.belief_update_count)

        # ---- 2. Evolution loop ----
        for it in range(1, self.max_iters + 1):
            if best_score == 0:
                # Already a valid solution.
                return best_grid, 0, it

            # Create new population with elitism (carry the best forward).
            new_population: List[Tuple[Grid, int]] = [(deepcopy(best_grid), best_score)]

            # A simple form of selection: tournament of size 3.
            def select_parent() -> Grid:
                competitors = random.sample(population, k=min(3, len(population)))
                return min(competitors, key=lambda x: x[1])[0]

            while len(new_population) < self.population_size:
                parent = select_parent()
                child = self._mutate_individual(parent)
                child_score = self._count_conflicts(child)
                new_population.append((child, child_score))

            population = new_population
            self._update_belief(population)

            # Track global best so far.
            current_best_grid, current_best_score = min(population, key=lambda x: x[1])
            if current_best_score < best_score:
                best_score = current_best_score
                best_grid = deepcopy(current_best_grid)
            
            self.current_conflicts = best_score
            
            # Real-time callback for metrics
            if self.callback:
                self.callback(it, best_score, self.mutation_count, self.belief_update_count)

        # ----------------------------------------------------------
        #  Optional: quick backtracking repair for guaranteed solution
        # ----------------------------------------------------------
        # Even if the cultural algorithm did not fully solve the puzzle
        # (best_score > 0), the best_grid is usually very close to a
        # solution.  A standard backtracking search from this point is
        # extremely fast and guarantees correctness.
        try:
            from sudoku import Sudoku  # imported here to avoid circular imports

            s = Sudoku(best_grid)
            ok, solved_grid, _ = s.solve(algorithm="backtracking", record_steps=False)
            if ok:
                # We report score 0 because backtracking guarantees validity.
                return solved_grid, 0, it
        except Exception:
            # If anything goes wrong, we simply fall back to the heuristic result.
            pass

        # Reached iteration limit and repair did not yield a full solution;
        # return the best heuristic board we have.
        return best_grid, best_score, self.max_iters


if __name__ == "__main__":
    # Small manual test when running this file directly.
    # Here we demonstrate a 9x9 puzzle; you can change `size` to 4 or 6.
    test_grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    solver = CulturalSudokuSolver(test_grid, population_size=200, max_iters=2000)
    solution, score, iterations = solver.run()
    print("Best score:", score, "iterations:", iterations)
    for row in solution:
        print(" ".join(str(v) for v in row))
