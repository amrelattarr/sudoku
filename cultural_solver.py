"""
cultural_solver.py
-------------------

A **PURE Cultural Algorithm** implementation for Sudoku (no backtracking or external help).

═══════════════════════════════════════════════════════════════════════════════
WHAT IS A CULTURAL ALGORITHM?
═══════════════════════════════════════════════════════════════════════════════
A Cultural Algorithm is an evolutionary algorithm that uses a "belief space"
to store and share knowledge learned from the best solutions. This knowledge
then guides the evolution of new solutions.

Think of it like a society:
- POPULATION: Individual candidate solutions (Sudoku boards)
- BELIEF SPACE: Cultural knowledge about what works (which numbers work in each cell)
- EVOLUTION: New generations learn from the belief space

═══════════════════════════════════════════════════════════════════════════════
KEY COMPONENTS (easy to understand):
═══════════════════════════════════════════════════════════════════════════════
1. **POPULATION**: Complete candidate Sudoku boards that respect clues
   
2. **BELIEF SPACE**: Histogram for each empty cell showing which values work well
   Example: belief[(2,3)] = {5: 10.0, 7: 3.0} means cell (2,3) often has 5 or 7
   
3. **FITNESS**: Count conflicts in columns and blocks (lower = better, 0 = solved)

4. **GENETIC OPERATORS**:
   - Selection: Tournament selection picks good parents
   - Crossover: Combine rows from two parents
   - Mutation: Change cells using belief space guidance + random changes
   - Repair: Fix row duplicates to maintain valid rows

═══════════════════════════════════════════════════════════════════════════════
EVOLUTION CYCLE (each generation):
═══════════════════════════════════════════════════════════════════════════════
  1. Evaluate all candidates by counting conflicts
  2. Update belief space from the top 20% best candidates (with decay!)
  3. Create new population:
     - Keep best solution (elitism)
     - 70% children via crossover + adaptive mutation
     - 30% children via mutation only
  4. Detect stagnation and apply escape mechanisms:
     - Adaptive mutation rates (increase when stuck)
     - Diversity injection (replace 20-50% every 20-100 stuck generations)
     - Belief decay (prevents overfitting to local optimum)
     - Simulated annealing (accept worse solutions when ≤5 conflicts and stuck ≥10 gens)
     - **BELIEF INVERSION** (try unpopular values when ≤5 conflicts and stuck ≥15 gens)
     - **BELIEF RESET** (clear poisoned beliefs when ≤5 conflicts and stuck ≥30 gens)
  5. Repeat until score = 0 (perfect solution) or max iterations

═══════════════════════════════════════════════════════════════════════════════
USAGE:
═══════════════════════════════════════════════════════════════════════════════
    solver = CulturalSudokuSolver(grid, population_size=200, max_iters=5000)
    solution, score, iters = solver.run()

where:
    - grid: NxN list of lists, 0 = empty cell
    - score: 0 means perfect solution (no conflicts)
    - iters: number of generations actually performed

═══════════════════════════════════════════════════════════════════════════════
SUPPORTED PUZZLE SIZES:
═══════════════════════════════════════════════════════════════════════════════
    - 4x4 Sudoku  (2x2 blocks)
    - 6x6 Sudoku  (2x3 blocks)
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
    def __init__(
        self,
        clues: Grid,
        population_size: int = 200,
        max_iters: int = 3000,
        callback=None,
        stop_event=None,
    ):
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
        self.stop_event = stop_event  # Optional cooperative cancellation (threading.Event-like)
        
        # Metrics tracking
        self.mutation_count = 0
        self.belief_update_count = 0
        self.initial_conflicts = None
        self.current_conflicts = None

        # Block shape depends on grid size.
        # 4x4 -> 2x2, 9x9 -> 3x3, 6x6 -> 2x3 (standard rectangular blocks).
        if self.size == 4:
            self.block_rows, self.block_cols = 2, 2
            # Small puzzles need fewer resources
            self.population_size = min(self.population_size, 50)
            self.max_iters = min(self.max_iters, 1500)  # Allow more for hard 4x4
        elif self.size == 6:
            self.block_rows, self.block_cols = 2, 3
            # Medium puzzles need moderate resources
            self.population_size = min(self.population_size, 120)
            self.max_iters = min(self.max_iters, 10000)  # Allow more for hard 6x6
        else:
            root = int(math.sqrt(self.size))
            if root * root != self.size:
                raise ValueError(f"Unsupported grid size {self.size}. Expected 4, 6 or 9.")
            self.block_rows = self.block_cols = root
            # 9x9 is the hardest case; needs MORE resources for PURE cultural algorithm
            # (no backtracking help). Larger population = more diversity.
            # More iterations = more time to converge to score 0.
            # IMPORTANT: Hard 9x9 puzzles may need 15,000+ iterations!
            self.population_size = min(self.population_size, 350)
            self.max_iters = min(self.max_iters, 20000)  # Allow up to 20k for very hard puzzles

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
        
        # Track stagnation for adaptive diversity control
        self.generations_without_improvement = 0
        self.last_best_score = float('inf')
        # Track true plateau length (only resets when best_score improves).
        # This should NOT reset due to diversity injection or annealing.
        self.plateau_generations = 0

        # Full restarts (multi-start) to escape persistent plateaus.
        # IMPORTANT: These are "hard" restarts: we do NOT keep any elite/best boards.
        # This matches the user's requirement: restart as if the algorithm was run again.
        self.restart_count = 0
        self.max_restarts = 8  # safety cap: prevents infinite restart loops

    # --------------------------------------------------------------
    #  BASIC HELPERS
    # --------------------------------------------------------------
    def _row_missing_numbers(self, row_index: int) -> List[int]:
        """Return which numbers (1..N) are missing from a given row (ignoring zeros)."""
        present = {v for v in self.clues[row_index] if v != 0}
        return [v for v in range(1, self.size + 1) if v not in present]

    def _initial_individual(self) -> Grid:
        """Create one complete candidate board that respects all clues.
        
        INITIALIZATION STRATEGY:
        ─────────────────────────────────────────────────────────────
        For Sudoku, we want to start with boards that are "partially valid":
        - All clue cells must keep their given values (mandatory)
        - All rows should have no duplicates (easy to enforce)
        - Columns and blocks will have conflicts (we'll evolve to fix these)
        
        ALGORITHM:
        ─────────────────────────────────────────────────────────────
        For each row:
            1. Keep all clue values in their positions
            2. Find which numbers are missing from the row
            3. Randomly shuffle those missing numbers
            4. Fill empty cells with the shuffled numbers
        
        EXAMPLE (4x4 Sudoku):
        ─────────────────────────────────────────────────────────────
        Clues:     [0, 3, 0, 0]   → Has 3, missing {1, 2, 4}
        Shuffle:   {1, 2, 4} → [4, 1, 2] (random)
        Result:    [4, 3, 1, 2]   → Valid row!
        
        GUARANTEES:
        ─────────────────────────────────────────────────────────────
        ✓ Every row is a permutation of 1..N (no row conflicts)
        ✓ All clue values are preserved
        ✗ Columns/blocks likely have conflicts (score > 0)
           → Evolution will fix these over generations!
        
        Returns:
            A complete NxN grid with valid rows
        """
        grid = deepcopy(self.clues)
        for r in range(self.size):
            missing = self._row_missing_numbers(r) 
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

    # ══════════════════════════════════════════════════════════════
    #  BELIEF SPACE (The "Cultural" Part!)
    # ══════════════════════════════════════════════════════════════
    def _update_belief(self, population: List[Tuple[Grid, int]], decay_factor: float = 0.8) -> None:
        """Update belief space from the ELITE part of the population.
        
        This is the KEY DIFFERENCE between Cultural Algorithms and regular GAs!
        
        THE BELIEF SPACE is like cultural knowledge passed down through generations.
        It remembers: "In cell (r,c), the value X appears often in good solutions."
        
        HOW IT WORKS:
        ─────────────────────────────────────────────────────────────
        1. CLEAR old beliefs (start fresh each generation)
           → This allows the belief to adapt as population improves
        
        2. IDENTIFY ELITES: Sort population by score, take top 20%
           → We only learn from the BEST solutions (ignore bad ones)
           → Top 20% means: if population=100, we learn from best 20
        
        3. BUILD HISTOGRAMS: For each elite solution:
           → Look at every empty cell (r, c)
           → See what number is there in this elite solution
           → Add to the histogram: belief[(r,c)][number] += weight
        
        4. WEIGHTING: Better solutions have more influence
           → A solution with score=0 has weight = 1.0 / (1+0) = 1.0
           → A solution with score=10 has weight = 1.0 / (1+10) ≈ 0.09
           → This means near-perfect solutions teach us more!
        
        EXAMPLE BELIEF STATE:
        ─────────────────────────────────────────────────────────────
        belief[(2, 3)] = {5: 12.5, 7: 3.2, 9: 1.1}
        → Cell at row 2, col 3 has value 5 in many good solutions (weight=12.5)
        → Value 7 appears sometimes (weight=3.2)
        → Value 9 appears rarely (weight=1.1)
        → When mutating, we'll likely try 5, sometimes 7, rarely 9
        
        Args:
            population: List of (grid, score) tuples for current generation
        """
        # DECAY old beliefs instead of clearing (prevents premature convergence!)
        # Decay factor: keep X% of old knowledge, add (1-X)% new knowledge
        # Smaller values forget faster (useful when belief becomes "poisoned").
        decay_factor = max(0.0, min(1.0, decay_factor))
        for pos in self.belief:
            for val in self.belief[pos]:
                self.belief[pos][val] *= decay_factor
        
        if not population:
            return

        # Sort by fitness: best solutions first (lowest score = best)
        pop_sorted = sorted(population, key=lambda x: x[1])
        top_k = max(1, len(pop_sorted) // 5)  # Top 20%
        elites = pop_sorted[:top_k]

        # Learn from each elite solution
        for grid, score in elites:
            # Calculate influence weight: better solutions have higher weight
            weight = 1.0 / (1.0 + score)
            
            # Update belief for each empty cell
            for (r, c) in self.mutable_positions:
                val = grid[r][c]  # What number is in this cell?
                cell_hist = self.belief.setdefault((r, c), {})  # Get histogram for this cell
                cell_hist[val] = cell_hist.get(val, 0.0) + weight  # Add weighted vote
        
        # Track how many times we've updated beliefs (for metrics/GUI)
        self.belief_update_count += 1

    def _sample_from_belief(self, pos: Position, invert: bool = False) -> int:
        """Sample a value for cell at position `pos` using WEIGHTED RANDOM SELECTION.
        
        This implements "belief-guided exploration" - we randomly pick values,
        but values that appear more often in good solutions are more likely.
        
        HOW IT WORKS:
        ─────────────────────────────────────────────────────────────
        Imagine we have: belief[(2,3)] = {5: 10.0, 7: 3.0, 9: 1.0}
        Total weight = 14.0
        
        We pick a random number between 0 and 14:
        - If random ∈ [0, 10): return 5   (71% chance)
        - If random ∈ [10, 13): return 7  (21% chance)
        - If random ∈ [13, 14): return 9  (7% chance)
        
        INVERT MODE (when stuck):
        If invert=True, we FLIP the weights - try unpopular values!
        - Value 5 (weight 10.0) becomes weight 4.0  (1/10.0 * 10 = 1, inverted)
        - Value 9 (weight 1.0) becomes weight 10.0  (1/1.0 * 10 = 10, highest!)
        This helps escape local optima by trying what we usually avoid.
        
        FALLBACK:
        If no belief exists for this cell yet (early generations), return random.
        
        Args:
            pos: (row, col) position of the cell
            invert: If True, prefer unpopular values (for escaping stuck states)
            
        Returns:
            A number from 1..N, weighted by belief (or inverted belief)
        """
        hist = self.belief.get(pos)
        
        # No belief for this cell yet? Return random value (exploration)
        if not hist:
            return random.randint(1, self.size)

        # Prepare for weighted random selection
        items = list(hist.items())  # [(value1, weight1), (value2, weight2), ...]
        values, weights_orig = zip(*items)  # Separate into two lists
        
        # INVERT weights if requested (try unpopular values when stuck!)
        if invert and len(values) > 1:
            # Invert: high weight becomes low, low becomes high
            max_weight = max(weights_orig)
            weights = tuple((max_weight - w + 0.1) for w in weights_orig)
        else:
            weights = weights_orig
        
        total = sum(weights)  # Total weight for normalization
        
        # Roulette wheel: pick a random point on the wheel
        rnd = random.uniform(0.0, total)
        acc = 0.0
        
        # Spin the wheel: accumulate weights until we hit our random point
        for value, w in zip(values, weights):
            acc += w
            if rnd <= acc:
                return value  # This is the chosen value!
        
        # Fallback (should rarely happen due to floating point)
        return values[-1]

    # --------------------------------------------------------------
    #  SIMULATED ANNEALING (Accept worse solutions to escape)
    # --------------------------------------------------------------
    def _simulated_annealing_mutation(self, grid: Grid, temperature: float) -> Grid:
        """Apply mutation that can ACCEPT WORSE solutions (escape local optima).
        
        Simulated Annealing allows accepting slightly worse solutions
        with probability based on temperature. This helps escape local optima.
        
        Args:
            grid: Current grid
            temperature: Higher temp = more willing to accept worse solutions
            
        Returns:
            Mutated grid (might be worse than input!)
        """
        current_grid = deepcopy(grid)
        current_score = self._count_conflicts(current_grid)
        
        # Try a few random swaps, accepting based on temperature
        num_attempts = 20  # Fast - just 20 attempts
        
        for _ in range(num_attempts):
            # Pick random row
            row_idx = random.randint(0, self.size - 1)
            mutable_in_row = [(row_idx, c) for c in range(self.size) 
                             if self.clues[row_idx][c] == 0]
            
            if len(mutable_in_row) < 2:
                continue
            
            # Swap two random cells
            pos1, pos2 = random.sample(mutable_in_row, 2)
            r1, c1 = pos1
            r2, c2 = pos2
            
            test_grid = deepcopy(current_grid)
            test_grid[r1][c1], test_grid[r2][c2] = test_grid[r2][c2], test_grid[r1][c1]
            test_score = self._count_conflicts(test_grid)
            
            # ACCEPT if better OR with probability based on temperature
            delta = test_score - current_score
            if delta <= 0:  # Better or equal
                current_grid = test_grid
                current_score = test_score
                if test_score == 0:
                    break
            elif random.random() < math.exp(-delta / temperature):
                # ACCEPT WORSE solution! (with probability)
                current_grid = test_grid
                current_score = test_score
        
        return current_grid
    
    # --------------------------------------------------------------
    #  CROSSOVER (Genetic Operator)
    # --------------------------------------------------------------
    def _crossover(self, parent1: Grid, parent2: Grid) -> Grid:
        """Combine two parent grids to create a child grid.
        
        For Sudoku, we use ROW-BASED CROSSOVER:
            - Randomly select which rows come from parent1 vs parent2
            - This preserves the row-validity that both parents have
            - Each row is already a permutation of 1..N in both parents
        
        Why this works:
            - Parent1 might have good column arrangements in some rows
            - Parent2 might have good block arrangements in other rows
            - Combining them gives the child a chance to inherit the best of both
        
        Args:
            parent1, parent2: Two complete Sudoku grids (both respect clues)
        
        Returns:
            A new grid that combines rows from both parents
        """
        child = deepcopy(self.clues)  # Start with clues only
        
        # For each row, randomly choose which parent to inherit from
        for r in range(self.size):
            if random.random() < 0.5:
                # Take this row from parent1
                child[r] = deepcopy(parent1[r])
            else:
                # Take this row from parent2
                child[r] = deepcopy(parent2[r])
        
        return child

    # --------------------------------------------------------------
    #  MUTATION + REPAIR
    # --------------------------------------------------------------
    def _repair_row(self, grid: Grid, r: int) -> None:
        """Ensure row r is a permutation of 1..N while respecting clues.
        
        WHY REPAIR IS NEEDED:
        After mutation or crossover, a row might have duplicate numbers.
        For example: [5, 3, 7, 3, 9, 1, 2, 8, 4] has two 3's but missing 6.
        
        REPAIR STRATEGY:
        1. Count how many times each number appears in the row
        2. Find which numbers are missing (should be there but aren't)
        3. Find positions with DUPLICATE values that we can change (non-clues only)
        4. Replace duplicates with missing numbers
        
        This guarantees every row has numbers 1..N exactly once (no row conflicts).
        
        Args:
            grid: The grid to modify (modified in-place)
            r: Row index to repair
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
        invert_belief: bool = False,
    ) -> Grid:
        """Create a slightly modified copy of `grid` using BELIEF-GUIDED MUTATION.
        
        This is where the CULTURAL part happens! The belief space guides mutations.

        For every mutable cell, we decide what value to place:
            1. With probability `random_mutation_rate` (default 5%):
               → Pick a RANDOM value from 1..N
               → This maintains diversity and helps escape local optima
            
            2. With probability `belief_influence` (default 70%):
               → Pick a value from the BELIEF SPACE
               → The belief space knows which values work well in each cell
               → This is learned from the best solutions we've seen
            
            3. Otherwise (remaining 25% by default):
               → Keep the ORIGINAL value from the parent
               → This preserves good patterns that already exist
        
        After modifying cells, we REPAIR each row to ensure it's still a valid
        permutation of 1..N (no duplicates in rows).
        
        Args:
            grid: Parent grid to mutate
            belief_influence: How often to use belief space (0.0 to 1.0)
            random_mutation_rate: How often to inject random values (0.0 to 1.0)
            invert_belief: If True, try UNPOPULAR values (escape stuck patterns)
        
        Returns:
            A new mutated grid
        
        SPECIAL MODE - BELIEF INVERSION (when stuck):
        ─────────────────────────────────────────────────────────────
        When stuck at 2-3 conflicts, the belief space is often POISONED
        with values that keep us stuck. Setting invert_belief=True flips
        the weights - we try values we normally avoid. This helps escape!
        """
        new_grid = deepcopy(grid)

        # Go through each empty cell (non-clue) and potentially modify it
        for (r, c) in self.mutable_positions:
            # First check: should we do a random mutation? (exploration)
            if random.random() < random_mutation_rate:
                new_grid[r][c] = random.randint(1, self.size)
            # Second check: should we use the belief space? (exploitation of knowledge)
            elif random.random() < belief_influence:
                # Use inverted belief when requested (try unpopular values to escape!)
                new_grid[r][c] = self._sample_from_belief((r, c), invert=invert_belief)
            # Otherwise: keep the parent's value (implicit else - do nothing)


        # Repair each row to avoid row conflicts.
        for r in range(self.size):
            self._repair_row(new_grid, r)

        # Track mutations
        self.mutation_count += 1

        return new_grid

    # --------------------------------------------------------------
    #  LOW-CONFLICT INTENSIFICATION (Memetic Local Search)
    # --------------------------------------------------------------
    def _cell_is_in_conflict(self, grid: Grid, r: int, c: int) -> bool:
        
        """Return True if grid[r][c] participates in a column or block duplicate."""
        v = grid[r][c]
        if v == 0:
            return False

        n = self.size

        # Column duplicate?
        col_count = 0
        for rr in range(n):
            if grid[rr][c] == v:
                col_count += 1
                if col_count > 1:
                    return True

        # Block duplicate?
        br = (r // self.block_rows) * self.block_rows
        bc = (c // self.block_cols) * self.block_cols
        block_count = 0
        for rr in range(br, br + self.block_rows):
            for cc in range(bc, bc + self.block_cols):
                if grid[rr][cc] == v:
                    block_count += 1
                    if block_count > 1:
                        return True

        return False

    def _row_mutable_cols(self, r: int) -> List[int]:
        """Return mutable column indices for a row (non-clues only)."""
        return [c for c in range(self.size) if self.clues[r][c] == 0]

    def _conflict_directed_row_swap_search(
        self,
        grid: Grid,
        max_attempts: int = 800,
        start_temperature: float = 0.35,
    ) -> Tuple[Grid, int]:
        """Local-search swaps within rows to escape the 'under-10-conflicts' stall.

        Why swaps?
        - Our representation keeps each row a permutation of 1..N.
        - Swapping two mutable cells in the SAME row preserves that constraint
          while allowing precise fixes in columns/blocks.

        Strategy:
        - Prefer rows containing conflicting mutable cells.
        - Propose swaps biased toward conflicting positions.
        - Accept improvements greedily; occasionally accept worse moves with a
          small SA-style probability (helps escape small loops).
        """
        current_score = self._count_conflicts(grid)
        best_grid = deepcopy(grid)
        best_score = current_score

        if best_score == 0:
            return best_grid, 0

        temp = max(0.05, start_temperature)
        n = self.size

        for t in range(max_attempts):
            # Cooperative cancellation
            if self.stop_event is not None and getattr(self.stop_event, "is_set", lambda: False)():
                break

            # Build list of candidate rows that contain conflicts in mutable positions
            conflict_rows: List[int] = []
            conflict_cols_by_row: Dict[int, List[int]] = {}
            for r in range(n):
                cols = self._row_mutable_cols(r)
                if len(cols) < 2:
                    continue
                conflict_cols = [c for c in cols if self._cell_is_in_conflict(grid, r, c)]
                if conflict_cols:
                    conflict_rows.append(r)
                    conflict_cols_by_row[r] = conflict_cols

            if conflict_rows:
                r = random.choice(conflict_rows)
                mutable_cols = self._row_mutable_cols(r)
                conflict_cols = conflict_cols_by_row.get(r, [])
                # When very close, do a stronger "best-swap-in-row" search to avoid thrashing.
                use_greedy = (current_score <= 6) or (random.random() < 0.25)
                if use_greedy:
                    best_pair = None
                    best_pair_score = float("inf")
                    # Evaluate all swap pairs in this row (<= 36 pairs for 9x9).
                    for i in range(len(mutable_cols) - 1):
                        for j in range(i + 1, len(mutable_cols)):
                            c1_try = mutable_cols[i]
                            c2_try = mutable_cols[j]
                            # Bias: prefer swaps involving at least one conflicting cell
                            if conflict_cols and (c1_try not in conflict_cols) and (c2_try not in conflict_cols):
                                continue
                            grid[r][c1_try], grid[r][c2_try] = grid[r][c2_try], grid[r][c1_try]
                            s_try = self._count_conflicts(grid)
                            grid[r][c1_try], grid[r][c2_try] = grid[r][c2_try], grid[r][c1_try]
                            if s_try < best_pair_score:
                                best_pair_score = s_try
                                best_pair = (c1_try, c2_try)
                                if best_pair_score == 0:
                                    break
                        if best_pair_score == 0:
                            break
                    if best_pair is None:
                        # Fall back to random
                        c1 = random.choice(conflict_cols) if conflict_cols else random.choice(mutable_cols)
                        c2 = random.choice([c for c in mutable_cols if c != c1])
                    else:
                        c1, c2 = best_pair
                else:
                    c1 = random.choice(conflict_cols) if conflict_cols else random.choice(mutable_cols)
                    c2_candidates = [c for c in mutable_cols if c != c1]
                    if not c2_candidates:
                        continue
                    c2 = random.choice(c2_candidates)
            else:
                # No explicit conflicts found (rare when score>0) → random row swap
                r = random.randrange(n)
                mutable_cols = self._row_mutable_cols(r)
                if len(mutable_cols) < 2:
                    continue
                c1, c2 = random.sample(mutable_cols, 2)

            # Propose swap (in-place)
            grid[r][c1], grid[r][c2] = grid[r][c2], grid[r][c1]
            new_score = self._count_conflicts(grid)

            delta = new_score - current_score
            accept = False
            if delta <= 0:
                accept = True
            else:
                # Small simulated annealing acceptance for exploration
                if random.random() < math.exp(-delta / temp):
                    accept = True

            if accept:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_grid = deepcopy(grid)
                    if best_score == 0:
                        return best_grid, 0
            else:
                # Revert swap
                grid[r][c1], grid[r][c2] = grid[r][c2], grid[r][c1]

            # Slowly cool temperature
            if (t + 1) % 100 == 0:
                temp = max(0.05, temp * 0.85)

        return best_grid, best_score

    # --------------------------------------------------------------
    #  HARD RESTART (Start from scratch, discard best)
    # --------------------------------------------------------------
    def _hard_restart(self) -> Tuple[List[Tuple[Grid, int]], Grid, int]:
        """Start the evolutionary process from scratch.

        This discards:
        - current population
        - belief space
        - current best grid
        - stagnation counters
        - mutation/belief update counters (metrics restart)
        """
        print("Hard restart")
        self.belief.clear()
        self.generations_without_improvement = 0
        self.last_best_score = float("inf")
        self.plateau_generations = 0
        self.mutation_count = 0
        self.belief_update_count = 0

        population: List[Tuple[Grid, int]] = []
        for _ in range(self.population_size):
            g = self._initial_individual()
            s = self._count_conflicts(g)
            population.append((g, s))

        self._update_belief(population, decay_factor=0.8)
        best_grid, best_score = min(population, key=lambda x: x[1])
        best_grid = deepcopy(best_grid)

        self.initial_conflicts = best_score
        self.current_conflicts = best_score
        return population, best_grid, best_score

    # ══════════════════════════════════════════════════════════════
    #  MAIN CULTURAL ALGORITHM LOOP
    # ══════════════════════════════════════════════════════════════
    def run(self) -> Tuple[Grid, int, int]:
        """Run the PURE Cultural Algorithm and return (best_grid, best_score, iters).
        
        STOPPING CONDITIONS:
            - SUCCESS: A perfect solution is found (score == 0)
            - TIMEOUT: `max_iters` generations have been completed
        
        RETURNS:
            - best_grid: The best Sudoku board found (2D list)
            - best_score: Number of conflicts (0 = perfect, valid solution)
            - iters: How many generations were actually performed
        
        ALGORITHM FLOW:
            Phase 1: Initialize population randomly
            Phase 2: Evolution loop (selection → crossover/mutation → evaluation)
            Phase 3: Return best solution found
        """
        # ══════════════════════════════════════════════════════════════
        # PHASE 1: INITIALIZATION
        # ══════════════════════════════════════════════════════════════
        # Create the initial population of random (but valid) Sudoku boards.
        # Each board respects all given clues and has valid rows (no duplicates).
        # The initial boards will have many column/block conflicts (high scores).
        
        # Start from scratch (initial population + belief space)
        population, best_grid, best_score = self._hard_restart()
        
        # Report initial state to GUI (if callback provided)
        if self.callback:
            self.callback(0, best_score, self.mutation_count, self.belief_update_count, 0)

        # ══════════════════════════════════════════════════════════════
        # PHASE 2: EVOLUTION LOOP (The Main Algorithm!)
        # ══════════════════════════════════════════════════════════════
        # Each iteration is one "generation" in the evolutionary process.
        # We repeat until we find a perfect solution (score=0) or run out of time.
        
        for it in range(1, self.max_iters + 1):
            # Cooperative cancellation: allow GUI to stop a running solve thread.
            if self.stop_event is not None and getattr(self.stop_event, "is_set", lambda: False)():
                # Return best-so-far and how far we got.
                return best_grid, best_score, it - 1

            # ┌─────────────────────────────────────────────────────────┐
            # │ EARLY STOPPING: Check if we already have a perfect solution
            # └─────────────────────────────────────────────────────────┘
            if best_score == 0:
                # SUCCESS! No conflicts means valid Sudoku solution
                return best_grid, 0, it

            # ┌─────────────────────────────────────────────────────────┐
            # │ STEP 2a: CREATE NEW GENERATION
            # └─────────────────────────────────────────────────────────┘
            # Start with elitism: always keep the best solution
            # This guarantees we NEVER lose progress (monotonic improvement)
            new_population: List[Tuple[Grid, int]] = [(deepcopy(best_grid), best_score)]

            # ┌─────────────────────────────────────────────────────────┐
            # │ STEP 2b: SELECTION MECHANISM
            # └─────────────────────────────────────────────────────────┘
            # TOURNAMENT SELECTION: Pick 3 random individuals, return the best
            # This creates "selection pressure" - good solutions are more likely
            # to become parents, but even mediocre ones have a small chance
            def select_parent() -> Grid:
                competitors = random.sample(population, k=min(3, len(population)))
                return min(competitors, key=lambda x: x[1])[0]

            # ┌─────────────────────────────────────────────────────────┐
            # │ STEP 2c: BREED NEW CHILDREN (Crossover + Mutation)
            # └─────────────────────────────────────────────────────────┘
            # ┌─────────────────────────────────────────────────────────┐
            # │ ADAPTIVE MUTATION: Increase randomness when stuck!
            # └─────────────────────────────────────────────────────────┘
            # Calculate adaptive mutation rates based on stagnation
            # When stuck for many generations, increase random exploration
            stagnation_factor = min(self.generations_without_improvement / 50.0, 2.0)
            
            # CRITICAL: Detect if stuck at low conflicts with poisoned belief space!
            # When stuck at ≤5 conflicts, the belief space is MISLEADING
            stuck_at_low_conflicts = (best_score > 0 and best_score <= 5 and 
                                     self.generations_without_improvement >= 15)
            
            # Base rates
            base_random_rate_crossover = 0.05  # Increased from 0.02
            base_random_rate_mutation = 0.10   # Increased from 0.05
            base_belief_influence = 0.6        # Decreased from 0.7 for more diversity
            
            # When stuck at low conflicts, DRASTICALLY change strategy
            if stuck_at_low_conflicts:
                # DISABLE belief influence (it's misleading!)
                adaptive_belief_influence = 0.1  # Almost completely random!
                # MASSIVE random mutation
                adaptive_random_rate_crossover = 0.6  # 60% random!
                adaptive_random_rate_mutation = 0.8   # 80% random!
                # INVERT belief (try unpopular values)
                use_inverted_belief = True
            else:
                # Normal adaptive rates (increase randomness when stuck)
                adaptive_random_rate_crossover = min(0.3, base_random_rate_crossover * (1 + stagnation_factor))
                adaptive_random_rate_mutation = min(0.4, base_random_rate_mutation * (1 + stagnation_factor))
                adaptive_belief_influence = max(0.3, base_belief_influence - (stagnation_factor * 0.1))
                use_inverted_belief = False
            
            # Fill the rest of the population with offspring
            while len(new_population) < self.population_size:
                # Mix of strategies: 70% crossover, 30% mutation-only
                # This balances exploitation (combining good solutions)
                # with exploration (random changes)
                
                if random.random() < 0.7:
                    # ═══════════════════════════════════════════════════
                    # CROSSOVER PATH (Exploitation)
                    # ═══════════════════════════════════════════════════
                    # Combine two good parents to create a child that
                    # hopefully inherits the best features of both
                    parent1 = select_parent()
                    parent2 = select_parent()
                    child = self._crossover(parent1, parent2)
                    
                    # Apply mutation with ADAPTIVE rates
                    # When stuck at low conflicts, use inverted belief!
                    child = self._mutate_individual(
                        child, 
                        belief_influence=adaptive_belief_influence,
                        random_mutation_rate=adaptive_random_rate_crossover,
                        invert_belief=use_inverted_belief
                    )
                else:
                    # ═══════════════════════════════════════════════════
                    # MUTATION-ONLY PATH (Exploration + Belief)
                    # ═══════════════════════════════════════════════════
                    # Take one parent and modify it using belief space
                    # This is where the CULTURAL part shines!
                    parent = select_parent()
                    child = self._mutate_individual(
                        parent,
                        belief_influence=adaptive_belief_influence,
                        random_mutation_rate=adaptive_random_rate_mutation,
                        invert_belief=use_inverted_belief
                    )
                
                # Evaluate the new child's fitness
                child_score = self._count_conflicts(child)
                new_population.append((child, child_score))

            # ┌─────────────────────────────────────────────────────────┐
            # │ STEP 2d: REPLACE OLD GENERATION
            # └─────────────────────────────────────────────────────────┘
            population = new_population
            
            # ┌─────────────────────────────────────────────────────────┐
            # │ STEP 2e: UPDATE BELIEF SPACE (Cultural Learning!)
            # └─────────────────────────────────────────────────────────┘
            # CRITICAL: When stuck at low conflicts for too long, 
            # the belief space is POISONED - clear it completely!
            if best_score > 0 and best_score <= 5 and self.generations_without_improvement >= 30:
                # NUCLEAR OPTION: Complete belief reset!
                # The old beliefs are keeping us stuck - forget everything!
                self.belief.clear()
            
            # Learn from the best individuals in the new generation
            # When we're close but stuck, forget faster to avoid "poisoned" beliefs
            belief_decay = 0.8
            if best_score <= 10 and self.generations_without_improvement >= 20:
                belief_decay = 0.55
            self._update_belief(population, decay_factor=belief_decay)

            # ┌─────────────────────────────────────────────────────────┐
            # │ STEP 2f: TRACK BEST SOLUTION & DETECT STAGNATION
            # └─────────────────────────────────────────────────────────┘
            # Check if this generation produced a new best solution
            current_best_grid, current_best_score = min(population, key=lambda x: x[1])
            if current_best_score < best_score:
                best_score = current_best_score
                best_grid = deepcopy(current_best_grid)
                self.generations_without_improvement = 0  # Reset stagnation counter
                self.plateau_generations = 0  # Reset true plateau only on improvement
            else:
                self.generations_without_improvement += 1  # Increment stagnation counter
                self.plateau_generations += 1  # True plateau length (does NOT reset on shakeups)
            
            # Update metrics for GUI
            self.current_conflicts = best_score

            # ┌─────────────────────────────────────────────────────────┐
            # │ HARD RESTARTS: Start completely from the beginning
            # └─────────────────────────────────────────────────────────┘
            # Requirement: do NOT keep the best boards (no elitism across restarts).
            # This is a full multi-start strategy: wipe population + belief space.
            if best_score > 0 and self.restart_count < self.max_restarts:
                # Trigger sooner when close to solution (last-mile stalls).
                if best_score <= 3:
                    restart_threshold = 120
                elif best_score <= 5:
                    restart_threshold = 180
                elif best_score <= 10:
                    restart_threshold = 260
                else:
                    restart_threshold = 650

                # IMPORTANT: use plateau_generations so diversity injection (which resets
                # generations_without_improvement at <=3 conflicts) doesn't prevent restarts.
                if self.plateau_generations >= restart_threshold:
                    self.restart_count += 1
                    population, best_grid, best_score = self._hard_restart()
                    # Continue with fresh society; generation counter `it` keeps going.
                    if self.callback:
                        self.callback(it, best_score, self.mutation_count, self.belief_update_count, 0)
                    continue

            # ┌─────────────────────────────────────────────────────────┐
            # │ LOW-CONFLICT INTENSIFICATION (<= 10 conflicts)
            # └─────────────────────────────────────────────────────────┘
            # Evolutionary operators often stall in the "last mile" because
            # reaching 0 conflicts requires a few very specific coordinated swaps.
            # When we're close (<=10) and stagnating, run a bounded memetic local
            # search on the best grid to try to finish the puzzle.
            if best_score > 0 and best_score <= 10 and self.generations_without_improvement >= 10:
                # Scale effort with closeness; run earlier to prevent long plateaus.
                # (More attempts when closer.)
                attempts = 500 + int((10 - best_score) * 200)
                attempts = min(4000, max(600, attempts))
                temp = 0.40 if best_score > 6 else 0.25

                candidate = deepcopy(best_grid)
                improved_grid, improved_score = self._conflict_directed_row_swap_search(
                    candidate,
                    max_attempts=attempts,
                    start_temperature=temp,
                )

                if improved_score < best_score:
                    best_grid = deepcopy(improved_grid)
                    best_score = improved_score
                    self.generations_without_improvement = 0
                    self.current_conflicts = best_score
                    if best_score == 0:
                        if self.callback:
                            self.callback(it, best_score, self.mutation_count, self.belief_update_count, self.plateau_generations)
                        return best_grid, 0, it
            
            # ┌─────────────────────────────────────────────────────────┐
            # │ DIVERSITY INJECTION: Fight premature convergence!
            # └─────────────────────────────────────────────────────────┘
            # If stuck for too long, inject random diversity to escape local optimum
            # When close to solution (few conflicts), be EXTREMELY aggressive!
            if best_score <= 3:
                stagnation_threshold = 20  # Very close - inject VERY soon
                replacement_ratio = 0.5    # Replace 50%!
            elif best_score <= 5:
                stagnation_threshold = 30  # Close - inject soon
                replacement_ratio = 0.4    # Replace 40%
            else:
                stagnation_threshold = 100 # Not close - normal threshold
                replacement_ratio = 0.2    # Replace 20%
            
            if self.generations_without_improvement >= stagnation_threshold:
                # Replace a percentage of population with completely random individuals
                pop_sorted = sorted(population, key=lambda x: x[1])
                num_to_replace = max(1, int(len(population) * replacement_ratio))
                
                for i in range(num_to_replace):
                    # Generate fresh random individual
                    new_grid = self._initial_individual()
                    new_score = self._count_conflicts(new_grid)
                    # Replace one of the worst individuals
                    population[-(i+1)] = (new_grid, new_score)
                
                # Reset stagnation counter after diversity injection
                self.generations_without_improvement = 0
            
            # ┌─────────────────────────────────────────────────────────┐
            # │ SIMULATED ANNEALING: Escape local optima!
            # └─────────────────────────────────────────────────────────┘
            # When close to solution (≤ 5 conflicts) and stuck,
            # apply simulated annealing to accept worse solutions temporarily
            if best_score > 0 and best_score <= 5 and self.generations_without_improvement >= 10:
                # Temperature based on how stuck we are
                # Higher temp when more stuck = accept worse solutions
                temp_factor = min(self.generations_without_improvement / 20.0, 3.0)
                temperature = 0.5 * (1.0 + temp_factor)  # Range: 0.5 to 2.0
                
                # Apply simulated annealing to best solution
                annealed_grid = self._simulated_annealing_mutation(best_grid, temperature)
                annealed_score = self._count_conflicts(annealed_grid)
                
                if annealed_score < best_score:
                    best_grid = annealed_grid
                    best_score = annealed_score
                    self.generations_without_improvement = 0
                    self.current_conflicts = best_score
                
                # Also apply to top 20% of population
                pop_sorted = sorted(population, key=lambda x: x[1])
                num_to_anneal = max(1, len(population) // 5)
                
                for i in range(num_to_anneal):
                    grid, score = pop_sorted[i]
                    if score > 0 and score <= 5:
                        annealed = self._simulated_annealing_mutation(grid, temperature)
                        annealed_score = self._count_conflicts(annealed)
                        
                        # Update in population
                        for j, (g, s) in enumerate(population):
                            if g is grid and s == score:
                                population[j] = (annealed, annealed_score)
                                if annealed_score == 0:
                                    best_grid = annealed
                                    best_score = 0
                                    self.generations_without_improvement = 0
                                break
            
            # Report progress to GUI (if callback provided)
            # Pass stagnation info so GUI can show when we're stuck
            if self.callback:
                # Show true plateau length (does not reset on diversity injection / annealing)
                self.callback(it, best_score, self.mutation_count, self.belief_update_count, self.plateau_generations)

        # ══════════════════════════════════════════════════════════════
        # PHASE 3: RETURN BEST RESULT
        # ══════════════════════════════════════════════════════════════
        # IMPORTANT: This is a PURE Cultural Algorithm implementation.
        #
        # If best_score > 0, we didn't find a perfect solution.
        # Solutions:
        #   - Increase max_iters (more generations)
        #   - Increase population_size (more diversity)
        #   - Tune crossover/mutation rates
        #   - Adjust belief_influence parameter
        
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
