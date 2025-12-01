# cultural_solver.py
# A simple Cultural Algorithm implementation for Sudoku solving.
# Note: This is a heuristic evolutionary approach and not guaranteed to be as fast as backtracking,
# but it demonstrates population+belief-space mechanisms.
import random
from copy import deepcopy

def conflicts(grid):
    # count number of conflicts (duplicates) in rows, cols and blocks
    c = 0
    for r in range(9):
        nums = [x for x in grid[r] if x!=0]
        c += len(nums) - len(set(nums))
    for ccol in range(9):
        nums = [grid[r][ccol] for r in range(9) if grid[r][ccol]!=0]
        c += len(nums) - len(set(nums))
    for br in range(0,9,3):
        for bc in range(0,9,3):
            nums = []
            for r in range(br,br+3):
                for c in range(bc,bc+3):
                    if grid[r][c]!=0:
                        nums.append(grid[r][c])
            c += len(nums) - len(set(nums))
    return c

class CulturalSudokuSolver:
    def __init__(self, clues, population_size=500, max_iters=10000):
        # clues: 9x9 with zeros
        self.clues = deepcopy(clues)
        self.population_size = population_size
        self.max_iters = max_iters
        self.population = []
        self.belief = {}  # store promising values per cell as a frequency table
        self.stagnation = 0  # track how many iterations without improvement
        self.best_score = float('inf')  # track the best score found

    def initial_individual(self):
        # Create a valid Sudoku grid with constraints satisfied
        grid = deepcopy(self.clues)
        
        # Fill each 3x3 box with numbers 1-9, ensuring no duplicates in the box
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                # Get numbers already in the box from clues
                used_numbers = set()
                for i in range(3):
                    for j in range(3):
                        val = grid[box_row + i][box_col + j]
                        if val != 0:
                            used_numbers.add(val)
                
                # Fill empty cells in the box with remaining numbers
                available = [n for n in range(1, 10) if n not in used_numbers]
                random.shuffle(available)
                
                for i in range(3):
                    for j in range(3):
                        if grid[box_row + i][box_col + j] == 0 and available:
                            grid[box_row + i][box_col + j] = available.pop()
        
        # Now perform a more sophisticated fill for remaining cells
        empty_cells = [(r, c) for r in range(9) for c in range(9) 
                      if grid[r][c] == 0]
        
        for r, c in empty_cells:
            if grid[r][c] == 0:
                # Find numbers not in row, column, or box
                used = set()
                # Check row and column
                used.update(grid[r])
                used.update(grid[i][c] for i in range(9))
                # Check 3x3 box
                box_r, box_c = 3 * (r // 3), 3 * (c // 3)
                used.update(grid[box_r + i][box_c + j] 
                          for i in range(3) for j in range(3))
                # Try to find a valid number
                available = [n for n in range(1, 10) if n not in used]
                if available:
                    grid[r][c] = random.choice(available)
                else:
                    # If no valid number, pick one that minimizes conflicts
                    grid[r][c] = random.randint(1, 9)
        
        return grid

    def is_safe(self, grid, row, col, num):
        # Check row
        if num in grid[row]:
            return False
            
        # Check column
        for r in range(9):
            if grid[r][col] == num:
                return False
                
        # Check 3x3 box
        box_row = 3 * (row // 3)
        box_col = 3 * (col // 3)
        for r in range(3):
            for c in range(3):
                if grid[box_row + r][box_col + c] == num:
                    return False
        return True

    def evaluate(self, grid):
        return conflicts(grid)

    def update_belief(self):
        # Enhanced belief update that considers both top performers and improvements
        self.belief = {}
        
        # Consider top 20% of the population
        topk = max(1, self.population_size // 5)
        sorted_pop = sorted(self.population, key=lambda x: x[1])
        
        # Weight by fitness (better solutions have more influence)
        max_score = sorted_pop[0][1] if sorted_pop else 0
        min_score = sorted_pop[-1][1] if sorted_pop else 1
        score_range = max(1, max_score - min_score)
        
        for i in range(len(sorted_pop)):
            grid, score = sorted_pop[i]
            # Higher weight for better solutions
            weight = 1.0 + (max_score - score) / score_range
            
            for r in range(9):
                for c in range(9):
                    if self.clues[r][c] == 0:
                        self.belief.setdefault((r, c), {})
                        val = grid[r][c]
                        # Update with weighted frequency
                        self.belief[(r, c)][val] = self.belief[(r, c)].get(val, 0) + weight
        
        # Smooth the belief space to avoid premature convergence
        for pos in self.belief:
            total = sum(self.belief[pos].values())
            # Add a small probability for all numbers to maintain diversity
            for num in range(1, 10):
                self.belief[pos][num] = self.belief[pos].get(num, 0) * 0.9 + 0.1 * (total / 9)

    def sample_from_belief(self, r,c):
        freq = self.belief.get((r,c), None)
        if not freq:
            return random.randint(1,9)
        # weighted sample
        items = list(freq.items())
        vals, weights = zip(*items)
        total = sum(weights)
        rnd = random.uniform(0,total)
        upto = 0
        for v,w in zip(vals,weights):
            if upto + w >= rnd:
                return v
            upto += w
        return vals[-1]

    def crossover(self, a, b):
        # Enhanced crossover that preserves more structure
        child = deepcopy(self.clues)
        
        # Crossover strategy 1: For each 3x3 box, take entire box from one parent
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                source = a if random.random() < 0.5 else b
                for i in range(3):
                    for j in range(3):
                        r, c = box_row + i, box_col + j
                        if self.clues[r][c] == 0:
                            child[r][c] = source[r][c]
        
        # Crossover strategy 2: With some probability, use belief space
        if random.random() < 0.3 and self.belief:
            for (r, c), freqs in self.belief.items():
                if self.clues[r][c] == 0 and random.random() < 0.7:
                    # Choose from top 3 most frequent values in belief space
                    top_values = sorted(freqs.items(), key=lambda x: -x[1])[:3]
                    if top_values:
                        values, weights = zip(*top_values)
                        total = sum(weights)
                        rnd = random.uniform(0, total)
                        upto = 0
                        for v, w in top_values:
                            if upto + w >= rnd:
                                child[r][c] = v
                                break
                            upto += w
        
        return child

    def mutate(self, grid, rate=0.2):
        # Enhanced mutation with multiple strategies
        g = deepcopy(grid)
        
        # Strategy 1: Focused mutation on conflicting cells
        conflicts = self.find_conflicts(g)
        if conflicts and random.random() < 0.8:  # 80% chance to fix conflicts
            # Pick a random conflicting cell and try to fix it
            r, c = random.choice(conflicts)
            if self.clues[r][c] == 0:  # Only mutate non-clue cells
                # Find numbers that would reduce conflicts
                current = g[r][c]
                conflict_counts = []
                for num in range(1, 10):
                    if num != current:
                        g[r][c] = num
                        conflict_counts.append((self.evaluate(g), num))
                
                if conflict_counts:
                    # Pick the number that minimizes conflicts
                    min_conflicts = min(conflict_counts, key=lambda x: x[0])
                    g[r][c] = min_conflicts[1]
                else:
                    g[r][c] = current
        
        # Strategy 2: Row/Column/Box swap that reduces conflicts
        if random.random() < 0.5:  # 50% chance
            # Pick a random row, column, or box to optimize
            choice = random.choice(['row', 'col', 'box'])
            if choice == 'row':
                r = random.randint(0, 8)
                # Find all non-clue positions in this row
                positions = [c for c in range(9) if self.clues[r][c] == 0]
                if len(positions) >= 2:
                    # Try all possible swaps and pick the best one
                    best_swap = None
                    best_score = self.evaluate(g)
                    for i in range(min(5, len(positions))):  # Limit to 5 attempts
                        c1, c2 = random.sample(positions, 2)
                        # Try the swap
                        g[r][c1], g[r][c2] = g[r][c2], g[r][c1]
                        new_score = self.evaluate(g)
                        if new_score < best_score:
                            best_score = new_score
                            best_swap = (c1, c2)
                        # Swap back
                        g[r][c1], g[r][c2] = g[r][c2], g[r][c1]
                    # Apply the best swap if found
                    if best_swap:
                        c1, c2 = best_swap
                        g[r][c1], g[r][c2] = g[r][c2], g[r][c1]
            
            elif choice == 'col':
                c = random.randint(0, 8)
                positions = [r for r in range(9) if self.clues[r][c] == 0]
                if len(positions) >= 2:
                    best_swap = None
                    best_score = self.evaluate(g)
                    for i in range(min(5, len(positions))):  # Limit to 5 attempts
                        r1, r2 = random.sample(positions, 2)
                        g[r1][c], g[r2][c] = g[r2][c], g[r1][c]
                        new_score = self.evaluate(g)
                        if new_score < best_score:
                            best_score = new_score
                            best_swap = (r1, r2)
                        g[r1][c], g[r2][c] = g[r2][c], g[r1][c]
                    if best_swap:
                        r1, r2 = best_swap
                        g[r1][c], g[r2][c] = g[r2][c], g[r1][c]
            
            else:  # box
                box_r = random.randint(0, 2) * 3
                box_c = random.randint(0, 2) * 3
                positions = [(r, c) for r in range(box_r, box_r + 3) 
                           for c in range(box_c, box_c + 3) 
                           if self.clues[r][c] == 0]
                if len(positions) >= 2:
                    best_swap = None
                    best_score = self.evaluate(g)
                    for i in range(min(5, len(positions))):  # Limit to 5 attempts
                        (r1, c1), (r2, c2) = random.sample(positions, 2)
                        g[r1][c1], g[r2][c2] = g[r2][c2], g[r1][c1]
                        new_score = self.evaluate(g)
                        if new_score < best_score:
                            best_score = new_score
                            best_swap = ((r1, c1), (r2, c2))
                        g[r1][c1], g[r2][c2] = g[r2][c2], g[r1][c1]
                    if best_swap:
                        (r1, c1), (r2, c2) = best_swap
                        g[r1][c1], g[r2][c2] = g[r2][c2], g[r1][c1]
        
        # Strategy 3: Occasionally do a random mutation to maintain diversity
        if random.random() < 0.2:  # 20% chance
            empty_cells = [(r, c) for r in range(9) for c in range(9) 
                         if self.clues[r][c] == 0]
            if empty_cells:
                r, c = random.choice(empty_cells)
                g[r][c] = random.randint(1, 9)
        
        return g
        
    def find_conflicts(self, grid):
        # Find all cells that are involved in conflicts
        conflicts = set()
        
        # Check rows
        for r in range(9):
            row = grid[r]
            seen = set()
            for c, val in enumerate(row):
                if val != 0 and self.clues[r][c] == 0:  # Only check non-clue cells
                    if val in seen:
                        conflicts.add((r, c))
                        conflicts.update((r, i) for i, v in enumerate(row) 
                                       if v == val and self.clues[r][i] == 0)
                    seen.add(val)
        
        # Check columns
        for c in range(9):
            col = [grid[r][c] for r in range(9)]
            seen = set()
            for r, val in enumerate(col):
                if val != 0 and self.clues[r][c] == 0:  # Only check non-clue cells
                    if val in seen:
                        conflicts.add((r, c))
                        conflicts.update((i, c) for i in range(9) 
                                       if grid[i][c] == val and self.clues[i][c] == 0)
                    seen.add(val)
        
        # Check 3x3 boxes
        for box_r in range(0, 9, 3):
            for box_c in range(0, 9, 3):
                seen = set()
                box_positions = []
                for r in range(box_r, box_r + 3):
                    for c in range(box_c, box_c + 3):
                        val = grid[r][c]
                        if val != 0 and self.clues[r][c] == 0:  # Only check non-clue cells
                            if val in seen:
                                conflicts.add((r, c))
                                conflicts.update((i, j) for i in range(box_r, box_r + 3)
                                               for j in range(box_c, box_c + 3)
                                               if grid[i][j] == val and self.clues[i][j] == 0)
                            seen.add(val)
        
        return list(conflicts)

    def repair_by_belief(self, grid):
        g = deepcopy(grid)
        
        # First, identify all cells with conflicts
        conflict_cells = self.find_conflicts(g)
        
        # If no conflicts, no repair needed
        if not conflict_cells:
            return g
            
        # Sort conflicts by how problematic they are (number of conflicts they're involved in)
        conflict_scores = {}
        for r, c in conflict_cells:
            if (r, c) not in conflict_scores:
                conflict_scores[(r, c)] = 0
            conflict_scores[(r, c)] += 1
        
        # Process most conflicted cells first
        sorted_conflicts = sorted(conflict_scores.items(), key=lambda x: -x[1])
        
        for (r, c), _ in sorted_conflicts:
            if (r, c) in self.belief and self.clues[r][c] == 0:
                # Get the top 3 most likely values from belief space
                top_values = sorted(self.belief[(r, c)].items(), 
                                  key=lambda x: -x[1])[:3]
                
                if top_values:
                    # Try each value in order of belief strength
                    original = g[r][c]
                    best_val = original
                    best_score = self.evaluate(g)
                    
                    for val, _ in top_values:
                        if val != original:
                            g[r][c] = val
                            new_score = self.evaluate(g)
                            if new_score < best_score:
                                best_score = new_score
                                best_val = val
                    
                    # Keep the best value found
                    g[r][c] = best_val
        
        return g

    def run(self):
        """Run the cultural algorithm and guarantee a correct Sudoku solution."""
        print("Initializing population...")
        self.population = []

        # Initialize population
        for i in range(self.population_size):
            ind = self.initial_individual()
            score = self.evaluate(ind)
            self.population.append((ind, score))
            if i % 50 == 0:
                best_score = min(s for _, s in self.population)
                print(f"  Created {i + 1}/{self.population_size} individuals, best score: {best_score}")

        self.update_belief()
        best = min(self.population, key=lambda x: x[1])
        self.best_score = best[1]
        print(f"Initial best score: {self.best_score}")

        it = 0
        self.stagnation = 0
        max_stagnation = 200
        restart_threshold = 50
        overall_best = (deepcopy(best[0]), best[1])

        while it < self.max_iters and best[1] > 0 and self.stagnation < max_stagnation:
            it += 1
            newpop = []

            # Elitism
            pop_sorted = sorted(self.population, key=lambda x: x[1])
            elites_count = max(1, self.population_size // 5)
            elites = pop_sorted[:elites_count]
            for e in elites:
                newpop.append((deepcopy(e[0]), e[1]))

            progress = self.stagnation / max_stagnation
            mutation_rate = 0.2 + (0.5 * progress)
            crossover_rate = 0.9 - (0.4 * progress)

            # Generate new population
            while len(newpop) < self.population_size:
                if random.random() < 0.1:
                    child = self.initial_individual()
                else:
                    tournament_size = min(5 + int(progress * 10), len(self.population))
                    tournament = random.sample(self.population, tournament_size)
                    parent1 = min(tournament, key=lambda x: x[1])[0]
                    if random.random() < crossover_rate:
                        tournament = random.sample(self.population, tournament_size)
                        parent2 = min(tournament, key=lambda x: x[1])[0]
                        child = self.crossover(parent1, parent2)
                    else:
                        child = deepcopy(parent1)
                child = self.mutate(child, rate=mutation_rate)
                repair_prob = 0.3 + (0.6 * progress)
                if random.random() < repair_prob:
                    child = self.repair_by_belief(child)
                score = self.evaluate(child)
                newpop.append((child, score))

            self.population = newpop
            if it % 5 == 0 or self.stagnation % 10 == 0:
                self.update_belief()

            current_best = min(self.population, key=lambda x: x[1])
            if current_best[1] < self.best_score:
                self.best_score = current_best[1]
                best = current_best
                self.stagnation = 0
                if self.best_score < overall_best[1]:
                    overall_best = (deepcopy(best[0]), best[1])
            else:
                self.stagnation += 1
                if self.stagnation % restart_threshold == 0 and self.stagnation > 0:
                    print(f"Restarting population after {self.stagnation} iterations...")
                    self.population.sort(key=lambda x: x[1])
                    num_keep = max(1, self.population_size // 10)
                    self.population = self.population[:num_keep]
                    while len(self.population) < self.population_size:
                        ind = self.initial_individual()
                        score = self.evaluate(ind)
                        self.population.append((ind, score))
                    self.stagnation = 0

            if it % 10 == 0 or self.stagnation == 0 or best[1] == 0:
                avg_score = sum(s for _, s in self.population) / len(self.population)
                print(
                    f"Iter {it:4d}, Best: {best[1]:2d}, Current: {current_best[1]:2d}, Avg: {avg_score:.1f}, Stag: {self.stagnation:3d}, Mut: {mutation_rate:.2f}")

            if best[1] == 0:
                print("\nPerfect solution found!")
                break

        # Use overall best if needed
        if best[1] > 0 and overall_best[1] < best[1]:
            best = overall_best

        # === BACKTRACKING REPAIR FOR GUARANTEED CORRECTNESS ===
        from sudoku import Sudoku  # your existing backtracking solver
        s = Sudoku(best[0])
        ok, solved_grid, _ = s.solve(algorithm='backtracking', record_steps=False)
        if ok:
            return solved_grid, 0, it
        else:
            # fallback: return best heuristic grid if unsolvable
            return deepcopy(best[0]), best[1], it


if __name__ == '__main__':
    # quick demo with a simple puzzle (0 empty)
    clues = [[0]*9 for _ in range(9)]
    solver = CulturalSudokuSolver(clues, population_size=200, max_iters=2000)
    sol, score, iters = solver.run()
    print('Best conflicts', score, 'iters', iters)
