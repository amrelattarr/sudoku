# generator.py
# Sudoku puzzle generator that creates a solved board and removes numbers
# while ensuring a unique solution (checks up to 2 solutions to detect non-unique).

import random
from copy import deepcopy

def _is_safe(grid, r, c, val):
    size = len(grid)
    
    # Determine block dimensions based on grid size
    if size == 9:
        block_rows, block_cols = 3, 3
    elif size == 6:
        block_rows, block_cols = 2, 3
    elif size == 4:
        block_rows, block_cols = 2, 2
    else:
        root = int(size ** 0.5)
        block_rows, block_cols = root, root
    
    # row
    if any(grid[r][j] == val for j in range(size)):
        return False
    # col
    if any(grid[i][c] == val for i in range(size)):
        return False
    # block
    br = (r // block_rows) * block_rows
    bc = (c // block_cols) * block_cols
    
    for i in range(br, min(br + block_rows, size)):
        for j in range(bc, min(bc + block_cols, size)):
            if grid[i][j] == val:
                return False
    return True

def generate_full_grid(size=9):
    """Generate a complete solved Sudoku grid using randomized backtracking."""
    grid = [[0] * size for _ in range(size)]

    def fill_cell(pos=0):
        if pos >= size * size:
            return True
        r, c = divmod(pos, size)
        if grid[r][c] != 0:
            return fill_cell(pos + 1)
        nums = list(range(1, size + 1))
        random.shuffle(nums)
        for n in nums:
            if _is_safe(grid, r, c, n):
                grid[r][c] = n
                if fill_cell(pos + 1):
                    return True
                grid[r][c] = 0
        return False

    fill_cell()
    return grid

def _count_solutions(grid, limit=2):
    """
    Count number of solutions for a Sudoku puzzle using backtracking.
    Stops early if count reaches `limit`.
    """
    g = deepcopy(grid)
    size = len(grid)
    count = 0

    def helper():
        nonlocal count, g
        # find empty
        found = False
        for i in range(size):
            for j in range(size):
                if g[i][j] == 0:
                    r, c = i, j
                    found = True
                    break
            if found:
                break
        if not found:
            count += 1
            return
        for val in range(1, size + 1):
            if _is_safe(g, r, c, val):
                g[r][c] = val
                helper()
                if count >= limit:
                    g[r][c] = 0
                    return
                g[r][c] = 0

    helper()
    return count

def remove_numbers_ensuring_unique(full_grid, removals_target):
    """
    Try to remove numbers from a full grid while preserving uniqueness.
    This uses a randomized removal order and only accepts a removal if
    the puzzle still has exactly 1 solution.
    """
    puzzle = deepcopy(full_grid)
    size = len(full_grid)
    # list of all cell coordinates
    cells = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(cells)

    removals = 0
    # Iterate cells; attempt to remove until reached target or exhausted cells
    for (r, c) in cells:
        if removals >= removals_target:
            break
        if puzzle[r][c] == 0:
            continue
        saved = puzzle[r][c]
        puzzle[r][c] = 0
        # Check if still has unique solution
        if _count_solutions(puzzle) == 1:
            removals += 1
        else:
            puzzle[r][c] = saved  # put it back

    return puzzle

def generate_random_puzzle(difficulty="medium", size=9):
    """
    Generate a random Sudoku puzzle.

    Args:
        difficulty: "easy" or "medium"
        size: size of the grid (4, 6, or 9)
    """
    # Define difficulty levels (percentage of cells to remove)
    difficulty_levels = {
        "easy": 0.4,    # ~40% cells removed
        "medium": 0.6,  # ~60% cells removed
    }

    # Backward compatibility: if older code passes "hard", treat it as "medium"
    if difficulty == "hard":
        difficulty = "medium"
    
    # Generate a complete grid
    full_grid = generate_full_grid(size)
    
    # Calculate how many cells to remove
    total_cells = size * size
    cells_to_remove = int(total_cells * difficulty_levels.get(difficulty, 0.6))
    
    # For very small grids, ensure we don't remove too many cells
    if size <= 4:
        min_clues = 4  # Minimum number of clues for 4x4
        cells_to_remove = min(cells_to_remove, total_cells - min_clues)
    
    # Remove numbers while ensuring unique solution
    puzzle = remove_numbers_ensuring_unique(full_grid, cells_to_remove)
    
    return puzzle

if __name__ == "__main__":
    # Test different grid sizes
    for size in [4, 6, 9]:
        print(f"\nTesting {size}x{size} grid:")
        p = generate_random_puzzle("medium", size)
        for row in p:
            print(" ".join(str(x) if x != 0 else "." for x in row))
