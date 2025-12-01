# generator.py
# Sudoku puzzle generator that creates a solved board and removes numbers
# while ensuring a unique solution (checks up to 2 solutions to detect non-unique).

import random
from copy import deepcopy

def _is_safe(grid, r, c, val):
    # row
    if any(grid[r][j] == val for j in range(9)):
        return False
    # col
    if any(grid[i][c] == val for i in range(9)):
        return False
    # block
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if grid[i][j] == val:
                return False
    return True

def generate_full_grid():
    """Generate a complete solved Sudoku grid using randomized backtracking."""
    grid = [[0] * 9 for _ in range(9)]

    def fill_cell(pos=0):
        if pos >= 81:
            return True
        r, c = divmod(pos, 9)
        if grid[r][c] != 0:
            return fill_cell(pos + 1)
        nums = list(range(1, 10))
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
    count = 0

    def helper():
        nonlocal count, g
        # find empty
        found = False
        for i in range(9):
            for j in range(9):
                if g[i][j] == 0:
                    r, c = i, j
                    found = True
                    break
            if found:
                break
        if not found:
            count += 1
            return
        for val in range(1, 10):
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
    # list of all cell coordinates
    cells = [(r, c) for r in range(9) for c in range(9)]
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
        sols = _count_solutions(puzzle, limit=2)
        if sols != 1:
            # revert if not unique
            puzzle[r][c] = saved
        else:
            removals += 1
    return puzzle

def generate_random_puzzle(difficulty="medium"):
    """
    Generate a random Sudoku puzzle.

    difficulty: "easy", "medium", or "hard"
    """
    full = generate_full_grid()

    if difficulty == "easy":
        removals = random.randint(35, 42)  # more clues -> easier
    elif difficulty == "hard":
        removals = random.randint(50, 58)  # fewer clues -> harder
    else:  # medium
        removals = random.randint(43, 50)

    puzzle = remove_numbers_ensuring_unique(full, removals_target=removals)
    return puzzle

if __name__ == "__main__":
    # quick test
    p = generate_random_puzzle("medium")
    for row in p:
        print(row)
