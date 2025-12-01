# sudoku.py
# Backtracking solver with step recording for visualization.
from copy import deepcopy

class Sudoku:
    def __init__(self, grid):
        self.grid = deepcopy(grid)  # 9x9 grid, 0 for empty
        self.steps = []  # (r,c,val, action) action: 'place' or 'remove'

    def find_empty(self):
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] == 0:
                    return r, c
        return None

    def valid(self, r, c, val):
        # row
        if any(self.grid[r][j] == val for j in range(9)):
            return False
        # col
        if any(self.grid[i][c] == val for i in range(9)):
            return False
        # block
        br, bc = 3*(r//3), 3*(c//3)
        for i in range(br, br+3):
            for j in range(bc, bc+3):
                if self.grid[i][j] == val:
                    return False
        return True

    def solve_backtracking(self, record_steps=True):
        found = self.find_empty()
        if not found:
            return True
        r, c = found
        for val in range(1, 10):
            if self.valid(r, c, val):
                self.grid[r][c] = val
                if record_steps:
                    self.steps.append((r, c, val, 'place'))
                if self.solve_backtracking(record_steps):
                    return True
                # backtrack
                self.grid[r][c] = 0
                if record_steps:
                    self.steps.append((r, c, val, 'remove'))
        return False

    def solve(self, algorithm='backtracking', **kwargs):
        self.steps.clear()
        if algorithm == 'backtracking':
            grid_copy = deepcopy(self.grid)
            solved = self.solve_backtracking(record_steps=kwargs.get('record_steps', True))
            return solved, self.grid, self.steps
        else:
            raise ValueError('Unsupported algorithm in sudoku.solve')

    def load(self, grid):
        self.grid = deepcopy(grid)
        self.steps.clear()

if __name__ == '__main__':
    # simple demo
    puzzle = [
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9],
    ]
    s = Sudoku(puzzle)
    ok, grid, steps = s.solve_backtracking(), s.grid, s.steps
    print('Solved:', ok)
