# sudoku.py
# Backtracking solver with step recording for visualization.
from copy import deepcopy

class Sudoku:
    def __init__(self, grid):
        self.grid = deepcopy(grid)  # NxN grid, 0 for empty
        self.size = len(grid)
        
        # Determine block dimensions based on grid size
        if self.size == 9:
            self.block_rows = 3
            self.block_cols = 3
        elif self.size == 6:
            self.block_rows = 2
            self.block_cols = 3
        elif self.size == 4:
            self.block_rows = 2
            self.block_cols = 2
        else:
            # Fallback for square grids
            self.block_rows = int(self.size ** 0.5)
            self.block_cols = int(self.size ** 0.5)
            
        self.steps = []  # (r,c,val, action) action: 'place' or 'remove'

    def find_empty(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == 0:
                    return r, c
        return None

    def valid(self, r, c, val):
        # row
        if any(self.grid[r][j] == val for j in range(self.size)):
            return False
        # col
        if any(self.grid[i][c] == val for i in range(self.size)):
            return False
        # block
        br = (r // self.block_rows) * self.block_rows
        bc = (c // self.block_cols) * self.block_cols
        
        for i in range(br, min(br + self.block_rows, self.size)):
            for j in range(bc, min(bc + self.block_cols, self.size)):
                if self.grid[i][j] == val:
                    return False
        return True

    def solve_backtracking(self, record_steps=True):
        found = self.find_empty()
        if not found:
            return True
        r, c = found
        for val in range(1, self.size + 1):
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
    # Test different grid sizes
    def test_sudoku_solver():
        # Test 9x9
        puzzle_9x9 = [
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
        
        # Test 6x6
        puzzle_6x6 = [
            [1,0,0,4,0,0],
            [0,2,0,0,5,0],
            [0,0,3,0,0,6],
            [4,0,0,1,0,0],
            [0,5,0,0,2,0],
            [0,0,6,0,0,3],
        ]
        
        # Test 4x4
        puzzle_4x4 = [
            [1,0,0,4],
            [0,2,3,0],
            [0,3,2,0],
            [4,0,0,1],
        ]
        
        for size, puzzle in [(9, puzzle_9x9), (6, puzzle_6x6), (4, puzzle_4x4)]:
            print(f"\nTesting {size}x{size} puzzle...")
            s = Sudoku(puzzle)
            ok = s.solve_backtracking(record_steps=False)
            print(f"  - Solved: {ok}")
            if ok:
                print("  - Solution:")
                for row in s.grid:
                    print("   ", " ".join(str(x) for x in row))
    
    test_sudoku_solver()
