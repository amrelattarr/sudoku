# gui.py
# Tkinter GUI to load a Sudoku puzzle, choose algorithm, and solve.

import tkinter as tk
from tkinter import messagebox, ttk
from sudoku import Sudoku
from cultural_solver import CulturalSudokuSolver
from generator import generate_random_puzzle
import time


class SudokuGUI:
    def __init__(self, root):
        self.root = root
        root.title('Sudoku Solver — Backtracking & Cultural Algorithm')

        self.entries = [[None] * 9 for _ in range(9)]
        self.puzzle = [[0] * 9 for _ in range(9)]

        self.build_grid()
        self.build_controls()

        # load a fresh random puzzle at start
        self.load_puzzle(generate_random_puzzle("medium"))

    # ----------------------------------------------------------
    #  UI GRID
    # ----------------------------------------------------------
    def build_grid(self):
        # Create main frame with padding
        main_frame = tk.Frame(self.root, bg='#000000')
        main_frame.pack(padx=20, pady=20)
        
        # Create a frame for the grid
        grid_frame = tk.Frame(main_frame, bg='#000000', padx=2, pady=2)
        grid_frame.pack()

        # Create a 9x9 grid of Entry widgets
        for r in range(9):
            for c in range(9):
                # Create a frame for each cell with border
                cell_frame = tk.Frame(
                    grid_frame,
                    bg='#000000',
                    highlightthickness=1,
                    highlightbackground='#000000'
                )
                cell_frame.grid(row=r, column=c, sticky='nsew')
                
                # Configure grid weights for resizing
                grid_frame.columnconfigure(c, weight=1)
                grid_frame.rowconfigure(r, weight=1)
                
                # Create the Entry widget
                e = tk.Entry(
                    cell_frame,
                    width=2,
                    justify='center',
                    font=('Arial', 20, 'bold'),
                    borderwidth=0,
                    relief='flat',
                    fg='#000000',
                    bg='#ffffff',
                    disabledforeground='#000000',
                    disabledbackground='#ffffff',
                    readonlybackground='#ffffff',
                    highlightthickness=0
                )
                e.pack(expand=True, fill='both')
                
                # Store the entry in the grid
                self.entries[r][c] = e
                
                # Set thicker borders for 3x3 blocks
                border_width = 2  # Thicker border for 3x3 blocks
                if r % 3 == 0:  # Top border of 3x3 block
                    cell_frame.grid(pady=(border_width, 0))
                if c % 3 == 0:  # Left border of 3x3 block
                    cell_frame.grid(padx=(border_width, 0))
                if r == 8:  # Bottom border of grid
                    cell_frame.grid(pady=(0, border_width))
                if c == 8:  # Right border of grid
                    cell_frame.grid(padx=(0, border_width))

    # ----------------------------------------------------------
    #  CONTROL PANEL
    # ----------------------------------------------------------
    def build_controls(self):
        # Main control frame with padding
        ctrl = tk.Frame(self.root, bg='#f0f0f0', padx=10, pady=10)
        ctrl.pack(fill='x', padx=5, pady=5)

        # Left frame for algorithm selection
        algo_frame = tk.Frame(ctrl, bg='#f0f0f0')
        algo_frame.pack(side='left', padx=5)
        
        # Algorithm chooser with better styling
        tk.Label(algo_frame, text='Algorithm:', bg='#f0f0f0', font=('Arial', 10, 'bold')).pack(anchor='w')
        self.algo_var = tk.StringVar(value='backtracking')
        
        algo_btn_frame = tk.Frame(algo_frame, bg='#f0f0f0')
        algo_btn_frame.pack(pady=2)
        
        tk.Radiobutton(algo_btn_frame, text='Backtracking', variable=self.algo_var, 
                      value='backtracking', bg='#f0f0f0', font=('Arial', 9),
                      selectcolor='#e1e1e1', indicatoron=0, width=15).pack(side='left', padx=2)
        tk.Radiobutton(algo_btn_frame, text='Cultural', variable=self.algo_var, 
                      value='cultural', bg='#f0f0f0', font=('Arial', 9),
                      selectcolor='#e1e1e1', indicatoron=0, width=15).pack(side='left', padx=2)

        # Middle frame for difficulty selection
        diff_frame = tk.Frame(ctrl, bg='#f0f0f0')
        diff_frame.pack(side='left', padx=20)
        
        # Difficulty selector with better styling
        tk.Label(diff_frame, text='Difficulty:', bg='#f0f0f0', font=('Arial', 10, 'bold')).pack(anchor='w')
        self.difficulty_var = tk.StringVar(value='medium')
        
        # Styled OptionMenu with modern look
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure the dropdown button
        style.configure('TMenubutton',
                      background='#ffffff',
                      foreground='#333333',
                      font=('Arial', 10, 'bold'),
                      borderwidth=0,
                      relief='flat',
                      padding=5,
                      width=10)
        
        # Configure the dropdown menu
        style.configure('TMenubutton.dropdown',
                      background='#ffffff',
                      foreground='#333333',
                      font=('Arial', 10),
                      relief='flat')
        
        # Create the dropdown menu
        diff_menu = tk.OptionMenu(diff_frame, self.difficulty_var, 'easy', 'medium', 'hard')
        diff_menu.config(
            bg='#f8f9fa',
            fg='#333333',
            activebackground='#e9ecef',
            activeforeground='#333333',
            highlightthickness=1,
            highlightbackground='#dee2e6',
            highlightcolor='#dee2e6',
            borderwidth=0,
            relief='flat',
            width=12,
            cursor='hand2',
            indicatoron=1
        )
        
        # Style the dropdown menu items
        diff_menu['menu'].config(
            bg='white',
            fg='#333333',
            activebackground='#e9ecef',
            activeforeground='#333333',
            font=('Arial', 10),
            bd=0,
            relief='flat',
            activeborderwidth=0
        )
        
        # Add hover effect
        def on_enter(e):
            diff_menu.config(bg='#e9ecef')
        
        def on_leave(e):
            diff_menu.config(bg='#f8f9fa')
        
        diff_menu.bind('<Enter>', on_enter)
        diff_menu.bind('<Leave>', on_leave)
        
        diff_menu.pack(pady=2, ipady=3)

        # Right frame for action buttons
        btn_frame = tk.Frame(ctrl, bg='#f0f0f0')
        btn_frame.pack(side='right', padx=5)
        
        # Styled buttons with better visual feedback
        button_style = {
            'font': ('Arial', 10, 'bold'),
            'width': 10,
            'bd': 0,
            'relief': 'raised',
            'padx': 5,
            'pady': 5,
            'cursor': 'hand2',
            'activebackground': '#357abd',
            'activeforeground': 'white',
            'highlightthickness': 2,
            'highlightbackground': '#f0f0f0',
            'highlightcolor': '#f0f0f0',
            'highlightbackground': 'white',
            'highlightcolor': 'white',
            'borderwidth': 0
        }
        
        # Restart button
        self.restart_btn = tk.Button(
            btn_frame, 
            text='⟳ Restart', 
            command=self.load_random,
            bg='#4a90e2', 
            fg='white',
            **button_style
        )
        self.restart_btn.pack(side='left', padx=5)
        
        # Add hover effects
        self.restart_btn.bind('<Enter>', lambda e: self.restart_btn.config(bg='#3a7bbd'))
        self.restart_btn.bind('<Leave>', lambda e: self.restart_btn.config(bg='#4a90e2'))
        
        # Solve button
        self.solve_btn = tk.Button(
            btn_frame, 
            text='▶ Solve', 
            command=self.solve,
            bg='#50c878', 
            fg='white',
            **button_style
        )
        self.solve_btn.pack(side='left', padx=5)
        
        # Add hover effects
        self.solve_btn.bind('<Enter>', lambda e: self.solve_btn.config(bg='#40b868'))
        self.solve_btn.bind('<Leave>', lambda e: self.solve_btn.config(bg='#50c878'))

        # Status label with better styling
        self.status_label = tk.Label(self.root, text='Ready', bg='#f0f0f0',
                                   font=('Arial', 10), fg='#333333', pady=5)
        self.status_label.pack(fill='x', padx=10, pady=(0, 5))

    # ----------------------------------------------------------
    #  BASIC GRID OPS
    # ----------------------------------------------------------
    def get_grid_from_entries(self):
        grid = [[0] * 9 for _ in range(9)]
        for r in range(9):
            for c in range(9):
                v = self.entries[r][c].get().strip()
                if v.isdigit():
                    iv = int(v)
                    if 1 <= iv <= 9:
                        grid[r][c] = iv
        return grid

    def load_puzzle(self, grid):
        self.puzzle = [row[:] for row in grid]
        for r in range(9):
            for c in range(9):
                val = grid[r][c]
                e = self.entries[r][c]
                e.delete(0, 'end')
                if val != 0:
                    e.insert(0, str(val))
                    e.config(fg='blue')
                else:
                    e.config(fg='black')

    # ----------------------------------------------------------
    #  PUZZLE GENERATION
    # ----------------------------------------------------------
    def load_random(self):
        diff = self.difficulty_var.get()
        self.status_label.config(text=f'Generating {diff} puzzle...')
        self.root.update()

        puzzle = generate_random_puzzle(diff)
        self.load_puzzle(puzzle)
        self.status_label.config(text=f'Loaded random {diff} puzzle.')

    # ----------------------------------------------------------
    #  VALIDATION
    # ----------------------------------------------------------
    def _validate_initial_grid(self, grid):
        """Check for mistakes in the initial puzzle before solving."""

        # Row check
        for r in range(9):
            seen = set()
            for c in range(9):
                v = grid[r][c]
                if v == 0:
                    continue
                if v in seen:
                    return False
                seen.add(v)

        # Column check
        for c in range(9):
            seen = set()
            for r in range(9):
                v = grid[r][c]
                if v == 0:
                    continue
                if v in seen:
                    return False
                seen.add(v)

        # Block check
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                seen = set()
                for r in range(br, br + 3):
                    for c in range(bc, bc + 3):
                        v = grid[r][c]
                        if v == 0:
                            continue
                        if v in seen:
                            return False
                        seen.add(v)

        return True

    # ----------------------------------------------------------
    #  SOLVER
    # ----------------------------------------------------------
    def solve(self):
        # Check if puzzle is already solved (no zeros left)
        current_grid = self.get_grid_from_entries()
        if all(cell != 0 for row in current_grid for cell in row):
            self.status_label.config(text='Puzzle is already solved!')
            return
            
        algo = self.algo_var.get()
        grid = current_grid

        # Store original empty positions to identify solved cells later
        self.original_empties = [(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0]

        # Validate initial clues
        if not self._validate_initial_grid(grid):
            messagebox.showerror('Invalid puzzle', 'Puzzle has conflicts in initial clues.')
            return

        self.status_label.config(text=f'Solving with {algo}...')
        self.root.update()
        start = time.time()

        # Backtracking
        if algo == 'backtracking':
            s = Sudoku(grid)
            ok, solved_grid, steps = s.solve(algorithm='backtracking', record_steps=False)
            elapsed = time.time() - start

            if ok:
                # First load the solved puzzle with default colors
                self.load_puzzle(solved_grid)
                # Then update colors for solved cells
                for r, c in self.original_empties:
                    self.entries[r][c].config(fg='black')
                self.status_label.config(text=f'Backtracking solved in {elapsed:.3f}s.')
            else:
                self.status_label.config(text='Backtracking could not solve this puzzle.')

        # Cultural Algorithm
        else:
            solver = CulturalSudokuSolver(grid, population_size=300, max_iters=5000)
            sol, score, iters = solver.run()
            elapsed = time.time() - start

            if score == 0:
                # First load the solved puzzle with default colors
                self.load_puzzle(sol)
                # Then update colors for solved cells
                for r, c in self.original_empties:
                    self.entries[r][c].config(fg='black')
                self.status_label.config(text=f'Cultural Algorithm solved in {iters} iterations ({elapsed:.2f}s).')
            else:
                self.load_puzzle(sol)
                self.status_label.config(text=f'Best solution has {score} conflicts — after {iters} iterations.')

# ----------------------------------------------------------
#  MAIN
# ----------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = SudokuGUI(root)
    root.mainloop()
