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
        root.title('Sudoku Solver')
        
        # Configure window size and minimum size
        root.geometry('1000x750')
        root.minsize(900, 650)
        
        # Configure grid weights to make the layout responsive
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Set application icon if available
        try:
            root.iconbitmap('icon.ico')  # You can add an icon file if desired
        except:
            pass

        self.grid_size = 9  # Default size
        self.entries = []
        self.puzzle = []
        self.grid_frame = None
        self.is_solving = False  # Flag to track if animation is running
        self.animation_speed = 50  # Delay in milliseconds between steps
        
        # Set application style
        self.set_style()
        
        self.build_controls()
        self.setup_grid()

        # Load a fresh random puzzle at start
        self.load_puzzle(generate_random_puzzle("medium"))

    # ----------------------------------------------------------
    #  UI GRID
    # ----------------------------------------------------------
    def clear_grid(self):
        """Clear the existing grid."""
        if hasattr(self, 'grid_frame') and self.grid_frame:
            self.grid_frame.destroy()
        self.entries = []
        self.puzzle = []

    def setup_grid(self, size=9):
        """Set up the grid with the specified size."""
        self.clear_grid()
        self.grid_size = size
        self.entries = [[None] * size for _ in range(size)]
        self.puzzle = [[0] * size for _ in range(size)]
        self.build_grid()

    def build_grid(self):
        # Create right frame for the grid if it doesn't exist
        if not hasattr(self, 'right_frame') or not self.right_frame.winfo_exists():
            self.right_frame = tk.Frame(self.main_container, bg='#f0f0f0')
            self.right_frame.pack(side='right', fill='both', expand=True, padx=(0, 20), pady=10)
        
        # Create a frame for the grid with a nice border (larger padding)
        self.grid_frame = tk.Frame(self.right_frame, bg='#000000', padx=3, pady=3,
                                 highlightthickness=1, highlightbackground='#cccccc')
        self.grid_frame.pack(expand=True, pady=20, padx=20)  # Added more padding

        # Calculate block size for the grid
        if self.grid_size == 9:
            block_rows, block_cols = 3, 3
        elif self.grid_size == 6:
            block_rows, block_cols = 2, 3
        elif self.grid_size == 4:
            block_rows, block_cols = 2, 2
        else:
            root = int(self.grid_size ** 0.5)
            block_rows, block_cols = root, root

        # Create grid of Entry widgets
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Create a frame for each cell with border (thinner border)
                cell_frame = tk.Frame(
                    self.grid_frame,
                    bg='#000000',
                    highlightthickness=0.5,
                    highlightbackground='#555555',
                    padx=0,
                    pady=0
                )
                cell_frame.grid(row=r, column=c, sticky='nsew')
                
                # Configure grid weights for resizing
                self.grid_frame.columnconfigure(c, weight=1)
                self.grid_frame.rowconfigure(r, weight=1)
                
                # Create the Entry widget (larger size)
                e = tk.Entry(
                    cell_frame,
                    width=3,
                    justify='center',
                    font=('Arial', 22, 'bold'),
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
                
                # Set thicker borders for blocks
                border_width = 2
                if r % block_rows == 0:  # Top border of block
                    cell_frame.grid(pady=(border_width, 0))
                if c % block_cols == 0:  # Left border of block
                    cell_frame.grid(padx=(border_width, 0))
                if r == self.grid_size - 1:  # Bottom border of grid
                    cell_frame.grid(pady=(0, border_width))
                if c == self.grid_size - 1:  # Right border of grid
                    cell_frame.grid(padx=(0, border_width))

    # ----------------------------------------------------------
    #  CONTROL PANEL
    # ----------------------------------------------------------
    def build_controls(self):
        # Create main container with left and right frames
        self.main_container = tk.Frame(self.root, bg='#f0f0f0')
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left frame for all controls (made more compact for smaller window)
        left_frame = tk.Frame(self.main_container, bg='#f0f0f0', padx=8, pady=8, width=140)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Title (smaller font and padding)
        title_frame = tk.Frame(left_frame, bg='#f0f0f0')
        title_frame.pack(fill='x', pady=(0, 8))
        tk.Label(title_frame, text='Sudoku Solver', bg='#f0f0f0', 
                font=('Arial', 12, 'bold')).pack(anchor='w')
        
        # Separator
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Grid size selection
        size_frame = tk.LabelFrame(left_frame, text='Grid Size', bg='#f0f0f0', 
                                 font=('Arial', 10, 'bold'), padx=5, pady=5)
        size_frame.pack(fill='x', pady=5)
        
        self.size_var = tk.IntVar(value=9)
        for size in [4, 6, 9]:
            rb = tk.Radiobutton(size_frame, text=f'{size}x{size}', variable=self.size_var, 
                              value=size, bg='#f0f0f0', font=('Arial', 10),
                              selectcolor='#e1e1e1', indicatoron=1, width=8,
                              command=self.on_size_change)
            rb.pack(anchor='w', pady=2)
        
        # Difficulty selection
        diff_frame = tk.LabelFrame(left_frame, text='Difficulty', bg='#f0f0f0',
                                 font=('Arial', 10, 'bold'), padx=5, pady=5)
        diff_frame.pack(fill='x', pady=5)
        
        self.difficulty_var = tk.StringVar(value='medium')
        for diff in ['easy', 'medium', 'hard']:
            rb = tk.Radiobutton(diff_frame, text=diff.capitalize(), variable=self.difficulty_var, 
                              value=diff, bg='#f0f0f0', font=('Arial', 10),
                              selectcolor='#e1e1e1', indicatoron=1, width=8)
            rb.pack(anchor='w', pady=2)
        
        # Algorithm selection
        algo_frame = tk.LabelFrame(left_frame, text='Algorithm', bg='#f0f0f0',
                                 font=('Arial', 10, 'bold'), padx=5, pady=5)
        algo_frame.pack(fill='x', pady=5)
        
        self.algo_var = tk.StringVar(value='backtracking')
        for algo in [('Backtracking', 'backtracking'), ('Cultural', 'cultural')]:
            rb = tk.Radiobutton(algo_frame, text=algo[0], variable=self.algo_var, 
                              value=algo[1], bg='#f0f0f0', font=('Arial', 10),
                              selectcolor='#e1e1e1', indicatoron=1, width=12)
            rb.pack(anchor='w', pady=2)
        
        # Animation speed control
        speed_frame = tk.LabelFrame(left_frame, text='Animation Speed', bg='#f0f0f0',
                                   font=('Arial', 10, 'bold'), padx=5, pady=5)
        speed_frame.pack(fill='x', pady=5)
        
        self.speed_var = tk.IntVar(value=50)
        speed_scale = tk.Scale(speed_frame, from_=1, to=200, orient='horizontal',
                              variable=self.speed_var, bg='#f0f0f0', 
                              font=('Arial', 9), length=120, showvalue=False,
                              command=self.update_speed)
        speed_scale.pack(fill='x', pady=2)
        
        self.speed_label = tk.Label(speed_frame, text='Medium', bg='#f0f0f0',
                                   font=('Arial', 9), fg='#555555')
        self.speed_label.pack(anchor='w')
        
        # Action buttons
        btn_frame = tk.Frame(left_frame, bg='#f0f0f0', pady=10)
        btn_frame.pack(fill='x', pady=(10, 0))
        
        # Button style (more compact)
        btn_style = {
            'font': ('Arial', 10, 'bold'),
            'width': 10,
            'bd': 0,
            'relief': 'raised',
            'padx': 6,
            'pady': 6,
            'cursor': 'hand2',
            'activeforeground': 'white',
            'highlightthickness': 0,
            'borderwidth': 0
        }
        
        # Restart button
        self.restart_btn = tk.Button(
            btn_frame, 
            text='⟳ New Puzzle', 
            command=self.load_random,
            bg='#4a90e2', 
            fg='white',
            **btn_style
        )
        self.restart_btn.pack(pady=5, fill='x')
        self.restart_btn.bind('<Enter>', lambda e: self.restart_btn.config(bg='#3a7bbd'))
        self.restart_btn.bind('<Leave>', lambda e: self.restart_btn.config(bg='#4a90e2'))
        
        # Solve button
        self.solve_btn = tk.Button(
            btn_frame, 
            text='▶ Solve', 
            command=self.solve,
            bg='#50c878', 
            fg='white',
            **btn_style
        )
        self.solve_btn.pack(pady=5, fill='x')
        self.solve_btn.bind('<Enter>', lambda e: self.solve_btn.config(bg='#40b868'))
        self.solve_btn.bind('<Leave>', lambda e: self.solve_btn.config(bg='#50c878'))
        
        # Stop button
        self.stop_btn = tk.Button(
            btn_frame, 
            text='■ Stop', 
            command=self.stop_animation,
            bg='#e74c3c', 
            fg='white',
            state='disabled',
            **btn_style
        )
        self.stop_btn.pack(pady=5, fill='x')
        self.stop_btn.bind('<Enter>', lambda e: self.stop_btn.config(bg='#c0392b') if self.is_solving else None)
        self.stop_btn.bind('<Leave>', lambda e: self.stop_btn.config(bg='#e74c3c') if self.is_solving else None)
        
        # Status label
        self.status_label = tk.Label(left_frame, text='Ready', bg='#f0f0f0',
                                   font=('Arial', 10), fg='#555555', pady=5,
                                   wraplength=180, justify='left', anchor='w')
        self.status_label.pack(fill='x', pady=(15, 0))

    # ----------------------------------------------------------
    #  ANIMATION SPEED
    # ----------------------------------------------------------
    def update_speed(self, val):
        """Update animation speed and label."""
        speed = int(val)
        self.animation_speed = speed
        if speed < 30:
            label = 'Very Fast'
        elif speed < 70:
            label = 'Fast'
        elif speed < 110:
            label = 'Medium'
        elif speed < 150:
            label = 'Slow'
        else:
            label = 'Very Slow'
        self.speed_label.config(text=label)
    
    # ----------------------------------------------------------
    #  BASIC GRID OPS
    # ----------------------------------------------------------
    def on_size_change(self):
        """Handle grid size change event."""
        new_size = self.size_var.get()
        if new_size != self.grid_size:
            self.setup_grid(new_size)
            self.load_random()

    def get_grid_from_entries(self):
        grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                v = self.entries[r][c].get().strip()
                if v.isdigit():
                    iv = int(v)
                    if 1 <= iv <= self.grid_size:
                        grid[r][c] = iv
        return grid

    def load_puzzle(self, grid):
        self.puzzle = [row[:] for row in grid]
        self.setup_grid(len(grid))  # Ensure grid matches puzzle size
        for r in range(self.grid_size):
            for c in range(self.grid_size):
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
        size = self.size_var.get()
        self.status_label.config(text=f'Generating {size}x{size} {diff} puzzle...')
        self.root.update()

        try:
            puzzle = generate_random_puzzle(diff, size)
            self.load_puzzle(puzzle)
            self.status_label.config(text=f'Loaded {size}x{size} {diff} puzzle.')
        except Exception as e:
            self.status_label.config(text=f'Error generating puzzle: {str(e)}')
            print(f"Error generating puzzle: {e}")
            # Fall back to a default puzzle if generation fails
            if size == 4:
                puzzle = [
                    [1, 0, 0, 4],
                    [0, 2, 3, 0],
                    [0, 3, 2, 0],
                    [4, 0, 0, 1],
                ]
            elif size == 6:
                puzzle = [
                    [1, 0, 0, 4, 0, 0],
                    [0, 2, 0, 0, 5, 0],
                    [0, 0, 3, 0, 0, 6],
                    [4, 0, 0, 1, 0, 0],
                    [0, 5, 0, 0, 2, 0],
                    [0, 0, 6, 0, 0, 3],
                ]
            else:  # 9x9
                puzzle = [
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
            self.load_puzzle(puzzle)
            self.status_label.config(text=f'Loaded default {size}x{size} puzzle.')

    # ----------------------------------------------------------
    #  ANIMATION
    # ----------------------------------------------------------
    def animate_steps(self, steps, solved_grid):
        """Animate the backtracking steps."""
        self.is_solving = True
        self.solve_btn.config(state='disabled')
        self.restart_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        def animate_step(index):
            if not self.is_solving or index >= len(steps):
                # Animation complete - show final solution
                self.finalize_solution(solved_grid)
                return
            
            r, c, val, action = steps[index]
            entry = self.entries[r][c]
            
            if action == 'place':
                # Show number being tried (orange color)
                entry.delete(0, 'end')
                entry.insert(0, str(val))
                entry.config(fg='#FF8C00')  # Dark orange for trying
                self.root.update()
            elif action == 'remove':
                # Show backtracking (red color, then remove)
                entry.config(fg='red')
                self.root.update()
                # Schedule removal after brief pause
                self.root.after(max(10, self.animation_speed // 3), 
                              lambda: entry.delete(0, 'end'))
            
            # Schedule next step
            self.root.after(self.animation_speed, lambda: animate_step(index + 1))
        
        # Start animation
        animate_step(0)
    
    def stop_animation(self):
        """Stop the current animation."""
        self.is_solving = False
        self.solve_btn.config(state='normal')
        self.restart_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text='Animation stopped.')
    
    def finalize_solution(self, solved_grid):
        """Show the final solved puzzle with appropriate colors."""
        self.is_solving = False
        self.solve_btn.config(state='normal')
        self.restart_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                entry = self.entries[r][c]
                val = solved_grid[r][c]
                entry.delete(0, 'end')
                if val != 0:
                    entry.insert(0, str(val))
                    # Color original clues in blue, solved cells in green
                    if (r, c) in self.original_empties:
                        entry.config(fg='green')
                    else:
                        entry.config(fg='blue')
        self.status_label.config(text=f'Solved! Animation complete.')
    
    # ----------------------------------------------------------
    #  VALIDATION
    # ----------------------------------------------------------
    def _validate_initial_grid(self, grid):
        """Check for mistakes in the initial puzzle before solving."""
        size = len(grid)
        block_row = 2 if size == 6 else int(size ** 0.5)  # 2x3 blocks for 6x6
        block_col = 3 if size == 6 else int(size ** 0.5)  # 2x3 blocks for 6x6

        # Row check
        for r in range(size):
            seen = set()
            for c in range(size):
                v = grid[r][c]
                if v == 0:
                    continue
                if v < 1 or v > size:  # Check if number is within valid range
                    return False
                if v in seen:
                    return False
                seen.add(v)

        # Column check
        for c in range(size):
            seen = set()
            for r in range(size):
                v = grid[r][c]
                if v == 0:
                    continue
                if v in seen:
                    return False
                seen.add(v)

        # Block check - handle 2x3 blocks for 6x6
        for br in range(0, size, block_row):
            for bc in range(0, size, block_col):
                seen = set()
                for r in range(br, br + block_row):
                    for c in range(bc, bc + block_col):
                        if r >= size or c >= size:  # Safety check
                            continue
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
        try:
            # Check if puzzle is already solved (no zeros left)
            current_grid = self.get_grid_from_entries()
            if all(cell != 0 for row in current_grid for cell in row):
                self.status_label.config(text='Puzzle is already solved!')
                return
                
            algo = self.algo_var.get()
            grid = current_grid

            # Store original empty positions to identify solved cells later
            self.original_empties = [(r, c) for r in range(self.grid_size) 
                                   for c in range(self.grid_size) if grid[r][c] == 0]

            # Validate initial clues
            if not self._validate_initial_grid(grid):
                messagebox.showerror('Invalid puzzle', 'Puzzle has conflicts in initial clues.')
                return

            self.status_label.config(text=f'Solving {self.grid_size}x{self.grid_size} with {algo}...')
            self.root.update()
            start = time.time()

            # Backtracking
            if algo == 'backtracking':
                s = Sudoku(grid)
                # Enable step recording for visualization
                ok, solved_grid, steps = s.solve(algorithm='backtracking', record_steps=True)
                elapsed = time.time() - start

                if ok:
                    self.status_label.config(text=f'Animating solution... ({len(steps)} steps)')
                    # Start animation
                    self.animate_steps(steps, solved_grid)
                else:
                    self.status_label.config(text='Backtracking could not solve this puzzle.')

            # Cultural Algorithm
            else:
                # Adjust population size based on grid size
                population_size = min(300, self.grid_size * 30)  # Smaller grid needs smaller population
                max_iters = 5000 if self.grid_size == 9 else 3000  # Fewer iterations for smaller grids
                
                try:
                    solver = CulturalSudokuSolver(grid, population_size=population_size, max_iters=max_iters)
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
                except Exception as e:
                    self.status_label.config(text=f'Error in Cultural Algorithm: {str(e)}')
                    print(f"Error in Cultural Algorithm: {e}")
        except Exception as e:
            self.status_label.config(text=f'Error: {str(e)}')
            print(f"Error in solve: {e}")

    def set_style(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.root.configure(bg='#f0f0f0')
        
        # Configure button styles
        style.configure('TButton',
                      font=('Arial', 10, 'bold'),
                      borderwidth=0,
                      relief='flat',
                      padding=5)
        
        style.map('TButton',
                 foreground=[('active', 'white')],
                 background=[('active', '#3a7bbd')])
        
        # Configure label styles
        style.configure('TLabel',
                      background='#f0f0f0',
                      font=('Arial', 10))
        
        # Configure frame styles
        style.configure('TFrame',
                      background='#f0f0f0')
        
        # Configure separator style
        style.configure('TSeparator',
                      background='#cccccc')

# ----------------------------------------------------------
#  MAIN
# ----------------------------------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = SudokuGUI(root)
    root.mainloop()
