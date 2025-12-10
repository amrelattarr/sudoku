# gui.py
# Tkinter GUI to load a Sudoku puzzle, choose algorithm, and solve.

import tkinter as tk
from tkinter import messagebox, ttk
from sudoku import Sudoku
from cultural_solver import CulturalSudokuSolver
from generator import generate_random_puzzle
import time
import datetime
import threading


class SudokuGUI:
    def __init__(self, root):
        self.root = root
        root.title('Sudoku Solver')
        
        # Configure window size and minimum size
        root.geometry('1100x800')
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
        
        # Performance metrics
        self.metrics = {
            'elapsed_time': 0,
            'steps': 0,
            'backtracks': 0,
            'generations': 0,
            'initial_conflicts': 0,
            'final_conflicts': 0,
            'mutations': 0,
            'belief_updates': 0
        }
        self.start_time = None
        
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
            
            # Configure grid layout for right frame
            self.right_frame.grid_rowconfigure(0, weight=1)
            self.right_frame.grid_columnconfigure(0, weight=1)
            self.right_frame.grid_columnconfigure(1, weight=0)
        
        # Create a frame for the grid with a nice border - using grid layout
        self.grid_frame = tk.Frame(self.right_frame, bg='#000000', padx=3, pady=3,
                                 highlightthickness=1, highlightbackground='#cccccc')
        self.grid_frame.grid(row=0, column=0, sticky='', padx=(20, 10), pady=20)
        
        # Build scrollable metrics panel (only once)
        if not hasattr(self, 'metrics_scroll_setup'):
            self.build_scrollable_metrics_panel(self.right_frame)
            self.metrics_scroll_setup = True

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
    
    def build_scrollable_metrics_panel(self, parent):
        """Build the scrollable performance metrics panel."""
        # Outer frame with title
        metrics_outer_frame = tk.LabelFrame(
            parent,
            text='Performance History',
            bg='#f0f0f0',
            font=('Arial', 12, 'bold'),
            fg='#2c3e50',
            padx=5,
            pady=5,
            relief='solid',
            borderwidth=1,
            width=280,
            height=500
        )
        metrics_outer_frame.grid(row=0, column=1, sticky='ns', padx=(10, 20), pady=20)
        metrics_outer_frame.grid_propagate(False)  # Prevent resizing based on content
        
        # Create canvas and scrollbar
        canvas_frame = tk.Frame(metrics_outer_frame, bg='#f0f0f0')
        canvas_frame.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(canvas_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        
        # Canvas
        self.metrics_canvas = tk.Canvas(
            canvas_frame,
            bg='#f0f0f0',
            yscrollcommand=scrollbar.set,
            highlightthickness=0,
            width=250
        )
        self.metrics_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.metrics_canvas.yview)
        
        # Container inside canvas for all metrics cards
        self.metrics_container = tk.Frame(self.metrics_canvas, bg='#f0f0f0')
        self.metrics_canvas_window = self.metrics_canvas.create_window(
            0, 0, window=self.metrics_container, anchor='nw'
        )
        
        # Update scrollregion when content changes
        def on_configure(event):
            self.metrics_canvas.configure(scrollregion=self.metrics_canvas.bbox('all'))
        self.metrics_container.bind('<Configure>', on_configure)
        
        # Bind mousewheel for scrolling only when mouse is over the canvas
        def on_mousewheel(event):
            self.metrics_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def bind_mousewheel(event):
            self.metrics_canvas.bind("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            self.metrics_canvas.unbind("<MouseWheel>")
        
        # Bind enter/leave events to control mousewheel scrolling
        self.metrics_canvas.bind("<Enter>", bind_mousewheel)
        self.metrics_canvas.bind("<Leave>", unbind_mousewheel)
        
        # Keep track of metric cards
        self.metrics_cards = []
        self.current_card = None
        self.should_auto_scroll = True  # Flag for auto-scroll on new cards
    
    def add_metrics_card(self, algorithm):
        """Add a new metrics card for the current solve."""
        # Create card frame
        card_frame = tk.Frame(
            self.metrics_container,
            bg='#ffffff',
            relief='solid',
            borderwidth=1,
            padx=10,
            pady=10
        )
        card_frame.pack(fill='x', pady=5, padx=5)
        
        # Timestamp
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        timestamp_label = tk.Label(
            card_frame,
            text=f'Solve at {timestamp}',
            bg='#ffffff',
            font=('Arial', 9),
            fg='#7f8c8d',
            anchor='w'
        )
        timestamp_label.pack(fill='x', pady=(0, 5))
        
        # Algorithm name
        algo_name = 'Backtracking' if algorithm == 'backtracking' else 'Cultural Evolution'
        algo_label = tk.Label(
            card_frame,
            text=f'Algorithm: {algo_name}',
            bg='#ffffff',
            font=('Arial', 10, 'bold'),
            fg='#3498db',
            anchor='w'
        )
        algo_label.pack(fill='x', pady=(0, 8))
        
        # Separator
        separator = ttk.Separator(card_frame, orient='horizontal')
        separator.pack(fill='x', pady=5)
        
        # Metrics container
        metrics_data = {}
        
        # Create the card's metric storage
        card = {
            'frame': card_frame,
            'algorithm': algorithm,
            'metrics': metrics_data,
            'timestamp': timestamp
        }
        
        self.metrics_cards.append(card)
        self.current_card = card
        
        # Force immediate scroll to bottom with multiple update calls
        def scroll_to_bottom():
            self.metrics_canvas.update_idletasks()
            self.metrics_canvas.configure(scrollregion=self.metrics_canvas.bbox('all'))
            self.metrics_canvas.yview_moveto(1.0)
        
        # Execute scroll immediately and after a short delay to ensure it works
        scroll_to_bottom()
        self.root.after(10, scroll_to_bottom)
        self.root.after(50, scroll_to_bottom)
        
        return card
    
    def add_metric_to_card(self, card, key, label_text, value):
        """Add a metric row to a specific card."""
        if key in card['metrics']:
            # Update existing metric
            card['metrics'][key]['value'].config(text=str(value))
        else:
            # Create new metric row
            row_frame = tk.Frame(card['frame'], bg='#ffffff')
            row_frame.pack(fill='x', pady=3)
            
            label = tk.Label(
                row_frame,
                text=label_text,
                bg='#ffffff',
                font=('Arial', 9, 'bold'),
                fg='#34495e',
                anchor='w',
                width=14
            )
            label.pack(side='left')
            
            value_label = tk.Label(
                row_frame,
                text=str(value),
                bg='#ffffff',
                font=('Arial', 9),
                fg='#2ecc71',
                anchor='e'
            )
            value_label.pack(side='right', fill='x', expand=True)
            
            card['metrics'][key] = {
                'frame': row_frame,
                'label': label,
                'value': value_label
            }
    
    def update_current_metric(self, key, label_text, value):
        """Update a metric in the current card (thread-safe)."""
        def update():
            if self.current_card:
                self.add_metric_to_card(self.current_card, key, label_text, value)
        # Use after() for thread-safe GUI updates
        self.root.after(0, update)

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
                    e.config(fg='black')  # Original clues in black
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
                    # Color original clues in black, solved cells in green
                    if (r, c) in self.original_empties:
                        entry.config(fg='blue')  # Solved cells in green
                    else:
                        entry.config(fg='black')  # Original clues in black
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
                # Create new metrics card for this solve
                self.add_metrics_card('backtracking')
                self.start_time = start
                
                # Run solver in separate thread
                def solve_backtracking_thread():
                    try:
                        # Create solver with callback
                        def backtrack_callback(event_type, steps, backtracks):
                            elapsed = time.time() - self.start_time
                            self.update_current_metric('elapsed_time', 'Elapsed Time:', f'{elapsed:.2f}s')
                            self.update_current_metric('steps', 'Steps:', f'{steps:,}')
                            self.update_current_metric('backtracks', 'Backtracks:', f'{backtracks:,}')
                        
                        s = Sudoku(grid, callback=backtrack_callback)
                        # Enable step recording for visualization
                        ok, solved_grid, steps = s.solve(algorithm='backtracking', record_steps=True)
                        elapsed = time.time() - start

                        # Schedule GUI updates on main thread
                        def finalize():
                            if ok:
                                # Final metric update
                                self.update_current_metric('elapsed_time', 'Elapsed Time:', f'{elapsed:.2f}s')
                                self.update_current_metric('steps', 'Steps:', f'{len(steps):,}')
                                self.update_current_metric('backtracks', 'Backtracks:', f'{s.backtrack_count:,}')
                                
                                self.status_label.config(text=f'Animating solution... ({len(steps)} steps)')
                                # Start animation
                                self.animate_steps(steps, solved_grid)
                            else:
                                self.status_label.config(text='Backtracking could not solve this puzzle.')
                        
                        self.root.after(0, finalize)
                    except Exception as e:
                        self.root.after(0, lambda: self.status_label.config(text=f'Error: {str(e)}'))
                        print(f"Error in backtracking thread: {e}")
                
                # Start thread
                threading.Thread(target=solve_backtracking_thread, daemon=True).start()

            # Cultural Algorithm
            else:
                # Create new metrics card for this solve
                self.add_metrics_card('cultural')
                self.start_time = start
                
                # Adjust population size based on grid size
                population_size = min(300, self.grid_size * 30)  # Smaller grid needs smaller population
                max_iters = 5000 if self.grid_size == 9 else 3000  # Fewer iterations for smaller grids
                
                # Run solver in separate thread
                def solve_cultural_thread():
                    try:
                        # Create solver with callback
                        def cultural_callback(generation, conflicts, mutations, belief_updates):
                            elapsed = time.time() - self.start_time
                            self.update_current_metric('elapsed_time', 'Elapsed Time:', f'{elapsed:.2f}s')
                            self.update_current_metric('generations', 'Generations:', f'{generation:,}')
                            
                            # Calculate improvement rate
                            if hasattr(self, '_initial_conflicts') and self._initial_conflicts > 0:
                                improvement = ((self._initial_conflicts - conflicts) / self._initial_conflicts) * 100
                                self.update_current_metric('improvement', 'Improvement:', f'{improvement:.1f}%')
                            else:
                                self._initial_conflicts = conflicts
                                self.update_current_metric('improvement', 'Improvement:', '0.0%')
                            
                            self.update_current_metric('mutations', 'Mutations:', f'{mutations:,}')
                            self.update_current_metric('belief_updates', 'Belief Updates:', f'{belief_updates:,}')
                        
                        solver = CulturalSudokuSolver(grid, population_size=population_size, 
                                                     max_iters=max_iters, callback=cultural_callback)
                        sol, score, iters = solver.run()
                        elapsed = time.time() - start

                        # Schedule GUI updates on main thread
                        def finalize():
                            try:
                                # Final metric update
                                self.update_current_metric('elapsed_time', 'Elapsed Time:', f'{elapsed:.2f}s')
                                self.update_current_metric('generations', 'Generations:', f'{iters:,}')
                                if hasattr(self, '_initial_conflicts') and self._initial_conflicts > 0:
                                    improvement = ((self._initial_conflicts - score) / self._initial_conflicts) * 100
                                    self.update_current_metric('improvement', 'Improvement:', f'{improvement:.1f}%')
                                self.update_current_metric('mutations', 'Mutations:', f'{solver.mutation_count:,}')
                                self.update_current_metric('belief_updates', 'Belief Updates:', f'{solver.belief_update_count:,}')

                                if score == 0:
                                    # First load the solved puzzle with default colors
                                    self.load_puzzle(sol)
                                    # Then update colors for solved cells
                                    for r, c in self.original_empties:
                                        self.entries[r][c].config(fg='blue')  # Solved cells in green
                                    self.status_label.config(text=f'Cultural Algorithm solved in {iters} iterations ({elapsed:.2f}s).')
                                else:
                                    self.load_puzzle(sol)
                                    self.status_label.config(text=f'Best solution has {score} conflicts — after {iters} iterations.')
                                
                                # Clean up
                                if hasattr(self, '_initial_conflicts'):
                                    delattr(self, '_initial_conflicts')
                            except Exception as e:
                                self.status_label.config(text=f'Error finalizing: {str(e)}')
                                print(f"Error in finalize: {e}")
                        
                        self.root.after(0, finalize)
                    except Exception as e:
                        self.root.after(0, lambda: self.status_label.config(text=f'Error in Cultural Algorithm: {str(e)}'))
                        print(f"Error in cultural thread: {e}")
                
                # Start thread
                threading.Thread(target=solve_cultural_thread, daemon=True).start()
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
