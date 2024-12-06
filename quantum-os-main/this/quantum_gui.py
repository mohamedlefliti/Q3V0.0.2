import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cirq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import json
import os
import multiprocessing as mp
from functools import partial
import logging
import hashlib

class CircuitCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key):
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least frequently used item
            min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = value
        self.access_count[key] = 1

class QuantumGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Quantum Computing Interface")
        self.root.geometry("1000x800")
        
        # Theme settings
        self.dark_mode = tk.BooleanVar(value=False)
        self.setup_theme()
        
        # Initialize components
        self.simulator = cirq.Simulator()
        self.current_circuit = []
        self.measurement_results = {}
        self.circuit_cache = CircuitCache()
        self.setup_gui()
        
        # Error logging
        self.setup_error_logging()
    
    def setup_error_logging(self):
        """Configure error logging"""
        self.logger = logging.getLogger('quantum_gui')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler('quantum_gui.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def get_circuit_hash(self):
        """Generate hash for current circuit configuration"""
        circuit_str = str(self.current_circuit)
        return hashlib.md5(circuit_str.encode()).hexdigest()
    
    def setup_theme(self):
        # Configure theme colors
        self.colors = {
            'light': {
                'bg': '#ffffff',
                'fg': '#000000',
                'button': '#f0f0f0',
                'highlight': '#0078d4'
            },
            'dark': {
                'bg': '#1e1e1e',
                'fg': '#ffffff',
                'button': '#333333',
                'highlight': '#0078d4'
            }
        }
        self.apply_theme()

    def apply_theme(self):
        theme = 'dark' if self.dark_mode.get() else 'light'
        style = ttk.Style()
        
        # Configure ttk styles
        style.configure('TFrame', background=self.colors[theme]['bg'])
        style.configure('TLabel', background=self.colors[theme]['bg'], foreground=self.colors[theme]['fg'])
        style.configure('TButton', background=self.colors[theme]['button'])
        
        # Apply colors to root window
        self.root.configure(bg=self.colors[theme]['bg'])
        
    def setup_gui(self):
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Circuit Designer Tab
        circuit_frame = ttk.Frame(self.notebook)
        self.notebook.add(circuit_frame, text='Circuit Designer')
        self.setup_circuit_designer(circuit_frame)
        
        # Results Viewer Tab
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text='Results Viewer')
        self.setup_results_viewer(results_frame)
        
        # Algorithm Library Tab
        algorithm_frame = ttk.Frame(self.notebook)
        self.notebook.add(algorithm_frame, text='Algorithm Library')
        self.setup_algorithm_library(algorithm_frame)
        
        # Noise Simulator Tab
        noise_frame = ttk.Frame(self.notebook)
        self.notebook.add(noise_frame, text='Noise Simulator')
        self.setup_noise_simulator(noise_frame)
        
        # Menu Bar
        self.setup_menu()
        
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Circuit", command=self.clear_circuit)
        file_menu.add_command(label="Save Circuit", command=self.save_circuit)
        file_menu.add_command(label="Load Circuit", command=self.load_circuit)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode, command=self.apply_theme)
        
        # Tools Menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Circuit Analyzer", command=self.analyze_circuit)
        tools_menu.add_command(label="State Tomography", command=self.state_tomography)
        tools_menu.add_command(label="Bloch Sphere", command=self.show_bloch_sphere)
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Tutorial", command=self.show_tutorial)
        help_menu.add_command(label="About", command=self.show_about)
        
    def setup_circuit_designer(self, parent):
        # Left Panel - Gates
        gate_frame = ttk.LabelFrame(parent, text="Quantum Gates")
        gate_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        # Single Qubit Gates
        ttk.Label(gate_frame, text="Single Qubit Gates").pack(pady=5)
        single_gates = ['H', 'X', 'Y', 'Z', 'T', 'S']
        for gate in single_gates:
            btn = ttk.Button(
                gate_frame,
                text=gate,
                command=lambda g=gate: self.add_gate(g)
            )
            btn.pack(pady=2, padx=5, fill='x')
            
        ttk.Separator(gate_frame, orient='horizontal').pack(fill='x', pady=5)
            
        # Two Qubit Gates
        ttk.Label(gate_frame, text="Two Qubit Gates").pack(pady=5)
        two_gates = ['CNOT', 'SWAP', 'CZ']
        for gate in two_gates:
            btn = ttk.Button(
                gate_frame,
                text=gate,
                command=lambda g=gate: self.add_gate(g)
            )
            btn.pack(pady=2, padx=5, fill='x')
            
        # Qubit Selection
        ttk.Separator(gate_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Label(gate_frame, text="Target Qubit").pack(pady=5)
        self.target_qubit = ttk.Spinbox(gate_frame, from_=0, to=4, width=5)
        self.target_qubit.pack(pady=2)
        
        # Right Panel - Circuit Display and Controls
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        # Circuit Display
        circuit_frame = ttk.LabelFrame(right_frame, text="Current Circuit")
        circuit_frame.pack(fill='both', expand=True, pady=5)
        
        self.circuit_text = tk.Text(circuit_frame, height=15)
        self.circuit_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Control Buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Button(
            control_frame,
            text="Run Circuit",
            command=self.run_circuit
        ).pack(side='left', padx=5)
        
        ttk.Button(
            control_frame,
            text="Clear Circuit",
            command=self.clear_circuit
        ).pack(side='left', padx=5)
        
        ttk.Button(
            control_frame,
            text="Visualize Circuit",
            command=self.visualize_circuit
        ).pack(side='left', padx=5)
        
        # Number of Shots
        ttk.Label(control_frame, text="Shots:").pack(side='left', padx=5)
        self.shots_var = tk.StringVar(value="1000")
        ttk.Entry(
            control_frame,
            textvariable=self.shots_var,
            width=8
        ).pack(side='left', padx=5)
        
    def setup_algorithm_library(self, parent):
        # Algorithm List
        algorithms_frame = ttk.LabelFrame(parent, text="Quantum Algorithms")
        algorithms_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        algorithms = [
            "Bell State",
            "GHZ State",
            "Quantum Fourier Transform",
            "Grover's Algorithm",
            "Quantum Teleportation"
        ]
        
        for algo in algorithms:
            ttk.Button(
                algorithms_frame,
                text=algo,
                command=lambda a=algo: self.load_algorithm(a)
            ).pack(pady=2, padx=5, fill='x')
            
        # Algorithm Description
        desc_frame = ttk.LabelFrame(parent, text="Algorithm Description")
        desc_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.algo_desc = tk.Text(desc_frame, wrap=tk.WORD)
        self.algo_desc.pack(fill='both', expand=True, padx=5, pady=5)
        
    def setup_noise_simulator(self, parent):
        """Setup the noise simulation interface"""
        # Main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Noise Parameters")
        params_frame.pack(fill='x', padx=5, pady=5)
        
        # T1 Relaxation
        t1_frame = ttk.Frame(params_frame)
        t1_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(t1_frame, text="T1 Relaxation Time (μs):").pack(side='left')
        self.t1_var = tk.StringVar(value="50")
        t1_entry = ttk.Entry(t1_frame, textvariable=self.t1_var, width=10)
        t1_entry.pack(side='left', padx=5)
        
        # T2 Dephasing
        t2_frame = ttk.Frame(params_frame)
        t2_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(t2_frame, text="T2 Dephasing Time (μs):").pack(side='left')
        self.t2_var = tk.StringVar(value="25")
        t2_entry = ttk.Entry(t2_frame, textvariable=self.t2_var, width=10)
        t2_entry.pack(side='left', padx=5)
        
        # Gate Error
        error_frame = ttk.Frame(params_frame)
        error_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(error_frame, text="Gate Error Rate (%):").pack(side='left')
        self.error_var = tk.StringVar(value="0.1")
        error_entry = ttk.Entry(error_frame, textvariable=self.error_var, width=10)
        error_entry.pack(side='left', padx=5)
        
        # Validation
        def validate_float(P):
            if P == "":
                return True
            try:
                float(P)
                return True
            except ValueError:
                return False
        
        vcmd = (self.root.register(validate_float), '%P')
        t1_entry.configure(validate='key', validatecommand=vcmd)
        t2_entry.configure(validate='key', validatecommand=vcmd)
        error_entry.configure(validate='key', validatecommand=vcmd)
        
        # Apply button with better styling
        btn_frame = ttk.Frame(params_frame)
        btn_frame.pack(fill='x', padx=5, pady=10)
        apply_btn = ttk.Button(
            btn_frame,
            text="Apply Noise Model",
            command=self.apply_noise_model,
            style='Accent.TButton'
        )
        apply_btn.pack(expand=True)
        
        # Create accent button style
        style = ttk.Style()
        style.configure('Accent.TButton', 
                       background='#0078D4',
                       foreground='white')
        
        # Information frame
        info_frame = ttk.LabelFrame(main_frame, text="Noise Model Information")
        info_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        info_text = tk.Text(info_frame, wrap=tk.WORD, height=8)
        info_text.pack(fill='both', expand=True, padx=5, pady=5)
        info_text.insert('1.0', """
Noise Parameter Guidelines:

• T1 Relaxation Time: Energy relaxation time
  - Typical range: 10-100 μs
  - Higher values mean less energy decay

• T2 Dephasing Time: Phase coherence time
  - Must be ≤ 2*T1
  - Typical range: 5-50 μs
  - Higher values mean less phase noise

• Gate Error Rate: Single-qubit gate error
  - Range: 0-100%
  - Typical values: 0.1-1%
  - Lower values mean more accurate gates
""")
        info_text.configure(state='disabled')
        
    def apply_noise_model(self):
        """Apply custom noise model to the circuit."""
        try:
            # Create a custom noise channel
            noise_params = {
                'T1': float(self.t1_var.get()) if self.t1_var.get() else 0,
                'T2': float(self.t2_var.get()) if self.t2_var.get() else 0
            }
            
            # Create a concrete noise model using cirq's built-in noise channels
            noise_model = cirq.NoiseModel.from_noise_model_like(
                cirq.depolarize(p=0.01)  # Default minimal noise
            )
            
            if noise_params['T1'] > 0:
                noise_model = cirq.NoiseModel.from_noise_model_like(
                    cirq.amplitude_damp(gamma=noise_params['T1'])
                )
            
            if noise_params['T2'] > 0:
                noise_model = cirq.NoiseModel.from_noise_model_like(
                    cirq.phase_damp(gamma=noise_params['T2'])
                )
            
            self.simulator = cirq.DensityMatrixSimulator(noise=noise_model)
            messagebox.showinfo("Success", "Noise model applied successfully")
            
        except (ValueError, TypeError) as e:
            messagebox.showerror(
                "Error",
                f"Error applying noise model: {str(e)}\nPlease ensure all parameters are valid numbers between 0 and 1."
            )
            self.logger.error(f"Error in noise model: {str(e)}")
            
    def add_gate(self, gate_name):
        target = self.target_qubit.get()
        if not target:
            messagebox.showerror("Error", "Please select a target qubit")
            return
        try:
            target_int = int(target)
            self.current_circuit.append((gate_name, target_int))
            self.update_circuit_display()
        except ValueError:
            messagebox.showerror("Error", "Invalid qubit number")
            
    def update_circuit_display(self):
        self.circuit_text.delete('1.0', 'end')
        for gate, target in self.current_circuit:
            self.circuit_text.insert('end', f"{gate} on q{target}\n")
            
    def save_circuit(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".qc",
            filetypes=[("Quantum Circuit", "*.qc")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.current_circuit, f)
                
    def load_circuit(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Quantum Circuit", "*.qc")]
        )
        if file_path:
            with open(file_path, 'r') as f:
                self.current_circuit = json.load(f)
                self.update_circuit_display()
                
    def analyze_circuit(self):
        if not self.current_circuit:
            messagebox.showinfo("Analysis", "No circuit to analyze")
            return
            
        analysis = {
            "Circuit Depth": len(self.current_circuit),
            "Gate Count": len(self.current_circuit),
            "Qubit Count": len(set(t for _, t in self.current_circuit))
        }
        
        msg = "Circuit Analysis:\n\n"
        for key, value in analysis.items():
            msg += f"{key}: {value}\n"
            
        messagebox.showinfo("Circuit Analysis", msg)
        
    def state_tomography(self):
        messagebox.showinfo(
            "State Tomography",
            "This feature will perform quantum state tomography on the current circuit."
        )
        
    def show_tutorial(self):
        tutorial = """
Quantum Circuit Designer Tutorial:

1. Select gates from the left panel
2. Choose target qubit(s)
3. Add gates to your circuit
4. Run the circuit to see results
5. Use the noise simulator to test circuit robustness
6. Save your circuits for later use

For more help, visit our documentation.
"""
        messagebox.showinfo("Tutorial", tutorial)
        
    def show_about(self):
        about_text = """
Quantum Computing Interface v1.0

A comprehensive quantum circuit design and simulation tool.
Created for educational and research purposes.

 2024 Quantum Computing Lab
"""
        messagebox.showinfo("About", about_text)
        
    def run_circuit(self):
        try:
            # Check cache first
            circuit_hash = self.get_circuit_hash()
            cached_results = self.circuit_cache.get(circuit_hash)
            
            if cached_results is not None:
                self.logger.info("Using cached results")
                self.measurement_results = cached_results
            else:
                # Create and run circuit
                self.logger.info("Running new circuit simulation")
                num_qubits = max(t for _, t in self.current_circuit) + 1
                qubits = cirq.LineQubit.range(num_qubits)
                circuit = cirq.Circuit()
                
                # Add gates
                for gate_name, target in self.current_circuit:
                    try:
                        if gate_name in ['H', 'X', 'Y', 'Z', 'T', 'S']:
                            circuit.append(getattr(cirq, gate_name)(qubits[target]))
                        elif gate_name in ['CNOT', 'SWAP', 'CZ'] and target + 1 < num_qubits:
                            gate = getattr(cirq, gate_name)
                            circuit.append(gate(qubits[target], qubits[target + 1]))
                    except Exception as e:
                        self.logger.error(f"Error adding gate {gate_name}: {str(e)}")
                        raise
                
                # Add measurements
                circuit.append(cirq.measure(*qubits, key='result'))
                
                # Run simulation
                shots = int(self.shots_var.get())
                try:
                    if len(circuit) > 10 or shots > 1000:
                        results = self.run_parallel_simulation(circuit, shots)
                    else:
                        results = self.simulator.run(circuit, repetitions=shots)
                    
                    self.measurement_results = results.histogram(key='result')
                    
                    # Cache results
                    self.circuit_cache.put(circuit_hash, self.measurement_results)
                    
                except Exception as e:
                    self.logger.error(f"Simulation error: {str(e)}")
                    raise
            
            # Display results
            self.display_results(num_qubits)
            
        except Exception as e:
            self.logger.exception("Error in circuit execution")
            messagebox.showerror("Error", str(e))
    
    def display_results(self, num_qubits):
        """Display measurement results"""
        try:
            self.notebook.select(1)  # Switch to Results tab
            self.results_text.delete('1.0', 'end')
            
            shots = int(self.shots_var.get())
            self.results_text.insert('end', f"Circuit Results ({shots} shots):\n\n")
            
            total_shots = sum(self.measurement_results.values())
            for bitstring, count in self.measurement_results.items():
                binary = format(bitstring, f'0{num_qubits}b')
                probability = count / total_shots
                self.results_text.insert('end', f"|{binary}⟩: {count} times ({probability:.3f})\n")
            
            # Plot results
            self.plot_results(num_qubits)
            
        except Exception as e:
            self.logger.error(f"Error displaying results: {str(e)}")
            raise
    
    def run_parallel_simulation(self, circuit, shots):
        """Run simulation in parallel for large circuits"""
        import multiprocessing as mp
        from functools import partial
        
        def simulate_chunk(chunk_size, simulator, circuit):
            return simulator.run(circuit, repetitions=chunk_size)
        
        # Determine optimal chunk size based on CPU cores
        num_cores = mp.cpu_count()
        chunk_size = max(1, shots // num_cores)
        chunks = [chunk_size] * (shots // chunk_size)
        if shots % chunk_size:
            chunks.append(shots % chunk_size)
        
        # Create process pool and run simulations
        with mp.Pool(num_cores) as pool:
            results = pool.map(
                partial(simulate_chunk, simulator=self.simulator, circuit=circuit),
                chunks
            )
        
        # Combine results
        combined_results = results[0]
        for r in results[1:]:
            combined_results.data = combined_results.data.append(r.data)
        
        return combined_results

    def plot_results(self, num_qubits):
        """Plot measurement results with enhanced visualization"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        total_shots = sum(self.measurement_results.values())
        states = [format(b, f'0{num_qubits}b') for b in self.measurement_results.keys()]
        probabilities = [count/total_shots for count in self.measurement_results.values()]
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(12, 5))
        
        # Bar plot
        ax1 = fig.add_subplot(121)
        ax1.bar(states, probabilities)
        ax1.set_title('Measurement Probabilities')
        ax1.set_xlabel('Quantum States')
        ax1.set_ylabel('Probability')
        plt.xticks(rotation=45)
        
        # Polar plot for phase visualization
        ax2 = fig.add_subplot(122, projection='polar')
        theta = np.linspace(0, 2*np.pi, len(states), endpoint=False)
        ax2.plot(theta, probabilities, 'o-')
        ax2.set_title('Phase Distribution')
        ax2.grid(True)
        
        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar for interactive plot manipulation
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        
    def setup_results_viewer(self, parent):
        # Results Text Area
        self.results_text = tk.Text(parent, height=10)
        self.results_text.pack(fill='x', padx=5, pady=5)
        
        # Plot Frame
        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
    def visualize_circuit(self):
        if not self.current_circuit:
            messagebox.showinfo("Visualization", "No circuit to visualize")
            return
            
        # Create visualization window
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Interactive Circuit Visualization")
        viz_window.geometry("1000x600")
        
        # Create circuit for visualization
        num_qubits = max(t for _, t in self.current_circuit) + 1
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()
        
        for gate_name, target in self.current_circuit:
            if gate_name in ['H', 'X', 'Y', 'Z', 'T', 'S']:
                circuit.append(getattr(cirq, gate_name)(qubits[target]))
            elif gate_name in ['CNOT', 'SWAP', 'CZ'] and target + 1 < num_qubits:
                gate = getattr(cirq, gate_name)
                circuit.append(gate(qubits[target], qubits[target + 1]))
        
        # Create figure for circuit diagram
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Draw circuit lines
        for i in range(num_qubits):
            ax.plot([0, len(self.current_circuit)], [i, i], 'k-', linewidth=1)
            ax.text(-0.5, i, f'q{i}: |0⟩', ha='right', va='center')
        
        # Draw gates
        for i, (gate_name, target) in enumerate(self.current_circuit):
            if gate_name in ['H', 'X', 'Y', 'Z', 'T', 'S']:
                ax.add_patch(plt.Rectangle((i-0.2, target-0.2), 0.4, 0.4, fill=True))
                ax.text(i, target, gate_name, ha='center', va='center', color='white')
            elif gate_name in ['CNOT', 'SWAP', 'CZ']:
                ax.add_patch(plt.Rectangle((i-0.2, target-0.2), 0.4, 0.4, fill=True))
                ax.add_patch(plt.Rectangle((i-0.2, target+0.8), 0.4, 0.4, fill=True))
                ax.plot([i, i], [target, target+1], 'k-', linewidth=2)
        
        ax.set_xlim(-1, len(self.current_circuit))
        ax.set_ylim(-0.5, num_qubits-0.5)
        ax.axis('off')
        
        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add zoom controls
        control_frame = ttk.Frame(viz_window)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(control_frame, text="Zoom:").pack(side='left')
        zoom = ttk.Scale(control_frame, from_=0.5, to=2.0, orient='horizontal')
        zoom.set(1.0)
        zoom.pack(side='left', padx=5)
        
        def update_zoom(*args):
            ax.set_xlim(-1*zoom.get(), len(self.current_circuit)*zoom.get())
            ax.set_ylim(-0.5*zoom.get(), (num_qubits-0.5)*zoom.get())
            canvas.draw()
        
        zoom.configure(command=update_zoom)
        
    def show_bloch_sphere(self):
        """Display interactive Bloch sphere visualization"""
        try:
            # Create window
            bloch_window = tk.Toplevel(self.root)
            bloch_window.title("Bloch Sphere Visualization")
            bloch_window.geometry("800x800")

            # Create figure
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Draw sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Plot main sphere
            ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
            
            # Draw circles
            theta = np.linspace(0, 2*np.pi, 100)
            
            # XY plane circle
            ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 'b--', alpha=0.5)
            # XZ plane circle
            ax.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta), 'r--', alpha=0.5)
            # YZ plane circle
            ax.plot(np.zeros_like(theta), np.cos(theta), np.sin(theta), 'g--', alpha=0.5)
            
            # Add axes
            # X axis (red)
            ax.quiver(-1.5, 0, 0, 3, 0, 0, color='red', arrow_length_ratio=0.1, alpha=0.5)
            # Y axis (green)
            ax.quiver(0, -1.5, 0, 0, 3, 0, color='green', arrow_length_ratio=0.1, alpha=0.5)
            # Z axis (blue)
            ax.quiver(0, 0, -1.5, 0, 0, 3, color='blue', arrow_length_ratio=0.1, alpha=0.5)
            
            # Add basis state labels
            ax.text(0, 0, 1.7, r'$|0\rangle$', fontsize=12)
            ax.text(0, 0, -1.7, r'$|1\rangle$', fontsize=12)
            ax.text(1.7, 0, 0, r'$|+\rangle$', fontsize=12)
            ax.text(-1.7, 0, 0, r'$|-\rangle$', fontsize=12)
            ax.text(0, 1.7, 0, r'$|+i\rangle$', fontsize=12)
            ax.text(0, -1.7, 0, r'$|-i\rangle$', fontsize=12)
            
            # Add axis labels
            ax.text(2, 0, 0, 'X', fontsize=12)
            ax.text(0, 2, 0, 'Y', fontsize=12)
            ax.text(0, 0, 2, 'Z', fontsize=12)
            
            # Set plot limits
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            
            # Remove background grid
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            # Initial view angle
            ax.view_init(elev=20, azim=45)
            
            # Embed in tkinter window
            canvas = FigureCanvasTkAgg(fig, master=bloch_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Add navigation toolbar
            toolbar = NavigationToolbar2Tk(canvas, bloch_window)
            toolbar.update()
            
            # Add rotation controls
            control_frame = ttk.Frame(bloch_window)
            control_frame.pack(fill='x', padx=5, pady=5)
            
            # Elevation control
            ttk.Label(control_frame, text="Elevation:").pack(side='left')
            elev = ttk.Scale(control_frame, from_=0, to=180, orient='horizontal')
            elev.set(20)
            elev.pack(side='left', padx=5, fill='x', expand=True)
            
            # Azimuth control
            ttk.Label(control_frame, text="Azimuth:").pack(side='left')
            azim = ttk.Scale(control_frame, from_=0, to=360, orient='horizontal')
            azim.set(45)
            azim.pack(side='left', padx=5, fill='x', expand=True)
            
            # Reset view button
            def reset_view():
                elev.set(20)
                azim.set(45)
                ax.view_init(elev=20, azim=45)
                canvas.draw()
            
            ttk.Button(
                control_frame,
                text="Reset View",
                command=reset_view
            ).pack(side='left', padx=5)
            
            # Update function
            def update_view(*args):
                ax.view_init(elev=elev.get(), azim=azim.get())
                canvas.draw()
            
            elev.configure(command=update_view)
            azim.configure(command=update_view)
            
            # Add state visualization if circuit exists
            if self.current_circuit:
                # Create and simulate minimal circuit
                num_qubits = max(t for _, t in self.current_circuit) + 1
                qubits = cirq.LineQubit.range(num_qubits)
                circuit = cirq.Circuit()
                
                # Add first qubit gates only
                for gate_name, target in self.current_circuit:
                    if target == 0:  # Only for first qubit
                        if gate_name in ['H', 'X', 'Y', 'Z', 'T', 'S']:
                            circuit.append(getattr(cirq, gate_name)(qubits[0]))
                
                # Simulate
                simulator = cirq.Simulator()
                result = simulator.simulate(circuit)
                
                # Get state for first qubit
                state = result.final_state_vector[0:2]
                
                # Convert to Bloch sphere coordinates
                theta = 2 * np.arccos(np.abs(state[0]))
                phi = np.angle(state[1]) - np.angle(state[0])
                
                # Plot state vector
                r = 1.0  # Unit sphere
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                # Add state vector arrow
                ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=2)
                
                # Update plot
                canvas.draw()
        
        except Exception as e:
            self.logger.error(f"Error in Bloch sphere: {str(e)}")
            messagebox.showerror(
                "Error",
                "Could not display Bloch sphere.\nError: " + str(e)
            )
            
    def clear_circuit(self):
        """Clear the current circuit and reset the display"""
        self.current_circuit = []
        self.circuit_text.delete('1.0', 'end')
        self.measurement_results = {}
        
        # Clear results if they exist
        if hasattr(self, 'results_text'):
            self.results_text.delete('1.0', 'end')
        
        # Clear plot if it exists
        if hasattr(self, 'plot_frame'):
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
    def load_algorithm(self, algorithm_name):
        """Load a predefined quantum algorithm"""
        self.clear_circuit()
        
        if algorithm_name == "Bell State":
            # Create Bell State: H(0), CNOT(0,1)
            self.current_circuit = [
                ('H', 0),
                ('CNOT', 0)
            ]
            self.algo_desc.delete('1.0', 'end')
            self.algo_desc.insert('end', """Bell State:
A fundamental quantum state of two qubits that exhibits quantum entanglement.
Circuit steps:
1. Apply Hadamard (H) gate to first qubit
2. Apply CNOT gate between first and second qubit
Expected outcome: Equal superposition of |00⟩ and |11⟩""")
            
        elif algorithm_name == "GHZ State":
            # Create GHZ State: H(0), CNOT(0,1), CNOT(1,2)
            self.current_circuit = [
                ('H', 0),
                ('CNOT', 0),
                ('CNOT', 1)
            ]
            self.algo_desc.delete('1.0', 'end')
            self.algo_desc.insert('end', """GHZ State:
A maximally entangled state of three qubits.
Circuit steps:
1. Apply Hadamard (H) gate to first qubit
2. Apply CNOT gates to create three-qubit entanglement
Expected outcome: Equal superposition of |000⟩ and |111⟩""")
            
        elif algorithm_name == "Quantum Fourier Transform":
            # Simple 2-qubit QFT: H(0), S(1), CNOT(0,1), H(1)
            self.current_circuit = [
                ('H', 0),
                ('S', 1),
                ('CNOT', 0),
                ('H', 1)
            ]
            self.algo_desc.delete('1.0', 'end')
            self.algo_desc.insert('end', """Quantum Fourier Transform (2-qubit):
Quantum version of the classical discrete Fourier transform.
Circuit steps:
1. Apply Hadamard gates
2. Apply phase rotations
3. Apply controlled operations
Expected outcome: Quantum state representing the Fourier transform""")
            
        elif algorithm_name == "Grover's Algorithm":
            # Simple 2-qubit Grover: H(0), H(1), X(0), X(1), H(1), CNOT(0,1), H(1), X(0), X(1), H(0), H(1)
            self.current_circuit = [
                ('H', 0), ('H', 1),
                ('X', 0), ('X', 1),
                ('H', 1),
                ('CNOT', 0),
                ('H', 1),
                ('X', 0), ('X', 1),
                ('H', 0), ('H', 1)
            ]
            self.algo_desc.delete('1.0', 'end')
            self.algo_desc.insert('end', """Grover's Algorithm (2-qubit):
Quantum algorithm for searching an unsorted database.
Circuit steps:
1. Initialize superposition
2. Apply oracle (marked state phase flip)
3. Apply diffusion operator
Expected outcome: Amplified amplitude of marked state""")
            
        elif algorithm_name == "Quantum Teleportation":
            # Quantum Teleportation: H(1), CNOT(1,2), CNOT(0,1), H(0), measure
            self.current_circuit = [
                ('H', 1),
                ('CNOT', 1),
                ('CNOT', 0),
                ('H', 0)
            ]
            self.algo_desc.delete('1.0', 'end')
            self.algo_desc.insert('end', """Quantum Teleportation:
Protocol for transmitting quantum states using entanglement.
Circuit steps:
1. Create Bell pair (H + CNOT)
2. Interact source qubit with Bell pair
3. Measure and apply corrections
Expected outcome: State transferred from first to third qubit""")
            
        # Update circuit display
        self.update_circuit_display()
        
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    gui = QuantumGUI()
    gui.run()
