import matplotlib.pyplot as plt
import numpy as np
import cirq
from qutip import Bloch
from typing import List, Dict

class QuantumVisualizer:
    def __init__(self):
        self.figure = None
        self.bloch_sphere = None
        
    def draw_circuit(self, circuit: cirq.Circuit, filename: str = None):
        """Draw quantum circuit using matplotlib."""
        self.figure = plt.figure(figsize=(12, 6))
        plt.title("Quantum Circuit Diagram")
        print(circuit)  # Cirq's built-in ASCII circuit drawing
        
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            
    def plot_bloch_sphere(self, state_vector: np.ndarray):
        """Visualize qubit states on Bloch sphere."""
        self.bloch_sphere = Bloch()
        
        # Convert state vector to Bloch sphere coordinates
        theta = 2 * np.arccos(np.abs(state_vector[0]))
        phi = np.angle(state_vector[1]) if len(state_vector) > 1 else 0
        
        # Add state vector to Bloch sphere
        self.bloch_sphere.add_states([theta, phi])
        self.bloch_sphere.render()
        
    def plot_probability_distribution(self, measurements: Dict[str, int], 
                                   total_shots: int):
        """Plot measurement probability distribution."""
        states = list(measurements.keys())
        probabilities = [count/total_shots for count in measurements.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(states, probabilities)
        plt.title("Measurement Probability Distribution")
        plt.xlabel("Quantum States")
        plt.ylabel("Probability")
        plt.show()
        
    def plot_noise_effects(self, ideal_results: np.ndarray, 
                          noisy_results: np.ndarray):
        """Visualize the effects of noise on quantum states."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Ideal State")
        plt.bar(range(len(ideal_results)), np.abs(ideal_results)**2)
        
        plt.subplot(1, 2, 2)
        plt.title("Noisy State")
        plt.bar(range(len(noisy_results)), np.abs(noisy_results)**2)
        
        plt.tight_layout()
        plt.show()
