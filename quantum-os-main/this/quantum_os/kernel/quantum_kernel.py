import cirq
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class QuantumTask:
    circuit: cirq.Circuit
    qubits: List[cirq.Qid]
    priority: int = 0
    noise_model: Optional[Dict] = None

class QuantumKernel:
    def __init__(self):
        self.task_queue = []
        self.available_qubits = [cirq.LineQubit(i) for i in range(10)]  # Default 10 qubits
        self.simulator = cirq.Simulator()

    def submit_task(self, task: QuantumTask) -> int:
        """Submit a quantum task to the kernel."""
        task_id = len(self.task_queue)
        self.task_queue.append(task)
        return task_id

    def execute_task(self, task_id: int) -> np.ndarray:
        """Execute a quantum task and return results."""
        if task_id >= len(self.task_queue):
            raise ValueError("Invalid task ID")
        
        task = self.task_queue[task_id]
        
        # Apply noise model if specified
        if task.noise_model:
            noisy_circuit = self.apply_noise_model(task.circuit, task.noise_model)
            result = self.simulator.simulate(noisy_circuit)
        else:
            result = self.simulator.simulate(task.circuit)
            
        return result.final_state_vector

    def apply_noise_model(self, circuit: cirq.Circuit, noise_model: Dict) -> cirq.Circuit:
        """Apply noise model to the circuit.
        
        Args:
            circuit: The quantum circuit to apply noise to
            noise_model: Dictionary containing noise parameters:
                - T1: Amplitude damping time
                - T2: Phase damping time
                Both parameters should be float values between 0 and 1
                
        Returns:
            cirq.Circuit: The circuit with noise model applied
            
        Raises:
            ValueError: If noise parameters are invalid
        """
        # Validate noise parameters
        noisy_circuit = circuit.copy()
        
        try:
            if 'T1' in noise_model:
                t1_value = float(noise_model['T1'])
                if not 0 <= t1_value <= 1:
                    raise ValueError("T1 value must be between 0 and 1")
                noisy_circuit.append(cirq.amplitude_damp(t1_value))
                
            if 'T2' in noise_model:
                t2_value = float(noise_model['T2'])
                if not 0 <= t2_value <= 1:
                    raise ValueError("T2 value must be between 0 and 1")
                noisy_circuit.append(cirq.phase_damp(t2_value))
                
            return noisy_circuit
            
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error applying noise model: {str(e)}. Please ensure all parameters are valid numbers between 0 and 1.")

    def allocate_qubits(self, num_qubits: int) -> List[cirq.Qid]:
        """Allocate virtual qubits for a task."""
        if num_qubits > len(self.available_qubits):
            raise ValueError("Not enough qubits available")
        return self.available_qubits[:num_qubits]
