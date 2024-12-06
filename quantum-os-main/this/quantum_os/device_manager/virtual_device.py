import cirq
import numpy as np
from typing import Dict, List, Optional
from enum import Enum

class DeviceType(Enum):
    GATE_BASED = "gate_based"
    ANNEALING = "annealing"

class VirtualQuantumDevice:
    def __init__(self, device_type: DeviceType, num_qubits: int):
        self.device_type = device_type
        self.num_qubits = num_qubits
        self.qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        self.noise_model = {}
        
    def set_noise_model(self, t1: float = None, t2: float = None, dephasing: float = None):
        """Set noise parameters for the virtual device."""
        if t1 is not None:
            self.noise_model['T1'] = t1
        if t2 is not None:
            self.noise_model['T2'] = t2
        if dephasing is not None:
            self.noise_model['dephasing'] = dephasing

    def get_available_qubits(self) -> List[cirq.Qid]:
        """Return list of available qubits."""
        return self.qubits

    def apply_noise(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Apply configured noise model to the circuit."""
        noisy_circuit = circuit.copy()
        
        if 'T1' in self.noise_model:
            noisy_circuit.append(cirq.amplitude_damp(self.noise_model['T1']))
        if 'T2' in self.noise_model:
            noisy_circuit.append(cirq.phase_damp(self.noise_model['T2']))
        if 'dephasing' in self.noise_model:
            for qubit in self.qubits:
                noisy_circuit.append(cirq.Z(qubit) ** self.noise_model['dephasing'])
                
        return noisy_circuit

class DeviceManager:
    def __init__(self):
        self.devices: Dict[str, VirtualQuantumDevice] = {}
        
    def create_device(self, name: str, device_type: DeviceType, num_qubits: int) -> VirtualQuantumDevice:
        """Create and register a new virtual quantum device."""
        device = VirtualQuantumDevice(device_type, num_qubits)
        self.devices[name] = device
        return device
    
    def get_device(self, name: str) -> Optional[VirtualQuantumDevice]:
        """Get a registered device by name."""
        return self.devices.get(name)
    
    def list_devices(self) -> List[str]:
        """List all registered devices."""
        return list(self.devices.keys())
