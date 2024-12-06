import cirq
import numpy as np

def create_bell_state():
    # Create two qubits
    q0, q1 = cirq.LineQubit.range(2)
    
    # Create a circuit
    circuit = cirq.Circuit(
        cirq.H(q0),  # Hadamard gate on q0
        cirq.CNOT(q0, q1)  # CNOT gate with q0 as control and q1 as target
    )
    
    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    
    print("Circuit operations:")
    print("1. Hadamard gate on qubit 0")
    print("2. CNOT gate between qubit 0 and qubit 1")
    
    print("\nFinal state probabilities:")
    state_vector = result.final_state_vector
    probabilities = np.abs(state_vector) ** 2
    
    states = ['00', '01', '10', '11']
    for state, prob in zip(states, probabilities):
        print(f"State |{state}>: {prob:.3f}")

if __name__ == '__main__':
    create_bell_state()
