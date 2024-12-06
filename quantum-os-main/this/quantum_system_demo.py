import cirq
import numpy as np
import matplotlib.pyplot as plt

class QuantumSystemDemo:
    def __init__(self):
        self.simulator = cirq.Simulator()
        
    def run_demo(self):
        print("=== Quantum Operating System Demo ===\n")
        
        # 1. Create Qubits
        print("1. Initializing Qubits...")
        q0, q1, q2 = cirq.LineQubit.range(3)
        print(f"Created 3 qubits: {q0}, {q1}, {q2}\n")
        
        # 2. Create GHZ State Circuit
        print("2. Creating GHZ State Circuit...")
        circuit = cirq.Circuit(
            cirq.H(q0),          # Hadamard on first qubit
            cirq.CNOT(q0, q1),   # CNOT between first and second qubit
            cirq.CNOT(q1, q2)    # CNOT between second and third qubit
        )
        print("Circuit created:")
        print("- H(q0): Creates superposition")
        print("- CNOT(q0,q1): Entangles first two qubits")
        print("- CNOT(q1,q2): Entangles third qubit\n")
        
        # 3. Simulate Circuit
        print("3. Simulating Circuit...")
        result = self.simulator.simulate(circuit)
        state_vector = result.final_state_vector
        
        # 4. Analyze Results
        print("4. Analyzing Results...")
        probabilities = np.abs(state_vector) ** 2
        
        states = ['000', '001', '010', '011', '100', '101', '110', '111']
        print("\nState Probabilities:")
        for state, prob in zip(states, probabilities):
            print(f"State |{state}> : {prob:.3f}")
            
        # 5. Visualize Results
        print("\n5. Creating Visualization...")
        plt.figure(figsize=(10, 6))
        plt.bar(states, probabilities)
        plt.title('GHZ State Probabilities')
        plt.xlabel('Quantum States')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('ghz_state_probabilities.png')
        print("Visualization saved as 'ghz_state_probabilities.png'\n")
        
        # 6. Run Multiple Measurements
        print("6. Running Multiple Measurements...")
        n_measurements = 1000
        measurement_circuit = circuit.copy()
        measurement_circuit.append(cirq.measure(q0, q1, q2, key='result'))
        
        results = self.simulator.run(measurement_circuit, repetitions=n_measurements)
        counts = results.histogram(key='result')
        
        print(f"\nMeasurement Results ({n_measurements} shots):")
        for bitstring, count in counts.items():
            print(f"State |{format(bitstring, '03b')}> : {count} times ({count/n_measurements:.3f})")

        # 7. Show Circuit Statistics
        print("\n7. Circuit Statistics:")
        print(f"- Number of qubits: 3")
        print(f"- Circuit depth: {len(circuit)}")
        print(f"- Number of gates: {sum(1 for moment in circuit for _ in moment)}")
        
        # 8. Quantum State Analysis
        print("\n8. Quantum State Analysis:")
        print("- Type: GHZ State")
        print("- Entanglement: Fully entangled state")
        print("- Expected measurement outcomes: Equal superposition of |000> and |111>")

if __name__ == '__main__':
    demo = QuantumSystemDemo()
    try:
        demo.run_demo()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
