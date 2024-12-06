import click
import cirq
from ..kernel.quantum_kernel import QuantumKernel, QuantumTask
from ..device_manager.virtual_device import DeviceManager, DeviceType
from ..instruction_manager.qir_manager import InstructionManager

class QuantumCLI:
    def __init__(self):
        self.kernel = QuantumKernel()
        self.device_manager = DeviceManager()
        self.instruction_manager = InstructionManager()

    @click.group()
    def cli():
        """Quantum Operating System Command Line Interface"""
        pass

    @cli.command()
    @click.option('--num-qubits', default=5, help='Number of qubits to initialize')
    def init_device(self, num_qubits):
        """Initialize a new quantum device"""
        device = self.device_manager.create_device(
            name="default",
            device_type=DeviceType.GATE_BASED,
            num_qubits=num_qubits
        )
        click.echo(f"Created quantum device with {num_qubits} qubits")
        return device

    @cli.command()
    @click.argument('circuit_file')
    def run_circuit(self, circuit_file):
        """Run a quantum circuit from a file"""
        try:
            # Load and validate circuit
            circuit = cirq.read_json(circuit_file)
            if not self.instruction_manager.validate_circuit(circuit):
                click.echo("Circuit validation failed")
                return

            # Create and submit task
            device = self.device_manager.get_device("default")
            if not device:
                click.echo("No quantum device initialized")
                return

            task = QuantumTask(
                circuit=circuit,
                qubits=device.get_available_qubits()
            )
            
            task_id = self.kernel.submit_task(task)
            result = self.kernel.execute_task(task_id)
            
            click.echo(f"Circuit executed successfully")
            click.echo(f"Result state vector: {result}")
            
        except Exception as e:
            click.echo(f"Error running circuit: {str(e)}")

    @cli.command()
    def list_devices(self):
        """List all available quantum devices"""
        devices = self.device_manager.list_devices()
        if not devices:
            click.echo("No devices registered")
            return
        
        for device_name in devices:
            device = self.device_manager.get_device(device_name)
            click.echo(f"Device: {device_name}")
            click.echo(f"Type: {device.device_type.value}")
            click.echo(f"Qubits: {device.num_qubits}")

if __name__ == '__main__':
    cli = QuantumCLI()
    cli.cli()
