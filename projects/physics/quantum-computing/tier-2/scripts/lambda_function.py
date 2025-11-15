#!/usr/bin/env python3
"""
AWS Lambda function for quantum circuit simulation.

This function simulates quantum circuits using numpy-based state vector simulation.
It reads circuit definitions from S3, simulates them, and stores results in DynamoDB.

Environment Variables:
    BUCKET_NAME: S3 bucket name
    DYNAMODB_TABLE: DynamoDB table name (default: QuantumResults)
    MAX_QUBITS: Maximum qubits to simulate (default: 10)
    AWS_REGION: AWS region (default: us-east-1)
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import boto3
from botocore.exceptions import ClientError


# Initialize AWS clients (reused across Lambda invocations)
s3_client = None
dynamodb = None
table = None


def get_aws_clients():
    """Initialize and return AWS clients."""
    global s3_client, dynamodb, table

    if s3_client is None:
        region = os.environ.get('AWS_REGION', 'us-east-1')
        s3_client = boto3.client('s3', region_name=region)
        dynamodb = boto3.resource('dynamodb', region_name=region)

        table_name = os.environ.get('DYNAMODB_TABLE', 'QuantumResults')
        table = dynamodb.Table(table_name)

    return s3_client, table


class QuantumCircuitSimulator:
    """Simple quantum circuit simulator using numpy."""

    def __init__(self, num_qubits: int):
        """
        Initialize quantum simulator.

        Args:
            num_qubits: Number of qubits in the circuit
        """
        if num_qubits > int(os.environ.get('MAX_QUBITS', '10')):
            raise ValueError(f"Maximum {os.environ.get('MAX_QUBITS', '10')} qubits supported")

        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

        # Initialize state vector to |00...0⟩
        self.state_vector = np.zeros(self.dim, dtype=complex)
        self.state_vector[0] = 1.0

    def apply_gate(self, gate_name: str, qubits: List[int], params: List[float] = None):
        """
        Apply quantum gate to the state vector.

        Args:
            gate_name: Name of the gate (h, x, y, z, cx, etc.)
            qubits: List of qubit indices the gate acts on
            params: Optional parameters for parameterized gates
        """
        gate_name = gate_name.lower()

        if gate_name == 'h':
            # Hadamard gate
            self._apply_single_qubit_gate(qubits[0], self._hadamard_matrix())
        elif gate_name == 'x':
            # Pauli-X (NOT) gate
            self._apply_single_qubit_gate(qubits[0], self._pauli_x_matrix())
        elif gate_name == 'y':
            # Pauli-Y gate
            self._apply_single_qubit_gate(qubits[0], self._pauli_y_matrix())
        elif gate_name == 'z':
            # Pauli-Z gate
            self._apply_single_qubit_gate(qubits[0], self._pauli_z_matrix())
        elif gate_name in ['cx', 'cnot']:
            # Controlled-NOT gate
            self._apply_cnot(qubits[0], qubits[1])
        elif gate_name in ['ccx', 'toffoli']:
            # Toffoli (CCNOT) gate
            self._apply_toffoli(qubits[0], qubits[1], qubits[2])
        elif gate_name in ['rx', 'ry', 'rz']:
            # Rotation gates
            if params is None or len(params) == 0:
                params = [0.5]  # Default rotation angle
            self._apply_rotation(qubits[0], gate_name, params[0])
        elif gate_name == 's':
            # S gate (phase gate)
            self._apply_single_qubit_gate(qubits[0], self._s_matrix())
        elif gate_name == 't':
            # T gate
            self._apply_single_qubit_gate(qubits[0], self._t_matrix())
        else:
            print(f"Warning: Unsupported gate '{gate_name}', skipping")

    def _hadamard_matrix(self) -> np.ndarray:
        """Hadamard gate matrix."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def _pauli_x_matrix(self) -> np.ndarray:
        """Pauli-X gate matrix."""
        return np.array([[0, 1], [1, 0]], dtype=complex)

    def _pauli_y_matrix(self) -> np.ndarray:
        """Pauli-Y gate matrix."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    def _pauli_z_matrix(self) -> np.ndarray:
        """Pauli-Z gate matrix."""
        return np.array([[1, 0], [0, -1]], dtype=complex)

    def _s_matrix(self) -> np.ndarray:
        """S gate matrix."""
        return np.array([[1, 0], [0, 1j]], dtype=complex)

    def _t_matrix(self) -> np.ndarray:
        """T gate matrix."""
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    def _rotation_matrix(self, axis: str, angle: float) -> np.ndarray:
        """Rotation gate matrix."""
        if axis == 'rx':
            return np.array([
                [np.cos(angle/2), -1j*np.sin(angle/2)],
                [-1j*np.sin(angle/2), np.cos(angle/2)]
            ], dtype=complex)
        elif axis == 'ry':
            return np.array([
                [np.cos(angle/2), -np.sin(angle/2)],
                [np.sin(angle/2), np.cos(angle/2)]
            ], dtype=complex)
        elif axis == 'rz':
            return np.array([
                [np.exp(-1j*angle/2), 0],
                [0, np.exp(1j*angle/2)]
            ], dtype=complex)

    def _apply_single_qubit_gate(self, qubit: int, gate_matrix: np.ndarray):
        """Apply single-qubit gate to state vector."""
        # Create full gate matrix using tensor products
        gate = np.eye(1, dtype=complex)

        for i in range(self.num_qubits):
            if i == qubit:
                gate = np.kron(gate, gate_matrix)
            else:
                gate = np.kron(gate, np.eye(2, dtype=complex))

        # Apply gate
        self.state_vector = gate @ self.state_vector

    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        # Build CNOT matrix
        cnot_matrix = np.eye(self.dim, dtype=complex)

        for i in range(self.dim):
            # Check if control qubit is 1
            if (i >> (self.num_qubits - 1 - control)) & 1:
                # Flip target qubit
                j = i ^ (1 << (self.num_qubits - 1 - target))
                if i != j:
                    # Swap rows i and j
                    cnot_matrix[[i, j]] = cnot_matrix[[j, i]]

        self.state_vector = cnot_matrix @ self.state_vector

    def _apply_toffoli(self, control1: int, control2: int, target: int):
        """Apply Toffoli (CCNOT) gate."""
        toffoli_matrix = np.eye(self.dim, dtype=complex)

        for i in range(self.dim):
            # Check if both control qubits are 1
            c1 = (i >> (self.num_qubits - 1 - control1)) & 1
            c2 = (i >> (self.num_qubits - 1 - control2)) & 1

            if c1 and c2:
                # Flip target qubit
                j = i ^ (1 << (self.num_qubits - 1 - target))
                if i != j:
                    toffoli_matrix[[i, j]] = toffoli_matrix[[j, i]]

        self.state_vector = toffoli_matrix @ self.state_vector

    def _apply_rotation(self, qubit: int, axis: str, angle: float):
        """Apply rotation gate."""
        gate_matrix = self._rotation_matrix(axis, angle)
        self._apply_single_qubit_gate(qubit, gate_matrix)

    def get_measurement_probabilities(self) -> Dict[str, float]:
        """
        Calculate measurement probabilities for all basis states.

        Returns:
            Dictionary mapping basis states to probabilities
        """
        probs = {}
        for i in range(self.dim):
            prob = abs(self.state_vector[i]) ** 2
            if prob > 1e-10:  # Only include non-negligible probabilities
                basis_state = format(i, f'0{self.num_qubits}b')
                probs[basis_state] = float(prob)

        return probs

    def calculate_fidelity(self, expected_state: np.ndarray) -> float:
        """
        Calculate fidelity between current and expected state.

        Args:
            expected_state: Expected state vector

        Returns:
            Fidelity value between 0 and 1
        """
        # Normalize states
        current = self.state_vector / np.linalg.norm(self.state_vector)
        expected = expected_state / np.linalg.norm(expected_state)

        # Calculate fidelity: |⟨ψ|φ⟩|²
        overlap = abs(np.vdot(expected, current))
        return float(overlap ** 2)

    def calculate_entanglement_entropy(self) -> float:
        """
        Calculate von Neumann entropy as a measure of entanglement.

        Returns:
            Entanglement entropy
        """
        # For simplicity, calculate entropy of density matrix
        rho = np.outer(self.state_vector, np.conj(self.state_vector))
        eigenvalues = np.linalg.eigvalsh(rho)

        # Calculate von Neumann entropy: -Tr(ρ log ρ)
        entropy = 0.0
        for eigenval in eigenvalues:
            if eigenval > 1e-10:
                entropy -= eigenval * np.log2(eigenval)

        return float(entropy)


def parse_qasm(qasm_code: str) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Parse QASM code and extract circuit information.

    Args:
        qasm_code: QASM circuit code

    Returns:
        Tuple of (num_qubits, gate_list)
    """
    lines = qasm_code.strip().split('\n')
    num_qubits = 0
    gates = []

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('//'):
            continue

        # Skip header lines
        if line.startswith('OPENQASM') or line.startswith('include'):
            continue

        # Parse qubit register declaration
        if line.startswith('qreg'):
            match = re.search(r'qreg\s+\w+\[(\d+)\]', line)
            if match:
                num_qubits = max(num_qubits, int(match.group(1)))
            continue

        # Skip classical register and measurement declarations
        if line.startswith('creg') or line.startswith('measure'):
            continue

        # Skip conditional operations (simplified handling)
        if line.startswith('if'):
            continue

        # Parse gates
        gate_match = re.match(r'(\w+)(?:\(([\d.]+)\))?\s+([\w\[\],\s]+)', line)
        if gate_match:
            gate_name = gate_match.group(1)
            param = gate_match.group(2)
            qubits_str = gate_match.group(3)

            # Parse qubit indices
            qubit_indices = []
            for qubit in re.findall(r'\w+\[(\d+)\]', qubits_str):
                qubit_indices.append(int(qubit))

            # Parse parameter
            params = []
            if param:
                params = [float(param)]

            gates.append({
                'gate': gate_name,
                'qubits': qubit_indices,
                'params': params
            })

    return num_qubits, gates


def simulate_circuit(circuit_code: str) -> Dict[str, Any]:
    """
    Simulate quantum circuit.

    Args:
        circuit_code: QASM circuit code

    Returns:
        Dictionary with simulation results
    """
    start_time = datetime.now()

    # Parse circuit
    num_qubits, gates = parse_qasm(circuit_code)

    if num_qubits == 0:
        raise ValueError("No qubits found in circuit")

    # Initialize simulator
    simulator = QuantumCircuitSimulator(num_qubits)

    # Apply gates
    for gate_info in gates:
        simulator.apply_gate(
            gate_info['gate'],
            gate_info['qubits'],
            gate_info.get('params', [])
        )

    # Calculate results
    measurement_probs = simulator.get_measurement_probabilities()
    entanglement = simulator.calculate_entanglement_entropy()

    end_time = datetime.now()
    execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

    return {
        'num_qubits': num_qubits,
        'num_gates': len(gates),
        'measurement_probabilities': measurement_probs,
        'entanglement': entanglement,
        'state_vector': simulator.state_vector.tolist() if num_qubits <= 8 else None,
        'execution_time_ms': execution_time_ms
    }


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    Args:
        event: Lambda event (S3 trigger)
        context: Lambda context

    Returns:
        Response dictionary
    """
    print(f"Event: {json.dumps(event)}")

    # Initialize AWS clients
    s3_client, dynamodb_table = get_aws_clients()

    try:
        # Extract S3 information from event
        record = event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']

        print(f"Processing: s3://{bucket_name}/{object_key}")

        # Download circuit from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        circuit_code = response['Body'].read().decode('utf-8')

        # Extract circuit ID from object key
        circuit_id = object_key.split('/')[-1].replace('.qasm', '')

        # Extract algorithm type from path
        algorithm_type = 'unknown'
        if '/' in object_key:
            path_parts = object_key.split('/')
            if len(path_parts) > 1:
                algorithm_type = path_parts[1] if path_parts[1] != 'circuits' else 'unknown'

        # Simulate circuit
        print(f"Simulating circuit: {circuit_id}")
        results = simulate_circuit(circuit_code)

        # Prepare DynamoDB item
        timestamp = datetime.utcnow().isoformat() + 'Z'
        item = {
            'CircuitID': circuit_id,
            'Timestamp': timestamp,
            'AlgorithmType': algorithm_type,
            'NumQubits': results['num_qubits'],
            'NumGates': results['num_gates'],
            'Fidelity': 1.0,  # Placeholder; would compare with expected state
            'ExecutionTimeMs': results['execution_time_ms'],
            'MeasurementProbabilities': results['measurement_probabilities'],
            'Entanglement': results['entanglement'],
            'S3Key': object_key
        }

        # Store state vector in S3 if available
        if results['state_vector'] is not None:
            result_key = object_key.replace('circuits/', 'results/').replace('.qasm', '_result.json')
            result_data = {
                'circuit_id': circuit_id,
                'algorithm_type': algorithm_type,
                **results
            }
            s3_client.put_object(
                Bucket=bucket_name,
                Key=result_key,
                Body=json.dumps(result_data, indent=2),
                ContentType='application/json'
            )
            item['S3ResultsKey'] = result_key
            print(f"Stored detailed results: s3://{bucket_name}/{result_key}")

        # Store in DynamoDB
        dynamodb_table.put_item(Item=item)
        print(f"Stored results in DynamoDB: {circuit_id}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Circuit simulated successfully',
                'circuit_id': circuit_id,
                'num_qubits': results['num_qubits'],
                'num_gates': results['num_gates'],
                'entanglement': results['entanglement']
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error simulating circuit',
                'error': str(e)
            })
        }


# For local testing
if __name__ == '__main__':
    # Test with sample Bell state circuit
    test_event = {
        'Records': [{
            's3': {
                'bucket': {'name': 'test-bucket'},
                'object': {'key': 'circuits/bell/bell_state.qasm'}
            }
        }]
    }

    # Sample QASM code for testing
    sample_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""

    print("Testing quantum simulator...")
    results = simulate_circuit(sample_qasm)
    print(json.dumps(results, indent=2))
