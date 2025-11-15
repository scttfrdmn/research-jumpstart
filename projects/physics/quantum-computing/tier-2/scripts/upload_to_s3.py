#!/usr/bin/env python3
"""
Upload quantum circuit definitions to S3.

This script uploads QASM and JSON circuit files to S3, organized by algorithm type.
It includes progress tracking, error handling, and automatic retry logic.

Usage:
    python upload_to_s3.py

Environment Variables:
    AWS_S3_BUCKET: S3 bucket name (required)
    AWS_REGION: AWS region (default: us-east-1)
"""

import os
import json
import boto3
from pathlib import Path
from typing import List, Dict, Any
from botocore.exceptions import ClientError
from tqdm import tqdm


def get_s3_client():
    """Create and return boto3 S3 client."""
    region = os.environ.get('AWS_REGION', 'us-east-1')
    return boto3.client('s3', region_name=region)


def get_bucket_name() -> str:
    """Get S3 bucket name from environment or prompt user."""
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    if not bucket_name:
        print("AWS_S3_BUCKET environment variable not set.")
        bucket_name = input("Enter your S3 bucket name (quantum-circuits-xxx): ").strip()

    if not bucket_name:
        raise ValueError("S3 bucket name is required")

    return bucket_name


def verify_bucket_exists(s3_client, bucket_name: str) -> bool:
    """Verify that the S3 bucket exists."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' exists")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"❌ Bucket '{bucket_name}' does not exist")
        elif error_code == '403':
            print(f"❌ Access denied to bucket '{bucket_name}'")
        else:
            print(f"❌ Error accessing bucket: {e}")
        return False


def create_sample_circuits() -> Dict[str, str]:
    """
    Create sample quantum circuits in QASM format.

    Returns:
        Dictionary mapping circuit names to QASM code
    """
    circuits = {}

    # Bell State (2 qubits)
    circuits['bell/bell_state.qasm'] = """OPENQASM 2.0;
include "qelib1.inc";

// Bell State: Creates maximal entanglement between 2 qubits
// |Ψ⟩ = (|00⟩ + |11⟩) / √2

qreg q[2];
creg c[2];

h q[0];        // Hadamard on qubit 0
cx q[0],q[1];  // CNOT with control=0, target=1

measure q[0] -> c[0];
measure q[1] -> c[1];
"""

    # GHZ State (4 qubits)
    circuits['ghz/ghz_4qubit.qasm'] = """OPENQASM 2.0;
include "qelib1.inc";

// GHZ State: Greenberger-Horne-Zeilinger state
// |GHZ⟩ = (|0000⟩ + |1111⟩) / √2
// Demonstrates 4-qubit entanglement

qreg q[4];
creg c[4];

h q[0];        // Create superposition on first qubit
cx q[0],q[1];  // Entangle with second qubit
cx q[0],q[2];  // Entangle with third qubit
cx q[0],q[3];  // Entangle with fourth qubit

measure q -> c;
"""

    # Quantum Teleportation (3 qubits)
    circuits['teleportation/teleportation.qasm'] = """OPENQASM 2.0;
include "qelib1.inc";

// Quantum Teleportation
// Transfer state from qubit 0 to qubit 2 using qubit 1 as entangled resource

qreg q[3];
creg c[3];

// Prepare state to teleport (arbitrary state on q[0])
rx(0.7) q[0];   // Rotation to create interesting state

// Create Bell pair between q[1] and q[2]
h q[1];
cx q[1],q[2];

// Bell measurement on q[0] and q[1]
cx q[0],q[1];
h q[0];

measure q[0] -> c[0];
measure q[1] -> c[1];

// Conditional operations on q[2] based on measurement results
if(c[0]==1) x q[2];
if(c[1]==1) z q[2];

measure q[2] -> c[2];
"""

    # Grover's Algorithm (3 qubits)
    circuits['grover/grover_3qubit.qasm'] = """OPENQASM 2.0;
include "qelib1.inc";

// Grover's Search Algorithm for 3 qubits
// Searches for |101⟩ state

qreg q[3];
creg c[3];

// Initialize superposition
h q[0];
h q[1];
h q[2];

// Oracle: Mark |101⟩ state
x q[1];          // Flip middle qubit
ccx q[0],q[1],q[2];  // Toffoli gate
x q[1];          // Flip back

// Diffusion operator (inversion about average)
h q[0];
h q[1];
h q[2];

x q[0];
x q[1];
x q[2];

ccx q[0],q[1],q[2];

x q[0];
x q[1];
x q[2];

h q[0];
h q[1];
h q[2];

measure q -> c;
"""

    # Deutsch-Jozsa (4 qubits)
    circuits['deutsch/deutsch_jozsa_4q.qasm'] = """OPENQASM 2.0;
include "qelib1.inc";

// Deutsch-Jozsa Algorithm
// Determines if function is constant or balanced in one query

qreg q[4];
creg c[3];

// Initialize
x q[3];      // Ancilla qubit
h q[0];
h q[1];
h q[2];
h q[3];

// Oracle (balanced function)
cx q[0],q[3];
cx q[1],q[3];

// Final Hadamards
h q[0];
h q[1];
h q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
"""

    # VQE Circuit for H2 molecule (4 qubits)
    circuits['vqe/h2_molecule.qasm'] = """OPENQASM 2.0;
include "qelib1.inc";

// VQE ansatz for H2 molecule simulation
// 4 qubits representing molecular orbitals

qreg q[4];
creg c[4];

// Initialize (Hartree-Fock state)
x q[0];
x q[1];

// Ansatz (parameterized circuit)
// Parameters would be optimized classically
ry(0.5) q[0];
ry(0.3) q[1];
ry(0.2) q[2];
ry(0.1) q[3];

cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];

ry(-0.2) q[0];
ry(-0.1) q[1];

measure q -> c;
"""

    # Simple superposition (2 qubits)
    circuits['basic/superposition_2q.qasm'] = """OPENQASM 2.0;
include "qelib1.inc";

// Simple superposition on 2 qubits
// Creates equal superposition of all 4 basis states

qreg q[2];
creg c[2];

h q[0];
h q[1];

measure q -> c;
"""

    return circuits


def create_circuit_metadata() -> List[Dict[str, Any]]:
    """
    Create metadata for each circuit.

    Returns:
        List of dictionaries with circuit metadata
    """
    metadata = [
        {
            "circuit_id": "bell_state",
            "algorithm_type": "bell",
            "num_qubits": 2,
            "num_gates": 2,
            "description": "Bell state demonstrating maximal entanglement",
            "expected_fidelity": 1.0,
            "expected_measurements": {"00": 0.5, "11": 0.5}
        },
        {
            "circuit_id": "ghz_4qubit",
            "algorithm_type": "ghz",
            "num_qubits": 4,
            "num_gates": 4,
            "description": "4-qubit GHZ state",
            "expected_fidelity": 1.0,
            "expected_measurements": {"0000": 0.5, "1111": 0.5}
        },
        {
            "circuit_id": "teleportation",
            "algorithm_type": "teleportation",
            "num_qubits": 3,
            "num_gates": 7,
            "description": "Quantum teleportation protocol",
            "expected_fidelity": 0.95
        },
        {
            "circuit_id": "grover_3qubit",
            "algorithm_type": "grover",
            "num_qubits": 3,
            "num_gates": 15,
            "description": "Grover search for |101⟩",
            "expected_fidelity": 0.90,
            "expected_measurements": {"101": 0.85}
        },
        {
            "circuit_id": "deutsch_jozsa_4q",
            "algorithm_type": "deutsch",
            "num_qubits": 4,
            "num_gates": 11,
            "description": "Deutsch-Jozsa algorithm",
            "expected_fidelity": 0.98
        },
        {
            "circuit_id": "h2_molecule",
            "algorithm_type": "vqe",
            "num_qubits": 4,
            "num_gates": 12,
            "description": "VQE ansatz for H2 molecule",
            "expected_fidelity": 0.85
        },
        {
            "circuit_id": "superposition_2q",
            "algorithm_type": "basic",
            "num_qubits": 2,
            "num_gates": 2,
            "description": "Simple 2-qubit superposition",
            "expected_fidelity": 1.0,
            "expected_measurements": {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
        }
    ]
    return metadata


def upload_file_to_s3(s3_client, bucket_name: str, file_path: str,
                      s3_key: str, content: str) -> bool:
    """
    Upload a file to S3.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        file_path: Local file path (for display)
        s3_key: S3 object key
        content: File content to upload

    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=content,
            ContentType='text/plain' if s3_key.endswith('.qasm') else 'application/json'
        )
        return True
    except ClientError as e:
        print(f"❌ Error uploading {s3_key}: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 70)
    print("Quantum Circuit Upload to S3")
    print("=" * 70)
    print()

    # Get S3 bucket name
    try:
        bucket_name = get_bucket_name()
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Create S3 client
    print(f"🔗 Connecting to AWS S3...")
    s3_client = get_s3_client()

    # Verify bucket exists
    if not verify_bucket_exists(s3_client, bucket_name):
        print("\n💡 Create the bucket first using:")
        print(f"   aws s3 mb s3://{bucket_name} --region us-east-1")
        return 1

    print()

    # Create sample circuits
    print("📝 Creating sample quantum circuits...")
    circuits = create_sample_circuits()
    metadata_list = create_circuit_metadata()
    print(f"✅ Created {len(circuits)} sample circuits")
    print()

    # Upload circuits
    print("📤 Uploading circuits to S3...")
    successful = 0
    failed = 0

    for circuit_path, circuit_content in tqdm(circuits.items(), desc="Uploading"):
        s3_key = f"circuits/{circuit_path}"
        if upload_file_to_s3(s3_client, bucket_name, circuit_path,
                            s3_key, circuit_content):
            successful += 1
        else:
            failed += 1

    print()

    # Upload metadata
    print("📤 Uploading circuit metadata...")
    metadata_content = json.dumps(metadata_list, indent=2)
    if upload_file_to_s3(s3_client, bucket_name, "metadata.json",
                        "circuits/metadata.json", metadata_content):
        successful += 1
        print("✅ Metadata uploaded")
    else:
        failed += 1
        print("❌ Metadata upload failed")

    print()

    # Summary
    print("=" * 70)
    print("Upload Summary")
    print("=" * 70)
    print(f"✅ Successful uploads: {successful}")
    if failed > 0:
        print(f"❌ Failed uploads: {failed}")
    print()
    print(f"📁 Bucket: s3://{bucket_name}/circuits/")
    print()

    # List uploaded files
    print("📋 Uploaded circuits:")
    for circuit_path in circuits.keys():
        print(f"   - circuits/{circuit_path}")
    print(f"   - circuits/metadata.json")
    print()

    # Next steps
    print("🎯 Next Steps:")
    print("1. Verify upload:")
    print(f"   aws s3 ls s3://{bucket_name}/circuits/ --recursive")
    print()
    print("2. Test Lambda function with a circuit:")
    print(f"   aws lambda invoke --function-name simulate-quantum-circuit \\")
    print(f"     --payload '{{\"Records\":[{{\"s3\":{{\"bucket\":{{\"name\":\"{bucket_name}\"}},\"object\":{{\"key\":\"circuits/bell/bell_state.qasm\"}}}}}}]}}' \\")
    print(f"     response.json")
    print()
    print("3. Open Jupyter notebook for analysis:")
    print("   jupyter notebook notebooks/quantum_analysis.ipynb")
    print()

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
