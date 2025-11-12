# Quantum Computing at Scale

**Tier 1 Flagship Project**

Run quantum algorithms on real quantum hardware using AWS Braket.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **VQE:** Variational Quantum Eigensolver for molecular ground states
- **QAOA:** Quantum optimization for combinatorial problems
- **QML:** Quantum machine learning with variational circuits
- **QPE:** Quantum phase estimation
- **Real Hardware:** IonQ (25 qubits), Rigetti (79 qubits), OQC (8 qubits)

## Cost Estimate

**$30-60** for complete set of experiments across multiple algorithms

## Technologies

- **Hardware:** AWS Braket (IonQ Aria, Rigetti Aspen-M-3, OQC Lucy)
- **Frameworks:** PennyLane, Qiskit, Cirq
- **AWS:** Braket, S3, SageMaker, Lambda
- **Applications:** Chemistry (H2, LiH), optimization, ML

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [VQE Tutorial](unified-studio/README.md#1-variational-quantum-eigensolver-vqe)
- [Hardware Comparison](unified-studio/README.md#quantum-hardware-on-aws-braket)
- [CloudFormation Template](unified-studio/cloudformation/quantum-stack.yml)
