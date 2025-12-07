# Quantum Computing Algorithms and Simulation

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** Quantum circuit simulations (in-memory)

## Research Goal

Learn fundamental quantum computing concepts by implementing and simulating quantum algorithms including quantum teleportation, Grover's search, Shor's factoring, and variational quantum eigensolvers (VQE) using Qiskit on classical hardware simulators.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/physics/quantum-computing/tier-0/quantum-algorithms.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/physics/quantum-computing/tier-0/quantum-algorithms.ipynb)

## What You'll Build

1. **Quantum circuit basics** (single/multi-qubit gates, measurement)
2. **Quantum teleportation** (entanglement and state transfer)
3. **Deutsch-Jozsa algorithm** (quantum vs classical advantage, 5 qubits)
4. **Grover's search** (quadratic speedup, 8-qubit simulation, 15-20 min)
5. **Shor's algorithm** (factor 15, 21 using quantum period finding)
6. **VQE for molecules** (H2 ground state energy, 20-30 min)

## Quantum Algorithms

**Implemented Algorithms:**
- **Quantum Teleportation:** Transfer quantum state via entanglement
- **Deutsch-Jozsa:** Exponential speedup for oracle problems
- **Bernstein-Vazirani:** Hidden string recovery in one query
- **Grover's Search:** O(√N) search vs classical O(N)
- **Shor's Factoring:** Exponential speedup for integer factorization
- **VQE:** Variational quantum eigensolver for molecular energies
- **QAOA:** Quantum approximate optimization algorithm

**Simulations:**
- Qiskit Aer statevector simulator
- Qiskit Aer qasm simulator
- Noise models for realistic quantum hardware

## Colab Considerations

This notebook works on Colab but you'll notice:
- **Limited qubits** (12-14 qubits max due to memory)
- **No real quantum hardware** (simulation only, no IBM Quantum access)
- **Slow VQE** (20-30 minute optimization on classical simulator)
- **No error correction** (ideal qubits only)
- **Small problem sizes** (factor 15, not cryptographic 2048-bit RSA)

These limitations prevent research on NISQ-era quantum computers.

## What's Included

- Single Jupyter notebook (`quantum-algorithms.ipynb`)
- Qiskit circuit construction
- Statevector and density matrix simulation
- Algorithm implementations (Grover, Shor, VQE)
- Quantum chemistry integration (PySCF, Qiskit Nature)
- Circuit visualization and analysis
- Performance comparison (quantum vs classical)

## Key Methods

- **Quantum Gates:** Hadamard, CNOT, Toffoli, phase gates
- **Entanglement:** Bell states, GHZ states
- **Quantum Fourier Transform:** Core of Shor's algorithm
- **Amplitude Amplification:** Grover's search mechanism
- **Variational Methods:** VQE, QAOA with classical optimization
- **Quantum Chemistry:** Molecular Hamiltonian simulation

## Simulation Outputs

1. **Circuit Diagrams:** Visual representation of quantum circuits
2. **Measurement Results:** Probability distributions and histograms
3. **Quantum Speedup:** Comparison with classical algorithms
4. **VQE Energies:** Molecular ground state energies (H2, LiH)
5. **Factorization Results:** Shor's algorithm output for small integers
6. **Bloch Sphere Visualizations:** Single-qubit state evolution

## Next Steps

**Need real quantum hardware?** This project uses simulation only:

- **Tier 1:** [Quantum computing with cloud access](../tier-1/) on Studio Lab
  - IBM Quantum Experience access (real quantum processors)
  - 20-qubit simulations with noise models
  - Larger VQE problems (H2O, LiH, BeH2)
  - Persistent quantum job management

- **Tier 2:** [AWS-integrated quantum computing](../tier-2/) with Braket
  - Amazon Braket simulators (SV1: 34 qubits, TN1: 50 qubits)
  - IonQ, Rigetti, OQC quantum hardware access
  - S3 storage for quantum experiment results
  - SageMaker for hybrid quantum-classical algorithms

- **Tier 3:** [Production quantum applications](../tier-3/) with CloudFormation
  - Large-scale VQE for drug discovery molecules
  - QAOA for optimization problems (100+ variables)
  - Quantum machine learning pipelines
  - Automated quantum experiment workflows
  - Integration with classical HPC for hybrid algorithms

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- Qiskit (quantum computing framework)
- Qiskit Aer (simulators)
- Qiskit Nature (quantum chemistry)
- PySCF (classical quantum chemistry)
- numpy, scipy
- matplotlib

**Note:** Simulations use classical hardware - no quantum computer access required
