# Quantum Computing at Scale - Tier 1 Flagship

**Duration:** 4-5 days | **Platform:** AWS Unified Studio | **Cost:** $30-60

Production-ready quantum computing research using AWS Braket with hybrid classical-quantum algorithms, variational quantum eigensolvers (VQE), and quantum machine learning.

## Overview

This flagship project demonstrates enterprise-scale quantum computing on AWS Braket, showcasing how to run quantum algorithms on real quantum hardware (IonQ, Rigetti, OQC) and high-performance simulators. Includes integration with SageMaker for hybrid algorithms and Bedrock AI for quantum circuit interpretation.

## What You'll Build

### Infrastructure
- **CloudFormation Stack:** Complete AWS infrastructure for quantum workloads
- **Braket Integration:** Access to quantum hardware and simulators
- **S3 Data Lake:** Quantum experiment results and circuit libraries
- **SageMaker Notebooks:** Hybrid classical-quantum workflows
- **Bedrock AI:** Circuit analysis and algorithm suggestions

### Quantum Algorithms
1. **Variational Quantum Eigensolver (VQE):** Molecular ground state energies
2. **Quantum Approximate Optimization Algorithm (QAOA):** Combinatorial optimization
3. **Quantum Machine Learning:** Variational quantum classifiers
4. **Quantum Phase Estimation:** Eigenvalue problems
5. **Grover's Search:** Unstructured search acceleration

## Quantum Hardware Access

**AWS Braket Devices:**

**Superconducting Qubits:**
- **Rigetti Aspen-M-3:** 79 qubits, gate-based
- **OQC Lucy:** 8 qubits, high connectivity

**Trapped Ion:**
- **IonQ Harmony:** 11 qubits, all-to-all connectivity
- **IonQ Aria:** 25 qubits, #AQ 23

**Simulators:**
- **SV1:** State vector simulator (34 qubits max)
- **TN1:** Tensor network simulator (50+ qubits)
- **DM1:** Density matrix simulator (noise modeling)
- **Local Simulator:** Free, unlimited usage

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AWS Unified Studio                     │
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌────────────────┐    │
│  │ SageMaker│───▶│  Braket  │───▶│ Bedrock Claude │    │
│  │ Notebooks│    │  Quantum │    │   Analysis     │    │
│  └──────────┘    └──────────┘    └────────────────┘    │
│        │              │                    │             │
│        └──────────────┴────────────────────┘             │
│                       │                                   │
│                       ▼                                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │              S3 Results Lake                     │    │
│  │  ┌──────────────┐  ┌──────────────┐            │    │
│  │  │   Quantum    │  │   Classical  │            │    │
│  │  │   Circuits   │  │   Results    │            │    │
│  │  └──────────────┘  └──────────────┘            │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  Quantum Hardware:                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  IonQ    │  │ Rigetti  │  │   OQC    │              │
│  │  Aria    │  │ Aspen-M  │  │  Lucy    │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

## Features

### 1. Variational Quantum Eigensolver (VQE)

Compute molecular ground state energies using hybrid quantum-classical optimization.

```python
from src.vqe import run_vqe
from braket.devices import LocalSimulator

# Define molecule (H2 dissociation curve)
molecule = 'H2'
distances = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

# Run VQE on simulator
device = LocalSimulator()
results = run_vqe(
    molecule=molecule,
    bond_distances=distances,
    ansatz='hardware_efficient',
    optimizer='COBYLA',
    device=device
)

# Plot dissociation curve
plot_dissociation_curve(results)
```

**Expected Results:**
- Ground state energy curve for H₂
- Comparison with exact diagonalization
- Convergence analysis
- Hardware vs simulator accuracy

### 2. Quantum Approximate Optimization Algorithm (QAOA)

Solve MaxCut problem on graphs using QAOA.

```python
from src.qaoa import run_qaoa
import networkx as nx

# Generate random graph
G = nx.erdos_renyi_graph(n=8, p=0.5, seed=42)

# Run QAOA
device = 'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony'
result = run_qaoa(
    graph=G,
    p=3,  # 3 QAOA layers
    device=device,
    shots=1000
)

# Visualize solution
plot_maxcut(G, result['cut'])
```

**Applications:**
- Graph partitioning
- Network optimization
- Clustering problems
- Supply chain logistics

### 3. Quantum Machine Learning

Train variational quantum classifier on real datasets.

```python
from src.qml import QuantumClassifier
from sklearn.datasets import make_moons

# Generate dataset
X_train, y_train = make_moons(n_samples=200, noise=0.1)

# Train quantum classifier
qc = QuantumClassifier(
    n_qubits=4,
    n_layers=3,
    device='arn:aws:braket:::device/quantum-simulator/amazon/sv1'
)

qc.fit(X_train, y_train, epochs=50)

# Evaluate
accuracy = qc.score(X_test, y_test)
print(f"Quantum classifier accuracy: {accuracy:.2%}")
```

**Features:**
- Data encoding circuits (amplitude/angle encoding)
- Parameterized quantum circuits (PQC)
- Gradient computation via parameter shift rule
- Hybrid optimization (quantum + classical)

### 4. Quantum Phase Estimation (QPE)

Estimate eigenvalues of unitary operators.

```python
from src.qpe import quantum_phase_estimation

# Phase estimation for simple unitary
unitary = create_rotation_gate(theta=np.pi/3)

# Run QPE
estimated_phase = quantum_phase_estimation(
    unitary=unitary,
    precision_bits=8,
    device=LocalSimulator()
)

print(f"True phase: {np.pi/3:.6f}")
print(f"Estimated phase: {estimated_phase:.6f}")
```

**Applications:**
- Shor's algorithm (factoring)
- Quantum simulation
- Chemistry (energy estimation)

### 5. Grover's Search

Accelerate unstructured search with quadratic speedup.

```python
from src.grover import grovers_search

# Search for marked item in database
database_size = 16  # 2^4 items
marked_items = [7, 11]  # Items to find

# Run Grover's algorithm
result = grovers_search(
    n_qubits=4,
    marked_items=marked_items,
    device=LocalSimulator()
)

print(f"Found items: {result['measured_states']}")
print(f"Success probability: {result['success_prob']:.2%}")
```

**Speedup:** O(√N) vs O(N) classical

## CloudFormation Stack

### Resources Created

```yaml
Resources:
  # S3 Buckets
  - QuantumResultsBucket: Circuit outputs and experiment data
  - CircuitLibraryBucket: Reusable quantum circuits

  # IAM Roles
  - BraketExecutionRole: Permissions for quantum devices
  - SageMakerRole: Notebook execution

  # SageMaker
  - NotebookInstance: ml.t3.xlarge for hybrid algorithms
  - TrainingJobs: For quantum ML models

  # Braket Resources
  - QuantumTaskQueue: Job management
  - HybridJobDefinition: Classical-quantum workflows

  # Bedrock
  - ClaudeModelAccess: Circuit analysis and interpretation
```

### Deployment

```bash
# Deploy quantum computing stack
aws cloudformation create-stack \
  --stack-name quantum-computing \
  --template-body file://cloudformation/quantum-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_IAM

# Wait for completion (3-5 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name quantum-computing

# Get outputs
aws cloudformation describe-stacks \
  --stack-name quantum-computing \
  --query 'Stacks[0].Outputs'
```

## Project Structure

```
quantum-computing/unified-studio/
├── README.md                      # This file
├── requirements.txt               # Python + quantum dependencies
├── setup.py                       # Package installation
│
├── cloudformation/
│   ├── quantum-stack.yml         # Main CFN template
│   └── parameters.json           # Stack parameters
│
├── src/
│   ├── __init__.py
│   ├── vqe.py                    # Variational Quantum Eigensolver
│   ├── qaoa.py                   # Quantum Approximate Optimization
│   ├── qml.py                    # Quantum Machine Learning
│   ├── qpe.py                    # Quantum Phase Estimation
│   ├── grover.py                 # Grover's search algorithm
│   ├── circuits.py               # Circuit building utilities
│   ├── optimization.py           # Classical optimizers
│   ├── visualization.py          # Quantum state/circuit plots
│   └── bedrock_client.py         # AI circuit interpretation
│
├── notebooks/
│   ├── 01_braket_introduction.ipynb
│   ├── 02_vqe_chemistry.ipynb
│   ├── 03_qaoa_optimization.ipynb
│   ├── 04_quantum_ml.ipynb
│   ├── 05_advanced_algorithms.ipynb
│   └── 06_hardware_experiments.ipynb
│
└── tests/
    ├── test_vqe.py
    ├── test_qaoa.py
    └── test_qml.py
```

## Key Experiments

### 1. H₂ Dissociation Curve (VQE)

**Objective:** Compute ground state energy of H₂ molecule at different bond distances

**Method:**
1. Map molecule to qubit Hamiltonian (Jordan-Wigner)
2. Prepare variational ansatz (UCCSD or hardware-efficient)
3. Optimize parameters to minimize energy expectation
4. Compare with classical exact diagonalization

**Circuit Depth:** 20-40 gates
**Qubits Required:** 4
**Shots per Distance:** 10,000
**Cost:** ~$10 on IonQ Harmony

**Expected Result:**
- Accurate ground state at equilibrium (0.74 Å)
- Dissociation limit matches theory
- Chemical accuracy (< 1.6 kcal/mol)

### 2. MaxCut on 10-Node Graph (QAOA)

**Objective:** Find maximum cut of random graph

**Method:**
1. Encode graph into cost Hamiltonian
2. Apply QAOA circuit with p=3 layers
3. Measure computational basis
4. Post-process to find best cut

**Circuit Depth:** ~60 gates
**Qubits Required:** 10
**Shots:** 5,000
**Cost:** ~$20 on IonQ Aria

**Expected Result:**
- 85-95% approximation ratio to optimal cut
- Better performance than classical greedy algorithms
- Visualization of cut distribution

### 3. Binary Classification (Quantum ML)

**Objective:** Train quantum classifier on moons dataset

**Method:**
1. Encode data into quantum state (amplitude encoding)
2. Apply trainable quantum circuit (3 layers)
3. Measure in computational basis
4. Optimize parameters via gradient descent

**Circuit Depth:** 15-25 gates
**Qubits Required:** 4
**Training Iterations:** 100
**Cost:** ~$30 on IonQ (or free on simulator)

**Expected Result:**
- Test accuracy: 90-95%
- Comparable to classical neural network
- Visualization of decision boundary

## Quantum Hardware Comparison

| Device | Qubits | Connectivity | Gate Fidelity | Cost per Task |
|--------|--------|--------------|---------------|---------------|
| IonQ Harmony | 11 | All-to-all | ~99.5% (1-qubit) | $0.30/task |
| IonQ Aria | 25 | All-to-all | ~99.9% (1-qubit) | $0.97/task |
| Rigetti Aspen-M-3 | 79 | Limited | ~99% (1-qubit) | $0.35/task |
| OQC Lucy | 8 | High | ~99% (1-qubit) | $0.30/task |
| SV1 Simulator | 34 | All-to-all | Exact | $0.075/min |

**Recommendation:** Start with SV1 simulator (free for development), then move to IonQ Harmony for hardware validation.

## AI-Powered Circuit Analysis

**Bedrock Claude Integration:**

```python
from src.bedrock_client import analyze_circuit

# Get AI analysis of quantum circuit
circuit = create_vqe_circuit(molecule='H2', distance=0.74)

analysis = analyze_circuit(
    circuit=circuit,
    context="VQE circuit for H2 molecule at equilibrium distance",
    metrics={'depth': 35, 'gate_count': 42, 'qubits': 4}
)

print(analysis)
# "This circuit implements a hardware-efficient ansatz suitable for NISQ devices.
#  The depth of 35 gates is reasonable for current ion trap systems which have
#  T2 times ~100μs. Consider reducing entangling gates for superconducting qubits.
#  The rotation gates at the end prepare measurement in the computational basis..."
```

**Use Cases:**
- Circuit optimization suggestions
- Hardware suitability analysis
- Algorithm explanation
- Error mitigation strategies
- Research direction recommendations

## Cost Optimization

### 1. Use Simulators for Development

```python
# Development (free)
device = LocalSimulator()

# Testing ($0.075/min)
device = 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'

# Production (real hardware)
device = 'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony'
```

### 2. Batch Jobs

```python
from braket.aws import AwsDevice

# Submit multiple circuits in single job
device = AwsDevice('arn:aws:braket:us-east-1::device/qpu/ionq/Harmony')

batch_job = device.run_batch(
    circuits=circuit_list,
    shots=1000,
    max_parallel=5
)

# Saves on task submission overhead
```

### 3. Hybrid Jobs for Long Runs

```python
from braket.jobs import hybrid_job

@hybrid_job(device='arn:aws:braket:::device/quantum-simulator/amazon/sv1')
def quantum_experiment():
    # Long-running experiment with multiple iterations
    for epoch in range(100):
        circuit = build_circuit(params)
        result = device.run(circuit, shots=1000).result()
        params = optimize(result)

    return params

# Runs as managed job, survives interruptions
job = quantum_experiment()
```

## Cost Breakdown

### One-Time Setup
- **CloudFormation Deployment:** Free
- **S3 Buckets:** $0.023/GB/month (< $1/month typical)

### Development (Free/Low Cost)
- **Local Simulator:** Free, unlimited
- **SV1 Simulator:** $0.075/min → ~$5 for typical development

### Production Experiments
- **VQE on IonQ Harmony:** 100 evaluations × $0.30 = $30
- **QAOA on IonQ Aria:** 50 runs × $0.97 = $48.50
- **QML Training:** Simulator only ≈ $10

### Total Estimated Cost
- **Development:** $5-10
- **Production Experiments:** $30-60
- **Monthly Maintenance:** < $5

**Total for Complete Project:** $40-75

## Scientific Applications

### 1. Quantum Chemistry

**Problem:** Calculate molecular properties for drug discovery

**Approach:**
- VQE for ground state energies
- Compare different ansätze (UCCSD vs hardware-efficient)
- Error mitigation techniques

**Molecules:**
- H₂ (hydrogen, 2 electrons → 4 qubits)
- LiH (lithium hydride, 4 electrons → 8 qubits)
- H₂O (water, 10 electrons → 20 qubits, requires simulator)

### 2. Optimization

**Problem:** Supply chain routing, portfolio optimization

**Approach:**
- QAOA for combinatorial problems
- Compare p=1, 2, 3 layers
- Benchmark against classical algorithms

**Problems:**
- MaxCut (graph partitioning)
- Traveling Salesman Problem (TSP)
- Vehicle Routing Problem (VRP)

### 3. Machine Learning

**Problem:** Classification on small datasets where quantum advantage possible

**Approach:**
- Quantum kernel methods
- Variational quantum classifiers
- Quantum feature maps

**Datasets:**
- Iris (classic ML benchmark)
- MNIST (small subset)
- Synthetic XOR/moons

## Extensions

### 1. Error Mitigation
```python
from braket.error_mitigation import zero_noise_extrapolation

# Extrapolate to zero noise
mitigated_result = zero_noise_extrapolation(
    circuit=circuit,
    noise_levels=[1.0, 1.5, 2.0],
    device=device
)
```

### 2. Quantum Simulation
```python
# Time evolution of quantum system
from src.simulation import trotterize

hamiltonian = create_ising_hamiltonian(n=8, J=1.0, h=0.5)
circuit = trotterize(hamiltonian, time=1.0, steps=10)
```

### 3. Advanced Algorithms
- **Quantum Fourier Transform:** Fast Fourier analysis
- **HHL Algorithm:** Solving linear systems
- **Quantum Annealing:** Via D-Wave integration

### 4. Real-Time Monitoring
```python
# Track quantum jobs
from src.monitoring import watch_quantum_job

watch_quantum_job(
    job_arn='arn:aws:braket:...',
    slack_webhook=os.environ['SLACK_WEBHOOK']
)
```

## Scientific References

1. **Peruzzo et al.** (2014). "A variational eigenvalue solver on a photonic quantum processor." *Nature Communications* 5: 4213.

2. **Farhi et al.** (2014). "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028.

3. **McClean et al.** (2016). "The theory of variational hybrid quantum-classical algorithms." *New Journal of Physics* 18: 023023.

4. **Havlíček et al.** (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature* 567: 209-212.

5. **Kandala et al.** (2017). "Hardware-efficient variational quantum eigensolver for small molecules." *Nature* 549: 242-246.

## Troubleshooting

### Issue: Circuit too deep for hardware

**Solution:** Use circuit optimization

```python
from braket.circuits import Circuit
from braket.transpiler import transpile

# Optimize circuit for target device
optimized = transpile(circuit, target_device='ionq')
print(f"Original depth: {circuit.depth}")
print(f"Optimized depth: {optimized.depth}")
```

### Issue: Quantum tasks timing out

**Solution:** Use hybrid jobs for long runs

```python
# Instead of running many tasks sequentially
for i in range(100):
    result = device.run(circuit, shots=1000).result()  # ❌ Slow

# Use hybrid job
@hybrid_job(device=device)
def batch_experiment():
    for i in range(100):
        result = device.run(circuit, shots=1000).result()  # ✅ Fast
    return results
```

### Issue: Poor convergence in VQE

**Solutions:**
1. Try different optimizers (COBYLA, SPSA, Adam)
2. Increase shots for better gradient estimates
3. Use simulator to debug before hardware
4. Check if ansatz expresses ground state

## Support

- **AWS Braket Documentation:** https://docs.aws.amazon.com/braket/
- **Braket Examples:** https://github.com/aws/amazon-braket-examples
- **Quantum Computing Support:** AWS Braket Console
- **Scientific Questions:** arXiv quantum-ph section

## License

This project is provided as educational material for AWS Research Jumpstart.

---

**Ready to run quantum algorithms at scale?**

Deploy the CloudFormation stack and start experimenting with real quantum hardware!
