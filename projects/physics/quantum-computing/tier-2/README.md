# Quantum Circuit Simulation with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $8-14 | **Platform:** AWS + Local machine

Simulate quantum circuits using serverless AWS services. Upload quantum circuit definitions to S3, process them with Lambda using lightweight simulation, and store results in DynamoDB for analysis—all without managing servers.

---

## What You'll Build

A cloud-native quantum computing simulation pipeline that demonstrates:

1. **Circuit Storage** - Upload quantum circuit definitions (QASM, JSON) to S3
2. **Serverless Simulation** - Lambda functions to simulate circuits in parallel
3. **Results Storage** - Store quantum states and measurements in DynamoDB
4. **Circuit Analysis** - Query and compare quantum algorithms using Athena
5. **Visualization** - Analyze circuit performance and fidelity in Jupyter

This bridges the gap between local quantum simulation (Tier 1) and production quantum infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter                                   │ │
│  │ • upload_to_s3.py - Upload quantum circuits              │ │
│  │ • lambda_function.py - Simulate circuits                 │ │
│  │ • query_results.py - Analyze simulation results          │ │
│  │ • quantum_analysis.ipynb - Circuit comparison            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB Table  │
│  │                  │  │                  │  │                  │
│  │ • Circuit QASM   │→ │ Quantum          │→ │ CircuitID        │
│  │ • Circuit JSON   │  │ simulation       │  │ AlgorithmType    │
│  │ • Results        │  │ (numpy-based)    │  │ NumQubits        │
│  └──────────────────┘  └──────────────────┘  │ Fidelity         │
│  ┌──────────────────┐                        │ ExecutionTime    │
│  │  Athena (SQL)    │                        └──────────────────┘
│  │                  │                        ┌──────────────────┐
│  │ Query circuit    │←─────────────────────→ │  IAM Role        │
│  │ performance      │                        │  (Permissions)   │
│  └──────────────────┘                        └──────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (numpy, boto3)
- Quantum computing fundamentals (qubits, gates, measurements)
- AWS basics (S3, Lambda, IAM, DynamoDB)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - numpy (quantum state vectors)
  - pandas (data manipulation)
  - matplotlib (visualization)
  - jupyter (analysis notebooks)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB table creation
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/physics/quantum-computing/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start (15 minutes)

### Step 1: Set Up AWS (10 minutes)
```bash
# Follow setup_guide.md for detailed instructions
# Creates:
# - S3 bucket: quantum-circuits-{your-id}
# - IAM role: lambda-quantum-simulator
# - Lambda function: simulate-quantum-circuit
# - DynamoDB table: QuantumResults
```

### Step 2: Upload Sample Circuits (2 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Simulate Circuits (2 minutes)
```bash
# Lambda processes circuits automatically
# Or trigger manually for testing
```

### Step 4: Query Results (1 minute)
```bash
python scripts/query_results.py
```

### Step 5: Visualize (5 minutes)
Open `notebooks/quantum_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Circuit Definition and Upload

**What's happening:**
- Create quantum circuits using standard formats (QASM, JSON)
- Define various algorithms: Bell states, GHZ, Grover, VQE, etc.
- Upload circuits to S3 organized by algorithm type
- Each circuit includes metadata: qubits, gates, expected output

**Files involved:**
- `setup_guide.md` - AWS setup instructions
- `scripts/upload_to_s3.py` - Batch upload automation

**Supported Circuit Types:**
- **Bell States** - Entanglement demonstrations (2 qubits)
- **GHZ States** - Multi-qubit entanglement (3-8 qubits)
- **Quantum Teleportation** - Information transfer (3 qubits)
- **Grover's Algorithm** - Quantum search (4-8 qubits)
- **Deutsch-Jozsa** - Oracle problems (4-6 qubits)
- **VQE Circuits** - Variational algorithms (2-6 qubits)

**Time:** 5-10 minutes

### 2. Lambda Simulation

**What's happening:**
- Lambda reads circuit definition from S3
- Parses QASM or JSON format
- Simulates quantum evolution using numpy
- Calculates state vector, measurement probabilities
- Computes fidelity and circuit metrics
- Stores results in DynamoDB and S3

**Lambda function flow:**
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'quantum-circuits-xxxx'},
        'object': {'key': 'circuits/grover/grover_3qubit.qasm'}
    }]
}

# Processing:
# 1. Parse circuit definition
# 2. Initialize quantum state |000...0⟩
# 3. Apply gates sequentially
# 4. Calculate final state vector
# 5. Compute measurement probabilities
# 6. Calculate fidelity vs. expected outcome

# Output stored in:
# - DynamoDB: CircuitID, metadata, metrics
# - S3: Full state vector (if < 10 qubits)
```

**Files involved:**
- `scripts/lambda_function.py` - Quantum simulation code
- `setup_guide.md` - Lambda deployment steps

**Simulation Capabilities:**
- **Qubits:** Up to 10 qubits (1024-dimensional state vector)
- **Gates:** H, X, Y, Z, CNOT, Toffoli, Phase, Rotation
- **Measurements:** Computational basis, probability distributions
- **Metrics:** Fidelity, purity, entanglement entropy

**Time:** 5-30 seconds per circuit (depends on qubits)

### 3. Results Storage

**What's happening:**
- Simulation results stored in DynamoDB for fast queries
- Full state vectors stored in S3 for detailed analysis
- Circuit metrics computed: fidelity, execution time, gate count
- Organized structure enables algorithm comparison

**DynamoDB Schema:**
```
QuantumResults Table:
├── CircuitID (Partition Key): "grover_3qubit_001"
├── Timestamp (Sort Key): "2025-01-14T10:30:00Z"
├── AlgorithmType: "grover"
├── NumQubits: 3
├── NumGates: 15
├── Fidelity: 0.987
├── ExecutionTimeMs: 245
├── MeasurementProbabilities: {|000⟩: 0.02, |101⟩: 0.96, ...}
├── Entanglement: 0.85
└── S3ResultsKey: "results/grover_3qubit_001.json"
```

**S3 Structure:**
```
s3://quantum-circuits-{your-id}/
├── circuits/                      # Input circuit definitions
│   ├── bell/
│   │   ├── bell_state.qasm
│   │   └── bell_state.json
│   ├── grover/
│   │   ├── grover_3qubit.qasm
│   │   └── grover_4qubit.qasm
│   ├── vqe/
│   │   └── h2_molecule.qasm
│   └── ghz/
│       └── ghz_5qubit.qasm
├── results/                       # Simulation outputs
│   ├── bell_state_result.json
│   ├── grover_3qubit_result.json
│   └── ...
└── logs/                          # Lambda execution logs
    └── simulation_log.txt
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for circuits by algorithm, qubits, fidelity
- Download detailed results from S3
- Compare algorithm performance
- Visualize quantum states and measurement distributions
- Generate publication-quality figures

**Files involved:**
- `notebooks/quantum_analysis.ipynb` - Main analysis notebook
- `scripts/query_results.py` - DynamoDB query utilities
- (Optional) Athena for SQL-based circuit queries

**Analysis Types:**
1. **Algorithm Comparison** - Compare Grover vs. brute force
2. **Scaling Analysis** - How fidelity degrades with qubits
3. **Gate Efficiency** - Gate count vs. circuit depth
4. **Entanglement Analysis** - Measure quantum correlations
5. **Error Analysis** - Simulation accuracy vs. theoretical

**Time:** 30-60 minutes analysis

---

## Project Files

```
tier-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # AWS setup instructions
├── cleanup_guide.md                   # Resource deletion guide
│
├── notebooks/
│   └── quantum_analysis.ipynb        # Main analysis notebook
│
├── scripts/
│   ├── upload_to_s3.py               # Upload circuits to S3
│   ├── lambda_function.py            # Lambda simulation function
│   ├── query_results.py              # Query DynamoDB results
│   └── __init__.py
│
└── sample_circuits/
    ├── bell_state.qasm               # Example Bell state
    ├── grover_3qubit.qasm            # Grover's algorithm
    ├── ghz_4qubit.qasm               # GHZ state
    └── vqe_h2.qasm                   # VQE for H2 molecule
```

---

## Cost Breakdown

**Total estimated cost: $8-14 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 500MB × 7 days | $0.08 |
| **S3 Requests** | ~500 PUT/GET requests | $0.03 |
| **Lambda Executions** | 200 invocations × 30 sec | $3.00 |
| **Lambda Compute** | 200 GB-seconds @ 1024MB | $3.30 |
| **DynamoDB Storage** | 100MB stored | $0.25 |
| **DynamoDB Read/Write** | 1000 requests | $1.25 |
| **Athena Queries** | 5 queries × 500MB scanned | $0.01 |
| **Data Transfer** | Download results (100MB) | $0.01 |
| **Total** | | **$7.93** |

**Cost optimization tips:**
1. Limit simulation to ≤10 qubits ($3.00 savings per extra qubit)
2. Delete S3 objects after analysis ($0.08 savings)
3. Use DynamoDB on-demand pricing (pay per request)
4. Set Lambda timeout to 2 minutes max
5. Batch circuit simulations together

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **DynamoDB**: 25GB storage free (always free)
- **Athena**: First 1TB scanned free (per month)

---

## Key Learning Objectives

### AWS Services
- ✅ S3 bucket creation and organization
- ✅ Lambda function deployment with numpy
- ✅ DynamoDB table design for NoSQL queries
- ✅ IAM role creation with least privilege
- ✅ CloudWatch monitoring and logs
- ✅ (Optional) Athena for serverless SQL

### Cloud Concepts
- ✅ Serverless architecture patterns
- ✅ Event-driven computing
- ✅ NoSQL database design
- ✅ Cost-conscious architecture
- ✅ Parallel computation strategies

### Quantum Computing Skills
- ✅ Quantum circuit simulation
- ✅ State vector calculations
- ✅ Measurement probability distributions
- ✅ Quantum algorithm analysis
- ✅ Fidelity and error metrics

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create DynamoDB table: 3 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5 minutes
- Configure triggers: 3 minutes
- **Subtotal setup: 28 minutes**

**Circuit Preparation:**
- Create/download circuits: 10 minutes
- Upload to S3: 5 minutes
- **Subtotal preparation: 15 minutes**

**Simulation:**
- Lambda processing: 10-20 minutes (200 circuits)
- Monitor execution: 5 minutes
- **Subtotal simulation: 15-25 minutes**

**Analysis:**
- Query results: 5 minutes
- Jupyter analysis: 40-60 minutes
- Generate visualizations: 15 minutes
- **Subtotal analysis: 60-80 minutes**

**Total time: 2-2.5 hours** (including setup)

---

## AWS Account Setup

### Prerequisites
1. Create AWS account: https://aws.amazon.com/
2. (Optional) Activate free tier
3. Create IAM user for programmatic access

### Local Setup
```bash
# Install Python 3.8+ (if needed)
python --version

# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter: Access Key ID
# Enter: Secret Access Key
# Enter: Region (us-east-1 recommended)
# Enter: Output format (json)
```

### Sample Data
- Sample quantum circuits provided in `sample_circuits/`
- Or create your own using Qiskit, Cirq, or OpenQASM

---

## Running the Project

### Option 1: Automated (Recommended)
```bash
# Step 1: Setup AWS services (follow prompts)
# See setup_guide.md for detailed instructions

# Step 2: Upload circuits
python scripts/upload_to_s3.py

# Step 3: Lambda processes automatically
# (Or trigger manually for testing)

# Step 4: Analyze results
jupyter notebook notebooks/quantum_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket
aws s3 mb s3://quantum-circuits-$(date +%s) --region us-east-1

# 2. Upload circuits
aws s3 cp sample_circuits/ s3://quantum-circuits-xxxx/circuits/ --recursive

# 3. Deploy Lambda (see setup_guide.md)

# 4. Create DynamoDB table (see setup_guide.md)

# 5. Run analysis notebook
jupyter notebook notebooks/quantum_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook (`quantum_analysis.ipynb`) includes:

### 1. Circuit Creation
- Generate Bell states, GHZ states, quantum teleportation
- Create Grover's algorithm circuits
- Define VQE ansatzes for molecules
- Export to QASM and JSON formats

### 2. Upload and Simulate
- Batch upload circuits to S3
- Trigger Lambda simulations
- Monitor execution progress
- Handle errors and retries

### 3. Results Analysis
- Query DynamoDB for circuit metadata
- Download state vectors from S3
- Calculate success probabilities
- Compare algorithm performance

### 4. Visualization
- Plot measurement probability distributions
- Visualize quantum state vectors (Bloch sphere, bar charts)
- Compare fidelity across algorithms
- Analyze scaling with qubit count
- Display circuit diagrams

### 5. Algorithm Comparison
- **Bell State**: Verify maximal entanglement
- **GHZ State**: Multi-qubit entanglement scaling
- **Grover**: Quadratic speedup demonstration
- **VQE**: Ground state energy estimation
- **Teleportation**: Quantum information transfer

---

## What You'll Discover

### Quantum Computing Insights
- How quantum algorithms achieve speedups
- Entanglement generation and measurement
- Quantum state evolution under gates
- Measurement collapse and probabilities
- Quantum algorithm benchmarking

### AWS Insights
- Serverless quantum simulation at scale
- Cost-effective cloud quantum research
- Parallel circuit evaluation
- NoSQL for quantum experiment data
- Event-driven quantum workflows

### Research Applications
- Algorithm development and testing
- Quantum error analysis
- Circuit optimization strategies
- Quantum chemistry simulations (VQE, QAOA)
- Quantum machine learning prototyping

---

## Next Steps

### Extend This Project
1. **More Algorithms**: Add Shor's algorithm, QAOA, quantum error correction
2. **Noise Models**: Simulate realistic quantum hardware noise
3. **Larger Circuits**: Use sparse matrix methods for 15+ qubits
4. **Optimization**: Implement circuit compilation and optimization
5. **Visualization**: Create interactive quantum state visualizations
6. **Real Hardware**: Interface with IBM Quantum or AWS Braket

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- AWS Braket integration for real quantum hardware
- Step Functions for complex quantum workflows
- Advanced monitoring and cost optimization
- Multi-region deployment for global access

---

## Troubleshooting

### Common Issues

**"botocore.exceptions.NoCredentialsError"**
```bash
# Solution: Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Key, region, output format
```

**"S3 bucket already exists"**
```bash
# Solution: Use a unique bucket name
s3://quantum-circuits-$(date +%s)-yourname
```

**"Lambda timeout" for large circuits**
```python
# Solution: Increase timeout in Lambda console
# Default: 30 seconds
# Recommended: 120 seconds (2 minutes)
# For 10+ qubit circuits: 300 seconds (5 minutes)
```

**"Lambda out of memory"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# For 8 qubits: 512 MB
# For 10 qubits: 1024 MB (1GB)
# For 12 qubits: 3008 MB (3GB)
# Note: State vector size = 2^n complex numbers
```

**"Circuit simulation error"**
```python
# Common causes:
# 1. Invalid QASM syntax - verify with parser
# 2. Unsupported gates - check Lambda implementation
# 3. Too many qubits - limit to 10 for standard Lambda
# Solution: Check CloudWatch logs for error details
```

**"DynamoDB query returns no results"**
```bash
# Solution: Verify data was written
aws dynamodb scan --table-name QuantumResults --limit 10

# Check Lambda CloudWatch logs for errors
aws logs tail /aws/lambda/simulate-quantum-circuit --follow
```

**"High AWS costs"**
```bash
# Solution: Check Lambda memory allocation
# Reduce to minimum needed: 256MB for ≤6 qubits

# Delete S3 objects after analysis
aws s3 rm s3://quantum-circuits-xxxx --recursive

# Use on-demand DynamoDB pricing
# Set billing alerts at $15 threshold
```

See `troubleshooting.md` for detailed solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://quantum-circuits-xxxx --recursive
aws s3 rb s3://quantum-circuits-xxxx

# Delete Lambda function
aws lambda delete-function --function-name simulate-quantum-circuit

# Delete DynamoDB table
aws dynamodb delete-table --table-name QuantumResults

# Delete IAM role
aws iam detach-role-policy --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy --role-name lambda-quantum-simulator \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam delete-role --role-name lambda-quantum-simulator
```

See `cleanup_guide.md` for detailed step-by-step instructions.

---

## Resources

### Quantum Computing
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [OpenQASM Specification](https://github.com/openqasm/openqasm)
- [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/)
- [Nielsen & Chuang Textbook](http://mmrc.amss.cas.cn/tlb/201702/W020170224608149940643.pdf)

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/latest/developerguide/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS Braket](https://aws.amazon.com/braket/) - Real quantum hardware

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `physics`, `quantum-computing`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `quantum-computing`

### Quantum Computing Help
- Qiskit Slack: https://qiskit.slack.com/
- Quantum Computing Stack Exchange: https://quantumcomputing.stackexchange.com/

---

## Cost Tracking

### Monitor Your Spending

```bash
# Check current AWS charges
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost"

# Set up billing alerts in AWS console:
# https://docs.aws.amazon.com/billing/latest/userguide/budgets-create.html
```

Recommended alerts:
- $10 threshold (warning)
- $20 threshold (warning)
- $30 threshold (critical)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $8-14 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable Lambda (pay per second) |
| **Qubits** | Limited by local RAM | Up to 10 qubits (12 with optimization) |
| **Parallelization** | Single notebook | Multiple Lambda functions |
| **Persistence** | Session-based | Permanent S3/DynamoDB storage |
| **Collaboration** | Limited | Full team access with IAM |
| **Results** | Local only | Queryable database |

---

## What's Next?

After completing this project:

**Skill Building**
- Advanced quantum algorithms (Shor, QAOA, VQE)
- Quantum error correction codes
- Lambda optimization for large circuits
- NoSQL database design patterns
- Cost optimization techniques

**Project Extensions**
- Real-time quantum circuit optimization
- Automated benchmarking pipelines
- Integration with quantum hardware (AWS Braket)
- Quantum machine learning workflows
- Multi-cloud quantum computing

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- AWS Braket integration for real quantum devices
- High-performance computing with AWS Batch
- Advanced monitoring and alerting
- Multi-region quantum workflows

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_quantum_tier2,
  title = {Quantum Circuit Simulation with S3 and Lambda: Tier 2},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

---

## License

Apache License 2.0 - See [LICENSE](../../../../LICENSE) for details.

---

**Ready to start?** Follow the [setup_guide.md](setup_guide.md) to get started!

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
