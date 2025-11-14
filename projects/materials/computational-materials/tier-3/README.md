# Computational Materials Science at Scale

**Tier 1 Flagship Project**

Accelerate materials discovery with density functional theory (DFT), high-throughput screening, and machine learning on AWS HPC infrastructure.

## Overview

This project demonstrates large-scale computational materials science on AWS, enabling researchers to:

- Run DFT calculations for electronic structure and materials properties
- Screen thousands of materials candidates using high-throughput workflows
- Predict materials properties with machine learning models
- Access massive materials databases (Materials Project, OQMD, AFLOW)
- Deploy HPC clusters optimized for quantum chemistry codes

Materials science is being revolutionized by computational approaches. AWS ParallelCluster and EC2 HPC instances enable researchers to run simulations that would take months on traditional hardware in days or hours. Combined with ML models trained on millions of known materials, we can discover new materials for batteries, catalysts, solar cells, and superconductors.

**Key Capabilities:**
- **DFT Calculations:** Quantum ESPRESSO, VASP, GPAW, ABINIT
- **Force Fields:** LAMMPS molecular dynamics, classical potentials
- **ML Models:** Crystal graph neural networks, composition-based prediction
- **Databases:** Materials Project (150K+ materials), OQMD (1M+), AFLOW (3.5M+)
- **HPC:** AWS ParallelCluster with EFA networking, FSx for Lustre
- **Workflows:** AiiDA, FireWorks, Jobflow for automation

## Table of Contents

- [Features](#features)
- [Cost Estimates](#cost-estimates)
- [Getting Started](#getting-started)
- [Applications](#applications)
  - [1. DFT Band Structure Calculations](#1-dft-band-structure-calculations)
  - [2. High-Throughput Screening](#2-high-throughput-screening)
  - [3. ML Materials Property Prediction](#3-ml-materials-property-prediction)
  - [4. Battery Materials Discovery](#4-battery-materials-discovery)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Performance](#performance)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Features

### DFT Calculations

- **Quantum ESPRESSO:** Plane-wave DFT for periodic systems
  - Band structure and density of states
  - Phonon calculations and lattice dynamics
  - Magnetic properties and spin-polarized calculations
  - Hybrid functionals (HSE06, PBE0) on GPU instances

- **VASP (Vienna Ab initio Simulation Package):**
  - Gold standard for solid-state calculations
  - Optimized for AWS Graviton3 processors (30% faster)
  - Hybrid functionals accelerated on P4d GPU instances

- **GPAW:** Real-space DFT with Python interface
  - Easy integration with ASE (Atomic Simulation Environment)
  - GPU acceleration for large systems
  - Time-dependent DFT for optical properties

### High-Throughput Workflows

- **AiiDA:** Automated workflows with provenance tracking
  - Submit 10,000+ calculations as AWS Batch array jobs
  - Automatic error recovery and resubmission
  - PostgreSQL database for results storage on RDS

- **FireWorks:** Lightweight workflow engine
  - MongoDB backend on DocumentDB
  - Dynamic workflows based on calculation results
  - Integration with Materials Project infrastructure

- **Jobflow:** Modern Python workflows
  - Native AWS Batch support
  - S3 storage for calculation outputs
  - Integration with Dask for parallel post-processing

### Machine Learning

- **Crystal Graph Convolutional Networks (CGCNN):**
  - Predict formation energy (MAE: 0.039 eV/atom)
  - Band gap prediction (MAE: 0.388 eV)
  - Elastic moduli and hardness
  - Train on 100K+ materials in hours using SageMaker

- **ALIGNN (Atomistic Line Graph Neural Network):**
  - State-of-the-art property prediction
  - Formation energy MAE: 0.020 eV/atom
  - Direct band gap MAE: 0.218 eV
  - Pre-trained models available

- **MEGNet (MatErials Graph Network):**
  - Multi-task learning across properties
  - Transfer learning for data-scarce properties
  - Uncertainty quantification with ensembles

### Materials Databases

- **Materials Project:**
  - 150,000+ calculated materials
  - REST API for programmatic access
  - Computed properties: energies, band gaps, elasticity
  - Phase diagrams and Pourbaix diagrams

- **OQMD (Open Quantum Materials Database):**
  - 1,000,000+ DFT calculations
  - Formation energies and stability
  - Direct download of CONTCAR and OUTCAR files

- **AFLOW:**
  - 3,500,000+ materials entries
  - Prototype-based crystal structure search
  - Machine-readable data with REST API

- **JARVIS (Joint Automated Repository):**
  - 75,000+ materials with high-accuracy DFT
  - 2D materials database (1,500+ monolayers)
  - AI models trained on JARVIS data

## Cost Estimates

### DFT Calculations

**Single Material (Relaxation + Band Structure):**
- **Small system (10-20 atoms):** $0.10-0.30 per material
  - Instance: c7g.4xlarge (16 vCPU Graviton3) @ $0.58/hr
  - Runtime: 10-30 minutes
  - Example: Simple oxides, binary compounds

- **Medium system (50-100 atoms):**  $2-8 per material
  - Instance: c7g.16xlarge (64 vCPU) @ $2.32/hr
  - Runtime: 1-3 hours
  - Example: Perovskites, battery cathodes

- **Large system (200-500 atoms):** $20-100 per material
  - Instance: hpc7g.16xlarge (64 vCPU, 200 Gbps EFA) @ $2.88/hr
  - Runtime: 6-30 hours
  - Example: Interfaces, defects, large unit cells

**GPU-Accelerated Hybrid DFT:**
- **Instance:** p4d.24xlarge (8x A100 GPUs) @ $32.77/hr
- **Speedup:** 10-50x over CPU-only
- **Cost per material:** $10-50 (1-2 hours per calculation)
- **Use cases:** Accurate band gaps, optical properties

### High-Throughput Screening

**1,000 Material Candidates:**
- **Relaxation only:** $100-500
  - c7g.4xlarge instances in parallel
  - ~100 concurrent jobs
  - Runtime: 6-12 hours total

- **Full property calculations:** $2,000-10,000
  - Relaxation + band structure + elastic constants
  - 50-100 concurrent jobs
  - Runtime: 1-3 days

**10,000 Material Candidates:**
- **Relaxation only:** $800-4,000
- **Full calculations:** $15,000-80,000
- **Recommendation:** Start with ML pre-screening to reduce to top 1,000

### Machine Learning

**Train CGCNN on 100K Materials:**
- **Instance:** ml.p4d.24xlarge (8x A100) @ $32.77/hr
- **Runtime:** 12-24 hours
- **Cost:** $400-800 per model

**Inference on 1M Candidates:**
- **Instance:** ml.g5.xlarge (1x A10G GPU) @ $1.01/hr
- **Runtime:** 2-4 hours
- **Cost:** $2-4 total
- **Use case:** Pre-screen before expensive DFT

### HPC Cluster (ParallelCluster)

**100-Node Cluster for 1 Week:**
- **Head node:** c7g.4xlarge @ $0.58/hr × 168 hr = $97
- **Compute:** 100 × hpc7g.8xlarge @ $1.44/hr × 168 hr = $24,192
- **Storage:** FSx for Lustre 100 TB @ $0.14/GB-month = $14,000
- **Total:** ~$38,000/week

**Optimization Strategies:**
- Use Spot instances (50-70% savings): $12,000-19,000/week
- Start with 10-node pilot ($2,400-3,800/week)
- Auto-scaling: scale to zero when idle

### Monthly Research Budget Examples

**Individual Researcher:**
- **Budget:** $500-2,000/month
- **Capacity:** 500-2,000 DFT relaxations or 10-50 full characterizations
- **Strategy:** Spot instances, auto-scaling, ML pre-screening

**Small Research Group (5-10 people):**
- **Budget:** $3,000-10,000/month
- **Capacity:** 3,000-10,000 DFT calculations or 100-500 full studies
- **Infrastructure:** ParallelCluster with auto-scaling, shared file system

**Large Consortium:**
- **Budget:** $20,000-100,000/month
- **Capacity:** 50,000-200,000 DFT calculations
- **Infrastructure:** Multi-cluster deployment, dedicated HPC support

## Getting Started

### Prerequisites

```bash
# Install AWS CLI
pip install awscli

# Install ParallelCluster
pip install aws-parallelcluster

# Install materials science packages
pip install pymatgen ase aiida-core mp-api
pip install torch torch-geometric  # For ML models
```

### 1. Set Up AWS ParallelCluster

Create a cluster configuration file (`pcluster-config.yaml`):

```yaml
Region: us-east-1
Image:
  Os: alinux2
HeadNode:
  InstanceType: c7g.4xlarge
  Networking:
    SubnetId: subnet-12345678
  Ssh:
    KeyName: my-key-pair
  Iam:
    AdditionalIamPolicies:
      - Policy: arn:aws:iam::aws:policy/AmazonS3FullAccess
Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: compute
      ComputeResources:
        - Name: graviton-cpu
          InstanceType: hpc7g.8xlarge
          MinCount: 0
          MaxCount: 100
      Networking:
        SubnetIds:
          - subnet-12345678
        PlacementGroup:
          Enabled: true
      ComputeSettings:
        LocalStorage:
          RootVolume:
            Size: 100
SharedStorage:
  - MountDir: /shared
    Name: fsx
    StorageType: FsxLustre
    FsxLustreSettings:
      StorageCapacity: 1200
      DeploymentType: SCRATCH_2
```

Deploy the cluster:

```bash
pcluster create-cluster --cluster-name materials-cluster \
  --cluster-configuration pcluster-config.yaml

# Wait for cluster creation
pcluster describe-cluster --cluster-name materials-cluster

# SSH into head node
pcluster ssh --cluster-name materials-cluster -i ~/.ssh/my-key-pair.pem
```

### 2. Install Quantum ESPRESSO

On the head node:

```bash
# Load modules
module load openmpi/4.1.4

# Download and compile Quantum ESPRESSO
cd /shared
wget https://github.com/QEF/q-e/releases/download/qe-7.2/qe-7.2-ReleasePack.tar.gz
tar xzf qe-7.2-ReleasePack.tar.gz
cd qe-7.2

# Configure for AWS Graviton3
./configure --enable-parallel --enable-openmp \
  CFLAGS="-O3 -march=armv8.4-a" \
  FFLAGS="-O3 -march=armv8.4-a"

make pw ph pp

# Test installation
export PATH=/shared/qe-7.2/bin:$PATH
pw.x --version
```

### 3. Run Your First DFT Calculation

Create a Slurm job script (`si_relax.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=si_relax
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --partition=compute

module load openmpi/4.1.4

# Set up paths
export PATH=/shared/qe-7.2/bin:$PATH
export ESPRESSO_PSEUDO=/shared/pseudopotentials

# Run calculation
mpirun -np 32 pw.x < si.relax.in > si.relax.out
```

Create input file (`si.relax.in`):

```fortran
&CONTROL
  calculation = 'vc-relax'
  prefix = 'silicon'
  pseudo_dir = './pseudo'
  outdir = './tmp'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 50.0
  ecutrho = 400.0
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press = 0.0
/
ATOMIC_SPECIES
  Si  28.086  Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (alat)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS (automatic)
  8 8 8 0 0 0
```

Submit the job:

```bash
sbatch si_relax.slurm
squeue  # Check job status
```

## Applications

### 1. DFT Band Structure Calculations

Calculate electronic band structure for a new material:

```python
from pymatgen.core import Structure, Lattice
from pymatgen.io.pwscf import PWInput
import boto3

# Define crystal structure
lattice = Lattice.cubic(4.05)  # MgO
structure = Structure(
    lattice,
    ["Mg", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]]
)

# Generate Quantum ESPRESSO input
pw_input = PWInput(
    structure,
    control={
        'calculation': 'scf',
        'prefix': 'mgo',
        'pseudo_dir': './pseudo',
        'outdir': './tmp'
    },
    system={
        'ecutwfc': 60,
        'ecutrho': 480,
        'occupations': 'smearing',
        'smearing': 'gaussian',
        'degauss': 0.02
    },
    electrons={
        'conv_thr': 1e-8,
        'mixing_beta': 0.7
    },
    kpoints_mode='automatic',
    kpoints_grid=(8, 8, 8)
)

# Write input file
with open('mgo.scf.in', 'w') as f:
    f.write(str(pw_input))

# Submit to AWS Batch
batch = boto3.client('batch')
response = batch.submit_job(
    jobName='mgo-scf',
    jobQueue='dft-queue',
    jobDefinition='qe-scf:1',
    containerOverrides={
        'command': ['mpirun', '-np', '32', 'pw.x', '-i', 'mgo.scf.in'],
        'environment': [
            {'name': 'INPUT_FILE', 'value': 'mgo.scf.in'},
            {'name': 'OUTPUT_S3', 'value': 's3://my-bucket/results/mgo/'}
        ]
    }
)

print(f"Job submitted: {response['jobId']}")
```

**Post-processing band structure:**

```python
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.pwscf import PWOutput

# Parse output
pw_out = PWOutput('mgo.bands.out')
bs = pw_out.get_band_structure()

# Plot bands
plotter = BSPlotter(bs)
plotter.save_plot('mgo_bands.png', img_format='png')

# Upload to S3
s3 = boto3.client('s3')
s3.upload_file('mgo_bands.png', 'my-bucket', 'results/mgo/bands.png')

# Key results
print(f"Band gap: {bs.get_band_gap()['energy']:.3f} eV")
print(f"CBM: {bs.get_cbm()}")
print(f"VBM: {bs.get_vbm()}")
```

**Performance:**
- Small unit cell (2-10 atoms): 5-15 minutes on c7g.4xlarge
- Medium (20-50 atoms): 30-120 minutes on c7g.16xlarge
- Large (100+ atoms): 3-12 hours on hpc7g.16xlarge with EFA

### 2. High-Throughput Screening

Screen 1,000 perovskite materials for photovoltaic applications:

```python
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from aiida import orm, engine
from aiida.plugins import WorkflowFactory
import itertools

# Generate candidate compositions
a_site = ['Ca', 'Sr', 'Ba']
b_site = ['Ti', 'Zr', 'Hf', 'Sn']
x_site = ['O', 'S', 'Se']

candidates = []
for a, b, x in itertools.product(a_site, b_site, x_site):
    formula = f"{a}{b}{x}3"
    # Check if stable using Materials Project data
    with MPRester() as mpr:
        entries = mpr.get_entries_in_chemsys([a, b, x, 'O'])
        pd = PhaseDiagram(entries)
        # Only include if potentially stable
        if pd.get_e_above_hull(entry) < 0.1:  # eV/atom
            candidates.append(formula)

print(f"Found {len(candidates)} potentially stable perovskites")

# Submit high-throughput AiiDA workflow
RelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')

for formula in candidates:
    # Create structure (using perovskite prototype)
    structure = create_perovskite_structure(formula)

    # Set up AiiDA calculation
    builder = RelaxWorkChain.get_builder()
    builder.structure = orm.StructureData(pymatgen_structure=structure)
    builder.code = orm.load_code('qe-pw@aws-cluster')
    builder.kpoints = orm.KpointsData()
    builder.kpoints.set_kpoints_mesh([6, 6, 6])

    builder.base.pw.parameters = orm.Dict(dict={
        'CONTROL': {
            'calculation': 'vc-relax',
        },
        'SYSTEM': {
            'ecutwfc': 60,
            'ecutrho': 480,
        },
    })

    # Submit to AWS Batch via AiiDA
    node = engine.submit(builder)
    print(f"Submitted {formula}: {node.uuid}")
```

**AWS Batch Integration:**

```python
# Custom AiiDA transport for AWS Batch
class AwsBatchTransport:
    def submit_job(self, input_file, num_cpus=32):
        batch = boto3.client('batch')

        # Upload input to S3
        s3_input = f"s3://dft-inputs/{uuid.uuid4()}/input.in"
        s3.upload_file(input_file, bucket, key)

        # Submit Batch job
        response = batch.submit_job(
            jobName=f'dft-{formula}',
            jobQueue='compute-queue',
            jobDefinition='qe-relax:1',
            arrayProperties={'size': 1},
            containerOverrides={
                'vcpus': num_cpus,
                'memory': num_cpus * 4000,  # 4 GB per vCPU
                'environment': [
                    {'name': 'S3_INPUT', 'value': s3_input},
                    {'name': 'S3_OUTPUT', 'value': f's3://dft-results/{formula}/'}
                ]
            }
        )

        return response['jobId']
```

**Results Analysis:**

```python
# Query completed calculations from AiiDA database
from aiida.orm import QueryBuilder, Node

qb = QueryBuilder()
qb.append(RelaxWorkChain, filters={'attributes.process_state': 'finished'})

results = []
for node in qb.all(flat=True):
    formula = node.inputs.structure.get_formula()
    energy = node.outputs.output_parameters.get_dict()['energy']
    bandgap = calculate_bandgap(node.outputs.bands)

    results.append({
        'formula': formula,
        'energy_per_atom': energy / node.inputs.structure.num_sites,
        'bandgap': bandgap,
        'volume': node.outputs.structure.get_cell_volume()
    })

# Filter for promising candidates
import pandas as pd
df = pd.DataFrame(results)
promising = df[(df['bandgap'] > 1.0) & (df['bandgap'] < 2.0)]
print(f"Found {len(promising)} materials with ideal bandgap for solar cells")

# Upload to S3 for further analysis
df.to_csv('screening_results.csv', index=False)
s3.upload_file('screening_results.csv', 'my-bucket', 'results/screening.csv')
```

**Cost Optimization:**

- Use Spot instances for fault-tolerant calculations (60-70% savings)
- Implement checkpointing to restart interrupted calculations
- Pre-screen with cheap ML models before expensive DFT

### 3. ML Materials Property Prediction

Train a crystal graph neural network to predict formation energies:

```python
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.ext.matproj import MPRester
import sagemaker
from sagemaker.pytorch import PyTorch

class CGCNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128):
        super().__init__()
        self.conv1 = CGConv(num_node_features, dim=hidden_dim, batch_norm=True)
        self.conv2 = CGConv(hidden_dim, dim=hidden_dim, batch_norm=True)
        self.conv3 = CGConv(hidden_dim, dim=hidden_dim, batch_norm=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)  # Aggregate to graph-level
        return self.fc(x)

# Prepare training data from Materials Project
def structure_to_graph(structure):
    """Convert pymatgen Structure to PyTorch Geometric Data"""
    # Node features: atomic number, group, period, electronegativity
    node_features = []
    for site in structure:
        element = site.specie
        features = [
            element.Z,
            element.group,
            element.row,
            element.X  # Electronegativity
        ]
        node_features.append(features)

    # Edge features: distance, Gaussian expansion
    edges = []
    edge_features = []
    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, r=5.0)  # 5 Å cutoff
        for neighbor in neighbors:
            j = neighbor.index
            edges.append([i, j])

            # Gaussian distance expansion
            distance = neighbor.nn_distance
            gaussian = torch.exp(-((distance - torch.arange(0, 5, 0.2))**2) / 0.2)
            edge_features.append(gaussian)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_features, dtype=torch.float)
    )
    return data

# Download training data
with MPRester() as mpr:
    entries = mpr.get_entries('*', property_data=['formation_energy_per_atom'])

dataset = []
for entry in entries[:100000]:  # 100K materials
    graph = structure_to_graph(entry.structure)
    graph.y = torch.tensor([[entry.data['formation_energy_per_atom']]],
                           dtype=torch.float)
    dataset.append(graph)

# Save dataset to S3
torch.save(dataset, 'cgcnn_dataset.pt')
s3.upload_file('cgcnn_dataset.pt', 'my-bucket', 'datasets/cgcnn_dataset.pt')

# Train on SageMaker
estimator = PyTorch(
    entry_point='train_cgcnn.py',
    role='arn:aws:iam::123456789:role/SageMakerRole',
    instance_type='ml.p4d.24xlarge',  # 8x A100 GPUs
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 200,
        'batch-size': 256,
        'learning-rate': 0.001,
        'hidden-dim': 128
    }
)

estimator.fit({'training': 's3://my-bucket/datasets/'})
```

**Training script (`train_cgcnn.py`):**

```python
import argparse
import torch
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def train(args):
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Load dataset
    dataset = torch.load(os.path.join(args.data_dir, 'cgcnn_dataset.pt'))

    # Split train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, len(dataset) - train_size - val_size]
    )

    # Distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = CGCNN(num_node_features=4, num_edge_features=25,
                  hidden_dim=args.hidden_dim)
    model = model.to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.L1Loss()  # MAE

    # Training loop
    best_val_mae = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        total_loss = 0
        for batch in train_loader:
            batch = batch.to(local_rank)
            optimizer.zero_grad()

            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        if local_rank == 0:
            model.eval()
            val_mae = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(local_rank)
                    pred = model(batch.x, batch.edge_index, batch.edge_attr,
                               batch.batch)
                    val_mae += criterion(pred, batch.y).item()

            val_mae /= len(val_loader)
            print(f"Epoch {epoch}: Val MAE = {val_mae:.4f} eV/atom")

            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.module.state_dict(),
                          os.path.join(args.model_dir, 'best_model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')

    args = parser.parse_args()
    train(args)
```

**Deploy for inference:**

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data='s3://my-bucket/models/cgcnn/model.tar.gz',
    role='arn:aws:iam::123456789:role/SageMakerRole',
    entry_point='inference.py',
    framework_version='2.0',
    py_version='py310'
)

predictor = model.deploy(
    instance_type='ml.g5.xlarge',
    initial_instance_count=1,
    endpoint_name='cgcnn-formation-energy'
)

# Predict for new materials
new_structure = Structure(...)  # Your candidate material
graph = structure_to_graph(new_structure)
prediction = predictor.predict(graph)
print(f"Predicted formation energy: {prediction:.3f} eV/atom")
```

**Performance:**
- Training on 100K materials: 12-24 hours on ml.p4d.24xlarge (~$400-800)
- Inference: 1,000 predictions/second on ml.g5.xlarge
- MAE: 0.039 eV/atom (matches published CGCNN results)
- Use case: Pre-screen 1M candidates in 15 minutes for $0.25

### 4. Battery Materials Discovery

Discover new lithium-ion cathode materials:

```python
from pymatgen.analysis.battery import BatteryAnalyzer
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Composition
import itertools

# Define search space
cations = ['Li', 'Na', 'K']
transition_metals = ['Co', 'Ni', 'Mn', 'Fe', 'V', 'Ti']
anions = ['O', 'F', 'S', 'PO4']

candidates = []
for c, tm, a in itertools.product(cations, transition_metals, anions):
    # Generate reasonable compositions
    if a == 'PO4':
        formula = f"{c}{tm}{a}"
    else:
        formula = f"{c}{tm}{a}2"

    comp = Composition(formula)
    if comp.is_element:
        continue

    candidates.append(formula)

print(f"Screening {len(candidates)} cathode candidates")

# Phase 1: ML pre-screening for voltage and stability
cgcnn_model = load_model('cgcnn-formation-energy')
voltage_model = load_model('voltage-predictor')

promising_candidates = []
for formula in candidates:
    structure = generate_structure_prototype(formula)

    # Predict formation energy
    formation_energy = cgcnn_model.predict(structure)

    # Predict voltage
    voltage = voltage_model.predict(structure)

    # Predict capacity
    comp = Composition(formula)
    capacity = calculate_theoretical_capacity(comp)

    # Filter criteria
    if (formation_energy < 0 and  # Stable
        voltage > 2.5 and voltage < 4.5 and  # Safe voltage window
        capacity > 150):  # mAh/g
        promising_candidates.append({
            'formula': formula,
            'voltage': voltage,
            'capacity': capacity,
            'energy_density': voltage * capacity
        })

# Sort by energy density
promising_candidates.sort(key=lambda x: x['energy_density'], reverse=True)
print(f"ML pre-screening: {len(promising_candidates)} promising materials")

# Phase 2: DFT calculations on top 100
top_candidates = promising_candidates[:100]

for candidate in top_candidates:
    # Submit DFT calculations
    # 1. Structure relaxation
    # 2. Delithiated structure (cathode)
    # 3. Calculate voltage from energy difference
    # 4. Calculate migration barriers

    submit_battery_workflow(candidate['formula'])

def submit_battery_workflow(formula):
    """Submit complete battery characterization workflow"""

    # Relaxation of lithiated structure
    relax_li = submit_qe_relax(formula, charge_state='lithiated')

    # Relaxation of delithiated structure
    delithiated_formula = remove_lithium(formula)
    relax_deli = submit_qe_relax(delithiated_formula, charge_state='delithiated')

    # Calculate voltage
    # V = -(E_delithiated - E_lithiated - n*E_Li) / n
    # where n = number of Li removed

    # Migration barrier with NEB (Nudged Elastic Band)
    neb_job = submit_neb_calculation(formula)

    # Compile results
    return {
        'formula': formula,
        'voltage_dft': calculate_voltage(relax_li, relax_deli),
        'migration_barrier': extract_barrier(neb_job),
        'volume_change': calculate_volume_change(relax_li, relax_deli)
    }
```

**Analysis of results:**

```python
# Query all completed battery calculations
results = query_aiida_database(workflow_name='battery_screening')

df = pd.DataFrame(results)

# Filter for promising cathodes
good_cathodes = df[
    (df['voltage_dft'] > 3.0) &  # High voltage
    (df['voltage_dft'] < 4.5) &  # Stable electrolyte
    (df['capacity'] > 150) &  # Good capacity
    (df['migration_barrier'] < 0.6) &  # Fast kinetics
    (df['volume_change'] < 10)  # Structural stability
]

print(f"Found {len(good_cathodes)} excellent cathode candidates")
print(good_cathodes.sort_values('energy_density', ascending=False).head(10))

# Upload to S3 for visualization
df.to_csv('battery_screening_results.csv')
s3.upload_file('battery_screening_results.csv', 'my-bucket',
               'results/battery_discovery.csv')
```

**Cost breakdown for discovery pipeline:**
- ML screening of 10,000 candidates: $1-5
- DFT calculations on 100 candidates: $500-2,000
- Full characterization of 10 materials: $1,000-5,000
- Total: $1,500-7,000 for complete discovery pipeline

**Expected timeline:**
- ML screening: 1-2 hours
- DFT relaxations: 1-2 days (parallel)
- Full characterization: 3-5 days (parallel)
- Total: ~1 week from idea to validated candidates

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Researchers                              │
│                    (Jupyter, CLI, API)                           │
└────────────┬────────────────────────────────────┬────────────────┘
             │                                    │
             v                                    v
┌─────────────────────────┐         ┌──────────────────────────┐
│  AWS ParallelCluster    │         │   AWS Batch              │
│  - Slurm scheduler      │         │   - Array jobs           │
│  - 100+ compute nodes   │         │   - Spot instances       │
│  - HPC instances        │         │   - Auto-scaling         │
│  - FSx for Lustre       │         └──────────┬───────────────┘
└────────┬────────────────┘                    │
         │                                      │
         v                                      v
┌─────────────────────────────────────────────────────────────────┐
│                         S3 Data Lake                             │
│  - Input structures     - DFT outputs     - ML models            │
│  - Pseudopotentials     - Results DB      - Training data        │
└─────────┬───────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────────┐
│                     Analysis & ML                                │
│  SageMaker: Train CGCNN, ALIGNN models                          │
│  Athena: Query results with SQL                                 │
│  Glue: Data catalog and ETL                                     │
│  QuickSight: Interactive dashboards                             │
└─────────────────────────────────────────────────────────────────┘
```

### Workflow Orchestration

**AiiDA Architecture:**

```
┌──────────────┐
│  AiiDA       │
│  (Head Node) │
└──────┬───────┘
       │
       ├─→ PostgreSQL (RDS) ─→ Store provenance, results
       │
       ├─→ RabbitMQ (Amazon MQ) ─→ Task queue
       │
       └─→ AWS Batch ─→ Submit calculations
                │
                └─→ S3 ─→ Store inputs/outputs
```

### Data Flow

1. **Input:** Researcher defines materials to study
2. **Structure generation:** Create initial geometries
3. **Submission:** Jobs sent to ParallelCluster or Batch
4. **Calculation:** DFT engines run on HPC instances
5. **Storage:** Outputs saved to S3
6. **Analysis:** Results processed with Python/ML
7. **Visualization:** Interactive plots and dashboards

## Data Sources

### Materials Databases

**Materials Project:**
- **Access:** Free API (request key at materialsproject.org)
- **Size:** 150,000+ materials
- **Properties:** Formation energy, band structure, elasticity, piezoelectricity
- **AWS hosting:** No (but fast API response)

```python
from mp_api.client import MPRester

with MPRester(api_key='YOUR_API_KEY') as mpr:
    # Get all perovskites
    docs = mpr.materials.summary.search(
        chemsys='Ca-Ti-O',
        fields=['material_id', 'formula_pretty', 'energy_per_atom', 'band_gap']
    )

    for doc in docs:
        print(f"{doc.formula_pretty}: {doc.band_gap} eV")
```

**Open Quantum Materials Database (OQMD):**
- **Access:** REST API (oqmd.org/api)
- **Size:** 1,000,000+ DFT calculations
- **Focus:** Formation energies and phase stability
- **Data format:** VASP input/output files

```python
import requests

url = 'http://oqmd.org/api/search'
params = {'composition': 'Fe2O3', 'format': 'json'}
response = requests.get(url, params=params)
data = response.json()

for entry in data['data']:
    print(f"OQMD ID: {entry['entry_id']}")
    print(f"Formation energy: {entry['delta_e']} eV/atom")
```

**AFLOW:**
- **Access:** REST API (aflowlib.org)
- **Size:** 3,500,000+ materials
- **Focus:** Crystal structures and prototypes
- **Unique:** Automatic structure classification

**JARVIS:**
- **Access:** Python API (jarvis-tools)
- **Size:** 75,000+ materials with high-accuracy DFT
- **Focus:** 2D materials, optical properties, defects
- **ML models:** Pre-trained ALIGNN models available

### Pseudopotentials

**SSSP (Standard Solid State Pseudopotentials):**
- **Location:** s3://sssp-pseudopotentials (user-hosted)
- **License:** Open source
- **Quality:** Precision and efficiency tested

**PseudoDojo:**
- **Download:** http://www.pseudo-dojo.org
- **Coverage:** Full periodic table
- **Formats:** ONCVPSP, Norm-conserving

**Upload to S3:**

```bash
# Download pseudopotentials
wget http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_psp8.tar.gz
tar xzf nc-sr-04_pbe_standard_psp8.tar.gz

# Upload to S3
aws s3 sync nc-sr-04_pbe_standard_psp8/ s3://my-bucket/pseudopotentials/
```

## Performance

### DFT Calculation Benchmarks

**Silicon (8 atoms, 8×8×8 k-points) - Relaxation:**
| Instance | vCPUs | Time | Cost | Speedup |
|----------|-------|------|------|---------|
| c6g.4xlarge (Graviton2) | 16 | 12 min | $0.14 | 1.0x |
| c7g.4xlarge (Graviton3) | 16 | 9 min | $0.09 | 1.3x |
| c7g.16xlarge (Graviton3) | 64 | 3 min | $0.12 | 4.0x |
| hpc7g.16xlarge (EFA) | 64 | 2.5 min | $0.12 | 4.8x |

**TiO₂ rutile (24 atoms, 6×6×6 k-points) - Band structure:**
| Instance | vCPUs | Time | Cost | Notes |
|----------|-------|------|------|-------|
| c7g.8xlarge | 32 | 45 min | $0.43 | Standard |
| c7g.16xlarge | 64 | 28 min | $0.54 | Better parallel |
| hpc7g.8xlarge | 32 | 35 min | $0.42 | EFA benefit minimal |
| p4d.24xlarge (GPU) | 96 + 8×A100 | 8 min | $4.37 | 5.6x faster |

**Large protein (500 atoms) - Hybrid DFT:**
| Instance | Time | Cost | Notes |
|----------|------|------|-------|
| c7g.16xlarge (CPU) | 18 hours | $41.76 | Standard |
| p4d.24xlarge (GPU) | 45 min | $24.58 | 24x speedup |
| p5.48xlarge (H100) | 25 min | $49.00 | 43x speedup |

**Key Insights:**
- Graviton3 (c7g) is 30% faster than Graviton2 for DFT
- GPU acceleration is essential for hybrid functionals (10-50x speedup)
- For small systems, single-node instances are most cost-effective
- For large systems or bulk calculations, use ParallelCluster with EFA

### High-Throughput Performance

**1,000 Material Screening (Relaxation Only):**
- **Configuration:** 50 × c7g.4xlarge instances (Spot)
- **Parallel jobs:** 800 concurrent
- **Runtime:** 8 hours
- **Cost:** $180 (with Spot discount)
- **Cost per material:** $0.18

**10,000 Material Screening (Full Characterization):**
- **Configuration:** 100 × hpc7g.8xlarge instances
- **Parallel jobs:** 1,600 concurrent
- **Runtime:** 3 days
- **Cost:** $35,000 (On-Demand) or $12,000 (Spot)
- **Cost per material:** $3.50 (On-Demand) or $1.20 (Spot)

### ML Model Training

**CGCNN on 100K Materials:**
- **Instance:** ml.p4d.24xlarge (8× A100 GPUs)
- **Runtime:** 18 hours
- **Cost:** $590
- **Validation MAE:** 0.039 eV/atom

**ALIGNN on 50K Materials:**
- **Instance:** ml.p4d.24xlarge
- **Runtime:** 24 hours
- **Cost:** $786
- **Validation MAE:** 0.020 eV/atom (state-of-the-art)

**Inference Performance:**
- **Instance:** ml.g5.xlarge (single A10G GPU)
- **Throughput:** 1,000 predictions/second
- **Cost:** $1.01/hour
- **Use case:** Screen 1M candidates in 15 minutes for $0.25

## Best Practices

### Cost Optimization

1. **Use Spot Instances for Fault-Tolerant Workloads:**
   - 60-70% cost savings
   - Ideal for high-throughput screening
   - Implement checkpointing for long calculations

2. **Right-Size Your Instances:**
   - Small systems: c7g.4xlarge (16 vCPU)
   - Medium systems: c7g.16xlarge (64 vCPU)
   - Large systems: hpc7g.16xlarge (64 vCPU with EFA)
   - Hybrid DFT: p4d.24xlarge (GPU acceleration)

3. **ML Pre-Screening:**
   - Train CGCNN for $500-800
   - Screen 1M candidates for $0.25
   - Run DFT only on top 0.1% (1,000 materials)
   - Total cost: ~$1,000 vs $100,000 for full DFT screening

4. **Storage Optimization:**
   - Use S3 Intelligent-Tiering for results
   - Delete intermediate wavefunction files
   - Use FSx for Lustre (SCRATCH_2) for temporary data
   - Compress outputs before uploading to S3

### Performance Optimization

1. **Parallelization Strategy:**
   - k-point parallelization: Best for small systems
   - Band parallelization: Good for large systems
   - Plane-wave parallelization: Use with many processors

2. **Convergence Testing:**
   - Test k-point mesh and energy cutoff convergence first
   - Use coarse settings for structure optimization
   - Switch to fine settings for final SCF

3. **Pseudopotential Selection:**
   - Use SSSP efficiency pseudopotentials for screening
   - Switch to precision pseudopotentials for final results
   - Pre-download to FSx for Lustre to avoid repeated S3 calls

### Workflow Best Practices

1. **Provenance Tracking:**
   - Use AiiDA to track all calculation inputs and outputs
   - Store results in structured database (PostgreSQL on RDS)
   - Tag calculations with project IDs for cost allocation

2. **Error Handling:**
   - Implement automatic error detection
   - Restart failed calculations with adjusted parameters
   - Set maximum retry limits to avoid infinite loops

3. **Data Management:**
   - Organize S3 with clear hierarchy: project/material/calctype/
   - Use Glue Data Catalog to track datasets
   - Implement data lifecycle policies (delete after 6 months)

## Troubleshooting

### Common DFT Issues

**Problem: SCF Convergence Failure**
```
Convergence NOT achieved after 100 iterations
```

**Solutions:**
1. Reduce mixing_beta (e.g., from 0.7 to 0.3)
2. Use mixing_mode = 'local-TF' for magnetic systems
3. Increase conv_thr temporarily (e.g., from 1e-8 to 1e-6)
4. Check if structure is reasonable (no overlapping atoms)

**Problem: Parallel Efficiency is Low**
```
Parallel efficiency: 30%
```

**Solutions:**
1. Reduce number of processors (diminishing returns after certain point)
2. Use different parallelization scheme (-npool, -ndiag flags)
3. Check if EFA is enabled for multi-node jobs
4. Ensure processors are in same placement group

### AWS-Specific Issues

**Problem: Spot Instance Interruptions**
```
AWS Batch job terminated: Spot instance interrupted
```

**Solutions:**
1. Enable checkpointing in DFT code
2. Use On-Demand capacity for critical calculations
3. Implement automatic restart with saved wavefunctions
4. Consider Spot Fleet with multiple instance types

**Problem: FSx for Lustre Performance**
```
Slow I/O performance during calculation
```

**Solutions:**
1. Use SCRATCH_2 deployment type (higher throughput)
2. Increase stripe count for large files: `lfs setstripe -c -1`
3. Mount with optimized options: `flock,noatime,nodiratime`
4. Pre-stage data to FSx before starting calculations

**Problem: S3 Transfer Timeouts**
```
Connection timeout when uploading results
```

**Solutions:**
1. Use S3 VPC endpoints for faster transfers
2. Implement multipart uploads for large files
3. Enable S3 Transfer Acceleration
4. Compress outputs before uploading

### Database Issues

**Problem: AiiDA Daemon Not Running**
```
Could not submit workflow: Daemon offline
```

**Solutions:**
```bash
# Check daemon status
verdi daemon status

# Restart daemon
verdi daemon restart

# Check logs
verdi daemon logshow
```

**Problem: PostgreSQL Connection Errors**
```
OperationalError: could not connect to server
```

**Solutions:**
1. Check RDS security group allows connections
2. Verify connection string in AiiDA config
3. Check if RDS instance is running
4. Test connection: `psql -h <rds-endpoint> -U aiida -d aiidadb`

## Additional Resources

### Documentation

- **Quantum ESPRESSO:** https://www.quantum-espresso.org/documentation
- **VASP:** https://www.vasp.at/wiki/
- **AiiDA:** https://aiida.readthedocs.io/
- **Materials Project:** https://docs.materialsproject.org/
- **AWS ParallelCluster:** https://docs.aws.amazon.com/parallelcluster/

### Tutorials

- **DFT Tutorial:** CECAM Electronic Structure Tutorials
- **AiiDA Tutorial:** https://aiida-tutorials.readthedocs.io/
- **Materials Project Workshop:** https://workshop.materialsproject.org/

### Pre-Trained ML Models

- **ALIGNN:** https://github.com/usnistgov/alignn
- **CGCNN:** https://github.com/txie-93/cgcnn
- **MEGNet:** https://github.com/materialsvirtuallab/megnet

### AWS Architecture Guides

- **HPC on AWS:** https://aws.amazon.com/hpc/
- **ParallelCluster Best Practices:** https://docs.aws.amazon.com/parallelcluster/latest/ug/best-practices.html
- **SageMaker for Materials:** https://aws.amazon.com/blogs/machine-learning/

## Support

For questions specific to this project:
- Create an issue on the GitHub repository
- Contact the AWS Research Jumpstart team

For AWS support:
- AWS Support Console (requires Support plan)
- AWS re:Post for community questions

For scientific computing questions:
- Materials Project Discussion Forum
- AiiDA Discourse
- Quantum ESPRESSO User Forum
