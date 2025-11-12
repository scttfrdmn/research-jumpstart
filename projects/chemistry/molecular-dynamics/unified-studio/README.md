# Molecular Dynamics at Scale - AWS Research Jumpstart

**Tier 1 Flagship Project**

Large-scale molecular dynamics (MD) simulations using GROMACS, AMBER, and OpenMM on AWS. Simulate protein folding, drug binding, membrane dynamics, and materials properties with millions of atoms. Integrate machine learning for enhanced sampling and accelerated simulations.

## Overview

This flagship project demonstrates how to run classical and GPU-accelerated molecular dynamics simulations on AWS infrastructure. We'll simulate biomolecular systems, perform free energy calculations, analyze conformational dynamics, and train neural network force fields using AWS Batch, ParallelCluster, and SageMaker.

### Key Features

- **MD Engines:** GROMACS, AMBER, OpenMM, NAMD, LAMMPS
- **System sizes:** From small molecules (100s atoms) to large complexes (millions of atoms)
- **GPU acceleration:** NVIDIA A100, V100 for 10-100x speedup
- **Enhanced sampling:** Metadynamics, replica exchange, adaptive sampling
- **ML integration:** Neural network potentials, AlphaFold2 for structure prediction
- **AWS services:** Batch, ParallelCluster, S3, FSx Lustre, SageMaker

### Scientific Applications

1. **Protein dynamics:** Folding, conformational changes, allosteric regulation
2. **Drug discovery:** Binding affinity, residence time, ADMET properties
3. **Membrane systems:** Lipid bilayers, ion channels, transporters
4. **Materials science:** Polymers, nanomaterials, catalysts
5. **Free energy calculations:** Binding free energy, solvation, pKa prediction

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Molecular Dynamics Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ PDB Database │      │ AlphaFold2   │      │ Drug         │
│ (Structures) │─────▶│ (Predicted)  │─────▶│ Libraries    │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   S3 Data Lake    │
                    │  (PDB, topologies,│
                    │   trajectories)   │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ AWS Batch     │   │ ParallelCluster   │   │ FSx Lustre │
│ (GPU p3/p4)   │   │ (MPI, HPC)        │   │ (Fast I/O) │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  MDAnalysis       │
                    │  (Trajectory      │
                    │   Analysis)       │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Free Energy  │   │ Neural Network    │   │ Visualization │
│ (FEP, TI)    │   │ Potentials (ML)   │   │ (VMD, PyMOL)  │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Bedrock (Claude)  │
                    │ Results           │
                    │ Interpretation    │
                    └───────────────────┘
```

## System Specifications

### MD Engines

| Engine | Best For | GPU Support | License |
|--------|----------|-------------|---------|
| GROMACS | Biomolecules, proteins | Excellent (CUDA) | GPL |
| AMBER | Nucleic acids, proteins | Good (CUDA) | Commercial/Academic |
| OpenMM | Python integration, custom forces | Excellent (CUDA/OpenCL) | MIT |
| NAMD | Large systems, membrane proteins | Good (CUDA) | Free (academic) |
| LAMMPS | Materials, polymers | Good (KOKKOS) | GPL |

### GPU Performance

**GROMACS benchmark (1.4M atom system, STMV virus):**
- CPU only (64 cores): ~10 ns/day
- 1x V100 GPU: ~100 ns/day (10x faster)
- 4x V100 GPUs: ~350 ns/day (35x faster)
- 1x A100 GPU: ~200 ns/day (20x faster)
- 4x A100 GPUs: ~750 ns/day (75x faster)

**Cost comparison (1 μs simulation):**
- CPU (c5.18xlarge, 72 vCPU): $36/day → $3,600 for 1 μs (100 days)
- GPU (p3.2xlarge, 1x V100): $3.06/hour → $310 for 1 μs (10 days)
- GPU (p4d.24xlarge, 8x A100): $32.77/hour → $420 for 1 μs (1.3 days)

**Recommendation:** Use GPU instances (p3.2xlarge or p3.8xlarge) for most workloads.

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# MD tools (for local testing)
# GROMACS: http://www.gromacs.org/
# OpenMM: http://openmm.org/
# VMD: https://www.ks.uiuc.edu/Research/vmd/

# Python dependencies
pip install -r requirements.txt
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name molecular-dynamics-stack \
  --template-body file://cloudformation/md-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion (10-15 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name molecular-dynamics-stack

# Get outputs
aws cloudformation describe-stacks \
  --stack-name molecular-dynamics-stack \
  --query 'Stacks[0].Outputs'
```

### Download Example Systems

```python
from src.system_setup import SystemBuilder
import MDAnalysis as mda

# Initialize builder
builder = SystemBuilder()

# Download protein structure from PDB
structure = builder.fetch_pdb('1UBQ')  # Ubiquitin
print(f"Loaded: {structure}")

# Or use AlphaFold2 predicted structure
structure = builder.fetch_alphafold('Q9Y4K3')  # Example UniProt ID

# Prepare system for simulation
system = builder.prepare_system(
    structure,
    forcefield='amber14sb',
    water_model='tip3p',
    box_type='cubic',
    box_padding=1.0,  # nm
    ion_concentration=0.15  # M NaCl
)

# Save prepared system
builder.save_system(system, 'ubiquitin_prepared.pdb')
```

## Core Simulations

### 1. Protein in Water

Classic MD simulation of a protein in explicit solvent.

```python
from src.md_simulations import GromacsSimulation

# Initialize simulation
sim = GromacsSimulation(
    structure='ubiquitin_prepared.pdb',
    topology='ubiquitin.top',
    forcefield='amber14sb'
)

# Energy minimization
sim.energy_minimization(
    max_steps=50000,
    emtol=1000.0  # kJ/mol/nm
)

# NVT equilibration (300K)
sim.equilibration_nvt(
    temperature=300,  # K
    duration=100,  # ps
    restraints='protein'  # Restrain protein, equilibrate water
)

# NPT equilibration (1 bar)
sim.equilibration_npt(
    temperature=300,
    pressure=1.0,  # bar
    duration=100  # ps
)

# Production run on AWS Batch (GPU)
job_id = sim.submit_production(
    duration=100,  # ns
    instance_type='p3.2xlarge',
    dt=0.002,  # ps (2 fs)
    output_freq=10,  # ps
    output_bucket='s3://my-bucket/trajectories/'
)

# Monitor job
sim.monitor_job(job_id)
```

### 2. Membrane Protein System

Simulate a protein embedded in lipid bilayer.

```python
from src.membrane_systems import MembraneBuilder

# Build membrane system
builder = MembraneBuilder()

# Insert protein into pre-equilibrated membrane
system = builder.build_membrane_protein(
    protein_pdb='gpcr.pdb',
    lipid_type='POPC',  # Phospholipid
    n_lipids=200,
    water_thickness=2.0,  # nm above/below
    ion_concentration=0.15
)

# Run simulation
sim = GromacsSimulation(system=system)
sim.run_protocol(
    minimization=True,
    equilibration_time=1000,  # ps
    production_time=100000,  # ps (100 ns)
    instance_type='p3.8xlarge'  # 4x V100
)
```

### 3. Ligand Binding

Simulate small molecule binding to protein.

```python
from src.ligand_binding import LigandDocking

# Prepare protein-ligand complex
docker = LigandDocking()

# Molecular docking (initial pose)
complex_pdb = docker.dock_ligand(
    receptor='protein.pdb',
    ligand='ligand.sdf',
    binding_site_center=[25.0, 30.0, 15.0],  # Coordinates
    binding_site_size=[20.0, 20.0, 20.0]  # Box dimensions
)

# MD simulation of complex
sim = GromacsSimulation(structure=complex_pdb)
sim.run_protocol(
    production_time=50000,  # 50 ns
    analysis=['rmsd', 'rmsf', 'contacts', 'hbonds']
)

# Analyze binding stability
contacts = sim.analyze_contacts(
    selection1='protein',
    selection2='resname LIG',
    cutoff=0.35  # nm
)

print(f"Persistent contacts: {len(contacts[contacts > 0.5 * sim.n_frames])}")
```

### 4. Free Energy Calculations

Compute binding free energy using alchemical methods.

```python
from src.free_energy import FreeEnergyCalculator

# Initialize calculator
fep = FreeEnergyCalculator(method='BAR')  # Bennett Acceptance Ratio

# Setup alchemical transformation
# Ligand in complex -> Ligand in water
fep.setup_transformation(
    complex_pdb='protein_ligand.pdb',
    ligand_residue='LIG',
    lambda_windows=20,  # Number of intermediate states
    soft_core=True  # Avoid singularities
)

# Run simulations for each lambda window (parallel on AWS Batch)
job_ids = fep.submit_lambda_windows(
    duration=5000,  # ps per window
    instances_per_window=1,
    instance_type='p3.2xlarge'
)

# Wait for completion and analyze
fep.wait_for_jobs(job_ids)
delta_g = fep.calculate_free_energy()

print(f"Binding free energy: {delta_g:.2f} ± {fep.uncertainty:.2f} kcal/mol")
print(f"Expected Kd: {fep.calculate_kd(delta_g):.2e} M")
```

### 5. Enhanced Sampling - Metadynamics

Accelerate sampling of rare events and calculate free energy landscapes.

```python
from src.enhanced_sampling import Metadynamics

# Setup metadynamics simulation
metad = Metadynamics()

# Define collective variables (CVs)
cvs = [
    {'type': 'distance', 'atoms': [10, 50]},  # Distance between residues
    {'type': 'rmsd', 'reference': 'native.pdb'}
]

# Run metadynamics
sim = metad.setup_simulation(
    structure='protein.pdb',
    collective_variables=cvs,
    hill_height=1.2,  # kJ/mol
    hill_width=[0.05, 0.05],  # nm, nm
    deposition_frequency=500  # steps
)

job_id = sim.submit_to_batch(
    duration=100000,  # ps (100 ns)
    instance_type='p3.2xlarge'
)

# Analyze free energy surface
sim.wait_for_completion(job_id)
fes = metad.calculate_free_energy_surface(
    temperature=300,
    grid_resolution=100
)

# Plot
metad.plot_fes(fes, save_path='free_energy_surface.png')
```

## Advanced Analysis

### Trajectory Analysis

```python
from src.analysis import TrajectoryAnalyzer
import MDAnalysis as mda

# Load trajectory
analyzer = TrajectoryAnalyzer(
    topology='system.pdb',
    trajectory='s3://bucket/trajectory.xtc'
)

# RMSD (structural drift)
rmsd = analyzer.calculate_rmsd(
    selection='protein and name CA',
    reference='initial'
)
analyzer.plot_rmsd(rmsd)

# RMSF (per-residue flexibility)
rmsf = analyzer.calculate_rmsf(
    selection='protein and name CA'
)
analyzer.plot_rmsf(rmsf)

# Radius of gyration (compactness)
rg = analyzer.calculate_rg(selection='protein')

# Principal component analysis
pca = analyzer.pca_analysis(
    selection='protein and name CA',
    n_components=3
)
analyzer.plot_pca_projection(pca, pc1=0, pc2=1)

# Secondary structure evolution
ss = analyzer.calculate_secondary_structure()
analyzer.plot_ss_timeline(ss)

# Contact maps
contacts = analyzer.calculate_contact_map(
    selection1='resid 1-100',
    selection2='resid 101-200',
    cutoff=0.8  # nm
)
analyzer.plot_contact_map(contacts)

# Hydrogen bonds
hbonds = analyzer.analyze_hydrogen_bonds(
    donors='protein',
    acceptors='protein or resname SOL',
    distance_cutoff=0.35,  # nm
    angle_cutoff=150  # degrees
)
```

### Machine Learning Potentials

Train neural network force fields for faster simulations.

```python
from src.ml_potentials import NeuralNetworkPotential
import torch

# Initialize ML potential (e.g., SchNet, PaiNN, MACE)
mlp = NeuralNetworkPotential(architecture='schnet')

# Prepare training data from MD trajectories
training_data = mlp.prepare_training_data(
    trajectories=['s3://bucket/traj1.xtc', 's3://bucket/traj2.xtc'],
    reference_forces='s3://bucket/forces.npy',  # From QM calculations
    max_samples=100000
)

# Train on SageMaker
training_job = mlp.train(
    data=training_data,
    instance_type='ml.p3.8xlarge',
    epochs=100,
    batch_size=32,
    learning_rate=1e-4
)

# Deploy as endpoint
endpoint = mlp.deploy(instance_type='ml.g4dn.xlarge')

# Run MD with ML potential (much faster than QM!)
from src.md_simulations import OpenMMSimulation

sim = OpenMMSimulation(structure='molecule.pdb')
sim.set_ml_potential(endpoint)
sim.run_production(duration=10000)  # ps
```

## AWS ParallelCluster for HPC

For very large simulations or parallel replica exchange.

```bash
# Create HPC cluster configuration
cat > cluster-config.yaml <<EOF
Region: us-east-1
Image:
  Os: alinux2
HeadNode:
  InstanceType: c5.2xlarge
  Networking:
    SubnetId: subnet-xxxxx
  Ssh:
    KeyName: my-key
Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: gpu
      ComputeResources:
        - Name: p3-8xl
          InstanceType: p3.8xlarge
          MinCount: 0
          MaxCount: 10
      Networking:
        SubnetIds:
          - subnet-xxxxx
      ComputeSettings:
        LocalStorage:
          RootVolume:
            Size: 100
SharedStorage:
  - MountDir: /fsx
    Name: fsx
    StorageType: FsxLustre
    FsxLustreSettings:
      StorageCapacity: 1200
      ImportPath: s3://my-bucket/data
EOF

# Create cluster
pcluster create-cluster \
  --cluster-name md-cluster \
  --cluster-configuration cluster-config.yaml

# Wait for creation
pcluster describe-cluster --cluster-name md-cluster

# SSH to head node
pcluster ssh --cluster-name md-cluster -i my-key.pem

# Submit GROMACS job
cat > submit_md.sh <<EOF
#!/bin/bash
#SBATCH --job-name=protein_md
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

module load gromacs/2023.1-cuda
gmx mdrun -deffnm production -ntomp 1 -nb gpu -pme gpu -npme 1
EOF

sbatch submit_md.sh
```

## Replica Exchange MD (REMD)

Enhanced sampling using multiple parallel replicas.

```python
from src.enhanced_sampling import ReplicaExchange

# Setup temperature replica exchange
remd = ReplicaExchange(
    n_replicas=16,
    temp_min=300,  # K
    temp_max=400,  # K
    exchange_frequency=1000  # steps
)

# Run on ParallelCluster or AWS Batch array jobs
job_ids = remd.submit_replicas(
    structure='protein.pdb',
    duration=50000,  # ps per replica
    instance_type='p3.2xlarge'  # 1 GPU per replica
)

# Monitor exchanges
remd.monitor_exchange_acceptance(job_ids)

# Analyze with WHAM (Weighted Histogram Analysis Method)
free_energy = remd.calculate_free_energy_profile(
    collective_variable='rmsd',
    temperature=300
)
```

## Integration with AlphaFold2

Use AI-predicted structures as starting points.

```python
from src.structure_prediction import AlphaFold2Runner

# Run AlphaFold2 on SageMaker
af2 = AlphaFold2Runner()

# Predict structure from sequence
job_id = af2.predict_structure(
    sequence='MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
    instance_type='ml.p3.2xlarge',
    use_templates=True,
    num_models=5
)

# Get predicted structures
structures = af2.get_results(job_id)
best_structure = structures[0]  # Highest pLDDT

print(f"Predicted structure confidence: {best_structure.plddt:.2f}")

# Use for MD
sim = GromacsSimulation(structure=best_structure.pdb_file)
sim.run_protocol(production_time=100000)
```

## Cost Estimate

**One-time setup:** $50-100

**Per-simulation costs:**
- Small protein (50 ns, p3.2xlarge): $15-20
- Large protein (100 ns, p3.8xlarge): $100-150
- Membrane system (100 ns, p3.8xlarge): $100-150
- Free energy (20 windows × 5 ns): $150-200
- Metadynamics (100 ns, p3.2xlarge): $30-40

**Large-scale campaign (drug screening, 100 ligands):**
- Docking: $10-20 (CPU)
- MD simulations (100 × 50 ns): $1,500-2,000
- Free energy (top 10 hits): $1,500-2,000
- **Total: $3,000-4,000**

**Cost optimization:**
- Use Spot instances (60-70% savings)
- Start with shorter simulations, extend if promising
- Use enhanced sampling to reduce simulation time
- Share GPU instances across multiple short simulations

## Performance Benchmarks

**GROMACS STMV benchmark (1.4M atoms):**
- c5.18xlarge (72 vCPU): 10 ns/day → 10 days for 100 ns → $360
- p3.2xlarge (1x V100): 100 ns/day → 1 day for 100 ns → $73
- p3.8xlarge (4x V100): 350 ns/day → 7 hours for 100 ns → $22
- p4d.24xlarge (8x A100): 750 ns/day → 3 hours for 100 ns → $98

**Recommendation: p3.8xlarge for best price/performance**

## CloudFormation Resources

The stack creates:

1. **S3 Buckets:**
   - `md-structures`: PDB files, topologies
   - `md-trajectories`: Simulation outputs
   - `md-analysis`: Analysis results

2. **AWS Batch:**
   - GPU compute environment (p3/p4 instances)
   - Job queues for GROMACS, AMBER, OpenMM
   - Job definitions with Docker containers

3. **FSx for Lustre:**
   - High-performance shared file system
   - Linked to S3 for auto-import/export

4. **SageMaker:**
   - Notebook instance for interactive analysis
   - Training jobs for ML potentials
   - Endpoints for AlphaFold2 and force fields

5. **ParallelCluster (optional):**
   - HPC cluster with Slurm scheduler
   - GPU compute nodes
   - Shared storage via FSx

## Example Research Applications

### 1. Protein Folding Pathways

**System:** Fast-folding protein (e.g., villin headpiece, 35 residues)
**Methods:** Multiple independent runs, Markov State Models
**Duration:** 100 × 1 μs simulations
**Cost:** $5,000-7,000

### 2. Drug-Target Binding

**System:** Kinase + small molecule inhibitor
**Methods:** Docking, MD, binding free energy (FEP)
**Duration:** 50 ns MD + 20 FEP windows
**Cost:** $200-300 per ligand

### 3. Membrane Channel Permeation

**System:** Ion channel in lipid bilayer (300K atoms)
**Methods:** Steered MD, umbrella sampling, PMF calculation
**Duration:** 50 windows × 20 ns
**Cost:** $500-800

### 4. Antibody-Antigen Recognition

**System:** Antibody Fab fragment + antigen epitope
**Methods:** Protein-protein docking, MD refinement, interface analysis
**Duration:** 100 ns MD
**Cost:** $100-150

## Visualization

```python
from src.visualization import MDVisualizer
import nglview

# Interactive 3D visualization in Jupyter
viz = MDVisualizer()

# Load and display
view = viz.show_trajectory(
    topology='system.pdb',
    trajectory='trajectory.xtc',
    selection='protein'
)

# Color by secondary structure
view.add_cartoon(selection='protein', color='ss')

# Show ligand as spheres
view.add_spacefill(selection='resname LIG')

# Animate
view.player.delay = 100  # ms between frames

# Save movie
viz.render_movie(
    topology='system.pdb',
    trajectory='trajectory.xtc',
    output='movie.mp4',
    resolution=(1920, 1080)
)
```

## Troubleshooting

### Simulation Instabilities

```python
# Check for common issues
from src.diagnostics import SimulationDiagnostics

diag = SimulationDiagnostics('trajectory.xtc')

# Energy conservation (NVE ensemble)
diag.check_energy_drift()

# Temperature fluctuations
diag.check_temperature_stability()

# Pressure fluctuations
diag.check_pressure_stability()

# Detect crashes or artifacts
diag.check_for_explosions()
```

### Performance Optimization

```bash
# GROMACS tuning
gmx mdrun -ntomp 1 -nb gpu -pme gpu -npme 1 -dlb yes -tunepme yes

# Check GPU utilization
nvidia-smi -l 1

# Profile simulation
gmx mdrun -deffnm run -ntomp 1 -nb gpu -timings
```

## Best Practices

1. **System preparation:** Use reliable force fields (AMBER, CHARMM, OPLS)
2. **Equilibration:** Gradually heat and equilibrate (don't rush!)
3. **Validation:** Compare to experimental data (X-ray, NMR, biochemical assays)
4. **Convergence:** Run multiple independent simulations
5. **Error bars:** Bootstrap or block averaging for uncertainties
6. **Cost control:** Start small, scale up after validation
7. **Reproducibility:** Save all parameters, random seeds, software versions

## References

### Software Documentation

- **GROMACS:** http://manual.gromacs.org/
- **AMBER:** https://ambermd.org/Manuals.php
- **OpenMM:** http://docs.openmm.org/
- **MDAnalysis:** https://docs.mdanalysis.org/
- **PyMOL:** https://pymolwiki.org/

### Key Papers

1. Abraham et al. (2015). "GROMACS: High performance molecular simulations." *SoftwareX*
2. Case et al. (2021). "AmberTools." *J. Chem. Inf. Model.*
3. Eastman et al. (2017). "OpenMM 7: Rapid development of high performance algorithms." *PLoS Comp. Biol.*
4. Shirts & Pande (2005). "Solvation free energies of amino acid side chain analogs." *J. Chem. Phys.*
5. Laio & Parrinello (2002). "Escaping free-energy minima." *PNAS*

## Next Steps

1. Deploy CloudFormation stack
2. Run test simulation (ubiquitin, 1 ns)
3. Validate output and check performance
4. Scale to production workloads
5. Implement custom analysis pipelines

---

**Tier 1 Project Status:** Production-ready
**Estimated Setup Time:** 4-6 hours
**Typical Simulation (50 ns):** 2-24 hours depending on system size
**Cost per simulation:** $15-150

For questions or issues, consult GROMACS documentation or AWS support.
