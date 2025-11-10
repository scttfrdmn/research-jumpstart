# Ecological Modeling & Population Dynamics

**Difficulty**: 🟡 Intermediate | **Time**: ⏱️ 3-4 hours (Studio Lab)

Model population dynamics, species interactions, and ecosystem processes using differential equations, agent-based models, and spatial analysis.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/environmental-science/ecological-modeling/studio-lab
conda env create -f environment.yml
conda activate ecological-modeling
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Implement classic population models (logistic growth, Lotka-Volterra)
- Simulate predator-prey dynamics
- Model species competition and mutualism
- Analyze spatial ecology (metapopulation, landscape models)
- Apply agent-based modeling (ABM) for individual organisms
- Fit models to real ecological data
- Assess ecosystem stability and resilience
- Visualize population trajectories and phase planes

## Key Analyses

1. **Single Species Dynamics**
   - Exponential growth model
   - Logistic growth (carrying capacity)
   - Allee effects (critical thresholds)
   - Age-structured populations (Leslie matrices)
   - Stochastic population models

2. **Species Interactions**
   - **Predator-Prey**: Lotka-Volterra equations
   - **Competition**: Competitive exclusion principle
   - **Mutualism**: Symbiotic relationships
   - **Host-Parasite**: Disease dynamics
   - **Functional responses**: Type I, II, III

3. **Spatial Ecology**
   - **Metapopulation models**: Patch occupancy
   - **Reaction-diffusion**: Spatial spread
   - **Landscape ecology**: Habitat fragmentation
   - **Species distribution models**: SDMs
   - **Migration patterns**: Individual-based tracking

4. **Community Ecology**
   - **Food webs**: Trophic cascades
   - **Succession**: Community assembly
   - **Biodiversity**: Species richness and evenness
   - **Niche theory**: Resource partitioning
   - **Neutral theory**: Hubbell's model

5. **Ecosystem Processes**
   - **Carbon cycling**: NPP, decomposition
   - **Nutrient dynamics**: Nitrogen, phosphorus
   - **Energy flow**: Trophic efficiency
   - **Ecosystem services**: Valuation

## Sample Datasets

### Included Examples
- **Lynx-Hare data**: Historic Hudson Bay Company records
- **Daphnia population**: Laboratory microcosm data
- **Bird abundance**: North American Breeding Bird Survey
- **Tree growth**: Forest Inventory and Analysis (FIA)

### Public Datasets
- [GBIF](https://www.gbif.org/): Global Biodiversity Information Facility
- [eBird](https://ebird.org/): Citizen science bird observations
- [LTER](https://lternet.edu/): Long Term Ecological Research
- [DataONE](https://www.dataone.org/): Earth and environmental science data

## Cost

**Studio Lab**: Free forever (public datasets, simulation models)
**Unified Studio**: ~$15-30 per analysis (AWS for large spatial datasets, S3 storage)

## Prerequisites

- Basic calculus (derivatives, integrals)
- Differential equations (ODEs) helpful
- Programming basics (Python or R)
- Ecology fundamentals (populations, communities, ecosystems)

## Use Cases

- **Conservation Biology**: Endangered species recovery
- **Wildlife Management**: Harvest quotas, population control
- **Invasive Species**: Spread prediction and management
- **Fisheries**: Sustainable catch limits
- **Climate Change**: Species range shifts
- **Disease Ecology**: Epidemic modeling
- **Restoration Ecology**: Ecosystem recovery planning

## Classic Models

### Exponential Growth
```
dN/dt = rN
```
- **N**: Population size
- **r**: Intrinsic growth rate
- **Result**: Unbounded growth (unrealistic long-term)

### Logistic Growth
```
dN/dt = rN(1 - N/K)
```
- **K**: Carrying capacity
- **Result**: S-shaped curve, stabilizes at K

### Lotka-Volterra Predator-Prey
```
dV/dt = rV - aVP        (Prey)
dP/dt = baVP - mP       (Predator)
```
- **V**: Prey population
- **P**: Predator population
- **r**: Prey growth rate
- **a**: Predation rate
- **b**: Conversion efficiency
- **m**: Predator mortality
- **Result**: Oscillating populations

### Competitive Lotka-Volterra
```
dN1/dt = r1*N1*(1 - N1/K1 - α*N2/K1)
dN2/dt = r2*N2*(1 - N2/K2 - β*N1/K2)
```
- **α, β**: Competition coefficients
- **Outcomes**: Coexistence, exclusion, or unstable equilibrium

## Typical Workflow

1. **Define Research Question**: What process are we modeling?
2. **Select Model Type**:
   - Analytical (ODEs)
   - Simulation (ABM, IBM)
   - Statistical (GLM, GAM)
   - Spatial (GIS, spatial statistics)
3. **Parameterize Model**:
   - Literature values
   - Fit to empirical data
   - Expert elicitation
   - Sensitivity analysis
4. **Run Simulations**:
   - Deterministic vs stochastic
   - Time series output
   - Ensemble runs for uncertainty
5. **Analyze Results**:
   - Equilibrium points
   - Stability analysis
   - Bifurcation diagrams
   - Phase plane plots
6. **Validate Model**:
   - Compare to observed data
   - Cross-validation
   - Out-of-sample prediction
7. **Scenario Analysis**:
   - Climate change impacts
   - Management interventions
   - Sensitivity to parameters

## Model Types

### Analytical Models
- **Pros**: Exact solutions, theoretical insights
- **Cons**: Limited to simple systems
- **Tools**: SymPy (Python), Mathematica

### Agent-Based Models (ABM)
- **Pros**: Individual heterogeneity, emergent behavior
- **Cons**: Computationally intensive, many parameters
- **Tools**: Mesa (Python), NetLogo

### Individual-Based Models (IBM)
- **Focus**: Track each organism explicitly
- **Applications**: Movement, foraging, life history
- **Example**: Fish schooling, bird migration

### Spatially Explicit Models
- **Raster-based**: Grid cells with states
- **Vector-based**: Polygons (habitat patches)
- **Applications**: Landscape connectivity, invasive spread
- **Tools**: GDAL, Rasterio, GeoPandas

## Analysis Techniques

### Stability Analysis
- **Equilibrium points**: dN/dt = 0
- **Eigenvalues**: Determine stability
- **Jacobian matrix**: Linearization at equilibrium
- **Lyapunov functions**: Energy methods

### Bifurcation Analysis
- Identify parameter values where dynamics change qualitatively
- Hopf bifurcation: Transition to oscillations
- Saddle-node bifurcation: Sudden transitions

### Sensitivity Analysis
- How does output vary with parameters?
- Local sensitivity: Partial derivatives
- Global sensitivity: Sobol indices, Morris method

### Parameter Estimation
- **Maximum Likelihood**: Fit model to data
- **Bayesian**: Posterior distributions
- **ABC (Approximate Bayesian Computation)**: Simulation-based
- **Optimization**: Minimize SSE, AIC, BIC

## Example Results

### Lynx-Hare Dynamics
- **Data**: 1845-1935 Hudson Bay Company records
- **Model**: Lotka-Volterra predator-prey
- **Finding**: 9-10 year population cycles
- **Parameters**: r_hare = 0.55, a = 0.028, b = 0.84, m = 0.40
- **Fit**: R² = 0.72, oscillations captured

### Moose-Wolf Isle Royale
- **System**: Isolated island (Lake Superior)
- **Observation**: 60+ years monitoring
- **Model**: Predator-prey with Allee effect
- **Result**: Predicts extinction risk for wolf population
- **Management**: Genetic rescue via wolf introduction

### Daphnia-Algae Microcosm
- **Controlled experiment**: Laboratory population
- **Model**: Consumer-resource with time delay
- **Finding**: Chaos in consumer-resource dynamics
- **Implication**: Deterministic chaos in ecology

## Spatial Ecology Applications

### Species Distribution Models (SDMs)
- **Input**: Occurrence data (presence/absence)
- **Predictors**: Climate (temp, precip), habitat, elevation
- **Methods**: MaxEnt, GLM, Random Forest
- **Output**: Habitat suitability maps
- **Application**: Range shift predictions under climate change

### Metapopulation Models
- **Patches**: Discrete habitat fragments
- **Dynamics**: Colonization and extinction
- **Levins model**: dp/dt = cp(1-p) - ep
- **Applications**: Conservation planning, corridor design

### Landscape Connectivity
- **Metrics**: Least-cost paths, circuit theory
- **Software**: Circuitscape, Conefor
- **Applications**: Wildlife corridor identification

## Advanced Topics

- **Stochastic differential equations (SDEs)**: Environmental noise
- **Time series analysis**: ARIMA, state-space models
- **Machine learning**: Neural ODEs, physics-informed neural networks
- **Adaptive dynamics**: Evolutionary game theory
- **Eco-evolutionary dynamics**: Rapid evolution effects
- **Tipping points**: Regime shifts, early warning signals

## Software Tools

### Python Libraries
- **scipy.integrate**: ODE solvers (odeint, solve_ivp)
- **mesa**: Agent-based modeling framework
- **geopandas/rasterio**: Spatial analysis
- **scikit-learn**: SDMs, classification
- **NetworkX**: Food web analysis

### R Packages
- **deSolve**: ODE solver
- **popbio**: Population matrix models
- **vegan**: Community ecology analysis
- **dismo/biomod2**: Species distribution modeling
- **igraph**: Network analysis

### Specialized Software
- **NetLogo**: ABM platform (GUI)
- **Vensim**: System dynamics modeling
- **Circuitscape**: Landscape connectivity
- **GAMA**: Spatially explicit ABM

## Conservation Applications

### Endangered Species Recovery
- Population viability analysis (PVA)
- Minimum viable population (MVP)
- Extinction risk assessment
- Captive breeding programs

### Invasive Species Management
- Spread rate prediction
- Control effort allocation
- Early detection networks
- Eradication feasibility

### Ecosystem Restoration
- Succession modeling
- Re-introduction success
- Trophic rewilding (e.g., wolf reintroduction)

## Resources

### Datasets
- [GBIF](https://www.gbif.org/): 2+ billion species occurrences
- [LTER](https://lternet.edu/): Long-term ecological data
- [DataONE](https://www.dataone.org/): Federated data network
- [eBird](https://ebird.org/): Bird observation data

### Books
- "A Primer of Ecology" (Gotelli)
- "Mathematical Ecology" (May & McLean)
- "Theoretical Ecology" (Case)
- "Spatial Ecology" (Tilman & Kareiva)

### Online Courses
- [Coursera: Introduction to Systems Science](https://www.coursera.org/learn/systems-science)
- [DataCamp: Ecological Models in Python](https://www.datacamp.com/)

### Scientific Societies
- [Ecological Society of America (ESA)](https://www.esa.org/)
- [British Ecological Society (BES)](https://www.britishecologicalsociety.org/)

## Community Contributions Welcome

This is a Tier 3 (starter) project. Contributions welcome:
- Complete Jupyter notebook tutorial
- Additional classic models (metapopulation, SIR disease)
- Agent-based modeling examples (Mesa)
- Spatial analysis tutorials (habitat suitability)
- Parameter estimation from real data
- Stochastic modeling examples
- Climate change scenario analysis

See [PROJECT_TEMPLATE.md](../../_template/HOW_TO_USE_THIS_TEMPLATE.md) for contribution guidelines.

## License

Apache 2.0 - Sample code
Ecological datasets: Check individual licenses (often CC-BY or public domain)

*Last updated: 2025-11-09*
