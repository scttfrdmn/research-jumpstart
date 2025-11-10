# Environmental Science - Ecology Modeling

**Duration:** 3 hours | **Level:** Intermediate | **Cost:** Free

Simulate predator-prey population dynamics using the Lotka-Volterra model and differential equations.

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws)

## Overview

Model ecological interactions between predators (foxes) and prey (rabbits) using classic population dynamics equations. Learn to simulate ecosystem behavior, analyze stability, and predict long-term outcomes—fundamental skills for ecology and conservation biology.

### What You'll Build
- Lotka-Volterra ODE solver
- Population trajectory simulator
- Phase space visualizer
- Equilibrium point calculator
- Vector field plotter

### Real-World Applications
- Wildlife management
- Conservation biology
- Fisheries management
- Pest control
- Ecosystem restoration

## Learning Objectives

✅ Understand predator-prey dynamics
✅ Solve differential equations numerically
✅ Perform phase space analysis
✅ Calculate equilibrium points
✅ Analyze system stability
✅ Interpret ecological oscillations
✅ Predict long-term population trends

## Dataset

**11-Year Rabbit-Fox Population Data**

| Year | Rabbits | Foxes | Vegetation Index |
|------|---------|-------|------------------|
| 2015 | 1,000 | 40 | 80 |
| 2016 | 1,100 | 42 | 78 |
| ... | ... | ... | ... |
| 2025 | 980 | 39 | 81 |

**Ecological Context:**
- **Setting**: Temperate forest ecosystem
- **Rabbit Dynamics**: Birth rate ~0.5/year, death rate depends on predation
- **Fox Dynamics**: Death rate ~0.3/year, growth depends on prey availability
- **Vegetation**: Indirectly affects carrying capacity

**Data Characteristics:**
- Coupled oscillations (rabbits lead, foxes follow)
- Phase lag: ~1-2 years
- Quasi-periodic cycles
- Realistic parameter values for small mammals

## Methods and Techniques

### 1. Lotka-Volterra Equations

**Classic Predator-Prey Model:**

```python
def lotka_volterra(state, t, alpha, beta, gamma, delta):
    """
    R: Prey (Rabbits)
    F: Predator (Foxes)

    dR/dt = alpha*R - beta*R*F     # Prey growth - Predation
    dF/dt = delta*R*F - gamma*F     # Predator growth - Death
    """
    R, F = state
    dR_dt = alpha * R - beta * R * F
    dF_dt = delta * R * F - gamma * F
    return [dR_dt, dF_dt]
```

**Parameters:**
- **α (alpha)**: Prey birth rate (0.5/year)
- **β (beta)**: Predation rate (0.002/interaction)
- **γ (gamma)**: Predator death rate (0.3/year)
- **δ (delta)**: Predator efficiency (0.0004/conversion)

### 2. Numerical Integration

**Solving ODEs:**
```python
from scipy.integrate import odeint

t = np.linspace(0, 50, 500)  # 50 years
initial_state = [1000, 40]   # Initial populations
solution = odeint(lotka_volterra, initial_state, t, args=(alpha, beta, gamma, delta))
```

**Methods:**
- **odeint**: Adaptive step-size integration
- **Accuracy**: Handles stiff equations
- **Output**: Population trajectories over time

### 3. Phase Space Analysis

**State Space Visualization:**
- X-axis: Prey population
- Y-axis: Predator population
- Trajectory: System evolution over time
- Closed loops: Periodic oscillations
- Spiral: Approach to equilibrium

```python
plt.plot(rabbits, foxes)
plt.xlabel('Rabbits')
plt.ylabel('Foxes')
```

### 4. Equilibrium Analysis

**Fixed Points:**
```python
# Non-trivial equilibrium
R_eq = gamma / delta  # ~750 rabbits
F_eq = alpha / beta   # ~250 foxes

# At equilibrium: dR/dt = 0, dF/dt = 0
```

**Stability:**
- **Neutral stability**: Lotka-Volterra orbits
- **Perturbations**: System oscillates indefinitely
- **Realistic models**: Add carrying capacity for stability

### 5. Vector Field

**Direction Field:**
```python
# At each (R, F) point, calculate derivatives
dR = alpha*R - beta*R*F
dF = delta*R*F - gamma*F

# Plot arrows showing system evolution
plt.quiver(R_grid, F_grid, dR, dF)
```

**Interpretation:**
- Arrows show population change direction
- Magnitude shows rate of change
- Reveals equilibria and attractors

## Notebook Structure

### Part 1: Introduction (15 min)
- Ecological background
- Predator-prey relationships
- Model assumptions and limitations

### Part 2: Data Exploration (20 min)
- Load population time series
- Visualize trends
- Identify oscillation patterns
- Calculate cycle periods

### Part 3: Lotka-Volterra Model (25 min)
- Equation derivation
- Parameter interpretation
- Implement ODE function
- Ecological meaning

### Part 4: Numerical Simulation (30 min)
- Set initial conditions
- Solve ODEs with odeint
- Plot population trajectories
- Compare with data

### Part 5: Phase Space Analysis (25 min)
- Create phase portrait
- Identify closed orbits
- Initial condition sensitivity
- Multiple trajectories

### Part 6: Equilibrium Analysis (25 min)
- Calculate fixed points
- Trivial vs. non-trivial equilibria
- Stability assessment
- Ecological interpretation

### Part 7: Vector Field (20 min)
- Create direction field
- Overlay trajectories
- Visualize dynamics
- Identify nullclines

### Part 8: Advanced Topics (15 min)
- Carrying capacity extension
- Functional responses
- Three-species models
- Real-world complications

**Total:** ~2.5-3 hours

## Key Results

### Population Dynamics

**Oscillation Characteristics:**
- **Period**: ~7-8 years per cycle
- **Rabbit Amplitude**: 600-1,400 individuals
- **Fox Amplitude**: 25-55 individuals
- **Phase Lag**: Foxes lag rabbits by ~1.5 years

**Ecological Interpretation:**
1. Rabbits increase → More food for foxes
2. Foxes increase → More predation on rabbits
3. Rabbits decrease → Less food for foxes
4. Foxes decrease → Less predation, rabbits recover
5. Cycle repeats

### Equilibrium Points

**Trivial Equilibrium:**
- R = 0, F = 0 (extinction)
- Unstable (if any population exists, growth occurs)

**Non-Trivial Equilibrium:**
- R* = 750 rabbits
- F* = 250 foxes
- Neutrally stable (orbits around equilibrium)

**Average Populations:**
- Mean rabbits: ~1,000 (varies by orbit)
- Mean foxes: ~40 (varies by orbit)

### Stability Analysis

**Lotka-Volterra System:**
- No damping → Oscillations persist indefinitely
- Initial conditions determine orbit size
- Conserved quantity: H(R,F) = constant

**Realistic Extensions:**
- Add carrying capacity → Limit cycles (stable oscillations)
- Add Allee effect → Minimum viable population
- Add refuge → Prey can hide from predators

## Visualizations

1. **Time Series Plot**: Rabbit and fox populations over time
2. **Phase Portrait**: Closed orbit in R-F space
3. **Vector Field**: Direction field with nullclines
4. **Multiple Trajectories**: Different initial conditions
5. **Parameter Sensitivity**: Effect of changing α, β, γ, δ
6. **Comparison Plot**: Model vs. observed data
7. **3D Surface**: Interaction strength across parameter space

## Extensions

### Modify the Model
- Change parameters and observe effects
- Add carrying capacity: dR/dt = αR(1 - R/K) - βRF
- Include prey refuge: dR/dt = αR - βR(F-F₀)
- Functional response: Type II (saturation)

### Additional Species
- Three-species model: Grass → Rabbit → Fox
- Competition: Two prey, one predator
- Mutualism: Both species benefit
- Parasitism: Host-parasite dynamics

### Real Ecological Data
- [GPDD](https://www.imperial.ac.uk/cpb/gpdd2/): Global Population Dynamics Database
- [Lynx-Hare Data](https://www.nceas.ucsb.edu/): Hudson's Bay Company records
- State wildlife surveys
- Camera trap population estimates

### Advanced Methods
- Stochastic models: Add randomness
- Spatial models: Metapopulations
- Agent-based models: Individual-based
- Parameter estimation: Fit to real data

## Scientific Background

### Lotka-Volterra History

**Alfred Lotka (1925)** and **Vito Volterra (1926)** independently developed these equations to explain:
- Oscillations in fish catch data (Adriatic Sea)
- Predator-prey cycles in nature
- Mathematical ecology foundations

**Classic Example**: Canadian lynx and snowshoe hare (Hudson's Bay Company data, 1845-1935)

### Model Assumptions

✅ **Exponential prey growth** (no carrying capacity)
✅ **Mass action predation** (random encounters)
✅ **Linear predator growth** (prey converts to predators)
✅ **Constant parameters** (no seasonality)
✅ **Well-mixed population** (no spatial structure)

⚠️ **Limitations**: Real ecosystems more complex

### Extensions

**Type II Functional Response:**
```python
predation_rate = beta * R / (1 + beta*h*R)  # Saturation
```

**Carrying Capacity:**
```python
dR_dt = alpha*R*(1 - R/K) - beta*R*F
```

**Allee Effect:**
```python
dR_dt = alpha*R*(R/A - 1) - beta*R*F  # Minimum viable population
```

## Resources

### Textbooks
- *Ecology: Concepts and Applications* by Molles
- *Mathematical Biology* by Murray
- *A Primer of Ecology* by Gotelli

### Software
- **R**: `deSolve` package for ODEs
- **MATLAB**: `ode45` solver
- **NetLogo**: Agent-based ecosystem models
- **Python**: scipy.integrate

### Datasets
- [GPDD](https://www.imperial.ac.uk/cpb/gpdd2/): Population time series
- [Living Planet Index](https://livingplanetindex.org/): Wildlife trends
- [eBird](https://ebird.org/): Bird population data

### Online Resources
- [EcoLab Models](https://ecolab.psu.edu/): Interactive simulations
- [Virtual Population Lab](https://www.iucnredlist.org/): Conservation tool

## Getting Started

```bash
cd projects/environmental-science/ecology-modeling/studio-lab

conda env create -f environment.yml
conda activate ecology

jupyter lab quickstart.ipynb
```

## FAQs

??? question "Why do populations oscillate?"
    Predator growth lags prey growth due to reproduction time. This lag creates out-of-phase oscillations.

??? question "Do real populations cycle like this?"
    Some do (lynx-hare, lemming-fox), but many don't. Real systems have more factors: weather, disease, competition, spatial structure.

??? question "What causes extinction?"
    In basic model: impossible (protected by equations). Realistic models add stochasticity, Allee effects, or environmental catastrophes.

??? question "How do I fit this to my data?"
    Use parameter optimization (scipy.optimize) to minimize difference between model and data.

---

**[Launch the notebook →](https://studiolab.sagemaker.aws)** 🌱
