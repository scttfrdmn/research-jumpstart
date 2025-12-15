# Marimo Integration Strategy

## Overview

Research Jumpstart will adopt **marimo** (reactive Python notebooks) alongside traditional Jupyter notebooks to provide modern, reproducible, git-friendly notebook experiences. This document outlines the integration strategy across different tiers.

## What is Marimo?

[marimo](https://marimo.io) is an open-source reactive Python notebook that addresses key limitations of traditional Jupyter notebooks:

- **Reactive execution**: Cells automatically update when dependencies change
- **Stored as pure Python**: `.py` files instead of JSON (git-friendly)
- **No hidden state**: Deterministic execution eliminates common notebook bugs
- **Three tools in one**: Notebook, script, and deployable web app
- **Built-in interactivity**: UI widgets without callbacks

**Why Marimo for Research Jumpstart?**
- Version control friendly (pure Python files)
- Reproducible research (no hidden state)
- Modern developer experience
- Aligns with MLOps best practices
- Open source (Apache 2.0 license)

## Tier-Based Strategy

### Tier 0: Jupyter Only

**Decision**: Keep Jupyter notebooks exclusively

**Rationale**:
- Maximum accessibility via Google Colab one-click launch
- Colab doesn't support Marimo natively
- Target audience (students/beginners) prioritize ease of access
- Already validated with 3 pilot projects

**Format**: `.ipynb` files only

**No action required** - existing tier-0 notebooks remain unchanged.

---

### Tier 1: Dual Format (Jupyter + Marimo Equally)

**Decision**: Provide both Jupyter and Marimo versions

**Rationale**:
- SageMaker Studio Lab supports Marimo via jupyter-server-proxy
- Persistent environments (4-8 hour sessions) match Marimo's workflow
- More serious researchers value reproducibility
- Gives users choice based on their needs

**Repository Structure**:
```
tier-1-project/
├── analysis.ipynb              # Traditional Jupyter notebook
├── analysis.marimo.py          # Marimo version (same analysis)
├── README.md                   # Documents both options
├── requirements.txt            # Shared dependencies
└── setup-marimo.sh            # Helper script for Studio Lab setup
```

**README Pattern**:
````markdown
## Run This Tutorial

### Option 1: Jupyter Notebook (SageMaker Studio Lab)
[![Open in Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](link-to-notebook)

Traditional Jupyter experience with persistent storage.

### Option 2: Marimo Notebook (SageMaker Studio Lab + Local)

**On SageMaker Studio Lab:**
```bash
# First time setup
pip install marimo jupyter-server-proxy
jupyter serverextension enable --py jupyter_server_proxy

# Start marimo
marimo edit analysis.marimo.py --host 0.0.0.0 --port 8888

# Access at: /proxy/8888/
```

**Local:**
```bash
pip install marimo
marimo edit analysis.marimo.py
# Opens at localhost:2718
```

**Why Marimo?**
- Reactive cells (automatic updates)
- Git-friendly (pure Python)
- Run as script: `python analysis.marimo.py`
````

**Implementation Plan**:
1. Create 1-2 Tier-1 Marimo prototypes
2. Test jupyter-server-proxy workflow on Studio Lab
3. Document lessons learned
4. Scale to all Tier-1 projects

---

### Tier 2-3: Decision Deferred

**Status**: No decision at this time

**Considerations for future**:
- Tier 2: AWS CloudFormation stacks - could benefit from Marimo's script execution
- Tier 3: Production workflows - Marimo's deployment features may be valuable
- Will revisit after Tier 1 implementation experience

---

## Technical Implementation

### Running Marimo on SageMaker Studio Lab

**Key Technology**: `jupyter-server-proxy`

This Jupyter extension allows accessing web applications (like Marimo's server) running on custom ports within SageMaker environments.

#### Setup Instructions

**1. Install Dependencies**
```bash
pip install marimo jupyter-server-proxy
```

**2. Enable jupyter-server-proxy**
```bash
jupyter serverextension enable --py jupyter_server_proxy
```

**3. Start Marimo Server**
```bash
marimo edit notebook.py --host 0.0.0.0 --port 8888
```

**4. Access Marimo UI**
Navigate to: `https://<your-domain>.studio.<region>.sagemaker.aws/jupyter/default/proxy/8888/`

#### Lifecycle Configuration (Optional)

For automated setup in SageMaker Studio, create a lifecycle configuration:

```bash
#!/bin/bash
set -eux

echo "Installing marimo and jupyter-server-proxy..."

pip install --upgrade pip
pip install marimo jupyter-server-proxy

# Enable proxy extension
jupyter serverextension enable --py jupyter_server_proxy --sys-prefix

# Create helper script
cat > /home/sagemaker-user/start-marimo.sh << 'EOF'
#!/bin/bash
PORT=${PORT:-8888}
NOTEBOOK=${1:-""}

if [ -z "$NOTEBOOK" ]; then
    echo "Starting marimo editor on port $PORT..."
    marimo edit --host 0.0.0.0 --port $PORT
else
    echo "Starting marimo with notebook: $NOTEBOOK on port $PORT..."
    marimo edit "$NOTEBOOK" --host 0.0.0.0 --port $PORT
fi
EOF

chmod +x /home/sagemaker-user/start-marimo.sh

echo "Marimo installation complete!"
```

### Converting Between Formats

**Jupyter → Marimo**:
```bash
marimo convert notebook.ipynb -o notebook.marimo.py
```

**Marimo → Jupyter** (if needed):
Export from Marimo UI or use programmatic conversion

**Best Practice**: Maintain both formats with equivalent content, test both regularly.

---

## Use Cases by Tier

### Tier 0 (Jupyter Only)
- **Use Case**: Quick 60-90 min tutorials for beginners
- **Platform**: Google Colab (free GPU/TPU)
- **Why Jupyter**: One-click launch, zero setup, maximum accessibility

### Tier 1 (Dual Format)
- **Use Case**: 4-8 hour deep dives, persistent experiments
- **Platform**: SageMaker Studio Lab (free persistent storage)
- **Why Both**:
  - Jupyter: Familiar interface, Studio Lab integration
  - Marimo: Reproducibility, git-friendly, reactive updates

### Tier 2 (TBD)
- **Use Case**: 2-3 day production deployments on AWS
- **Platform**: AWS (CloudFormation/CDK)
- **Decision pending**: Based on Tier 1 experience

### Tier 3 (TBD)
- **Use Case**: Enterprise/ongoing production workloads
- **Platform**: AWS with monitoring/automation
- **Decision pending**: Based on Tier 1 & 2 experience

---

## Guidelines for Creating Marimo Notebooks

### 1. Cell Structure
```python
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, pd, np

@app.cell
def __(mo):
    mo.md("# Your Analysis Title")
    return

# More cells...

if __name__ == "__main__":
    app.run()
```

### 2. Reactive Design Principles
- Design cells to depend explicitly on other cells' outputs
- Avoid global state mutations
- Use `mo.ui` components for interactivity
- Let Marimo handle execution order

### 3. Conversion Checklist
When converting Jupyter → Marimo:
- [ ] Remove cell magic commands (`%%time`, `%matplotlib inline`)
- [ ] Extract imports to first cell
- [ ] Convert widgets to `mo.ui` components
- [ ] Test reactive execution by modifying upstream cells
- [ ] Ensure script execution works: `python notebook.py`

### 4. Testing Both Formats
```bash
# Test Jupyter
jupyter nbconvert --execute notebook.ipynb

# Test Marimo as notebook
marimo edit notebook.marimo.py

# Test Marimo as script
python notebook.marimo.py

# Test Marimo as app
marimo run notebook.marimo.py
```

---

## Marimo vs Jupyter: When to Use Each

### Use Marimo When:
✅ Building interactive dashboards
✅ Creating reproducible research
✅ Working with version control/Git
✅ Need reactive updates across cells
✅ Want to deploy as web app
✅ Creating reusable Python modules

### Use Jupyter When:
✅ Quick exploratory analysis
✅ Using SageMaker-specific features
✅ Team already invested in Jupyter ecosystem
✅ Need specific Jupyter extensions
✅ Teaching beginners (Google Colab)

**Best Practice**: Provide both options and let users choose based on their workflow.

---

## Resources

### Documentation
- [Marimo Official Docs](https://docs.marimo.io)
- [Marimo Examples](https://marimo.io/examples)
- [SageMaker Studio Lab Guide](https://studiolab.sagemaker.aws)
- [jupyter-server-proxy](https://jupyter-server-proxy.readthedocs.io/)

### Community
- [Marimo GitHub](https://github.com/marimo-team/marimo) (16k+ stars)
- [Marimo Discord](https://discord.gg/JE7nhX6mD8)

### Infrastructure as Code
- See `marimo-chat.md` for Terraform and AWS CDK examples
- Reference implementation for AWS deployment

---

## Roadmap

### Phase 1: Documentation (Current)
- [x] Document strategy
- [ ] Create setup guides
- [ ] Define quality standards

### Phase 2: Tier-1 Prototype (Next)
- [ ] Select 2 Tier-1 projects for conversion
- [ ] Create Marimo versions
- [ ] Test on SageMaker Studio Lab
- [ ] Document lessons learned

### Phase 3: Scale Tier-1 (Future)
- [ ] Convert all Tier-1 projects to dual format
- [ ] Create automated conversion pipeline
- [ ] Update all README files

### Phase 4: Evaluate Tier 2-3 (Future)
- [ ] Assess Marimo deployment features
- [ ] Test CloudFormation integration
- [ ] Make decision on Tier 2-3 strategy

---

## Success Metrics

### Tier-1 Adoption Metrics:
- **Usage**: % of users choosing Marimo vs Jupyter
- **Reproducibility**: Success rate of notebook execution
- **Git activity**: Cleaner diffs with `.py` format
- **User feedback**: Survey responses on experience

### Quality Standards:
- [ ] Both formats execute successfully
- [ ] Equivalent outputs in both formats
- [ ] Clear documentation for both paths
- [ ] Helper scripts tested on Studio Lab

---

## FAQ

**Q: Why not use Marimo for Tier 0?**
A: Google Colab doesn't support Marimo, and Tier 0 prioritizes one-click accessibility for beginners.

**Q: Can users run Marimo locally?**
A: Yes! `pip install marimo && marimo edit notebook.py` works on any machine.

**Q: Is Marimo free?**
A: Yes, Marimo is open source (Apache 2.0). The cloud service (molab) is currently free as well.

**Q: What about Modal Notebooks?**
A: Modal Notebooks are proprietary/commercial. Marimo is open source, aligning better with Research Jumpstart's philosophy.

**Q: Do we maintain two codebases?**
A: Yes, for Tier 1. The benefits (reproducibility, git-friendliness) outweigh the maintenance cost. We'll automate conversion where possible.

**Q: What if jupyter-server-proxy doesn't work?**
A: Users can always run Marimo locally or use molab (free cloud hosting for Marimo).

---

## Contact

For questions about Marimo integration:
- Create an issue in the Research Jumpstart repository
- Reference this strategy document
- Tag with `marimo` label
