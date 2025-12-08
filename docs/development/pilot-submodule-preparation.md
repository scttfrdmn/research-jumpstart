# Pilot Submodule Content Preparation

**Date:** 2025-12-07
**Status:** Preparation phase for pilot migration
**Pilot Projects:** Astronomy (exoplanet), Climate (LSTM), Genomics (variant calling)

---

## Repository Names

1. **Astronomy:** `rj-astronomy-exoplanet-detection-tier0`
2. **Climate:** `rj-climate-temperature-forecasting-tier0`
3. **Genomics:** `rj-genomics-variant-calling-tier0`

---

## Content Structure for Each Submodule

Each submodule repository will contain:

```
rj-{domain}-{project}-tier0/
├── README.md                    # Repo-specific README with DOI, quick start
├── LICENSE                      # Apache 2.0
├── .gitignore                   # Python, Jupyter, data files
├── requirements.txt             # Python dependencies
├── {notebook-name}.ipynb        # Main Jupyter notebook
└── data/                        # (Optional) Sample data or download scripts
    └── README.md
```

---

## Pilot 1: Astronomy - Exoplanet Detection

### Repository: `rj-astronomy-exoplanet-detection-tier0`

**Source files (from main repo):**
- `projects/astronomy/sky-survey/tier-0/exoplanet-transit-detection.ipynb`
- `projects/astronomy/sky-survey/tier-0/requirements.txt`
- `projects/astronomy/sky-survey/tier-0/README.md` (needs modification)

**New files to create:**
- `LICENSE` (Apache 2.0)
- `.gitignore`

**README.md modifications needed:**
- Add Zenodo DOI badge section (placeholder for DOI minting)
- Update GitHub URLs to point to submodule repo
- Add "Part of Research Jumpstart" footer with link to main repo
- Update Colab/Studio Lab launch buttons to point to submodule repo

**Content summary:**
- Duration: 60-90 minutes
- Dataset: ~1.5GB TESS light curves
- Method: CNN for transit detection
- Performance: 95% accuracy, 90% precision, 85% recall

---

## Pilot 2: Climate - Temperature Forecasting (NEW AI/ML)

### Repository: `rj-climate-temperature-forecasting-tier0`

**Source files (from main repo):**
- `projects/climate-science/ensemble-analysis/tier-0-ml/climate-temperature-forecasting-lstm.ipynb`
- `projects/climate-science/ensemble-analysis/tier-0-ml/requirements.txt`
- `projects/climate-science/ensemble-analysis/tier-0-ml/README.md` (needs modification)

**New files to create:**
- `LICENSE` (Apache 2.0)
- `.gitignore`

**README.md modifications needed:**
- Add Zenodo DOI badge section
- Update GitHub URLs to point to submodule repo
- Update Colab/Studio Lab launch buttons
- Add "Part of Research Jumpstart" footer

**Content summary:**
- Duration: 60-90 minutes
- Dataset: NOAA GISTEMP monthly temperature data (1880-2024)
- Method: LSTM encoder-decoder for multi-step forecasting
- Output: 6-year forecast (2025-2030) with uncertainty quantification

**Note:** This is a NEW AI/ML project created during pilot preparation.

---

## Pilot 3: Genomics - Variant Calling

### Repository: `rj-genomics-variant-calling-tier0`

**Source files (from main repo):**
- `projects/genomics/variant-analysis/tier-0/genomics-variant-calling.ipynb`
- `projects/genomics/variant-analysis/tier-0/requirements.txt`
- `projects/genomics/variant-analysis/tier-0/README.md` (needs modification)

**New files to create:**
- `LICENSE` (Apache 2.0)
- `.gitignore`

**README.md modifications needed:**
- Add Zenodo DOI badge section
- Update GitHub URLs to point to submodule repo
- Update Colab/Studio Lab launch buttons
- Add "Part of Research Jumpstart" footer

**Content summary:**
- Duration: 60-90 minutes
- Dataset: ~1.5GB 1000 Genomes BAM files (chromosome 20 subset)
- Method: CNN for variant calling from pileup tensors
- Performance: Comparable to GATK HaplotypeCaller

---

## Standard README Template for Submodules

Each submodule README should follow this structure:

```markdown
# {Project Title}

[![DOI](https://zenodo.org/badge/DOI/{ZENODO_DOI}.svg)](https://doi.org/{ZENODO_DOI})
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/rj-{domain}-{project}-tier0/blob/main/{notebook}.ipynb)

**Duration:** {60-90 minutes}
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)

## Overview

{Brief project description - 2-3 sentences}

## Quick Start

### Run in Google Colab
{Colab instructions}

### Run in SageMaker Studio Lab
{Studio Lab instructions}

### Run Locally
{Local setup instructions}

## What You'll Learn

{Bullet points of key learning outcomes}

## Dataset

{Dataset description with size, source, format}

## Requirements

{Python version and key dependencies}

## Key Results

{Expected outcomes and performance metrics}

## Next Steps

This project is **Tier 0** in the Research Jumpstart framework. For more advanced workflows:

- **Tier 1:** {Brief tier-1 description} - [Learn more](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/{domain}/{project}/tier-1)
- **Tier 2:** {Brief tier-2 description} - [Learn more](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/{domain}/{project}/tier-2)
- **Tier 3:** {Brief tier-3 description} - [Learn more](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/{domain}/{project}/tier-3)

## Citation

If you use this project in your research, please cite:

\`\`\`bibtex
@software{rj_{domain}_{project}_tier0,
  title = {{Project Title}},
  author = {Research Jumpstart Community},
  year = {2025},
  doi = {{ZENODO_DOI}},
  url = {https://github.com/scttfrdmn/rj-{domain}-{project}-tier0}
}
\`\`\`

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
```

---

## Standard .gitignore

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Data files (users download locally)
*.csv
*.fits
*.bam
*.bai
*.vcf
*.h5
*.hdf5
*.zarr
*.nc
*.npz

# Model checkpoints
*.pt
*.pth
*.ckpt
*.pb
*.h5
*.keras

# Environment
.venv/
venv/
env/
ENV/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
```

---

## LICENSE Template (Apache 2.0)

```
Copyright 2025 Research Jumpstart Community

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Preparation Checklist

### Before Creating GitHub Repos:

- [x] Identify source files for each pilot project
- [x] Create new AI/ML notebook for climate (LSTM)
- [ ] Prepare modified READMEs with:
  - [ ] Zenodo DOI placeholder badges
  - [ ] Updated GitHub URLs (point to submodule repos)
  - [ ] Updated Colab/Studio Lab launch buttons
  - [ ] "Part of Research Jumpstart" footer
- [ ] Create LICENSE files (Apache 2.0)
- [ ] Create .gitignore files
- [ ] Verify all notebooks run successfully

### After Creating GitHub Repos:

- [ ] Push content to each submodule repo
- [ ] Test Colab/Studio Lab launch buttons
- [ ] Create v1.0.0 release tags
- [ ] Enable Zenodo integration
- [ ] Mint DOIs via Zenodo
- [ ] Update README badges with actual DOIs
- [ ] Add submodules to main research-jumpstart repo
- [ ] Update main repo tier-0 READMEs to reference submodules
- [ ] Test cloning with `--recurse-submodules`

---

## GitHub Repository Creation Commands

Once preparation is complete, create repos with:

```bash
# 1. Create repos on GitHub (via web interface or gh CLI)
gh repo create scttfrdmn/rj-astronomy-exoplanet-detection-tier0 --public --description "Exoplanet transit detection with TESS using CNNs (Tier-0)"
gh repo create scttfrdmn/rj-climate-temperature-forecasting-tier0 --public --description "Climate temperature forecasting with LSTM deep learning (Tier-0)"
gh repo create scttfrdmn/rj-genomics-variant-calling-tier0 --public --description "Variant calling with deep learning on 1000 Genomes data (Tier-0)"

# 2. Clone and populate each repo
# (Detailed steps in separate migration script)

# 3. Add as submodules to main repo
cd /path/to/research-jumpstart
git submodule add https://github.com/scttfrdmn/rj-astronomy-exoplanet-detection-tier0.git projects/astronomy/sky-survey/tier-0-submodule
git submodule add https://github.com/scttfrdmn/rj-climate-temperature-forecasting-tier0.git projects/climate-science/ensemble-analysis/tier-0-ml-submodule
git submodule add https://github.com/scttfrdmn/rj-genomics-variant-calling-tier0.git projects/genomics/variant-analysis/tier-0-submodule
```

---

## Next Steps

1. **Complete README modifications** for all 3 pilots
2. **Create LICENSE and .gitignore** files
3. **Test all notebooks** to ensure they run successfully
4. **Create GitHub repositories** (requires user action - can't be automated in Claude Code)
5. **Populate submodule repos** with prepared content
6. **Integrate submodules** into main research-jumpstart repo
7. **Set up Zenodo** and mint DOIs
8. **Validate workflow** end-to-end
9. **Document learnings** for scaling to remaining 22 projects

---

**Status:** Ready for README modifications and file preparation
**Blocker:** GitHub repository creation requires user action (gh CLI or web interface)
