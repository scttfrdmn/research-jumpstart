# Research Jumpstart: Tier Structure Design

**Version:** 2.0 (4-Tier Model)
**Updated:** 2025-11-13

## Overview

The Research Jumpstart project uses a 4-tier progression model (Tier 0-3) that guides researchers from quick demos to production-scale AWS infrastructure:

```
Tier 0 → Tier 1 → Tier 2 → Tier 3
(10-30 min) → (1-2 hours) → (2-4 hours) → (4-5 days)
Free → Free → AWS Starter → AWS Production
Colab/Studio Lab → Studio Lab → Simple AWS → Full CloudFormation
```

## Tier Definitions

### Tier 0: Quick Start Demo
**Duration:** 10-30 minutes
**Environment:** Google Colab OR SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Goal:** Immediate value demonstration

**Characteristics:**
- Single Jupyter notebook (`.ipynb`)
- Runs in free cloud notebooks (Colab, Studio Lab)
- Uses public datasets only
- No AWS infrastructure required
- Focuses on core research concepts
- "Click and run" experience
- Demonstrates value in under 30 minutes

**Example Content:**
- Load public dataset
- Run basic analysis
- Create visualizations
- Show research insights
- Educational explanations

**Technologies:**
- Python, pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy, statsmodels
- Public APIs (no authentication needed)
- Pre-trained models from HuggingFace, torchvision
- No cloud storage (ephemeral data only)

**Deliverables:**
- Single `.ipynb` notebook
- Minimal README with "Open in Colab" badge
- Inline documentation and explanations

---

### Tier 1: SageMaker Studio Lab Advantages
**Duration:** 1-2 hours
**Environment:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Goal:** Showcase Studio Lab advantages over Colab

**Characteristics:**
- Multiple notebooks or small project structure
- Demonstrates Studio Lab advantages:
  - **Better compute:** Longer sessions, better specs
  - **Persistence:** Projects saved between sessions
  - **Collaboration:** Better sharing and team features
- Still no AWS infrastructure (S3, Lambda, etc.)
- Slightly larger datasets (can be stored in Studio Lab)
- More complex analysis workflows
- Can use Studio Lab's persistent storage (15GB)

**Example Content:**
- Multi-notebook workflows
- Larger dataset analysis (stored in Studio Lab)
- Longer-running computations
- Save/load model checkpoints
- Team collaboration examples
- Persistent environment setup

**Technologies:**
- Same as Tier 0, plus:
- Persistent storage for datasets/models
- Multi-file project structure
- Virtual environment management
- Git integration
- Slightly heavier libraries (can persist installs)

**Deliverables:**
- Multiple `.ipynb` notebooks (2-4)
- Simple Python modules for shared code
- README with Studio Lab setup instructions
- Data download/preparation scripts
- Saved model checkpoints

**Restrictions:**
- NO AWS SDK (boto3) calls
- NO S3, Lambda, or other AWS services
- Must work entirely within Studio Lab environment
- No AWS account or credentials needed

---

### Tier 2: AWS Starter (formerly Tier 3)
**Duration:** 2-4 hours
**Environment:** AWS account required
**Cost:** $5-15
**Goal:** Introduction to AWS services

**Characteristics:**
- Uses AWS services (S3, Lambda, etc.)
- Simple AWS infrastructure
- Manual setup or simple deployment scripts
- Boto3 for AWS SDK access
- Small-scale datasets on S3
- Serverless functions
- No CloudFormation (yet)

**Example Content:**
- Upload data to S3
- Process data with Lambda functions
- Query data with Athena
- Store results in DynamoDB
- Send notifications with SNS
- Basic cost optimization

**Technologies:**
- boto3 (AWS SDK)
- S3, Lambda, Athena, DynamoDB, SNS, SQS
- CloudWatch for monitoring
- IAM for permissions (manual setup)

**Deliverables:**
- Jupyter notebooks + Python scripts
- AWS setup instructions (manual steps)
- Simple deployment scripts (Python/Bash)
- Cost estimation
- Cleanup instructions

---

### Tier 3: Production Flagship (formerly Tier 1)
**Duration:** 4-5 days
**Environment:** AWS account required
**Cost:** $50-500/month (depends on scale)
**Goal:** Production-ready research infrastructure

**Characteristics:**
- Full CloudFormation infrastructure-as-code
- Multiple AWS services integrated
- Scalable architecture (1K-1M+ data points)
- Distributed computing (Batch, EMR, Dask)
- Database integration (RDS, DynamoDB)
- Machine learning pipelines (SageMaker)
- Monitoring and alerting
- Security best practices
- Cost optimization strategies

**Example Content:**
- Complete CloudFormation templates
- Distributed data processing
- ML model training at scale
- Real-time data ingestion
- Automated workflows
- Production monitoring
- Disaster recovery

**Technologies:**
- All AWS services
- CloudFormation (500-1000+ lines)
- Distributed computing frameworks
- ML frameworks at scale
- Monitoring and observability
- Security hardening

**Deliverables:**
- CloudFormation templates
- Complete Python package structure
- Comprehensive documentation
- Cost estimates by scale
- Deployment automation
- Monitoring dashboards
- Multiple application examples

---

## Learning Path

### For Complete Beginners
1. **Start with Tier 0:** Run a quick demo in Colab (no setup)
2. **Move to Tier 1:** Try SageMaker Studio Lab (free, better features)
3. **Consider Tier 2:** If you need AWS services (requires budget)
4. **Scale to Tier 3:** When ready for production (requires time + budget)

### For AWS-Ready Researchers
1. **Skip Tier 0/1:** If already comfortable with notebooks
2. **Start at Tier 2:** Learn AWS basics
3. **Deploy Tier 3:** Production infrastructure

### For Production Teams
1. **Deploy Tier 3 directly:** CloudFormation templates ready to go
2. **Reference Tier 2:** For understanding component services
3. **Share Tier 0/1:** With students and new team members

---

## Migration Path

### Current State (January 2025)
- 21 domains × 3 tiers = 63 projects
- Tier 3 (Starter): 2-4 hours, simple AWS
- Tier 2 (Complete): 2-3 days, moderate AWS
- Tier 1 (Flagship): 4-5 days, full CloudFormation

### New State (Target)
- 21 domains × 4 tiers = 84 projects
- Tier 0 (Quick): 10-30 min, Colab/Studio Lab
- Tier 1 (Studio Lab): 1-2 hours, Studio Lab only
- Tier 2 (Starter): 2-4 hours, simple AWS (renamed from Tier 3)
- Tier 3 (Flagship): 4-5 days, full CloudFormation (renamed from Tier 1)

### Renaming Strategy
```
OLD → NEW
projects/domain/project-name/tier-3/ → projects/domain/project-name/tier-2/
projects/domain/project-name/tier-1/ → projects/domain/project-name/tier-3/
```

**New additions:**
- `projects/domain/project-name/tier-0/` (NEW)
- `projects/domain/project-name/tier-1/` (NEW)

---

## Project Count

| Tier | Count | Total Duration | Status |
|------|-------|----------------|--------|
| Tier 0 | 21 | 3.5-10.5 hours | TO CREATE |
| Tier 1 | 21 | 21-42 hours | TO CREATE |
| Tier 2 | 21 | 42-84 hours | EXISTS (rename from Tier 3) |
| Tier 3 | 21 | 84-105 days | EXISTS (rename from Tier 1) |
| **Total** | **84** | | |

**New projects to create:** 42 (21 Tier 0 + 21 Tier 1)

---

## File Structure

### Tier 0 Structure
```
projects/domain/project-name/tier-0/
├── README.md                    # Quick overview
├── quick-demo.ipynb             # Single notebook
└── requirements.txt             # Python dependencies
```

### Tier 1 Structure
```
projects/domain/project-name/tier-1/
├── README.md                    # Studio Lab setup guide
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_analysis.ipynb
│   └── 03_visualization.ipynb
├── src/
│   ├── __init__.py
│   └── utils.py                 # Shared utilities
├── requirements.txt
└── environment.yml              # Conda environment
```

### Tier 2 Structure (Existing Tier 3)
```
projects/domain/project-name/tier-2/
├── README.md
├── notebooks/
├── src/
├── scripts/                     # AWS setup scripts
├── requirements.txt
└── aws-setup-guide.md
```

### Tier 3 Structure (Existing Tier 1)
```
projects/domain/project-name/tier-3/
├── README.md
├── unified-studio/
│   ├── README.md               # Comprehensive guide
│   ├── cloudformation/
│   │   ├── stack.yml           # Infrastructure
│   │   └── parameters.json
│   ├── src/
│   ├── requirements.txt
│   └── setup.py
└── cloudformation/ (symlink)
```

---

## Implementation Plan

### Phase 1: Design (CURRENT)
- ✅ Define tier structure
- ✅ Document characteristics
- ✅ Plan migration strategy

### Phase 2: Create Tier 0 Projects
- 21 quick demo notebooks (10-30 min each)
- Colab-compatible, Studio Lab-compatible
- No AWS dependencies

### Phase 3: Create Tier 1 Projects
- 21 Studio Lab projects (1-2 hours each)
- Showcase persistence, compute, collaboration
- Still no AWS integration

### Phase 4: Rename Existing Tiers
- Rename Tier 3 → Tier 2
- Rename Tier 1 → Tier 3
- Update all documentation

### Phase 5: Update Roadmap
- Update FULL_MATRIX_ROADMAP.md
- Update all READMEs
- Create migration guide for existing users

---

## Success Metrics

### Tier 0
- Runs in under 30 minutes
- Zero setup required
- Clear educational value
- Works on Colab AND Studio Lab

### Tier 1
- Demonstrates Studio Lab advantages
- Takes 1-2 hours to complete
- Shows persistent storage benefits
- Highlights collaboration features
- Still $0 cost (no AWS account needed)

### Tier 2
- Uses basic AWS services
- Costs under $15
- Manual setup is manageable
- Clear path to Tier 3

### Tier 3
- Production-ready
- Fully automated deployment
- Comprehensive documentation
- Scales to research needs

---

## Questions & Decisions

### Why 4 tiers instead of 3?
- **Accessibility:** Tier 0 removes all barriers (no account, no cost, no setup)
- **Free tier optimization:** Tier 1 shows Studio Lab advantages over Colab
- **Progressive complexity:** Each tier builds on previous knowledge
- **Clear value demonstration:** Researchers see value in minutes, not days

### Why separate Colab and Studio Lab tiers?
- **Tier 0:** Both platforms, maximum accessibility
- **Tier 1:** Studio Lab only, shows competitive advantages
- **Educational path:** Guides users toward Studio Lab without forcing it

### Why no AWS integration in Tier 1?
- **Cost barrier:** Keeps Tier 1 at $0
- **Focus on compute:** Showcases Studio Lab features, not AWS features
- **Clear distinction:** Tier 1 = free, Tier 2 = AWS begins

---

## Next Steps

1. ✅ Complete this design document
2. Create first Tier 0 project (Climate Science quick demo)
3. Create first Tier 1 project (Climate Science Studio Lab)
4. Get user feedback on approach
5. Scale to all 21 domains
6. Execute renaming migration
7. Update all documentation
