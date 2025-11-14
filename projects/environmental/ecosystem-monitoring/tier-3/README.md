# Unified Studio Production Version

This directory contains the production-ready version of Multi-Sensor Environmental Monitoring, designed for SageMaker Unified Studio with full AWS integration.

## What's Included

### Notebooks
- `01_data_access.ipynb` - Direct access to satellite archives on S3
- `02_classification.ipynb` - Production land cover classification
- `03_change_detection.ipynb` - Temporal analysis and change detection
- `04_bedrock_integration.ipynb` - AI-assisted interpretation with Claude

### Source Code
- `src/data_access.py` - S3 utilities and data loading
- `src/classification.py` - Classification algorithms
- `src/change_detection.py` - Change detection methods
- `src/visualization.py` - Publication-quality plotting
- `src/bedrock_client.py` - Amazon Bedrock integration

### Infrastructure
- `cloudformation/environmental-stack.yml` - Infrastructure as code
- `cloudformation/parameters.json` - Stack configuration

## Prerequisites

- AWS account with billing enabled
- SageMaker Unified Studio domain
- Familiarity with Studio Lab version (complete it first!)

## Setup

### 1. Deploy Infrastructure

```bash
cd cloudformation

aws cloudformation create-stack \
  --stack-name environmental-monitoring \
  --template-body file://environmental-stack.yml \
  --parameters file://parameters.json \
  --capabilities CAPABILITY_IAM

# Wait for stack creation (10-15 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name environmental-monitoring
```

### 2. Launch Unified Studio

1. Open AWS Console → SageMaker Unified Studio
2. Navigate to `environmental-monitoring` domain
3. Launch JupyterLab environment (ml.t3.xlarge)

### 3. Clone Repository

```bash
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/environmental/ecosystem-monitoring/unified-studio
```

### 4. Install Dependencies

```bash
conda env create -f environment.yml
conda activate environmental-production
```

### 5. Run Notebooks

Execute notebooks in order:
1. Data access
2. Classification
3. Change detection
4. Bedrock integration

## What's Different from Studio Lab

### Real Data Access
- Direct S3 access to Landsat/Sentinel archives
- No data download or storage needed
- Process hundreds of scenes
- Any region, any time period

### Production Scale
- Multi-sensor fusion (optical + radar + LiDAR)
- Continental-scale analysis possible
- Distributed processing with EMR
- Automated workflows

### AI Integration
- Claude 3 via Amazon Bedrock
- Automated report generation
- Result interpretation
- Literature context

### Collaboration
- Shared environments
- Version controlled outputs
- Team workflows
- Reproducible research

## Cost Estimate

Typical analysis (50km × 50km, 5 years, 3 sensors):
- Compute: $0.45-0.60 (3-4 hours)
- Storage: $0.10-0.20/month
- Bedrock: $2-4 per analysis
- EMR (if needed): $10-15
- **Total: $13-20 per analysis**

## Architecture

```
SageMaker Unified Studio
├── JupyterLab (ml.t3.xlarge)
│   └── Notebooks + source code
├── S3 Data Access
│   ├── Landsat (s3://landsat-pds)
│   ├── Sentinel-2 (s3://sentinel-2-l2a)
│   └── Sentinel-1 (s3://sentinel-1-l1c)
├── Processing (optional EMR)
│   └── Distributed geospatial processing
├── Bedrock
│   └── Claude 3 for interpretation
└── S3 Output
    └── Results, figures, reports
```

## Troubleshooting

### S3 Access Issues
```bash
# Check IAM role permissions
aws iam get-role --role-name SageMakerUnifiedStudioRole

# Required policies:
# - AmazonS3ReadOnlyAccess (for public data)
# - AmazonBedrockFullAccess
```

### Cost Management
```bash
# Set up billing alerts
aws budgets create-budget \
  --account-id ACCOUNT_ID \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

### Performance Optimization
- Use spot instances for EMR (60-80% savings)
- Cache frequently-used imagery subsets
- Process multiple regions in single run
- Delete intermediate results

## Next Steps

- Explore extension ideas in [main README](../README.md#extension-ideas)
- Scale to larger regions
- Add more sensors
- Implement automated monitoring

## Support

- **AWS Support**: Open ticket in Console
- **GitHub Issues**: Tag with `unified-studio`
- **Documentation**: See [main README](../README.md)

---

Built for SageMaker Unified Studio with [Claude Code](https://claude.com/claude-code)
