# Student Learning Analytics and Prediction

**Flagship Project** ⭐ | **Difficulty**: 🟢 Beginner | **Time**: ⏱️⏱️ 4-8 hours (Studio Lab)

Analyze student learning patterns and predict outcomes using machine learning without managing complex institutional data infrastructure. Perfect introduction to educational data science and learning analytics.

---

## What Problem Does This Solve?

Education researchers and institutional analysts routinely need to:
- Predict student dropout risk for early intervention
- Analyze learning pathways and progression patterns
- Quantify effectiveness of interventions
- Compare outcomes across courses, programs, or institutions

**Traditional approach problems**:
- Student data = **terabytes** across multiple systems (LMS, SIS, admissions)
- Accessing even one semester of logs = weeks of data requests and approvals
- Multi-institution analysis requires complex data sharing agreements
- Updating analysis when new data arrives = start over from scratch

**This project shows you how to**:
- Access educational datasets efficiently (MOOC logs, institutional data)
- Process student interaction sequences with deep learning
- Train LSTM models for dropout prediction and pathway analysis
- Generate actionable intervention recommendations
- Scale from single-course analysis (free) to multi-institution studies (production)

---

## What You'll Learn

### Educational Research Skills
- Learning analytics and educational data mining
- Student success prediction and early warning systems
- Learning pathway analysis and sequence modeling
- Intervention effectiveness evaluation
- Cross-institutional comparison methods

### Machine Learning Skills
- Sequential modeling with LSTMs and Transformers
- Imbalanced classification (handling rare dropout events)
- Feature engineering from behavioral data
- Ensemble modeling across institutions
- Model interpretability for actionable insights

### Technical Skills
- Jupyter notebook workflows
- Conda environment management
- Time series and sequence data processing
- Publication-quality visualization
- Git version control for research

---

## Prerequisites

### Required Knowledge
- **Education research**: Basic understanding of learning analytics
- **Python**: Familiarity with pandas, NumPy, matplotlib
- **None required**: No cloud experience needed!

### Optional (Helpful)
- Experience with sequence modeling or NLP
- Basic command line skills
- Git basics

### Technical Requirements

**Studio Lab (Free Tier)**
- SageMaker Studio Lab account ([request here](https://studiolab.sagemaker.aws))
- No AWS account needed
- No credit card required

**Unified Studio (Production)**
- AWS account with billing enabled
- Estimated cost: $15-25 per analysis (see Cost Estimates section)
- SageMaker Unified Studio access

---

## Quick Start

### Option 1: Studio Lab (Free - Start Here!)

Perfect for learning, testing, and small-scale analysis.

**Launch in 3 steps**:

1. **Request Studio Lab account** (if you don't have one)
   - Visit https://studiolab.sagemaker.aws
   - Create account with email
   - Approval time varies (can be instant to several days)

2. **Clone this repository**
   ```bash
   git clone https://github.com/research-jumpstart/research-jumpstart.git
   cd research-jumpstart/projects/education/learning-analytics/studio-lab
   ```

3. **Set up environment and run**
   ```bash
   # Create conda environment (one time)
   conda env create -f environment.yml
   conda activate learning-analytics

   # Launch notebook
   jupyter notebook quickstart.ipynb
   ```

**What's included in Studio Lab version**:
- ✅ Complete workflow demonstration
- ✅ Simulated MOOC data (~50,000 students, 12-week course)
- ✅ Sample data generation (simulated for educational purposes)
- ✅ All analysis techniques: dropout prediction, pathway analysis, intervention optimization
- ✅ LSTM model training
- ✅ Comprehensive documentation

**Limitations**:
- ⚠️ Uses simulated data (not real institutional data)
- ⚠️ Limited to single-course analysis (vs multi-institution)
- ⚠️ Smaller dataset (~1.5GB vs 10GB+)
- ⚠️ 15GB storage, 12-hour sessions

**Time to complete**: 4-6 hours (including environment setup and exploring code)

---

### Option 2: Unified Studio (Production)

Full-scale learning analytics with real institutional data from multiple sources.

**Prerequisites**:
- AWS account with billing enabled
- SageMaker Unified Studio domain set up
- Familiarity with Studio Lab version (complete it first!)
- Institutional data sharing agreements (if using real data)

**Quick launch**:

1. **Deploy infrastructure** (one-time setup)
   ```bash
   cd unified-studio/cloudformation
   aws cloudformation create-stack \
     --stack-name learning-analytics \
     --template-body file://learning-analytics-stack.yml \
     --parameters file://parameters.json \
     --capabilities CAPABILITY_IAM
   ```

2. **Launch Unified Studio**
   - Open SageMaker Unified Studio
   - Navigate to learning-analytics domain
   - Launch JupyterLab environment

3. **Run analysis notebooks**
   ```bash
   cd unified-studio/notebooks
   # Follow notebooks in order:
   # 01_data_ingestion.ipynb       - Load institutional data
   # 02_feature_engineering.ipynb  - Process interaction logs
   # 03_model_training.ipynb       - Train LSTM models
   # 04_pathway_analysis.ipynb     - Learning pathway prediction
   # 05_intervention_optimization.ipynb - Optimize intervention timing
   # 06_bedrock_integration.ipynb  - AI-assisted interpretation
   ```

**What's included in Unified Studio version**:
- ✅ Real institutional data pipelines (S3 integration)
- ✅ Multi-institution ensemble models
- ✅ Multiple courses and programs
- ✅ Longitudinal tracking (multi-semester)
- ✅ Distributed processing with EMR
- ✅ AI-assisted analysis via Amazon Bedrock
- ✅ Automated report generation
- ✅ Production-ready code modules

**Cost estimate**: $15-25 per analysis (see detailed breakdown below)

**Time to complete**:
- First time setup: 2-3 hours
- Each subsequent analysis: 45-90 minutes

---

## Project Structure

```
learning-analytics/
├── README.md                          # This file
├── tier-0/                           # Beginner tier (Colab/Studio Lab)
│   └── README.md                     # Tier 0 documentation
├── tier-1/                           # Intermediate tier (Studio Lab only)
│   ├── README.md                     # Tier 1 documentation
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   └── saved_models/
├── studio-lab/                        # Free tier version
│   ├── quickstart.ipynb              # Main analysis notebook
│   ├── environment.yml               # Conda dependencies
│   └── README.md                     # Studio Lab specific docs
├── unified-studio/                    # Production version
│   ├── notebooks/
│   ├── src/
│   ├── cloudformation/
│   ├── environment.yml
│   └── README.md
├── workshop/                          # Half-day workshop materials
│   ├── slides.pdf
│   ├── exercises/
│   └── solutions/
└── assets/
    ├── architecture-diagram.png
    ├── sample-outputs/
    └── cost-calculator.xlsx
```

---

## Cost Estimates

### Studio Lab: $0 (Always Free)

- No AWS account required
- No credit card needed
- No hidden costs
- 15GB storage, 12-hour sessions

**When Studio Lab is enough**:
- Learning educational data science
- Teaching/workshops
- Prototyping analysis workflows
- Single-course or small-scale studies

---

### Unified Studio: $15-25 per Analysis

**Realistic cost breakdown for typical analysis**:
(5 institutions, 100K students, 5 years of data, ensemble models)

| Service | Usage | Cost |
|---------|-------|------|
| **Data Storage (S3)** | 10GB institutional data | $0.23/month |
| **Compute (Jupyter)** | ml.t3.xlarge, 3 hours | $0.45 |
| **Processing (Glue)** | ETL jobs for data harmonization | $2-3 |
| **Training (SageMaker)** | LSTM training | $5-8 |
| **Bedrock (Claude 3)** | Report generation | $3-5 |
| **Total per analysis** | | **$11-17** |

**Monthly costs if running regularly**:
- 5 analyses/month: $55-85
- 10 analyses/month: $110-170
- Storage (persistent): $2-5/month

**Cost optimization tips**:
1. Use spot instances for training (save 60-80%)
2. Delete intermediate results (keep only final outputs)
3. Process multiple cohorts in single run
4. Cache frequently-used feature sets
5. Use ml.t3.medium for lighter analyses

**When Unified Studio is worth it**:
- Need real institutional data (not simulated)
- Multi-institution comparative analysis
- Longitudinal tracking (multi-semester)
- Regular analysis updates (semester-by-semester)
- Collaboration with institutional research teams

---

## Getting Started

Choose your path:

1. **New to learning analytics?** Start with [Tier 0](tier-0/README.md) - Single-course dropout prediction (60-90 min, free)

2. **Ready for multi-institution analysis?** Move to [Tier 1](tier-1/README.md) - Cross-institutional ensemble models (4-8 hours, free Studio Lab)

3. **Need production deployment?** Explore [Unified Studio](unified-studio/README.md) - Full-scale infrastructure ($15-25 per analysis)

---

## Additional Resources

### Educational Data Mining & Learning Analytics

- **Journal of Educational Data Mining**: https://jedm.educationaldatamining.org/
- **Journal of Learning Analytics**: https://learning-analytics.info/
- **Society for Learning Analytics Research**: https://www.solaresearch.org/

### Datasets

- **MOOC Data**: https://moocdata.cn/data/user-activity
- **Open University Learning Analytics Dataset**: https://analyse.kmi.open.ac.uk/open_dataset

### Technical Resources

- **TensorFlow Time Series**: https://www.tensorflow.org/tutorials/structured_data/time_series
- **PyTorch LSTM Tutorial**: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_learning_analytics,
  title = {Student Learning Analytics and Prediction: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../../LICENSE) file for details.

**Note on Educational Data**: When working with real student data, ensure compliance with FERPA and institutional IRB requirements.

---

*Last updated: 2025-11-13 | Research Jumpstart v1.0.0*
