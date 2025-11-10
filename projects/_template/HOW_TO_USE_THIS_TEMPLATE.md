# How to Use This Project Template

This template provides the structure and patterns for creating new Research Jumpstart projects. Follow this guide to create a high-quality project that fits the community standards.

---

## Quick Start

1. **Copy the template**:
   ```bash
   cp -r projects/_template projects/[domain]/[project-name]
   cd projects/[domain]/[project-name]
   ```

2. **Fill in README.md** using the bracketed placeholders

3. **Create Studio Lab version** (minimum viable project)

4. **Test thoroughly**

5. **Submit PR** with completed project

---

## Required Components

### ✅ Minimum Viable Project (Studio Lab Only)

To submit a project, you MUST include:

**1. Main README.md** (see template)
- Problem statement
- Learning objectives
- Prerequisites
- Quick start instructions
- Key features
- Cost estimates
- Troubleshooting
- Resources

**2. Studio Lab version**:
- `quickstart.ipynb` - Complete working notebook
- `environment.yml` - Conda environment specification
- `sample_data.[ext]` - Sample dataset (if needed)
- `README.md` - Studio Lab specific guide

**3. Basic documentation**:
- Clear setup instructions
- Expected outputs
- Common issues and solutions

### 🎯 Highly Recommended

**4. Unified Studio version** (for production scale):
- Python modules in `src/`
- CloudFormation template
- Production README
- Requirements.txt

**5. Assets**:
- Architecture diagram (text or visual)
- Sample output screenshots/files
- Cost calculator

---

## Pattern to Follow

Looking at existing projects, here's the established pattern:

### Project Tiers

**Tier 1: Flagship** (Climate, Social Media)
- Complete Studio Lab notebook (500+ lines)
- Complete Unified Studio modules (2,000+ lines)
- CloudFormation infrastructure
- Comprehensive documentation (1,000+ lines README)
- Time investment: 4-5 days

**Tier 2: Complete** (Target for most)
- Complete Studio Lab notebook (300+ lines)
- Basic Unified Studio structure
- Clear documentation (500+ lines README)
- Time investment: 2-3 days

**Tier 3: Starter** (Genomics, Digital Humanities, Medical)
- README with clear workflow outline
- Environment specification
- Sample data description
- Time investment: 2-4 hours

**All tiers welcome!** Start with Tier 3, community can enhance to Tier 2/1.

---

## README Template Guide

### Section-by-Section Instructions

#### Header
```markdown
# [Project Name]

**Difficulty**: 🟢/🟡/🔴 | **Time**: ⏱️ X-Y hours

[One sentence summary]
```

**Guidelines**:
- Choose appropriate difficulty: 🟢 (no cloud/domain exp), 🟡 (some exp helpful), 🔴 (significant exp needed)
- Time estimate should be realistic for Studio Lab version
- One sentence should capture the essence

#### What Problem Does This Solve?

**Purpose**: Convince researchers this is worth their time

**What to include**:
- Real pain points they experience
- Why traditional approaches don't work
- How cloud helps solve specific problems

**Example from Climate project**:
> "Climate scientists routinely need to analyze multiple climate models...
> Traditional approach: Download terabytes, requires institutional storage...
> This project: Access S3 directly, no downloads, analyze 20+ models"

#### What You'll Learn

**Purpose**: Clear learning objectives across three categories

**Pattern**:
1. Domain skills (specific to the field)
2. Data science/computing skills (technical)
3. Cloud computing skills (AWS/cloud patterns)

**Keep it specific**: "Multi-model ensemble analysis" not "Climate stuff"

#### Prerequisites

**Be honest**:
- What domain knowledge is actually required?
- What Python level? (Basic = loops/functions, Intermediate = pandas/numpy, Advanced = ML frameworks)
- Clearly state "No cloud experience needed!"

#### Quick Start

**Two versions**: Studio Lab (free) + Unified Studio (production)

**Studio Lab section must include**:
- Copy-paste setup commands
- ✅ Checklist of what's included
- ⚠️ Clear limitations
- Realistic time estimate

**Example**:
```markdown
**What's included**:
- ✅ Sample data for 3 models
- ✅ Complete analysis workflow
- ✅ Publication-quality figures

**Limitations**:
- ⚠️ Sample data (not real)
- ⚠️ Limited to 3 models
- ⚠️ 4GB RAM constraint
```

#### Cost Estimates

**Be brutally honest**:
- Always show Studio Lab is $0
- Give realistic Unified Studio costs with **specific scenario**
- Break down by AWS service
- Include monthly costs for different usage levels
- Add cost optimization tips

**Example structure**:
```markdown
**Realistic cost breakdown** (20 models, 1 variable, 1 region):

| Service | Usage | Cost |
|---------|-------|------|
| S3 Data | Read-only | $0 |
| Compute | 4 hours | $2.40 |
| Total | | **$2.40** |
```

#### Troubleshooting

**Include real issues** users will hit:
- Environment setup problems
- Memory errors
- Common mistakes
- Platform-specific issues

**Format**:
- Problem statement or error message
- Cause (if not obvious)
- Step-by-step solution

#### Extension Ideas

**Purpose**: Give users next steps after completing the project

**Pattern** (12 ideas total):
- Beginner: 3-4 ideas (2-4 hours each)
- Intermediate: 3-4 ideas (4-8 hours each)
- Advanced: 3-4 ideas (8+ hours each)

---

## Studio Lab Notebook Pattern

Based on successful projects, follow this structure:

### Required Cells

1. **Title and Overview** (Markdown)
   - Project name
   - What you'll build
   - Time estimate

2. **Setup & Imports** (Code)
   - All imports
   - Configuration
   - Data downloads (NLTK, models, etc.)

3. **Load Data** (Code)
   - Load sample data
   - Basic exploration
   - Display summary statistics

4. **Main Analysis Sections** (3-6 sections)
   - Each section: Markdown explanation + Code implementation
   - Clear section headers
   - Visualizations where appropriate

5. **Summary** (Markdown + Code)
   - Key findings
   - Summary statistics
   - Interpretation

6. **Next Steps** (Markdown)
   - How to customize
   - How to scale up
   - Learning resources

### Code Quality Standards

**Comments**:
- Docstrings for all functions
- Inline comments for complex logic
- Explain WHY not just WHAT

**Variables**:
- Descriptive names (`ensemble_mean` not `em`)
- Constants in UPPERCASE
- Configuration at top

**Output**:
- Print clear status messages
- Use f-strings for formatting
- Include ✓ checkmarks for completion

**Visualizations**:
- Always label axes
- Include titles
- Use colorblind-friendly palettes
- `plt.tight_layout()` before `plt.show()`

---

## Environment Specification

Pattern for `environment.yml`:

```yaml
name: [project-env-name]
channels:
  - conda-forge
  - defaults
dependencies:
  # Python version
  - python=3.10

  # Core data science (if needed)
  - numpy=1.24.3
  - pandas=2.0.3
  - scipy=1.11.1

  # Domain-specific packages
  - [package]=[[version]]
  - [package]=[[version]]

  # Visualization
  - matplotlib=3.7.2
  - seaborn=0.12.2

  # Jupyter
  - ipykernel=6.25.0
  - ipywidgets=8.1.0

  # pip packages (if conda unavailable)
  - pip:
    - [package]==[version]
```

**Guidelines**:
- Pin major.minor versions (not patch)
- Group packages logically
- Comment each group
- Keep it minimal (only what's needed)

---

## Sample Data Guidelines

### DO:
- ✅ Keep it small (< 10MB for Studio Lab)
- ✅ Use CSV, JSON, or simple formats
- ✅ Include README explaining data source
- ✅ Synthetic data is fine for education
- ✅ Public domain or properly licensed

### DON'T:
- ❌ Include large files (> 50MB)
- ❌ Use copyrighted data without permission
- ❌ Include sensitive/private information
- ❌ Require external downloads in Studio Lab
- ❌ Use proprietary formats without tools

### Sources for Public Data:
- AWS Open Data Registry
- Kaggle (with proper attribution)
- Government datasets (data.gov, etc.)
- Academic repositories
- Generate synthetic data

---

## Unified Studio Pattern

If creating production version, follow this structure:

### Python Modules (`src/`)

**Minimum**:
1. `__init__.py` - Package exports
2. `data_access.py` - S3/data loading
3. `[analysis].py` - Core analysis functions
4. `visualization.py` - Plotting utilities

**Optional**:
5. `[specialized].py` - Domain-specific module
6. `bedrock_client.py` - AI integration

**Each module should have**:
- Module docstring
- Imports
- Functions with docstrings
- Type hints
- Logging configuration

### CloudFormation Template

**Minimum resources**:
- S3 bucket for results
- IAM role with proper permissions
- CloudWatch log group

**Common additions**:
- SNS topic for notifications
- Cost monitoring alarms
- Parameters for customization

---

## Testing Checklist

Before submitting, verify:

### Studio Lab Version
- [ ] Conda environment creates without errors
- [ ] All cells run without errors
- [ ] Output is as expected
- [ ] Visualizations display correctly
- [ ] Time estimate is accurate
- [ ] Works on fresh Studio Lab instance

### Documentation
- [ ] README.md is complete (no [brackets] left)
- [ ] All links work
- [ ] Code examples are correct
- [ ] Prerequisites are accurate
- [ ] Cost estimates are realistic

### Code Quality
- [ ] Functions have docstrings
- [ ] Variables are well-named
- [ ] Comments explain complex logic
- [ ] No hardcoded paths
- [ ] Consistent style

---

## Submitting Your Project

1. **Create PR** with:
   - Completed project in `projects/[domain]/[project-name]/`
   - Updated `docs/projects/index.md` with your project
   - Clear PR description explaining the project

2. **PR description should include**:
   - Domain and project name
   - What it does (1-2 sentences)
   - Completeness level (Tier 1/2/3)
   - Any special setup requirements
   - Screenshots (if applicable)

3. **Be responsive** to review feedback

4. **Join the community** on Discussions

---

## Examples to Follow

**Best examples**:
- Climate Science (Tier 1 - flagship)
- Social Media Analysis (Tier 1 - flagship)
- Genomics Variant Analysis (Tier 3 - starter)

**Study these** to understand the patterns.

---

## Getting Help

**Questions about the template**:
- Open issue with `template` label
- Ask in Discussions under "Contributing"
- Review existing projects for patterns

**Need project review**:
- Open draft PR early
- Tag with `feedback-wanted`
- Be specific about what needs help

---

## Community Standards

Your project should:
- Be educational (not just production code)
- Include sample data or clear data access
- Work in Studio Lab (free tier)
- Follow the established patterns
- Be well-documented
- Be honest about limitations
- Include proper attribution

**Remember**: Perfect is the enemy of good. A Tier 3 starter is valuable!

---

*Last updated: 2025-11-09*
