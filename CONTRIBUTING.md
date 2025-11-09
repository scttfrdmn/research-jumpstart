# Contributing to Research Jumpstart

Thank you for your interest in contributing to Research Jumpstart! This project exists to help researchers transition to cloud computing, and we welcome contributions from researchers, educators, developers, and anyone who wants to help.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Contributing a Research Project](#contributing-a-research-project)
- [Improving Documentation](#improving-documentation)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Review Process](#review-process)
- [Recognition](#recognition)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to hello@researchjumpstart.org.

## How Can I Contribute?

### 🚀 Contributing a Research Project

**Who**: Researchers who have cloud-based workflows to share

**What**: Add a new research project to the catalog

**Why**: Help other researchers in your domain make the leap to cloud computing

[Jump to detailed guide →](#contributing-a-research-project)

### 📚 Improving Documentation

**Who**: Anyone who can write clearly

**What**: Fix typos, clarify instructions, add examples, improve guides

**Why**: Clear documentation is critical for researchers learning new tools

[Jump to detailed guide →](#improving-documentation)

### 🐛 Reporting Bugs

**Who**: Anyone who finds an issue

**What**: Report problems with notebooks, documentation, or tools

**Why**: Help us maintain quality and fix issues quickly

[Jump to detailed guide →](#reporting-bugs)

### 💡 Suggesting Enhancements

**Who**: Anyone with ideas for improvement

**What**: Propose new features, projects, or improvements

**Why**: Community input shapes the project's direction

[Jump to detailed guide →](#suggesting-enhancements)

### 🔧 Developing Tools

**Who**: Developers

**What**: Build utilities, scripts, automation, testing infrastructure

**Why**: Better tools help everyone contribute more easily

### 🎓 Creating Educational Content

**Who**: Educators, trainers

**What**: Workshop materials, tutorials, videos, teaching guides

**Why**: Multiple learning formats help different audiences

---

## Contributing a Research Project

This is the most impactful way to contribute! Here's how to add a research project to the catalog.

### Prerequisites

Before starting:
- [ ] You have a working cloud-based research workflow
- [ ] You've tested it yourself (at least the Studio Lab version)
- [ ] The workflow solves a real research problem
- [ ] You can create both Studio Lab and Unified Studio versions
- [ ] You have 4-8 hours to create comprehensive documentation

### Project Requirements

Every project must include:

**✅ Required Components**:
1. **Studio Lab Version** (free tier)
   - Working Jupyter notebook
   - Sample/representative data
   - Runs in 2-4 hours on CPU
   - environment.yml with exact package versions

2. **Unified Studio Version** (production)
   - Multiple notebooks or Python scripts
   - CloudFormation template for deployment
   - Access to full-scale datasets
   - Production-ready code with error handling

3. **Comprehensive README**
   - Problem statement ("what pain does this solve?")
   - Learning objectives
   - Prerequisites (knowledge and tools)
   - Quick launch instructions
   - Architecture diagram
   - Cost estimates (realistic and honest)
   - Transition pathway (Studio Lab → Unified Studio)
   - Troubleshooting guide
   - Extension ideas

4. **Documentation**
   - Code comments explaining each step
   - Data sources clearly documented
   - Environment dependencies specified
   - Example outputs included

**⚠️ Quality Standards**:
- Code must actually run (we will test!)
- Documentation must be clear to non-experts
- Cost estimates must be honest and realistic
- Troubleshooting guide must cover common issues
- No secrets or credentials in code

### Step-by-Step Guide

#### Step 1: Choose Your Domain

Select from existing domains or propose a new one:
- Climate Science
- Genomics
- Medical Research
- Social Sciences
- Digital Humanities
- Physics & Astronomy
- _[See full list in README](README.md#featured-projects)_

#### Step 2: Use the Project Template

```bash
# Copy the template
cp -r projects/_template/ projects/your-domain/your-project-name/

# Navigate to your project
cd projects/your-domain/your-project-name/
```

The template includes:
```
your-project-name/
├── README.md                    # Main project documentation
├── studio-lab/                  # Free tier version
│   ├── notebook.ipynb
│   └── environment.yml
├── unified-studio/              # Production version
│   ├── notebooks/
│   ├── src/
│   ├── cloudformation/
│   └── environment.yml
├── workshop/                    # Optional workshop materials
│   └── workshop.md
└── assets/                      # Images, diagrams, examples
    └── architecture.png
```

#### Step 3: Create Studio Lab Version

**Goal**: Working notebook that anyone can run for free

```python
# Start with this cell in your notebook
"""
Project: [Your Project Name]
Domain: [Your Domain]
Description: [One-sentence description]

This is the Studio Lab version - runs on free tier with sample data.
For production scale, see unified-studio/ directory.
"""

import warnings
warnings.filterwarnings('ignore')

# Check environment
import sys
print(f"Python: {sys.version}")
```

**Best Practices**:
- Use pre-processed sample data (include in repo or document how to get it)
- Target 2-4 hour completion time
- Test on actual Studio Lab before submitting
- Add progress indicators (print statements showing progress)
- Include expected outputs so users know if it's working
- Handle errors gracefully with helpful messages

#### Step 4: Create Unified Studio Version

**Goal**: Production-ready workflow that scales

**Structure your notebooks**:
1. `01_data_access.ipynb` - Connect to full datasets
2. `02_analysis.ipynb` - Core analysis logic
3. `03_visualization.ipynb` - Publication-quality figures
4. `04_bedrock_integration.ipynb` - AI-assisted interpretation (if applicable)

**Create reusable modules** (`src/`):
```python
# src/data_processing.py
"""
Data processing functions for [Project Name]

Functions:
    load_data: Load and validate input data
    preprocess: Clean and prepare data for analysis
    validate: Check data quality and completeness
"""

def load_data(source, **kwargs):
    """
    Load data from S3 or local path

    Args:
        source (str): S3 path or local file path
        **kwargs: Additional arguments for data loading

    Returns:
        data: Loaded and validated data

    Raises:
        ValueError: If data validation fails
    """
    # Implementation
    pass
```

**Create CloudFormation template**:
```yaml
# cloudformation/template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Research Jumpstart - [Project Name]'

Parameters:
  ProjectName:
    Type: String
    Default: 'research-jumpstart-project'
    Description: 'Name for your project resources'

Resources:
  # Define S3 buckets, IAM roles, SageMaker resources, etc.

Outputs:
  # Output important resource names/URLs
```

#### Step 5: Write Comprehensive Documentation

Use the template in `projects/_template/README.md` and fill in:

**Problem Statement** (2-3 paragraphs):
- What researchers currently struggle with
- Why traditional approaches are limiting
- What this project enables

**Example**:
```markdown
## The Problem This Solves

Researchers analyzing climate models typically spend 2-3 months
downloading CMIP6 data before doing any science. A single model
can be 500GB, and publications require 15-20 model ensembles.
Laptops crash on datasets this large, and local HPC clusters
have limited storage quotas.

This project enables researchers to analyze 20+ climate models
without downloading a single file, reducing time-to-results from
months to days.
```

**Cost Estimates** (be honest!):
```markdown
## Cost Estimate

**Studio Lab**: $0

**Unified Studio**:
- Data access (S3 reads): $0 (AWS Open Data)
- Compute (EMR, 4 hours, spot instances): $12-18
- Storage (500GB results): $2/month
- Bedrock (report generation): $3-5
- **Total: ~$20 per analysis**

**Compare to Alternatives:**
- Grad student time saved: 160 hours × $30/hr = $4,800
- Your time saved: 40 hours × $75/hr = $3,000
- **Net benefit: $7,780 per project**
```

**Troubleshooting** (cover real issues):
```markdown
## Common Issues

### Issue: "Kernel died during processing"
**Cause**: Out of memory - trying to load too much data at once
**Solution**: Process data in chunks instead of loading all at once
```python
# Bad: Loads everything into memory
all_data = load_all_files()

# Good: Process one file at a time
for file in files:
    data = load_file(file)
    result = process(data)
    save_result(result)
    del data  # Free memory
```
**Cost impact**: May need to restart, adds ~15 minutes
```

#### Step 6: Test Everything

**Before submitting, verify**:
- [ ] Studio Lab notebook runs start-to-finish without errors
- [ ] Unified Studio notebooks run successfully
- [ ] CloudFormation template deploys without errors
- [ ] All links in README work
- [ ] Example outputs are included
- [ ] Environment files have exact package versions
- [ ] Cost estimates are realistic
- [ ] Troubleshooting covers issues you actually encountered
- [ ] Code is well-commented
- [ ] No secrets/credentials in code

**Testing checklist**:
```bash
# Test Studio Lab version
cd studio-lab/
jupyter nbconvert --to notebook --execute notebook.ipynb

# Test environment recreation
conda env create -f environment.yml
conda activate your-env
# Run notebooks

# Validate CloudFormation
cd unified-studio/cloudformation/
aws cloudformation validate-template --template-body file://template.yaml
```

#### Step 7: Submit Your Project

1. **Fork the repository**
   ```bash
   # On GitHub: Click "Fork"
   git clone https://github.com/YOUR-USERNAME/research-jumpstart
   cd research-jumpstart
   ```

2. **Create a branch**
   ```bash
   git checkout -b add-project-your-project-name
   ```

3. **Add your project**
   ```bash
   git add projects/your-domain/your-project-name/
   git commit -m "Add [Your Project Name] project

   - Studio Lab version with sample data
   - Unified Studio version with CloudFormation
   - Comprehensive documentation
   - Tested and working
   "
   ```

4. **Push and create Pull Request**
   ```bash
   git push origin add-project-your-project-name
   # On GitHub: Create Pull Request
   ```

5. **Fill out the PR template**
   - Describe your project
   - Link to any related issues
   - Check all boxes in the PR checklist
   - Add screenshots of example outputs

#### Step 8: Respond to Review

A maintainer will review your project and may request changes:
- Answer questions promptly
- Make requested changes
- Re-test after changes
- Be patient - reviews take time!

Once approved, your project will be merged and added to the catalog!

---

## Improving Documentation

Documentation improvements are always welcome!

### Types of Documentation

- **Guides**: Getting started, transition guides, tutorials
- **Reference**: API docs, configuration options
- **Examples**: Code examples, sample outputs
- **FAQ**: Common questions and answers

### How to Contribute

1. **Find something to improve**
   - Typos or grammatical errors
   - Confusing explanations
   - Missing information
   - Outdated content

2. **Make your changes**
   ```bash
   # Edit the file
   # For docs: edit files in docs/
   # For project docs: edit project README
   ```

3. **Submit a Pull Request**
   - Small fixes: direct PR is fine
   - Large changes: open an issue first to discuss

### Documentation Style Guide

- **Be clear and concise**: Researchers are busy
- **Use examples**: Show, don't just tell
- **Be honest**: Don't oversell or hide limitations
- **Be welcoming**: Remember your audience may be new to cloud
- **Use "you" language**: "You will learn..." not "Users will learn..."
- **Avoid jargon**: Explain technical terms when needed
- **Include context**: Why is this important?

**Good example**:
```markdown
## Setting Up AWS Account

Before you can use Unified Studio, you'll need an AWS account.
This takes about 10 minutes and requires a credit card (for
verification, not charges yet).

1. Go to aws.amazon.com
2. Click "Create an AWS Account"
3. ...
```

**Bad example**:
```markdown
## AWS Account

Users must provision an AWS account via the standard IAM workflow
prior to leveraging SageMaker Unified Studio capabilities.
```

---

## Reporting Bugs

Found something broken? Please let us know!

### Before Submitting

1. **Search existing issues**: Someone may have already reported it
2. **Try to reproduce**: Can you make it happen consistently?
3. **Gather information**: Error messages, screenshots, environment details

### Creating a Bug Report

Use the bug report template:

**Title**: Clear, specific description (e.g., "Studio Lab notebook crashes on cell 5")

**Environment**:
- Platform: Studio Lab / Unified Studio / Local
- Browser: Chrome 120 / Firefox 121 / etc.
- Project: Which project has the issue

**Steps to Reproduce**:
1. Open notebook X
2. Run cells 1-4
3. Run cell 5
4. See error

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Error Messages**: Copy/paste full error message

**Screenshots**: If applicable

**Additional Context**: Anything else relevant

---

## Suggesting Enhancements

Have an idea to make Research Jumpstart better?

### Types of Enhancements

- New features
- New research domains
- Tool improvements
- Website improvements
- New documentation
- Integration ideas

### Submitting a Suggestion

1. **Check existing issues**: Someone may have suggested it already
2. **Open an issue** with the "Feature Request" template
3. **Describe**:
   - What problem does this solve?
   - Who would benefit?
   - How would it work?
   - Are there alternatives?

### What Makes a Good Suggestion

- **Solves a real problem**: Addresses actual researcher pain points
- **Fits the mission**: Helps researchers transition to cloud
- **Feasible**: Can realistically be implemented
- **Clear**: Easy to understand what you're proposing

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- conda or virtualenv
- (Optional) AWS account for testing

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/research-jumpstart/research-jumpstart
   cd research-jumpstart
   ```

2. **Create development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install MkDocs and extensions**
   ```bash
   pip install mkdocs mkdocs-material
   ```

4. **Run local documentation server**
   ```bash
   mkdocs serve
   # Open http://127.0.0.1:8000
   ```

5. **Run tests**
   ```bash
   # Validate projects
   python scripts/validate_project.py projects/climate-science/ensemble-analysis/

   # Test cost calculator
   python tools/cost-calculator/calculator.py --help
   ```

---

## Style Guidelines

### Python Code

- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable names
- Add docstrings to all functions
- Include type hints where helpful
- Keep functions focused and short

**Example**:
```python
def calculate_ensemble_mean(models: list, variable: str) -> xr.DataArray:
    """
    Calculate multi-model ensemble mean

    Args:
        models: List of model names to include
        variable: Climate variable (e.g., 'tas', 'pr')

    Returns:
        Ensemble mean as xarray DataArray

    Raises:
        ValueError: If no models provided or variable not found
    """
    if not models:
        raise ValueError("At least one model required")

    # Load data for each model
    data = [load_model(m, variable) for m in models]

    # Calculate mean
    ensemble_mean = xr.concat(data, dim='model').mean(dim='model')

    return ensemble_mean
```

### Jupyter Notebooks

- Add markdown cells explaining each section
- Use clear section headers
- Include progress indicators
- Show example outputs
- Clean up outputs before committing (large images)
- Keep cell outputs for key results

### Markdown

- Use clear hierarchical headers (# ## ###)
- Include table of contents for long documents
- Use code blocks with language specified
- Add alt text to images
- Use relative links within the repository

### Git Commits

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Keep commits focused (one logical change)
- Reference issues when applicable

**Good commit messages**:
```
Add climate ensemble analysis project

- Studio Lab version with 3-model sample
- Unified Studio version with full CMIP6 access
- CloudFormation template for deployment
- Comprehensive documentation and troubleshooting
```

```
Fix memory error in variant calling notebook

- Process chromosomes one at a time instead of all at once
- Add progress indicators
- Update troubleshooting guide

Fixes #42
```

---

## Review Process

### What to Expect

1. **Initial Review** (2-5 days)
   - Maintainer checks that submission is complete
   - May request missing components

2. **Technical Review** (1-2 weeks)
   - Code is tested
   - Documentation is reviewed
   - Quality is assessed

3. **Revisions** (as needed)
   - You address feedback
   - May require multiple rounds

4. **Approval & Merge**
   - Once approved, your contribution is merged!
   - You're added to contributors list

### Review Criteria

**Projects**:
- ✅ Code actually runs
- ✅ Documentation is clear and complete
- ✅ Follows project template
- ✅ Includes both Studio Lab and Unified Studio versions
- ✅ Cost estimates are realistic
- ✅ No secrets or credentials
- ✅ Code is well-commented

**Documentation**:
- ✅ Clear and concise
- ✅ Accurate information
- ✅ Follows style guide
- ✅ Links work
- ✅ Appropriate for audience

**Tools/Code**:
- ✅ Works as intended
- ✅ Includes tests
- ✅ Documented
- ✅ Follows style guide
- ✅ Error handling

---

## Recognition

### Contributors

All contributors are recognized in:
- Contributors list on website
- GitHub contributors page
- Project-specific attribution

### Types of Recognition

- **Author**: Created a complete project
- **Contributor**: Made significant improvements
- **Reviewer**: Helped review contributions
- **Maintainer**: Ongoing project maintenance

### Special Thanks

Major contributors may be:
- Invited to join steering committee
- Featured in success stories
- Invited to present at conferences
- Acknowledged in publications about Research Jumpstart

---

## Questions?

- 💬 [GitHub Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)
- 📧 Email: hello@researchjumpstart.org
- 📅 Office Hours: Tuesdays 2-4pm PT

Thank you for contributing to Research Jumpstart! Together, we're making cloud computing accessible to researchers everywhere. 🚀
