# Studio Lab Quickstart

Get started with Research Jumpstart in under 5 minutes - no AWS account, no credit card, completely free.

!!! success "What You'll Accomplish"
    By the end of this guide, you'll have:

    - ✅ Studio Lab account (approval time varies - can be instant to several days)
    - ✅ Working Jupyter environment
    - ✅ Your first research project running
    - ✅ Understanding of cloud-based workflows

**Time Required**: 2-5 minutes setup + wait for account approval

---

## Step 1: Request Studio Lab Account

### Create Your Account

1. **Go to Studio Lab**

   Visit [https://studiolab.sagemaker.aws](https://studiolab.sagemaker.aws)

2. **Click "Request free account"**

3. **Fill out the form**
   - Email address (use your institutional email for faster approval)
   - Name
   - How you'll use Studio Lab (be specific - "research" is better than "learning")
   - Organization (optional but recommended)

4. **Submit and wait**
   - Approval time varies: some are instant, others may take several days
   - Check your email for approval notice
   - Check spam folder if you don't hear back

!!! tip "Improve Your Chances"
    - Use institutional (.edu) email if available
    - Be specific about your research use case
    - Include your university/organization name
    - Provide clear justification for your intended use

### While You Wait

- ⏱️ **Browse [projects](../projects/index.md)** to choose your first project
- 📚 **Read [platform comparison](platform-comparison.md)** to understand options
- 🎥 **Watch [video tutorials](../resources/videos.md)** to see Studio Lab in action

---

## Step 2: Activate Your Account

Once you receive the approval email:

1. **Click the activation link** in your email

2. **Create your password**
   - Use a strong password
   - Save it in your password manager

3. **Sign in** to Studio Lab

4. **You're in!** 🎉

---

## Step 3: Launch Your Environment

### First Time Setup

1. **Choose your compute**

   === "CPU (Recommended for most)"
       - 4 vCPU, 16GB RAM
       - Good for data analysis, visualization
       - 12-hour sessions
       - **Start with this**

   === "GPU (For deep learning)"
       - 4 vCPU, 16GB RAM + NVIDIA T4 GPU
       - For training models, neural networks
       - Limited hours per month
       - **Use only if needed**

2. **Click "Start runtime"**
   - Takes 2-3 minutes to start
   - Environment is being created
   - ☕ Get coffee while you wait

3. **Click "Open project"**
   - Opens JupyterLab interface
   - Familiar if you've used Jupyter before
   - Clean, empty workspace

---

## Step 4: Clone Your First Project

### Option A: Use the Terminal

1. **Open a terminal**
   - Click the "+" button (top left)
   - Click "Terminal" under "Other"

2. **Clone a Research Jumpstart project**
   ```bash
   git clone https://github.com/research-jumpstart/research-jumpstart.git
   cd research-jumpstart/projects/
   ```

3. **Choose a domain** (e.g., climate-science)
   ```bash
   cd climate-science/ensemble-analysis/studio-lab/
   ```

4. **Open the notebook**
   - Navigate in file browser
   - Double-click `notebook.ipynb`

### Option B: Use Git UI (Easier)

1. **Click the Git icon** (left sidebar, looks like a branch)

2. **Click "Clone a Repository"**

3. **Paste URL**:
   ```
   https://github.com/research-jumpstart/research-jumpstart.git
   ```

4. **Navigate to project**:
   `projects/climate-science/ensemble-analysis/studio-lab/`

---

## Step 5: Run Your First Project

### Set Up Environment

1. **Create conda environment** (first time only)

   In terminal:
   ```bash
   cd research-jumpstart/projects/climate-science/ensemble-analysis/studio-lab/
   conda env create -f environment.yml
   conda activate climate-analysis
   ```

   This takes 5-10 minutes. ☕ Another coffee break!

2. **Select kernel in notebook**
   - Open notebook
   - Click kernel name (top right)
   - Select `climate-analysis` from dropdown
   - If not visible, click "Refresh" first

### Run the Analysis

1. **Read the introduction** (first few cells)
   - Understand what the project does
   - Check prerequisites
   - Review expected runtime

2. **Run cell by cell**
   - Click first code cell
   - Press `Shift+Enter` to run
   - Wait for output before running next
   - Read comments as you go

3. **Monitor progress**
   - `[*]` means cell is running
   - `[1]` means cell finished
   - Progress bars show time remaining

!!! warning "Session Limits"
    Studio Lab sessions timeout after 12 hours of inactivity. Your work auto-saves, but long-running cells will be interrupted. For jobs > 12 hours, consider [Unified Studio](aws-account-setup.md).

### Troubleshooting First Run

??? failure "Kernel died / restarted"
    **Cause**: Out of memory or code error

    **Solution**:
    - Restart kernel: Kernel → Restart Kernel
    - Re-run cells from beginning
    - Check you're using sample data (not full dataset)

??? failure "ModuleNotFoundError"
    **Cause**: Missing package or wrong environment

    **Solution**:
    ```bash
    # In terminal
    conda activate climate-analysis
    conda install missing-package
    ```

??? failure "Cells running very slowly"
    **Cause**: Shared resources during peak times

    **Solution**:
    - Be patient, it will finish
    - Try during off-peak hours (early morning US time)
    - CPU jobs are generally reliable, just slower sometimes

??? failure "Data not found"
    **Cause**: Sample data not in expected location

    **Solution**:
    - Check you're in correct directory
    - Verify path in notebook matches data location
    - Some projects include data downloads in first cell

---

## Step 6: Save Your Work

### Version Control (Recommended)

1. **Initialize Git** (if not already cloned)
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   git add .
   git commit -m "Completed climate analysis project"
   ```

2. **Push to your own repo**
   ```bash
   # Fork research-jumpstart on GitHub first
   git remote set-url origin https://github.com/YOUR-USERNAME/research-jumpstart.git
   git push
   ```

### Manual Backup

1. **Download important results**
   - Right-click file in browser
   - Download
   - Save to your computer

2. **Export notebooks**
   - File → Download
   - Saves as `.ipynb` file

---

## Step 7: Explore More Projects

### Browse by Domain

Now that you've completed one project, try others:

- 🌍 [Climate Science](../projects/climate-science.md)
- 🧬 [Genomics](../projects/genomics.md)
- 🏥 [Medical Research](../projects/medical.md)
- 📊 [Social Sciences](../projects/social-science.md)
- 🔬 [Physics & Astronomy](../projects/physics.md)
- 🎨 [Digital Humanities](../projects/digital-humanities.md)

[View all 20+ domains →](../projects/all-domains.md)

### Customize a Project

Try adapting a project to your needs:

1. **Change the region** (for geographic analyses)
2. **Modify variables** (different measurements)
3. **Add your own data** (small files < 10GB)
4. **Adjust parameters** (time periods, thresholds)

---

## Common Workflows

### Daily Research Routine

```bash
# 1. Start Studio Lab session (2 min)
# 2. Activate environment
conda activate your-env

# 3. Pull latest from Git
git pull

# 4. Work on analysis (up to 12 hours)
# 5. Commit progress
git add .
git commit -m "Daily progress"
git push

# 6. Stop session when done (saves costs... wait, it's free!)
```

### Collaborating with Others

1. **Share via Git**
   - Push to GitHub
   - Collaborator clones repo
   - Both work independently
   - Merge changes via pull requests

2. **Share results**
   - Export figures/data
   - Email or shared drive
   - GitHub releases for versions

---

## Studio Lab Tips & Tricks

### Maximize Your 15GB Storage

```bash
# Check storage usage
du -sh *

# Clean up old conda environments
conda env list
conda env remove -n old-env

# Clear cache
conda clean --all
pip cache purge

# Remove large datasets after analysis
rm -rf data/large-files/
```

### Optimize for 12-Hour Limit

**Checkpoint Long Runs**:
```python
# Save intermediate results
import pickle

# After each major step
with open('checkpoint_step1.pkl', 'wb') as f:
    pickle.dump(intermediate_result, f)

# Resume from checkpoint if needed
with open('checkpoint_step1.pkl', 'rb') as f:
    intermediate_result = pickle.load(f)
```

### Use Screen for Background Jobs

```bash
# Start screen session
screen -S analysis

# Run long command
python long_analysis.py

# Detach (Ctrl+A, then D)
# Session continues in background

# Reattach later
screen -r analysis
```

### Speed Up Conda Environment Creation

```bash
# Use mamba (faster conda)
conda install mamba -n base -c conda-forge

# Create env with mamba instead
mamba env create -f environment.yml
```

---

## Understanding the Interface

### JupyterLab Layout

```
┌─────────────────────────────────────────────────────────┐
│ [File] [Edit] [View] [Run] [Kernel] [Tabs] [Settings]  │ Menu
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐  ┌───────────────────────────────────┐  │
│  │         │  │                                   │  │
│  │  File   │  │                                   │  │
│  │ Browser │  │       Notebook                    │  │
│  │         │  │       (Main work area)            │  │
│  │         │  │                                   │  │
│  │         │  │                                   │  │
│  │         │  │                                   │  │
│  └─────────┘  └───────────────────────────────────┘  │
│   Sidebar            Main Area                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Features**:
- **File Browser** (left): Navigate files and folders
- **Git Panel** (left): Version control
- **Running Terminals** (left): See active sessions
- **Launcher** (+button): Open new notebooks, terminals
- **Kernel Indicator** (top right): Shows running/idle

---

## Keyboard Shortcuts

**Notebook Shortcuts** (Command mode - press Esc first):

| Shortcut | Action |
|----------|--------|
| `Shift+Enter` | Run cell, select next |
| `Ctrl+Enter` | Run cell, stay |
| `A` | Insert cell above |
| `B` | Insert cell below |
| `D,D` (press twice) | Delete cell |
| `M` | Convert to Markdown |
| `Y` | Convert to Code |
| `Shift+M` | Merge cells |

**JupyterLab Shortcuts**:

| Shortcut | Action |
|----------|--------|
| `Ctrl+B` | Toggle left sidebar |
| `Ctrl+Shift+C` | Open command palette |
| `Ctrl+Shift+L` | Toggle line numbers |

[Full shortcut reference →](https://jupyterlab.readthedocs.io/en/stable/user/interface.html#keyboard-shortcuts)

---

## When to Transition to Unified Studio

Consider transitioning when:

✅ **Data Size**: Your datasets exceed 10GB
✅ **Runtime**: Analyses take > 12 hours
✅ **Team**: Need to collaborate with others
✅ **Production**: Ready for publication-quality work
✅ **AI**: Want Bedrock integration for reports
✅ **Scale**: Need distributed processing

[Learn How to Transition →](../transition-guides/studio-lab-to-unified.md)

---

## Getting Help

### Within Studio Lab

**Built-in Help**:
- Help menu → JupyterLab Reference
- Help menu → Notebook Reference
- Hover over functions for docstrings

### Research Jumpstart Community

- 💬 [GitHub Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)
- 🐛 [Report Issues](https://github.com/research-jumpstart/research-jumpstart/issues)
- 📧 [Email](mailto:hello@researchjumpstart.org)
- 📅 [Office Hours](../community/office-hours.md) (Tuesdays)

### Common Questions

[Browse FAQ →](../resources/faq.md)

---

## Next Steps

### Complete 2-3 More Projects

Build confidence by trying projects in different domains:

1. Pick a project in your field
2. Pick a project in an adjacent field
3. Pick a project that seems challenging

### Share What You Learned

- Write a blog post about your experience
- Present to your lab group
- Help a colleague get started

### Decide on Transition

After 2-3 weeks with Studio Lab:
- Evaluate if cloud works for your research
- Check if you hit Studio Lab limitations
- Plan transition to Unified Studio if needed

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  STUDIO LAB QUICK REFERENCE                     │
├─────────────────────────────────────────────────┤
│  Start runtime → Open project → Clone repo      │
│  conda env create -f environment.yml           │
│  conda activate env-name                        │
│  Open notebook → Select kernel → Run cells      │
│  git add . && git commit -m "msg" && git push  │
│  Stop runtime when done                         │
├─────────────────────────────────────────────────┤
│  LIMITS:                                        │
│  - 15GB storage                                 │
│  - 12-hour sessions                             │
│  - CPU/GPU compute                              │
│  - Sample data only                             │
├─────────────────────────────────────────────────┤
│  HELP:                                          │
│  - GitHub Discussions                           │
│  - docs: getting-started/studio-lab-quickstart │
│  - office-hours: Tuesdays 2-4pm PT             │
└─────────────────────────────────────────────────┘
```

---

**Ready to start?** [Request your free Studio Lab account](https://studiolab.sagemaker.aws) now! 🚀
