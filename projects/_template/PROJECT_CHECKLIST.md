# New Project Checklist

Use this checklist when creating a new Research Jumpstart project.

---

## Planning Phase

- [ ] Choose domain and project topic
- [ ] Verify it solves a real research problem
- [ ] Confirm data is available (public or can generate sample)
- [ ] Estimate time to complete (be realistic)
- [ ] Check it doesn't duplicate existing project

---

## Minimum Viable Project (Required)

### Documentation
- [ ] README.md completed (no [brackets] remaining)
- [ ] Problem statement is clear and compelling
- [ ] Learning objectives are specific
- [ ] Prerequisites are accurate
- [ ] Quick start works (tested)
- [ ] Cost estimates are realistic
- [ ] Troubleshooting includes common issues
- [ ] Extension ideas are actionable

### Studio Lab Version
- [ ] `quickstart.ipynb` runs end-to-end
- [ ] All cells execute without errors
- [ ] Visualizations display correctly
- [ ] Output matches expectations
- [ ] Time estimate is accurate (<= stated time)
- [ ] `environment.yml` is complete
- [ ] Environment creates successfully
- [ ] All dependencies resolve
- [ ] Sample data included (if needed)
- [ ] Data is < 10MB
- [ ] Studio Lab README created

### Code Quality
- [ ] Functions have docstrings
- [ ] Complex logic has comments
- [ ] Variables are descriptive
- [ ] No hardcoded paths
- [ ] Consistent code style
- [ ] Status messages use ✓ checkmarks
- [ ] Error handling where appropriate

---

## Enhanced Project (Recommended)

### Unified Studio Version
- [ ] Python modules in `src/`
- [ ] `__init__.py` with exports
- [ ] CloudFormation template
- [ ] `requirements.txt`
- [ ] `setup.py`
- [ ] Unified Studio README
- [ ] Production cost estimates

### Assets
- [ ] Architecture diagram (text or visual)
- [ ] Sample outputs
- [ ] Assets README

---

## Testing Phase

### Functional Testing
- [ ] Fresh Studio Lab instance
- [ ] Clean conda environment
- [ ] Run all cells sequentially
- [ ] Verify all outputs
- [ ] Test with different parameters
- [ ] Check error messages are helpful

### Documentation Testing
- [ ] All links work
- [ ] Code examples are correct
- [ ] Setup commands are copy-paste ready
- [ ] Time estimate is accurate
- [ ] Prerequisites match actual requirements

### User Experience
- [ ] Clear what the project does
- [ ] Easy to get started
- [ ] Helpful error messages
- [ ] Logical flow
- [ ] Satisfying completion

---

## Pre-Submission

### Final Review
- [ ] Spell check documentation
- [ ] Remove debug code/comments
- [ ] Clear all notebook outputs
- [ ] Remove any TODO comments
- [ ] Check for sensitive information
- [ ] Verify licensing is correct

### Repository Updates
- [ ] Add project to `docs/projects/index.md`
- [ ] Update project count
- [ ] Add to appropriate domain section
- [ ] Include difficulty and time estimate

### PR Preparation
- [ ] Write clear PR title
- [ ] Complete PR description
- [ ] Tag with appropriate labels
- [ ] Add screenshots (if applicable)
- [ ] Link to related issues

---

## Post-Submission

### Community Engagement
- [ ] Respond to review comments promptly
- [ ] Make requested changes
- [ ] Update documentation based on feedback
- [ ] Thank reviewers

### After Merge
- [ ] Share project in Discussions
- [ ] Monitor for issues
- [ ] Help users with questions
- [ ] Consider blog post or tutorial

---

## Quick Quality Check

**Ask yourself**:

1. ✅ **Would I use this if I found it?**
   - Is it clear what it does?
   - Is setup easy?
   - Does it work?

2. ✅ **Does it teach something?**
   - Clear learning objectives?
   - Good code examples?
   - Helpful comments?

3. ✅ **Is it honest?**
   - Realistic time estimates?
   - Clear limitations?
   - Accurate cost estimates?

4. ✅ **Can others build on it?**
   - Well-documented?
   - Extensible?
   - Good structure?

If you answer "no" to any, improve before submitting.

---

## Tier Classification

**Which tier is your project?**

### Tier 1: Flagship (4-5 days)
- [ ] Complete Studio Lab (500+ lines notebook)
- [ ] Complete Unified Studio (2,000+ lines modules)
- [ ] CloudFormation infrastructure
- [ ] Comprehensive docs (1,000+ lines README)
- [ ] Multiple visualizations
- [ ] Tested end-to-end

### Tier 2: Complete (2-3 days)
- [ ] Complete Studio Lab (300+ lines notebook)
- [ ] Basic Unified Studio structure
- [ ] Clear documentation (500+ lines README)
- [ ] Core visualizations
- [ ] Tested with users

### Tier 3: Starter (2-4 hours)
- [ ] README with workflow outline
- [ ] Environment specification
- [ ] Sample data description
- [ ] Quick start instructions

**All tiers are valuable! Choose based on your time and goals.**

---

## Remember

- **Perfect is the enemy of good**: A working Tier 3 helps the community
- **Start small**: You can always enhance later
- **Ask for help**: Community is here to support
- **Test thoroughly**: Your reputation is on the line
- **Be responsive**: Engage with users and reviewers
- **Have fun**: You're helping researchers worldwide!

---

*Use this checklist to ensure your project meets community standards.*

*Last updated: 2025-11-09*
