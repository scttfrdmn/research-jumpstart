# Zenodo Integration Guide for Tier-0 Submodules

**Date:** 2025-12-14
**Status:** Releases created, ready for Zenodo DOI minting

---

## ✅ Prerequisites Complete

All 3 pilot repositories now have:
- ✅ v1.0.0 tags created and pushed
- ✅ GitHub releases published
- ✅ Release notes with features and citations

**Repositories:**
1. [rj-astronomy-exoplanet-detection-tier0](https://github.com/scttfrdmn/rj-astronomy-exoplanet-detection-tier0/releases/tag/v1.0.0)
2. [rj-climate-temperature-forecasting-tier0](https://github.com/scttfrdmn/rj-climate-temperature-forecasting-tier0/releases/tag/v1.0.0)
3. [rj-genomics-variant-calling-tier0](https://github.com/scttfrdmn/rj-genomics-variant-calling-tier0/releases/tag/v1.0.0)

---

## 🔗 Step 1: Enable Zenodo Integration (Manual - 5 minutes)

Zenodo integration must be enabled through the web interface:

### 1.1 Log in to Zenodo
- Go to: https://zenodo.org/
- Click "Log in" → "Log in with GitHub"
- Authorize Zenodo to access your GitHub repositories

### 1.2 Enable Repositories
- After login, go to: https://zenodo.org/account/settings/github/
- You'll see a list of your GitHub repositories
- Find and flip the switch **ON** for:
  - ☐ `scttfrdmn/rj-astronomy-exoplanet-detection-tier0`
  - ☐ `scttfrdmn/rj-climate-temperature-forecasting-tier0`
  - ☐ `scttfrdmn/rj-genomics-variant-calling-tier0`

**What this does:** Zenodo will start watching these repos for new releases.

---

## 📦 Step 2: Trigger Zenodo DOI Creation (Automatic)

Once Zenodo integration is enabled, DOIs are created automatically:

### 2.1 Zenodo Detects Existing Releases
- Since you already have v1.0.0 releases, Zenodo will **automatically** create DOI records
- This happens within **5-10 minutes** of enabling integration
- Check progress: https://zenodo.org/account/settings/github/

### 2.2 Verify DOI Records Created
- Go to: https://zenodo.org/
- Click on your profile (top right)
- Click "Uploads"
- You should see 3 new uploads (one for each repo)

---

## 🔍 Step 3: Get Your DOIs (Manual - 2 minutes)

For each repository:

### 3.1 Find the DOI Record
- Go to https://zenodo.org/ and search for your repository name
- OR go directly to your uploads: https://zenodo.org/account/settings/github/
- Click on each repository to see its DOI record

### 3.2 Identify Concept DOI vs Version DOI

Zenodo creates **two DOIs** for each repo:

```
Concept DOI (permanent):
  ├─ Points to "latest version" always
  ├─ Format: 10.5281/zenodo.1234567
  └─ Use this in READMEs and citations

Version DOI (specific):
  ├─ Points to v1.0.0 specifically
  ├─ Format: 10.5281/zenodo.1234568
  └─ Use for reproducibility (optional)
```

**Which to use?** → **Concept DOI** (it always points to the latest version)

### 3.3 Copy DOIs
Copy the **Concept DOI** for each repository:

```
Astronomy: 10.5281/zenodo._______ (fill in after Zenodo creates it)
Climate:   10.5281/zenodo._______ (fill in after Zenodo creates it)
Genomics:  10.5281/zenodo._______ (fill in after Zenodo creates it)
```

---

## 📝 Step 4: Update README Badges (I Can Automate This)

Once you have the DOIs, I'll update all README files:

### Current Placeholder:
```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

### Updated with Real DOI:
```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)
```

---

## 🤖 Step 5: Automated Badge Updates (Once You Have DOIs)

**After you get the DOIs from Zenodo, give them to me and I'll:**

1. Update README.md in each submodule repo
2. Commit and push the changes
3. Create v1.0.1 tags if needed (or just push to main)
4. Update Zenodo records automatically

---

## 📊 Expected Timeline

| Step | Time | Status |
|------|------|--------|
| Create tags and releases | 5 min | ✅ DONE |
| Enable Zenodo integration | 5 min | ⏳ MANUAL (you do this) |
| Zenodo creates DOIs | 5-10 min | ⏳ AUTOMATIC (Zenodo does this) |
| Get DOIs from Zenodo | 2 min | ⏳ MANUAL (you copy DOIs) |
| Update README badges | 2 min | ⏳ AUTOMATED (I do this) |
| **Total** | **~20 minutes** | |

---

## 🎯 Verification Checklist

After completing all steps, verify:

- [ ] All 3 repos show "DOI" badge on Zenodo settings page
- [ ] Each repo has a DOI record on Zenodo
- [ ] DOI badges in README display correctly
- [ ] Clicking DOI badge takes you to Zenodo record
- [ ] Zenodo record shows correct metadata (title, authors, description)
- [ ] Citation export works (BibTeX, EndNote, etc.)

---

## 🚨 Troubleshooting

### Problem: Zenodo doesn't see my releases
**Solution:** Make sure releases are **public** (not draft)
```bash
# Check release status
gh release view v1.0.0 --repo scttfrdmn/rj-astronomy-exoplanet-detection-tier0
```

### Problem: DOI not created after 10 minutes
**Solution:** Manually trigger DOI creation
- Go to Zenodo settings: https://zenodo.org/account/settings/github/
- Click "Sync now" button next to the repository
- Wait 2-3 minutes and refresh

### Problem: Wrong metadata in Zenodo record
**Solution:** Edit Zenodo record
- Go to your Zenodo upload
- Click "Edit"
- Update title, description, authors
- Click "Publish" to save changes

---

## 📚 Resources

- **Zenodo GitHub Integration Guide:** https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- **Zenodo DOI Documentation:** https://help.zenodo.org/
- **Example Zenodo Record:** https://zenodo.org/record/1234567 (template)

---

## 🎉 Next Steps After DOIs Are Minted

Once badges are updated:

1. **Test the workflow:**
   - Click DOI badges in each README
   - Verify Zenodo pages load correctly
   - Test BibTeX export

2. **Announce the releases:**
   - Tweet/post about the pilot repos with DOIs
   - Share in relevant communities (astronomy, climate, genomics)
   - Get user feedback

3. **Document lessons learned:**
   - What worked well?
   - Any Zenodo issues?
   - Update migration plan

4. **Prepare for full-scale migration:**
   - Create remaining 7 AI/ML notebooks
   - Migrate remaining 22 tier-0 projects
   - Automate where possible

---

## 💡 Tips for Scaling to 22 More Projects

### Efficiency Gains
1. **Batch Zenodo Integration:** Enable all repos at once in Zenodo settings
2. **Badge Updates:** Script badge updates for bulk processing
3. **Release Automation:** Create templates for release notes

### Quality Checks
1. **Verify DOI links** before announcing
2. **Check citation exports** (BibTeX format)
3. **Test Colab buttons** with DOI badges

---

**Current Status:** Releases created ✅, awaiting Zenodo integration ⏳

**Your Next Action:**
1. Go to https://zenodo.org/account/settings/github/
2. Enable integration for all 3 pilot repos
3. Wait 5-10 minutes for DOI creation
4. Copy the Concept DOIs
5. Give them to me, and I'll update all badges automatically

---

**Prepared by:** Claude Code
**Date:** 2025-12-14
