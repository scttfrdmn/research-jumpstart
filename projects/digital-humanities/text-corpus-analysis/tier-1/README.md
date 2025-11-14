# Multi-Language Multilingual Corpus Analysis

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB multi-language historical texts

## Research Goal

Perform cross-lingual analysis on a multilingual historical text corpus using ensemble transformer models. Compare writing styles across languages, identify translation patterns, and analyze cultural differences in literary expression.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB multilingual dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Complex environment** (multilingual tokenizers and models)

## What This Enables

Real research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of multilingual texts **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache preprocessed data and embeddings

### ⚡ Long-Running Training
- Train 5-6 multilingual transformer models
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### 🧪 Reproducible Environments
- Conda environment with multilingual dependencies
- Persists between sessions
- No reinstalling tokenizers
- Team members use identical setup

### 📊 Iterative Analysis
- Save cross-lingual analysis results
- Build on previous runs
- Refine models incrementally
- Collaborative research workflows

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download multilingual corpus (~10GB total)
   - Languages: English, French, German, Spanish, Italian, Russian
   - Cache in persistent storage
   - Preprocess and tokenize for each language
   - Generate parallel text alignments

2. **Ensemble Transformer Training** (5-6 hours)
   - Train mBERT for cross-lingual style analysis
   - Fine-tune XLM-RoBERTa for authorship
   - Train language-specific models
   - Checkpoint every epoch
   - Parallel training workflows

3. **Cross-Lingual Analysis** (90 min)
   - Compare writing styles across languages
   - Identify translation patterns
   - Analyze cultural differences in expression
   - Map stylistic features across languages

4. **Results Synthesis** (45 min)
   - Generate comparative visualizations
   - Identify universal vs. language-specific patterns
   - Quantify translation effects
   - Publication-ready figures and tables

## Datasets

**Multilingual Historical Corpus**
- **Languages:** 6 (English, French, German, Spanish, Italian, Russian)
- **Authors per language:** 8-10 major authors
- **Period:** 1800-1920
- **Text types:** Novels, essays, poetry, non-fiction
- **Total size:** ~10GB plain text files
- **Parallel texts:** 500+ translated works for alignment
- **Storage:** Cached in Studio Lab's 15GB persistent storage

**Sources:**
- Project Gutenberg (English)
- Wikisource (multilingual)
- European Literature Archives
- Public domain translations

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/digital-humanities/text-corpus-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate dh-multilingual

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_corpus_preparation.ipynb` - Download and cache multilingual data
2. `02_model_training.ipynb` - Train ensemble transformers
3. `03_cross_lingual_analysis.ipynb` - Analyze across languages
4. `04_visualization.ipynb` - Create comparative visualizations

## Key Features

### Persistence Example
```python
# Save multilingual model checkpoint (persists between sessions!)
model.save_pretrained('saved_models/xlm_roberta_style_v1/')
tokenizer.save_pretrained('saved_models/xlm_roberta_style_v1/')

# Load in next session
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('saved_models/xlm_roberta_style_v1/')
tokenizer = AutoTokenizer.from_pretrained('saved_models/xlm_roberta_style_v1/')
```

### Longer Computations
```python
# Run intensive cross-lingual analysis that would timeout on Colab
# Studio Lab: 12-hour GPU sessions, 4-hour CPU sessions
results = compare_styles_across_languages(
    languages=['en', 'fr', 'de', 'es', 'it', 'ru'],
    n_authors_per_lang=10
)
```

### Shared Utilities
```python
# Import from project modules (better code organization)
from src.multilingual_utils import load_parallel_texts, align_translations
from src.style_analysis import cross_lingual_style_vector
from src.visualization import create_language_comparison_plot
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_corpus_preparation.ipynb   # Download multilingual data
│   ├── 02_model_training.ipynb       # Train ensemble models
│   ├── 03_cross_lingual_analysis.ipynb  # Cross-language analysis
│   └── 04_visualization.ipynb        # Comparative visualizations
│
├── src/
│   ├── __init__.py
│   ├── multilingual_utils.py         # Multilingual data utilities
│   ├── style_analysis.py             # Cross-lingual style analysis
│   ├── translation_patterns.py       # Translation detection
│   └── visualization.py              # Plotting functions
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded corpora by language
│   ├── processed/                    # Preprocessed texts
│   ├── parallel/                     # Aligned parallel texts
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    ├── mbert/                        # Multilingual BERT
    ├── xlm_roberta/                  # XLM-RoBERTa
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB multilingual data** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Multilingual env** | ❌ Reinstall each time | ✅ Conda persists |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |
| **Cross-lingual cache** | ❌ No persistence | ✅ Cache embeddings |

**Bottom line:** Cross-lingual research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 20 minutes (one-time)
- Data download: 60 minutes (one-time, ~10GB multilingual)
- Environment setup: 15 minutes (one-time, multilingual tokenizers)
- Model training: 5-6 hours
- Analysis: 2-3 hours
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Key Methods

### Cross-Lingual Style Analysis
- **Multilingual embeddings:** Map texts to shared semantic space
- **Style transfer detection:** Identify translation effects
- **Universal features:** Find language-independent patterns
- **Language-specific traits:** Quantify cultural differences

### Ensemble Models
- **mBERT:** Multilingual BERT for style comparison
- **XLM-RoBERTa:** Cross-lingual authorship attribution
- **Language-specific:** Fine-tuned models per language
- **Ensemble voting:** Combine predictions for robustness

### Analysis Techniques
- **Parallel text alignment:** Match original-translation pairs
- **Style vector comparison:** Measure cross-lingual distances
- **Translation pattern mining:** Identify systematic changes
- **Cultural analysis:** Examine language-specific expressions

## Research Questions

This project enables investigation of:

1. **Universal vs. Cultural Style**
   - Which stylistic features transcend languages?
   - What patterns are language-specific?
   - How do cultural contexts shape expression?

2. **Translation Effects**
   - How does translation alter authorial style?
   - Can we detect translated vs. original texts?
   - What stylistic elements are preserved/lost?

3. **Temporal Patterns**
   - Do languages evolve similarly over time?
   - Are there parallel trends across cultures?
   - How do translation norms change historically?

4. **Authorship Across Languages**
   - Can we attribute authorship cross-lingually?
   - Do authors have consistent multi-lingual styles?
   - How does language affect perceived authorship?

## Expected Results

Based on similar research, you should observe:

- **85-90% accuracy** in cross-lingual authorship attribution
- **Clear clustering** by language in style embeddings
- **Translation effects** detectable with 75-80% accuracy
- **Universal patterns** in sentence structure and pacing
- **Language-specific traits** in vocabulary and formality

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3, Lambda, Comprehend) - $15-35
  - Store 100GB+ multilingual corpora
  - Automated translation pipelines
  - AWS Translate integration
  - Cross-lingual search with OpenSearch

- **Tier 3:** Production infrastructure with CloudFormation - $100-800/month
  - Multiple languages and domains (TB+)
  - Distributed NLP clusters
  - Real-time translation and style detection
  - AI-powered comparative analysis with Bedrock

## Resources

### Multilingual NLP
- [Hugging Face Multilingual Models](https://huggingface.co/models?pipeline_tag=text-classification&language=multilingual)
- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [mBERT Documentation](https://github.com/google-research/bert/blob/master/multilingual.md)

### Digital Humanities
- [European Literature Archives](https://www.europeana.eu/)
- [Multilingual Digital Humanities](https://dhq.digitalhumanities.org/)
- [Translation Studies Resources](https://www.translationstudies.net/)

### Studio Lab
- [SageMaker Studio Lab Documentation](https://studiolab.sagemaker.aws/docs)
- [Getting Started Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)
- [Community Forum](https://github.com/aws/studio-lab-examples)

## Troubleshooting

### Environment Issues
```bash
# Reset multilingual environment
conda env remove -n dh-multilingual
conda env create -f environment.yml

# If tokenizer issues persist
pip install --force-reinstall transformers tokenizers
```

### Storage Full
```bash
# Check usage by language
du -sh data/raw/*

# Clean old preprocessed files
rm -rf data/processed/old_*
rm -rf saved_models/checkpoints/epoch_*

# Keep only final models
```

### Memory Issues
```python
# Process languages in batches if hitting memory limits
languages = ['en', 'fr', 'de', 'es', 'it', 'ru']
batch_size = 2

for i in range(0, len(languages), batch_size):
    batch = languages[i:i+batch_size]
    results = process_language_batch(batch)
    save_intermediate_results(results, f'batch_{i//batch_size}')
```

### Session Timeout
Data and checkpoints persist! Just restart and continue where you left off.

```python
# Resume from checkpoint
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    'saved_models/xlm_roberta_style_v1/checkpoint-epoch-2'
)
```

## Publications Using Similar Methods

- Rybicki, J. (2012). "The great mystery of the (almost) invisible translator"
- Underwood, T. (2019). "Distant Horizons: Digital Evidence and Literary Change"
- Baroni, M. et al. (2014). "Don't count, predict! A systematic comparison"
- Kestemont, M. et al. (2016). "Authenticating the writings of Julius Caesar"

---

**🤖 Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
