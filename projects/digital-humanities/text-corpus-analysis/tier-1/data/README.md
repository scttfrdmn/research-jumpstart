# Data Directory

This directory contains the multilingual text corpus data.

## Structure

```
data/
├── raw/                    # Downloaded raw texts by language
│   ├── en/                # English texts
│   ├── fr/                # French texts
│   ├── de/                # German texts
│   ├── es/                # Spanish texts
│   ├── it/                # Italian texts
│   └── ru/                # Russian texts
├── processed/             # Preprocessed and tokenized texts
│   ├── en/
│   ├── fr/
│   └── ...
└── parallel/              # Aligned parallel texts (original-translation pairs)
    ├── en-fr/            # English-French pairs
    ├── en-de/            # English-German pairs
    └── ...
```

## Data Sources

- **Project Gutenberg**: English texts
- **Wikisource**: Multilingual texts
- **European Literature Archives**: Historical texts
- **Public domain translations**: Parallel corpora

## Size

- Total: ~10GB
- Per language: ~1.5-2GB
- Parallel texts: ~2GB

## Note

Data files are not tracked in Git (see .gitignore).
Run `01_corpus_preparation.ipynb` to download and prepare data.
