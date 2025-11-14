# Large-Scale Corpus Linguistics

**Tier 1 Flagship Project**

Computational analysis of language across billions of words with diachronic semantics, dialectology, and multilingual comparison on AWS.

## Overview

This project demonstrates large-scale computational linguistics on AWS, enabling researchers to:

- Analyze corpora with billions of words using distributed computing
- Track semantic change across centuries with diachronic word embeddings
- Compare dialects and language varieties across regions
- Extract n-grams, collocations, and lexical bundles at scale
- Train multilingual models across 100+ languages
- Perform concordance search on massive corpora in real-time
- Study morphosyntactic patterns with dependency parsing
- Analyze language variation by register, genre, and social factors

Computational corpus linguistics has been transformed by big data and machine learning. AWS EMR and Spark enable processing of billion-word corpora in hours. Modern NLP models (BERT, GPT) trained on hundreds of billions of words reveal subtle patterns in language use. This project provides production-ready infrastructure for linguistic research at unprecedented scale.

**Key Capabilities:**
- **N-gram Analysis:** Extract frequencies, collocations, trends across time/geography
- **Diachronic Semantics:** Track word meaning changes over centuries
- **Morphosyntax:** POS tagging, dependency parsing with spaCy, Stanza, UDPipe
- **Embeddings:** Word2Vec, FastText, GloVe, BERT at scale
- **Dialectology:** Regional variation analysis with geographically-tagged data
- **Multilingual:** Cross-linguistic comparison across language families
- **Real-Time Search:** Concordance and KWIC with Elasticsearch
- **Data Sources:** COCA, BNC, Google Books, Common Crawl, OpenSubtitles

## Table of Contents

- [Features](#features)
- [Cost Estimates](#cost-estimates)
- [Getting Started](#getting-started)
- [Applications](#applications)
  - [1. Diachronic Semantic Shift Detection](#1-diachronic-semantic-shift-detection)
  - [2. Large-Scale Collocation Analysis](#2-large-scale-collocation-analysis)
  - [3. Dialectal Variation Analysis](#3-dialectal-variation-analysis)
  - [4. Multilingual Semantic Space](#4-multilingual-semantic-space)
  - [5. Register and Genre Classification](#5-register-and-genre-classification)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Performance](#performance)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Features

### N-gram Extraction and Analysis

- **Distributed Processing:** Extract 1-5 grams from billion-word corpora with EMR Spark
- **Frequency Analysis:** Count occurrences across entire corpus
- **Temporal Trends:** Track usage over decades (Google Books style)
- **Geographical Variation:** Compare across regions
- **Register Comparison:** Academic vs news vs social media

### Morphosyntactic Analysis

- **POS Tagging:** spaCy for 60+ languages, Stanza for 70+ languages
- **Dependency Parsing:** Universal Dependencies format
- **Lemmatization:** Reduce words to base forms
- **Morphological Analysis:** Case, tense, number, gender
- **Syntactic Patterns:** Extract constructions (passive voice, relative clauses)

### Semantic Analysis

- **Word Embeddings:** Train Word2Vec, FastText on custom corpora
- **Contextual Embeddings:** BERT, RoBERTa, XLM-R for 100+ languages
- **Diachronic Embeddings:** Temporal alignment to track semantic shift
- **Semantic Similarity:** Cosine distance in embedding space
- **Analogies:** Solve proportional analogies (king:queen :: man:?)
- **Polysemy Detection:** Cluster word senses

### Collocation and Lexical Bundles

- **Association Measures:** PMI, log-likelihood, t-score, MI3
- **Statistical Significance:** Chi-square, Fisher's exact test
- **Lexical Bundles:** Frequent 3-5 word sequences
- **Network Analysis:** Collocation networks with NetworkX
- **Cross-Register Comparison:** Identify register-specific collocations

### Diachronic Linguistics

- **Historical Corpora:** Google Books (1500-2019), COHA (1820-2019)
- **Semantic Change Detection:** Identify words with shifting meanings
- **Grammaticalization:** Track function word emergence
- **Lexical Replacement:** Document obsolescence (e.g., "wireless" → "radio")
- **Phonological Change:** Analyze spelling variation as proxy

### Multilingual and Cross-Linguistic

- **Parallel Corpora:** Europarl, ParaCrawl for 50+ language pairs
- **Translation Equivalents:** Map concepts across languages
- **Typology:** Compare how languages encode concepts
- **Language Families:** Phylogenetic analysis
- **Code-Switching:** Identify and analyze mixed-language text

### Concordance and KWIC

- **Elasticsearch Index:** Sub-second search on billion-word corpora
- **KWIC Display:** Keyword in context with configurable window
- **Regular Expressions:** Complex pattern matching
- **Metadata Filtering:** By time period, author, genre
- **Export:** Results to CSV, JSON for further analysis

## Cost Estimates

### Small Corpus (100M words)

**Examples:** Single language variety, one genre, limited time period
- **EMR Cluster:** m5.xlarge × 3 nodes, 5 hours = $3
- **S3 Storage:** 50 GB × $0.023/GB/month = $1/month
- **Elasticsearch:** t3.small.search, single node = $33/month
- **SageMaker Training:** ml.p3.2xlarge × 2 hours = $6
- **Total:** $40-60/month ongoing, $10-20 per analysis run

**Use Cases:** Dissertation corpus, pilot study, single author analysis

### Medium Corpus (1B words)

**Examples:** COCA, large genre collection, multi-decade corpus
- **EMR Cluster:** r5.2xlarge × 5 nodes, 20 hours = $40
- **S3 Storage:** 500 GB × $0.023/GB/month = $12/month
- **Elasticsearch:** r5.large.search × 3 nodes = $373/month
- **SageMaker Training:** ml.p3.8xlarge × 8 hours = $96
- **Athena Queries:** $50/month
- **Total:** $450-600/month ongoing, $150-250 per analysis run

**Use Cases:** Large research project, cross-register comparison, diachronic study

### Large Corpus (10B+ words)

**Examples:** Multi-language corpus, web-scale data, comprehensive diachronic analysis
- **EMR Cluster:** r5.4xlarge × 20 nodes, 50 hours = $800
- **S3 Storage:** 5 TB × $0.023/GB/month = $115/month
- **Elasticsearch:** r5.xlarge.search × 10 nodes = $1,867/month
- **SageMaker Training:** ml.p4d.24xlarge × 24 hours = $786
- **Athena Queries:** $200/month
- **Total:** $2,500-4,000/month ongoing, $1,500-3,000 per analysis run

**Use Cases:** Language change over centuries, comprehensive dialectology, multilingual typology

### Massive Corpus (100B+ words, Google Books scale)

**Examples:** Google Books Ngrams, Common Crawl subset, comprehensive web language
- **EMR Cluster:** r5.8xlarge × 50 nodes, 200 hours = $20,000
- **S3 Storage:** 50 TB × $0.023/GB/month = $1,150/month
- **Elasticsearch:** r5.2xlarge.search × 30 nodes = $11,202/month
- **SageMaker Training:** ml.p4d.24xlarge × 100 hours = $3,277
- **Athena Queries:** $1,000/month
- **Total:** $15,000-25,000/month ongoing, $10,000-30,000 per analysis run

**Use Cases:** Comprehensive language documentation, web-scale sociolinguistics, large-scale diachronic analysis

### Cost Optimization Strategies

1. **Use Spot Instances for EMR:** 50-70% savings on processing
2. **S3 Lifecycle Policies:** Move to Glacier after analysis complete
3. **Elasticsearch Snapshots:** Stop cluster when not querying, restore from snapshot
4. **Pre-trained Embeddings:** Use existing models (spaCy, Gensim) instead of training
5. **Incremental Processing:** Process in batches, save intermediate results
6. **Query Optimization:** Use Athena partitioning by year/genre for faster queries

## Getting Started

### Prerequisites

```bash
# Install AWS CLI
pip install awscli

# Install linguistic analysis packages
pip install spacy nltk gensim transformers datasets
pip install pyspark elasticsearch pandas numpy scipy scikit-learn

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf  # Transformer-based

# Download NLTK data
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

### 1. Deploy CloudFormation Stack

```bash
aws cloudformation create-stack \
  --stack-name corpus-linguistics \
  --template-body file://cloudformation/linguistics-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_IAM

# Wait for stack creation
aws cloudformation wait stack-create-complete \
  --stack-name corpus-linguistics

# Get outputs
aws cloudformation describe-stacks \
  --stack-name corpus-linguistics \
  --query 'Stacks[0].Outputs'
```

### 2. Upload Corpus Data to S3

```python
import boto3
import os

s3 = boto3.client('s3')
bucket = 'corpus-linguistics-raw-123456789'

# Upload text files
corpus_dir = '/path/to/corpus'
for root, dirs, files in os.walk(corpus_dir):
    for file in files:
        if file.endswith('.txt'):
            local_path = os.path.join(root, file)
            s3_key = f"raw-texts/{file}"
            s3.upload_file(local_path, bucket, s3_key)
            print(f"Uploaded {file}")
```

### 3. Launch EMR Cluster for Processing

```bash
aws emr create-cluster \
  --name "Corpus Linguistics Processing" \
  --release-label emr-6.12.0 \
  --applications Name=Spark Name=Hadoop \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceCount=1,InstanceType=r5.2xlarge \
    InstanceGroupType=CORE,InstanceCount=4,InstanceType=r5.2xlarge \
  --use-default-roles \
  --log-uri s3://corpus-linguistics-processed-123456789/emr-logs/
```

### 4. Run First Analysis

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram, CountVectorizer

# Initialize Spark
spark = SparkSession.builder \
    .appName("Corpus N-gram Analysis") \
    .getOrCreate()

# Load corpus from S3
df = spark.read.text("s3://corpus-linguistics-raw-123456789/raw-texts/")

# Tokenize
from pyspark.sql.functions import split, lower
tokens_df = df.withColumn("tokens", split(lower("value"), "\\s+"))

# Extract trigrams
ngram = NGram(n=3, inputCol="tokens", outputCol="trigrams")
trigrams_df = ngram.transform(tokens_df)

# Count frequencies
from pyspark.sql.functions import explode
trigram_counts = trigrams_df \
    .select(explode("trigrams").alias("trigram")) \
    .groupBy("trigram") \
    .count() \
    .orderBy("count", ascending=False)

# Show top 20
trigram_counts.show(20, truncate=False)

# Save results
trigram_counts.write.parquet("s3://corpus-linguistics-processed-123456789/trigrams/")
```

## Applications

### 1. Diachronic Semantic Shift Detection

Track how word meanings change over time using historical corpora and temporally-aligned word embeddings.

**Research Question:** How has the meaning of "gay" shifted from the 1800s to today?

#### Data Preparation

```python
import boto3
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# Load Google Books data by decade
s3 = boto3.client('s3')
bucket = 'corpus-linguistics-raw-123456789'

decades = range(1800, 2020, 10)
corpora_by_decade = {}

for decade in decades:
    # Download texts from this decade
    prefix = f'google-books/{decade}s/'
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    texts = []
    for obj in response.get('Contents', []):
        text = s3.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read().decode('utf-8')
        texts.append(text)

    corpora_by_decade[decade] = texts

print(f"Loaded {len(corpora_by_decade)} decades of text")
```

#### Train Decade-Specific Embeddings

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

embeddings_by_decade = {}

for decade, texts in corpora_by_decade.items():
    # Tokenize all texts
    sentences = []
    for text in texts:
        sentences.extend([word_tokenize(sent.lower()) for sent in text.split('.')])

    # Train Word2Vec
    model = Word2Vec(
        sentences,
        vector_size=300,
        window=5,
        min_count=50,  # Only words appearing 50+ times
        workers=16,
        epochs=10,
        sg=1  # Skip-gram
    )

    embeddings_by_decade[decade] = model
    print(f"Trained {decade}s: {len(model.wv.index_to_key)} words")
```

#### Align Embeddings Across Time (Procrustes)

```python
from scipy.linalg import orthogonal_procrustes

def align_embeddings(source_model, target_model, anchor_words):
    """Align source embeddings to target using Procrustes"""

    # Get vectors for anchor words (stable meanings)
    source_vecs = []
    target_vecs = []

    for word in anchor_words:
        if word in source_model.wv and word in target_model.wv:
            source_vecs.append(source_model.wv[word])
            target_vecs.append(target_model.wv[word])

    source_matrix = np.array(source_vecs)
    target_matrix = np.array(target_vecs)

    # Compute optimal rotation matrix
    R, _ = orthogonal_procrustes(source_matrix, target_matrix)

    # Apply rotation to all source vectors
    aligned_vectors = {}
    for word in source_model.wv.index_to_key:
        aligned_vectors[word] = source_model.wv[word] @ R

    return aligned_vectors

# Anchor words (stable meanings across time)
anchor_words = [
    'man', 'woman', 'water', 'tree', 'house', 'eat', 'drink', 'big', 'small',
    'one', 'two', 'three', 'mother', 'father', 'sun', 'moon', 'day', 'night'
]

# Align all decades to 2010s as reference
reference_model = embeddings_by_decade[2010]
aligned_embeddings = {2010: reference_model.wv}

for decade in sorted(corpora_by_decade.keys())[:-1]:
    aligned = align_embeddings(
        embeddings_by_decade[decade],
        reference_model,
        anchor_words
    )
    aligned_embeddings[decade] = aligned
    print(f"Aligned {decade}s to 2010s")
```

#### Calculate Semantic Distance Over Time

```python
def semantic_distance_over_time(word, aligned_embeddings):
    """Calculate cosine distance of word across decades"""

    decades = sorted(aligned_embeddings.keys())
    distances = []

    # Use 1900s as baseline
    baseline_decade = 1900
    if baseline_decade not in aligned_embeddings:
        baseline_decade = decades[0]

    if word not in aligned_embeddings[baseline_decade]:
        return None

    baseline_vec = aligned_embeddings[baseline_decade][word]

    for decade in decades:
        if word in aligned_embeddings[decade]:
            decade_vec = aligned_embeddings[decade][word]

            # Cosine distance (1 - cosine similarity)
            dist = cosine(baseline_vec, decade_vec)
            distances.append((decade, dist))

    return distances

# Analyze "gay"
gay_distances = semantic_distance_over_time('gay', aligned_embeddings)

# Plot
decades, distances = zip(*gay_distances)
plt.figure(figsize=(12, 6))
plt.plot(decades, distances, marker='o', linewidth=2)
plt.xlabel('Decade')
plt.ylabel('Semantic Distance from 1900s')
plt.title('Semantic Shift of "gay" Over Time')
plt.grid(True, alpha=0.3)
plt.savefig('semantic_shift_gay.png', dpi=300, bbox_inches='tight')

print(f"Maximum shift occurred in {decades[np.argmax(distances)]}: {max(distances):.3f}")
```

#### Find Nearest Neighbors Over Time

```python
def nearest_neighbors_by_decade(word, embeddings_by_decade, top_n=5):
    """Show nearest neighbors in each decade"""

    results = {}
    for decade, model in embeddings_by_decade.items():
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=top_n)
            results[decade] = neighbors

    return results

# Get neighbors for "gay"
gay_neighbors = nearest_neighbors_by_decade('gay', embeddings_by_decade, top_n=10)

for decade in sorted(gay_neighbors.keys()):
    print(f"\n{decade}s - Nearest neighbors of 'gay':")
    for neighbor, similarity in gay_neighbors[decade]:
        print(f"  {neighbor}: {similarity:.3f}")
```

**Expected Results:**
- 1800s-1900s: "merry", "cheerful", "lively", "festive"
- 1950s-1970s: Transitional period with both meanings
- 1990s-2010s: "lesbian", "homosexual", "bisexual", "LGBT", "queer"

#### Statistical Significance Testing

```python
from scipy.stats import ttest_ind

def test_semantic_shift_significance(word, aligned_embeddings, early_decades, late_decades):
    """Test if semantic shift is statistically significant"""

    # Get neighbors in early vs late periods
    early_neighbors = []
    late_neighbors = []

    reference_vec = None
    for decade in sorted(aligned_embeddings.keys()):
        if word in aligned_embeddings[decade]:
            if reference_vec is None:
                reference_vec = aligned_embeddings[decade][word]

            vec = aligned_embeddings[decade][word]
            dist = cosine(reference_vec, vec)

            if decade in early_decades:
                early_neighbors.append(dist)
            elif decade in late_decades:
                late_neighbors.append(dist)

    # t-test
    t_stat, p_value = ttest_ind(early_neighbors, late_neighbors)

    return {
        'early_mean': np.mean(early_neighbors),
        'late_mean': np.mean(late_neighbors),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Test "gay"
result = test_semantic_shift_significance(
    'gay',
    aligned_embeddings,
    early_decades=range(1800, 1950, 10),
    late_decades=range(1970, 2020, 10)
)

print(f"Semantic shift test for 'gay':")
print(f"  Early period mean distance: {result['early_mean']:.4f}")
print(f"  Late period mean distance: {result['late_mean']:.4f}")
print(f"  t-statistic: {result['t_statistic']:.4f}")
print(f"  p-value: {result['p_value']:.6f}")
print(f"  Significant: {result['significant']}")
```

#### Upload Results to S3

```python
# Save embeddings
for decade, model in embeddings_by_decade.items():
    model_path = f'/tmp/embeddings_{decade}.model'
    model.save(model_path)
    s3.upload_file(
        model_path,
        bucket,
        f'models/diachronic/word2vec_{decade}.model'
    )

# Save semantic distance data
df_distances = pd.DataFrame(gay_distances, columns=['decade', 'distance'])
df_distances.to_csv('/tmp/gay_semantic_shift.csv', index=False)
s3.upload_file(
    '/tmp/gay_semantic_shift.csv',
    bucket,
    'results/diachronic/gay_semantic_shift.csv'
)
```

**Performance:**
- Training 10 Word2Vec models (1800s-2010s): 4-6 hours on EMR (5 × r5.2xlarge)
- Alignment with Procrustes: 5-10 minutes
- Semantic distance calculation: Seconds
- Total cost: $40-60

### 2. Large-Scale Collocation Analysis

Identify statistically significant word pairs using association measures on billion-word corpora.

**Research Question:** What are the strongest collocations in academic vs conversational English?

#### Extract Bigrams with Spark

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, count, lower
from pyspark.ml.feature import NGram

spark = SparkSession.builder \
    .appName("Collocation Analysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()

# Load corpus with register annotations
df = spark.read.parquet("s3://corpus-linguistics-processed-123456789/annotated-corpus/")

# Filter by register
academic_df = df.filter(col("register") == "academic")
conversation_df = df.filter(col("register") == "conversation")

def extract_bigrams(df, register_name):
    # Tokenize
    from pyspark.sql.functions import split
    tokens_df = df.withColumn("tokens", split(lower("text"), "\\s+"))

    # Extract bigrams
    ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
    bigrams_df = ngram.transform(tokens_df)

    # Count frequencies
    bigram_freq = bigrams_df \
        .select(explode("bigrams").alias("bigram")) \
        .groupBy("bigram") \
        .agg(count("*").alias("freq")) \
        .orderBy("freq", ascending=False)

    # Save
    bigram_freq.write.mode("overwrite").parquet(
        f"s3://corpus-linguistics-processed-123456789/bigrams/{register_name}/"
    )

    return bigram_freq

academic_bigrams = extract_bigrams(academic_df, "academic")
conversation_bigrams = extract_bigrams(conversation_df, "conversation")

print(f"Academic bigrams: {academic_bigrams.count()}")
print(f"Conversation bigrams: {conversation_bigrams.count()}")
```

#### Calculate Association Measures

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def calculate_pmi(w1_freq, w2_freq, bigram_freq, corpus_size):
    """Calculate Pointwise Mutual Information"""

    p_w1 = w1_freq / corpus_size
    p_w2 = w2_freq / corpus_size
    p_w1_w2 = bigram_freq / corpus_size

    if p_w1 * p_w2 == 0:
        return 0

    pmi = np.log2(p_w1_w2 / (p_w1 * p_w2))
    return pmi

def calculate_log_likelihood(w1_freq, w2_freq, bigram_freq, corpus_size):
    """Calculate log-likelihood ratio"""

    # 2x2 contingency table
    a = bigram_freq  # w1 and w2 together
    b = w1_freq - a  # w1 without w2
    c = w2_freq - a  # w2 without w1
    d = corpus_size - a - b - c  # neither

    contingency_table = np.array([[a, b], [c, d]])

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    log_likelihood = chi2

    return log_likelihood, p_value

def calculate_t_score(w1_freq, w2_freq, bigram_freq, corpus_size):
    """Calculate t-score"""

    p_w1 = w1_freq / corpus_size
    p_w2 = w2_freq / corpus_size
    p_w1_w2 = bigram_freq / corpus_size

    expected = p_w1 * p_w2 * corpus_size

    if expected == 0:
        return 0

    t_score = (bigram_freq - expected) / np.sqrt(bigram_freq)
    return t_score

# Load bigram frequencies
academic_bigrams_df = pd.read_parquet(
    "s3://corpus-linguistics-processed-123456789/bigrams/academic/"
)

# Load unigram frequencies
unigram_freq = pd.read_parquet(
    "s3://corpus-linguistics-processed-123456789/unigrams/academic/"
)
unigram_dict = dict(zip(unigram_freq['word'], unigram_freq['freq']))

corpus_size = unigram_freq['freq'].sum()

# Calculate association measures for each bigram
results = []
for idx, row in academic_bigrams_df.iterrows():
    bigram = row['bigram']
    w1, w2 = bigram.split()

    bigram_freq = row['freq']
    w1_freq = unigram_dict.get(w1, 0)
    w2_freq = unigram_dict.get(w2, 0)

    if bigram_freq < 5:  # Minimum frequency threshold
        continue

    pmi = calculate_pmi(w1_freq, w2_freq, bigram_freq, corpus_size)
    log_lik, p_value = calculate_log_likelihood(w1_freq, w2_freq, bigram_freq, corpus_size)
    t_score = calculate_t_score(w1_freq, w2_freq, bigram_freq, corpus_size)

    results.append({
        'bigram': bigram,
        'freq': bigram_freq,
        'pmi': pmi,
        'log_likelihood': log_lik,
        't_score': t_score,
        'p_value': p_value
    })

df_results = pd.DataFrame(results)

# Sort by log-likelihood (most commonly used)
df_results_sorted = df_results.sort_values('log_likelihood', ascending=False)

print("Top 20 collocations in academic English:")
print(df_results_sorted[['bigram', 'freq', 'log_likelihood', 'pmi']].head(20))

# Save results
df_results_sorted.to_csv('/tmp/academic_collocations.csv', index=False)
s3.upload_file(
    '/tmp/academic_collocations.csv',
    'corpus-linguistics-processed-123456789',
    'results/collocations/academic_collocations.csv'
)
```

#### Compare Across Registers

```python
def calculate_keyness(freq1, size1, freq2, size2):
    """Calculate keyness (log-likelihood) for word/bigram across two corpora"""

    a = freq1
    b = freq2
    c = size1 - a
    d = size2 - b

    contingency_table = np.array([[a, b], [c, d]])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Effect size (effect direction)
    e1 = (a + b) * size1 / (size1 + size2)

    if a > e1:
        keyness = chi2  # Positive keyness (more in corpus 1)
    else:
        keyness = -chi2  # Negative keyness (more in corpus 2)

    return keyness, p_value

# Load both registers
academic_collocs = pd.read_csv('/tmp/academic_collocations.csv')
conversation_collocs = pd.read_parquet(
    "s3://corpus-linguistics-processed-123456789/bigrams/conversation/"
)

academic_size = academic_bigrams_df['freq'].sum()
conversation_size = conversation_collocs['freq'].sum()

# Calculate keyness for each bigram
academic_dict = dict(zip(academic_collocs['bigram'], academic_collocs['freq']))
conversation_dict = dict(zip(conversation_collocs['bigram'], conversation_collocs['freq']))

all_bigrams = set(academic_dict.keys()) | set(conversation_dict.keys())

keyness_results = []
for bigram in all_bigrams:
    freq_academic = academic_dict.get(bigram, 0)
    freq_conversation = conversation_dict.get(bigram, 0)

    if freq_academic + freq_conversation < 10:
        continue

    keyness, p_value = calculate_keyness(
        freq_academic, academic_size,
        freq_conversation, conversation_size
    )

    keyness_results.append({
        'bigram': bigram,
        'freq_academic': freq_academic,
        'freq_conversation': freq_conversation,
        'keyness': keyness,
        'p_value': p_value
    })

df_keyness = pd.DataFrame(keyness_results)
df_keyness_sorted = df_keyness.sort_values('keyness', ascending=False)

print("\nTop 20 collocations distinctive of academic English:")
print(df_keyness_sorted[['bigram', 'keyness', 'p_value']].head(20))

print("\nTop 20 collocations distinctive of conversational English:")
print(df_keyness_sorted[['bigram', 'keyness', 'p_value']].tail(20))
```

**Expected Results:**
- Academic: "research shows", "findings indicate", "statistically significant", "present study"
- Conversational: "you know", "I mean", "sort of", "kind of"

**Performance:**
- Bigram extraction on 1B words: 2-3 hours on EMR (5 × r5.2xlarge)
- Association measure calculation: 30-60 minutes
- Total cost: $30-50

### 3. Dialectal Variation Analysis

Compare language use across geographical regions using geographically-tagged social media data.

**Research Question:** What are the lexical differences between American and British English?

#### Load Geographically-Tagged Corpus

```python
import pandas as pd
import geopandas as gpd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Load tweets with location metadata
tweets_df = pd.read_parquet("s3://corpus-linguistics-raw-123456789/twitter/geotagged/")

# Filter for US and UK
us_tweets = tweets_df[tweets_df['country'] == 'US']
uk_tweets = tweets_df[tweets_df['country'] == 'UK']

print(f"US tweets: {len(us_tweets)}")
print(f"UK tweets: {len(uk_tweets)}")

# Sample 100K each for balance
us_sample = us_tweets.sample(100000, random_state=42)
uk_sample = uk_tweets.sample(100000, random_state=42)

combined = pd.concat([us_sample, uk_sample])
```

#### Extract Lexical Features

```python
# Vectorize with TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=10,
    max_df=0.5,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(combined['text'])
feature_names = vectorizer.get_feature_names_out()

# Create labels
y = combined['country'].values

print(f"Feature matrix shape: {X.shape}")
```

#### Identify Distinctive Features

```python
from scipy.stats import mannwhitneyu

def find_distinctive_words(X, y, feature_names, n_top=50):
    """Find words most distinctive of each variety"""

    us_mask = (y == 'US')
    uk_mask = (y == 'UK')

    us_X = X[us_mask].toarray()
    uk_X = X[uk_mask].toarray()

    results = []
    for i, word in enumerate(feature_names):
        us_vals = us_X[:, i]
        uk_vals = uk_X[:, i]

        us_mean = us_vals.mean()
        uk_mean = uk_vals.mean()

        # Mann-Whitney U test
        if us_vals.sum() > 0 and uk_vals.sum() > 0:
            u_stat, p_value = mannwhitneyu(us_vals, uk_vals, alternative='two-sided')

            results.append({
                'word': word,
                'us_mean': us_mean,
                'uk_mean': uk_mean,
                'diff': us_mean - uk_mean,
                'p_value': p_value
            })

    df_results = pd.DataFrame(results)
    df_results = df_results[df_results['p_value'] < 0.001]  # Significant only

    # Top US-distinctive
    us_distinctive = df_results.sort_values('diff', ascending=False).head(n_top)

    # Top UK-distinctive
    uk_distinctive = df_results.sort_values('diff', ascending=True).head(n_top)

    return us_distinctive, uk_distinctive

us_words, uk_words = find_distinctive_words(X, y, feature_names)

print("Top US-distinctive words:")
print(us_words[['word', 'us_mean', 'uk_mean', 'diff']].head(20))

print("\nTop UK-distinctive words:")
print(uk_words[['word', 'us_mean', 'uk_mean', 'diff']].head(20))
```

**Expected Results:**
- US: "mom", "gotten", "color", "apartment", "candy", "soccer", "fall" (season)
- UK: "mum", "colour", "flat", "sweets", "football", "autumn", "lorry", "loo"

#### Visualize Dialectal Differences

```python
# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(12, 8))
us_mask = (y == 'US')
plt.scatter(X_pca[us_mask, 0], X_pca[us_mask, 1],
            alpha=0.3, s=1, c='blue', label='US English')
plt.scatter(X_pca[~us_mask, 0], X_pca[~us_mask, 1],
            alpha=0.3, s=1, c='red', label='UK English')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Dialectal Variation: US vs UK English')
plt.legend()
plt.tight_layout()
plt.savefig('dialect_pca.png', dpi=300)
```

#### Regional Analysis Within US

```python
# Load US tweets with state information
us_tweets_detailed = tweets_df[tweets_df['country'] == 'US']

# Group by state
states_of_interest = ['CA', 'TX', 'NY', 'FL', 'MA', 'LA', 'MN', 'WA']
state_samples = []

for state in states_of_interest:
    state_df = us_tweets_detailed[us_tweets_detailed['state'] == state]
    if len(state_df) > 5000:
        sample = state_df.sample(5000, random_state=42)
        state_samples.append(sample)

combined_states = pd.concat(state_samples)

# Vectorize
vectorizer_states = TfidfVectorizer(max_features=2000, min_df=5)
X_states = vectorizer_states.fit_transform(combined_states['text'])

# MDS for 2D visualization
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)

# Calculate pairwise distances between states
from sklearn.metrics.pairwise import cosine_distances

state_centroids = []
for state in states_of_interest:
    mask = combined_states['state'] == state
    centroid = X_states[mask].mean(axis=0)
    state_centroids.append(centroid)

state_centroids = np.vstack(state_centroids)
distances = cosine_distances(state_centroids)

# MDS projection
state_coords = mds.fit_transform(distances)

# Plot
plt.figure(figsize=(10, 8))
for i, state in enumerate(states_of_interest):
    plt.scatter(state_coords[i, 0], state_coords[i, 1], s=200, alpha=0.7)
    plt.annotate(state, (state_coords[i, 0], state_coords[i, 1]),
                fontsize=12, ha='center')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('Regional Linguistic Variation Within US')
plt.tight_layout()
plt.savefig('us_regional_variation.png', dpi=300)
```

**Performance:**
- Feature extraction on 200K tweets: 5-10 minutes
- Statistical testing: 5-10 minutes
- PCA/MDS: 1-2 minutes
- Total: 20-30 minutes
- Cost: $2-5 (single machine)

### 4. Multilingual Semantic Space

Train cross-lingual word embeddings to compare how different languages encode concepts.

**Research Question:** How do languages differ in their semantic organization of kinship terms?

#### Load Parallel Corpora

```python
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel

# Load parallel corpus (Europarl)
# English-Spanish-French-German-Italian
languages = ['en', 'es', 'fr', 'de', 'it']

parallel_data = {}
for lang in languages:
    dataset = load_dataset('europarl_bilingual', f'en-{lang}')
    parallel_data[lang] = dataset['train']
    print(f"Loaded {len(parallel_data[lang])} sentence pairs for {lang}")
```

#### Train Multilingual Embeddings

```python
# Use pre-trained multilingual BERT
model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def get_word_embedding(word, language_code):
    """Extract BERT embedding for a word"""

    # Tokenize
    inputs = tokenizer(word, return_tensors='pt').to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()

    return embedding

# Extract embeddings for kinship terms
kinship_terms = {
    'en': ['mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather',
           'aunt', 'uncle', 'daughter', 'son', 'niece', 'nephew'],
    'es': ['madre', 'padre', 'hermana', 'hermano', 'abuela', 'abuelo',
           'tía', 'tío', 'hija', 'hijo', 'sobrina', 'sobrino'],
    'fr': ['mère', 'père', 'sœur', 'frère', 'grand-mère', 'grand-père',
           'tante', 'oncle', 'fille', 'fils', 'nièce', 'neveu'],
    'de': ['Mutter', 'Vater', 'Schwester', 'Bruder', 'Großmutter', 'Großvater',
           'Tante', 'Onkel', 'Tochter', 'Sohn', 'Nichte', 'Neffe'],
    'it': ['madre', 'padre', 'sorella', 'fratello', 'nonna', 'nonno',
           'zia', 'zio', 'figlia', 'figlio', 'nipote', 'nipote']
}

embeddings = {}
for lang, terms in kinship_terms.items():
    embeddings[lang] = {}
    for term in terms:
        emb = get_word_embedding(term, lang)
        embeddings[lang][term] = emb
        print(f"Got embedding for {lang}:{term}")
```

#### Analyze Semantic Structure

```python
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

def analyze_semantic_structure(embeddings_dict, language):
    """Analyze how a language organizes kinship terms"""

    terms = list(embeddings_dict.keys())
    vectors = np.array([embeddings_dict[term] for term in terms])

    # Compute similarity matrix
    sim_matrix = cosine_similarity(vectors)

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=terms, yticklabels=terms,
                cmap='viridis', vmin=0, vmax=1, annot=True, fmt='.2f')
    plt.title(f'Kinship Term Similarity: {language}')
    plt.tight_layout()
    plt.savefig(f'kinship_similarity_{language}.png', dpi=300)

    # Hierarchical clustering
    plt.figure(figsize=(10, 6))
    linkage_matrix = linkage(vectors, method='ward')
    dendrogram(linkage_matrix, labels=terms, leaf_font_size=10)
    plt.title(f'Hierarchical Clustering of Kinship Terms: {language}')
    plt.xlabel('Term')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'kinship_dendrogram_{language}.png', dpi=300)

    return sim_matrix

# Analyze each language
for lang in languages:
    sim_matrix = analyze_semantic_structure(embeddings[lang], lang)
```

#### Compare Across Languages

```python
def compare_semantic_organization(embeddings_by_lang):
    """Compare how different languages organize the semantic space"""

    # For each language, compute within-category vs between-category similarity
    results = []

    categories = {
        'parent': [0, 1],  # mother, father
        'sibling': [2, 3],  # sister, brother
        'grandparent': [4, 5],
        'aunt/uncle': [6, 7],
        'child': [8, 9],
        'niece/nephew': [10, 11]
    }

    for lang, emb_dict in embeddings_by_lang.items():
        terms = list(emb_dict.keys())
        vectors = np.array([emb_dict[term] for term in terms])

        sim_matrix = cosine_similarity(vectors)

        # Within-category similarity
        within_sims = []
        for category, indices in categories.items():
            for i in indices:
                for j in indices:
                    if i < j:
                        within_sims.append(sim_matrix[i, j])

        # Between-category similarity
        between_sims = []
        for cat1_indices in categories.values():
            for cat2_indices in categories.values():
                if cat1_indices != cat2_indices:
                    for i in cat1_indices:
                        for j in cat2_indices:
                            between_sims.append(sim_matrix[i, j])

        results.append({
            'language': lang,
            'within_mean': np.mean(within_sims),
            'between_mean': np.mean(between_sims),
            'diff': np.mean(within_sims) - np.mean(between_sims)
        })

    return pd.DataFrame(results)

df_comparison = compare_semantic_organization(embeddings)
print(df_comparison)

# Plot
plt.figure(figsize=(10, 6))
x = np.arange(len(languages))
width = 0.35

plt.bar(x - width/2, df_comparison['within_mean'], width, label='Within-category')
plt.bar(x + width/2, df_comparison['between_mean'], width, label='Between-category')

plt.xlabel('Language')
plt.ylabel('Mean Cosine Similarity')
plt.title('Semantic Organization of Kinship Terms Across Languages')
plt.xticks(x, languages)
plt.legend()
plt.tight_layout()
plt.savefig('cross_linguistic_kinship.png', dpi=300)
```

**Expected Insights:**
- Some languages may have tighter within-category clustering (more distinct categories)
- Gender distinctions may be encoded differently across languages
- Generational structure (parent > child > grandchild) should be similar across languages

**Performance:**
- Loading mBERT model: 1-2 minutes
- Embedding extraction for 60 words: 5-10 minutes
- Analysis and visualization: 5 minutes
- Total: 15-20 minutes
- Cost: $5-10 (ml.p3.2xlarge on SageMaker for 1 hour)

### 5. Register and Genre Classification

Train classifiers to automatically identify text register using linguistic features.

**Research Question:** Can we classify texts by register with high accuracy?

#### Load Multi-Register Corpus

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load corpus with register labels
corpus_df = pd.read_parquet("s3://corpus-linguistics-processed-123456789/multi-register-corpus/")

# Registers: academic, news, fiction, web, social_media, legal, medical
print("Register distribution:")
print(corpus_df['register'].value_counts())

# Sample 10K from each register for balance
balanced_samples = []
for register in corpus_df['register'].unique():
    register_df = corpus_df[corpus_df['register'] == register]
    sample = register_df.sample(min(10000, len(register_df)), random_state=42)
    balanced_samples.append(sample)

balanced_df = pd.concat(balanced_samples)
print(f"\nBalanced corpus size: {len(balanced_df)}")
```

#### Feature Engineering

```python
import spacy

nlp = spacy.load('en_core_web_sm')

def extract_linguistic_features(text):
    """Extract linguistic features for register classification"""

    doc = nlp(text)

    features = {
        # Lexical features
        'avg_word_length': np.mean([len(token.text) for token in doc]),
        'type_token_ratio': len(set([token.text for token in doc])) / len(doc),

        # Syntactic features
        'avg_sentence_length': np.mean([len(list(sent)) for sent in doc.sents]),
        'noun_ratio': len([t for t in doc if t.pos_ == 'NOUN']) / len(doc),
        'verb_ratio': len([t for t in doc if t.pos_ == 'VERB']) / len(doc),
        'adj_ratio': len([t for t in doc if t.pos_ == 'ADJ']) / len(doc),
        'adv_ratio': len([t for t in doc if t.pos_ == 'ADV']) / len(doc),
        'pron_ratio': len([t for t in doc if t.pos_ == 'PRON']) / len(doc),

        # Dependency features
        'passive_voice': len([t for t in doc if t.dep_ == 'nsubjpass']) / len(list(doc.sents)),
        'subordinate_clauses': len([t for t in doc if t.dep_ == 'mark']) / len(list(doc.sents)),

        # Named entities
        'ner_density': len(doc.ents) / len(doc)
    }

    return features

# Extract features for sample (this is slow, use Spark for full corpus)
sample_size = 1000
sample_df = balanced_df.sample(sample_size, random_state=42)

linguistic_features = []
for text in sample_df['text']:
    if len(text) > 10:  # Skip very short texts
        features = extract_linguistic_features(text[:5000])  # Limit length
        linguistic_features.append(features)
    else:
        linguistic_features.append(None)

features_df = pd.DataFrame([f for f in linguistic_features if f is not None])
```

#### Train Classifier

```python
# Combine TF-IDF with linguistic features
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_tfidf = vectorizer.fit_transform(balanced_df['text'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, balanced_df['register'],
    test_size=0.2, random_state=42, stratify=balanced_df['register']
)

# Train multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results[name] = {
        'model': clf,
        'accuracy': accuracy,
        'predictions': y_pred
    }

# Best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.3f}")
```

#### Fine-Tune BERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# Prepare data for BERT
train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

# Create label mapping
label2id = {label: i for i, label in enumerate(balanced_df['register'].unique())}
id2label = {i: label for label, i in label2id.items()}

train_df['label'] = train_df['register'].map(label2id)
test_df['label'] = test_df['register'].map(label2id)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

# Tokenize
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"\nBERT Fine-tuning Results:")
print(f"Accuracy: {results['eval_accuracy']:.3f}")
```

#### Interpret Features

```python
# Feature importance from Random Forest
rf_model = results['Random Forest']['model']
feature_importance = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

# Top features
top_indices = np.argsort(feature_importance)[-50:]
top_features = [(feature_names[i], feature_importance[i]) for i in top_indices]

print("\nTop 20 most important features for register classification:")
for feature, importance in reversed(top_features[-20:]):
    print(f"  {feature}: {importance:.4f}")

# Visualize confusion matrix
y_pred_best = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=balanced_df['register'].unique(),
            yticklabels=balanced_df['register'].unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix: {best_model_name}')
plt.tight_layout()
plt.savefig('register_confusion_matrix.png', dpi=300)
```

**Expected Results:**
- Logistic Regression: 80-85% accuracy
- Random Forest: 85-90% accuracy
- BERT Fine-tuned: 90-95% accuracy

**Most Distinctive Features by Register:**
- Academic: "research", "findings", "significant", "analysis", "data"
- News: "said", "according", "reported", "yesterday", "officials"
- Fiction: "she", "her", "his", "said", narrative past tense
- Web: "click", "here", "page", "search", "site"
- Social Media: "lol", "omg", "ur", emoticons, informal spelling
- Legal: "shall", "hereby", "pursuant", "aforementioned", "thereof"
- Medical: "patient", "treatment", "diagnosis", "mg", "prescribed"

**Performance:**
- Feature extraction (linguistic): 2-4 hours for 70K texts
- TF-IDF vectorization: 5-10 minutes
- Traditional ML training: 5-15 minutes
- BERT fine-tuning: 2-3 hours on ml.p3.2xlarge
- Total cost: $30-50

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Researchers                              │
│              (Jupyter, EMR Notebooks, REST API)                 │
└────────────┬────────────────────────────────────┬────────────────┘
             │                                    │
             v                                    v
┌─────────────────────────┐         ┌──────────────────────────┐
│  EMR Spark Cluster      │         │   SageMaker              │
│  - N-gram extraction    │         │   - Embedding training   │
│  - POS tagging at scale │         │   - BERT fine-tuning     │
│  - Collocation analysis │         │   - Register classifiers │
│  - Distributed spaCy    │         └──────────┬───────────────┘
└────────┬────────────────┘                    │
         │                                      │
         v                                      v
┌─────────────────────────────────────────────────────────────────┐
│                         S3 Data Lake                             │
│  - Raw corpora          - Processed features    - Embeddings     │
│  - N-gram frequencies   - POS-tagged texts      - Trained models │
└─────────┬───────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────────┐
│                 Elasticsearch Cluster                            │
│                 - Concordance search                             │
│                 - KWIC (keyword in context)                      │
│                 - Sub-second queries on billions of words        │
└─────────────────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────────┐
│                     Analysis & Visualization                     │
│  Athena: SQL queries on linguistic features                     │
│  QuickSight: Frequency trends, geographic visualization          │
│  Lambda: On-demand analysis APIs                                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion:** Raw texts uploaded to S3 from corpus sources
2. **Processing:** EMR Spark processes at scale (tokenization, POS tagging, n-grams)
3. **Indexing:** Elasticsearch ingests processed texts for search
4. **Training:** SageMaker trains embeddings and classifiers
5. **Analysis:** Athena queries, QuickSight dashboards, custom notebooks
6. **Search:** Elasticsearch provides concordance and KWIC

## Data Sources

### Open Corpora

**Corpus of Contemporary American English (COCA):**
- **Size:** 1 billion words (1990-2019)
- **Genres:** Spoken, fiction, popular magazines, newspapers, academic
- **Access:** Free samples, full corpus requires license
- **Format:** Annotated with POS tags, lemmas
- **Use:** Diachronic analysis, genre comparison

**British National Corpus (BNC):**
- **Size:** 100 million words
- **Composition:** 90% written, 10% spoken
- **Period:** 1980s-1990s
- **Access:** Free XML version available
- **Use:** British English reference corpus

**Google Books Ngrams:**
- **Size:** 500+ billion words (1500-2019)
- **Languages:** English, Spanish, French, German, Italian, Russian, Hebrew, Chinese
- **Format:** N-gram frequencies by year
- **Access:** Free download (multiple terabytes)
- **Use:** Large-scale diachronic analysis

**Common Crawl:**
- **Size:** 100+ billion webpages
- **Update:** Monthly crawls
- **Languages:** 100+ languages
- **Access:** Free S3 bucket (requester pays)
- **Format:** WARC files
- **Use:** Web language, contemporary usage

**OpenSubtitles:**
- **Size:** 4+ billion sentences
- **Languages:** 60+ languages
- **Domain:** Movie and TV subtitles
- **Access:** Free download
- **Use:** Conversational language, multilingualanalysis

### Specialized Corpora

**Universal Dependencies:**
- **Size:** 200+ treebanks, 100+ languages
- **Annotation:** Morphology, syntax (dependency trees)
- **Access:** Free GitHub repository
- **Use:** Cross-linguistic syntactic analysis

**ParaCrawl:**
- **Size:** Billions of parallel sentences
- **Languages:** 50+ language pairs with English
- **Access:** Free download
- **Use:** Translation, cross-linguistic semantics

**Twitter/Reddit APIs:**
- **Size:** Real-time, millions of posts daily
- **Languages:** All major languages
- **Metadata:** Location, time, user demographics
- **Access:** API (rate-limited)
- **Use:** Contemporary language, dialectology, sociolinguistics

**Project Gutenberg:**
- **Size:** 70,000+ books
- **Period:** Mostly pre-1928 (out of copyright)
- **Languages:** English, other European languages
- **Access:** Free download
- **Use:** Historical English, literary analysis

## Performance

### N-gram Extraction

**100M words:**
- **Setup:** EMR with 5 × r5.2xlarge nodes
- **Runtime:** 30-45 minutes
- **Throughput:** 2-3 million words/minute
- **Cost:** $4-6

**1B words:**
- **Setup:** EMR with 10 × r5.2xlarge nodes
- **Runtime:** 2-3 hours
- **Throughput:** 5-8 million words/minute
- **Cost:** $30-45

**10B words:**
- **Setup:** EMR with 20 × r5.4xlarge nodes
- **Runtime:** 8-12 hours
- **Throughput:** 15-20 million words/minute
- **Cost:** $200-300

### POS Tagging

**spaCy Performance (single machine):**
- **Speed:** 10,000-50,000 words/second depending on model
- **Models:** en_core_web_sm (fastest), en_core_web_trf (most accurate)
- **Scaling:** Distribute with Spark UDFs on EMR

**1M words:**
- **Single machine:** 1-5 minutes
- **Cost:** $0.50-1

**100M words:**
- **EMR (5 nodes):** 1-2 hours
- **Cost:** $8-15

### Embedding Training

**Word2Vec on 1B words:**
- **Instance:** ml.c5.18xlarge (72 vCPU)
- **Runtime:** 4-8 hours
- **Cost:** $25-50
- **Result:** 300-dim vectors, 500K vocabulary

**BERT Fine-Tuning:**
- **Instance:** ml.p3.8xlarge (4× V100 GPUs)
- **Dataset:** 100K labeled examples
- **Runtime:** 3-6 hours
- **Cost:** $40-80

### Elasticsearch Performance

**Concordance Search:**
- **Cluster:** 3 × r5.large.search
- **Corpus:** 1B words indexed
- **Query Latency:** 50-200ms for simple queries
- **Complex Regex:** 500-2000ms
- **Cost:** $373/month (continuous)

**Indexing Throughput:**
- **Rate:** 5,000-10,000 documents/second
- **1B words:** 4-8 hours to index
- **Storage:** ~2-3x corpus size (with replicas)

## Best Practices

### Corpus Design

1. **Representativeness:**
   - Ensure corpus represents target population/language variety
   - Balance across genres, time periods, speakers if studying variation
   - Document sampling methodology clearly

2. **Size:**
   - 1M words minimum for stable lexical statistics
   - 10M+ words for syntactic patterns
   - 100M+ words for rare constructions, diachronic analysis
   - 1B+ words for distributional semantics

3. **Metadata:**
   - Include time, genre, author/speaker demographics, geography
   - Enable stratified sampling and subgroup analysis
   - Document metadata schema clearly

### Tokenization

1. **Unicode Handling:**
   - Normalize to NFC or NFD consistently
   - Handle non-ASCII characters properly
   - Consider language-specific rules (e.g., Arabic, Chinese)

2. **Contractions and Clitics:**
   - Decide on splitting strategy ("don't" → "do" + "n't" or keep as one)
   - Be consistent across corpus
   - Document decision for reproducibility

3. **Punctuation:**
   - Consider whether to remove or keep
   - Sentence-final punctuation affects sentence segmentation
   - Emoticons and emojis: keep for social media corpora

### Statistical Significance

1. **Frequency Thresholds:**
   - Filter out hapax legomena (frequency = 1) for many analyses
   - Use minimum frequency of 5-10 for reliable statistics
   - Higher thresholds (50+) for association measures

2. **Multiple Testing:**
   - Apply Bonferroni or false discovery rate (FDR) correction
   - When testing thousands of words, adjust p-value threshold

3. **Effect Size:**
   - Don't rely on p-values alone
   - Report effect sizes (Cohen's d, log-likelihood, etc.)
   - Large corpora find significant effects even for tiny differences

### Reproducibility

1. **Corpus Versioning:**
   - Document exact corpus version and date accessed
   - Archive corpus or provide DOI if possible
   - Note any sampling or filtering applied

2. **Code and Parameters:**
   - Share analysis scripts (GitHub, OSF)
   - Document hyperparameters for embeddings, classifiers
   - Use random seeds for reproducible sampling

3. **Preprocessing:**
   - Document all preprocessing steps
   - Save preprocessed data for replication
   - Note software versions (spaCy, NLTK, etc.)

### Cost Optimization

1. **EMR Spot Instances:**
   - 50-70% cost savings
   - Set bid price appropriately
   - Handle interruptions gracefully

2. **S3 Lifecycle:**
   - Move raw corpora to Glacier after processing
   - Keep frequently accessed processed data in S3 Standard
   - Delete intermediate results after analysis

3. **Elasticsearch:**
   - Stop cluster when not querying
   - Create snapshot to S3
   - Restore from snapshot when needed (10-30 minutes)
   - Can save $10,000+/month for large clusters

4. **Pre-trained Models:**
   - Use spaCy, Gensim pre-trained models when possible
   - Download Word2Vec, GloVe, BERT from public sources
   - Only train custom embeddings if corpus is specialized

## Troubleshooting

### Character Encoding Issues

**Problem:** Garbled text with strange characters
```
Invalid start byte at position X
```

**Solutions:**
1. Detect encoding with `chardet`:
```python
import chardet

with open('corpus.txt', 'rb') as f:
    result = chardet.detect(f.read(100000))
    encoding = result['encoding']

# Read with detected encoding
df = pd.read_csv('corpus.txt', encoding=encoding)
```

2. Try common encodings: UTF-8, Latin-1, Windows-1252
3. Use `errors='ignore'` or `errors='replace'` to skip problematic characters

### Memory Issues with Large Embeddings

**Problem:** Out of memory when loading Word2Vec models

**Solutions:**
1. Use memory-mapped files:
```python
from gensim.models import KeyedVectors

# Save in word2vec format
model.wv.save_word2vec_format('embeddings.txt', binary=True)

# Load with memory mapping
wv = KeyedVectors.load_word2vec_format('embeddings.txt', binary=True, mmap='r')
```

2. Load only needed words:
```python
# Load subset of vocabulary
wv = KeyedVectors.load_word2vec_format('embeddings.txt', binary=True,
                                       limit=100000)  # Top 100K words
```

3. Use larger instance (r5.8xlarge, r5.16xlarge)

### Elasticsearch Cluster Issues

**Problem:** Cluster status RED, some shards unassigned

**Solutions:**
1. Check cluster health:
```bash
curl -X GET "localhost:9200/_cluster/health?pretty"
```

2. Increase replica count if you have multiple nodes:
```bash
curl -X PUT "localhost:9200/_settings" -H 'Content-Type: application/json' -d'
{
  "index": {
    "number_of_replicas": 2
  }
}
'
```

3. Allocate unassigned shards:
```bash
curl -X POST "localhost:9200/_cluster/reroute?pretty"
```

4. If persistent: delete and recreate index

**Problem:** Slow queries on large corpus

**Solutions:**
1. Use filters instead of queries when possible (faster)
2. Increase `index.refresh_interval` during bulk indexing
3. Add more nodes to cluster
4. Use specific fields instead of `_all`
5. Optimize query with `explain` API

### EMR Job Failures

**Problem:** Spark job fails with out-of-memory errors

**Solutions:**
1. Increase executor memory:
```python
spark = SparkSession.builder \
    .config("spark.executor.memory", "32g") \
    .config("spark.driver.memory", "16g") \
    .getOrCreate()
```

2. Increase number of partitions:
```python
df = df.repartition(1000)  # More partitions = less memory per partition
```

3. Use larger instance types (r5.4xlarge, r5.8xlarge)

4. Enable dynamic allocation:
```python
.config("spark.dynamicAllocation.enabled", "true")
.config("spark.shuffle.service.enabled", "true")
```

## Additional Resources

### Corpus Linguistics Textbooks

- **Biber, D., Conrad, S., & Reppen, R. (1998).** *Corpus Linguistics: Investigating Language Structure and Use.* Cambridge University Press.
- **McEnery, T., & Hardie, A. (2011).** *Corpus Linguistics: Method, Theory and Practice.* Cambridge University Press.
- **Gries, S. T. (2017).** *Quantitative Corpus Linguistics with R.* Routledge.

### NLP Tools and Libraries

- **spaCy:** https://spacy.io/ - Industrial-strength NLP in Python
- **NLTK:** https://www.nltk.org/ - Natural Language Toolkit
- **Gensim:** https://radimrehurek.com/gensim/ - Topic modeling and word embeddings
- **Stanza:** https://stanfordnlp.github.io/stanza/ - Stanford NLP toolkit, 70+ languages
- **UDPipe:** https://ufal.mff.cuni.cz/udpipe - Universal Dependencies parsing

### Corpus Resources

- **Linguistic Data Consortium (LDC):** https://www.ldc.upenn.edu/ - Large collection of corpora
- **CLARIN:** https://www.clarin.eu/ - European research infrastructure for language resources
- **Universal Dependencies:** https://universaldependencies.org/ - Cross-linguistic treebanks

### Research Communities

- **International Corpus Linguistics Conference:** Biennial conference
- **Corpus Linguistics mailing list:** CORPORA@LISTSERV.UOTTAWA.CA
- **Association for Computational Linguistics (ACL):** https://www.aclweb.org/

### Online Courses

- **Corpus Linguistics** (Lancaster University) - Available on FutureLearn
- **Applied Text Mining in Python** (Michigan) - Coursera
- **Natural Language Processing Specialization** (Deeplearning.AI) - Coursera

### AWS Resources

- **AWS EMR Best Practices:** https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-longrunning-transient.html
- **Amazon Elasticsearch Service:** https://docs.aws.amazon.com/elasticsearch-service/
- **SageMaker Examples:** https://github.com/aws/amazon-sagemaker-examples
