# Historical Text Corpus Analysis at Scale - AWS Research Jumpstart

**Tier 1 Flagship Project**

Large-scale analysis of historical texts, literary corpora, and digitized manuscripts using natural language processing, topic modeling, and machine learning on AWS. Process millions of documents from digital archives, perform linguistic analysis, track cultural trends, and discover patterns across centuries of human expression.

## Overview

This flagship project demonstrates how to analyze massive text corpora from digital humanities collections using AWS services. We'll work with historical newspapers, literary works, parliamentary proceedings, and social media archives to perform distant reading, stylometry, sentiment analysis, and cultural evolution studies at scale.

### Key Features

- **Massive corpora:** HathiTrust (17M volumes), Google Books, Internet Archive, Project Gutenberg
- **OCR processing:** Tesseract, Amazon Textract for historical documents
- **NLP at scale:** Topic modeling (LDA, BERTopic), named entity recognition, word embeddings
- **Linguistic analysis:** Sentiment trends, vocabulary evolution, stylometric fingerprinting
- **ML integration:** BERT, GPT for semantic analysis, Claude via Bedrock for interpretation
- **AWS services:** S3, Comprehend, Textract, SageMaker, Athena, OpenSearch

### Scientific Applications

1. **Cultural evolution:** Track concept spread across time and geography
2. **Authorship attribution:** Identify anonymous authors through stylometry
3. **Historical linguistics:** Document language change over centuries
4. **Social history:** Analyze newspapers for public opinion trends
5. **Literary analysis:** Distant reading of thousands of novels

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Text Corpus Analysis Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ HathiTrust   │      │ Internet     │      │ Project      │
│ (17M vols)   │─────▶│ Archive      │─────▶│ Gutenberg    │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   S3 Data Lake    │
                    │  (Text files,     │
                    │   metadata)       │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ Textract OCR  │   │ Comprehend        │   │ SageMaker  │
│ (Historical)  │   │ (NER, Sentiment)  │   │ (Topic ML) │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  OpenSearch       │
                    │  (Full-text       │
                    │   search)         │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Topic Models │   │ Word Embeddings   │   │ Stylometry    │
│ (LDA, BERT)  │   │ (Word2Vec, BERT)  │   │ (Authorship)  │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Bedrock (Claude)  │
                    │ Interpretation    │
                    │ & Insights        │
                    └───────────────────┘
```

## Major Text Corpora

### 1. HathiTrust Digital Library

**What:** Massive digital library of scanned books
**Size:** 17+ million volumes, 8 million books
**Coverage:** 1500s-present, multiple languages
**Access:** Public domain texts available via API
**Format:** Plain text, OCR quality varies
**URL:** https://www.hathitrust.org/

**Collections:**
- Historical newspapers (19th-20th century)
- Scientific journals (medicine, natural science)
- Literary works (novels, poetry, drama)
- Government documents

### 2. Project Gutenberg

**What:** Free ebooks, mostly public domain
**Size:** 70,000+ books
**Languages:** Multiple (English majority)
**Quality:** High-quality text
**Format:** Plain text, EPUB, HTML
**URL:** https://www.gutenberg.org/

**Strengths:**
- Clean, accurate text
- Well-structured metadata
- Easy bulk download

### 3. Internet Archive

**What:** Digital library with books, images, video
**Size:** 35+ million books and texts
**OCR:** Available for most scanned items
**Access:** Open API
**URL:** https://archive.org/

### 4. Google Books Ngrams

**What:** Word frequency data from 8M books
**Years:** 1500-2019
**Languages:** Multiple corpora
**Size:** Billions of n-grams
**URL:** https://books.google.com/ngrams/

**Use cases:**
- Track word usage over time
- Cultural evolution studies
- Linguistic change

### 5. Historical Newspapers

- **Chronicling America:** 16M+ US newspaper pages (1789-1963)
- **Times Digital Archive:** Complete Times of London (1785-2019)
- **Trove (Australia):** 200M+ Australian newspaper articles

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# Python dependencies
pip install -r requirements.txt
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name text-analysis-stack \
  --template-body file://cloudformation/text-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name text-analysis-stack
```

### Download Sample Corpus

```python
from src.data_ingestion import GutenbergLoader, HathiTrustLoader

# Download from Project Gutenberg
gutenberg = GutenbergLoader(bucket_name='my-text-corpus')

# Download all works by Jane Austen
austen_books = gutenberg.download_author_works(
    author='Austen, Jane',
    output_path='s3://my-text-corpus/austen/'
)

# Or specific book
pride = gutenberg.download_book(
    book_id=1342,  # Pride and Prejudice
    format='txt'
)

# Download from HathiTrust (public domain)
hathi = HathiTrustLoader(api_key='YOUR_KEY')

# Search for 19th century American novels
results = hathi.search(
    query='american novel',
    year_range=[1800, 1900],
    language='eng',
    max_results=1000
)

# Download full texts
hathi.download_volumes(
    volume_ids=[r['htid'] for r in results],
    output_bucket='s3://my-text-corpus/american-novels/'
)
```

## Core Analyses

### 1. Text Preprocessing and Normalization

Clean and prepare historical texts for analysis.

```python
from src.preprocessing import TextPreprocessor
import boto3

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Load raw text (possibly with OCR errors)
raw_text = open('historical_document.txt').read()

# Clean text
cleaned = preprocessor.clean_text(
    raw_text,
    remove_headers=True,  # Remove running headers/footers
    fix_hyphenation=True,  # Rejoin hyphenated words at line breaks
    normalize_whitespace=True,
    remove_page_numbers=True
)

# OCR error correction
corrected = preprocessor.correct_ocr_errors(
    cleaned,
    dictionary='english',
    context_aware=True  # Use surrounding words for correction
)

# Historical spelling normalization (Shakespeare → modern)
normalized = preprocessor.normalize_historical_spelling(
    corrected,
    target_period='modern'
)

# Tokenization
tokens = preprocessor.tokenize(
    normalized,
    method='spacy',  # or 'nltk', 'word_tokenize'
    pos_tag=True,
    lemmatize=True
)

# Process entire corpus in parallel (AWS Batch)
from src.batch_processing import CorpusProcessor

processor = CorpusProcessor(job_queue='text-processing-queue')

job_ids = processor.process_corpus(
    input_bucket='s3://my-corpus/raw/',
    output_bucket='s3://my-corpus/processed/',
    preprocessing_steps=['clean', 'ocr_correct', 'tokenize'],
    n_parallel=100
)

processor.wait_for_completion(job_ids)
```

### 2. Topic Modeling at Scale

Discover themes and topics across large corpora.

```python
from src.topic_modeling import TopicModeler
import sagemaker

# Initialize topic modeler
modeler = TopicModeler()

# Load preprocessed corpus
corpus = modeler.load_corpus('s3://my-corpus/processed/')

# Method 1: LDA (Latent Dirichlet Allocation)
lda_model = modeler.train_lda(
    corpus=corpus,
    n_topics=50,
    alpha='auto',
    eta='auto',
    passes=10,
    distributed=True,  # Use AWS Batch for large corpora
    instance_type='c5.4xlarge'
)

# Get topics
topics = lda_model.get_topics(n_words=20)
for topic_id, words in enumerate(topics):
    print(f"Topic {topic_id}: {', '.join(words)}")

# Document-topic distribution
doc_topics = lda_model.get_document_topics(
    documents=['s3://my-corpus/processed/doc1.txt']
)

# Method 2: BERTopic (transformer-based)
bert_model = modeler.train_bertopic(
    corpus=corpus,
    embedding_model='all-MiniLM-L6-v2',
    min_topic_size=10,
    nr_topics='auto',
    instance_type='ml.p3.2xlarge'  # GPU for embeddings
)

# Visualize topics
viz = modeler.visualize_topics(bert_model)
viz.save('topic_visualization.html')

# Topic evolution over time
temporal_topics = modeler.analyze_topic_evolution(
    model=lda_model,
    documents=corpus,
    timestamps=corpus.metadata['year'],
    time_bins=list(range(1800, 2000, 10))  # Decade bins
)

# Plot topic trends
modeler.plot_topic_trends(
    temporal_topics,
    topic_ids=[0, 5, 12],  # Select interesting topics
    save_path='topic_trends.png'
)
```

### 3. Named Entity Recognition and Linking

Extract people, places, and organizations from texts.

```python
from src.entity_extraction import EntityExtractor

# Initialize extractor
extractor = EntityExtractor()

# Method 1: AWS Comprehend (simple, scalable)
text = open('historical_speech.txt').read()

entities = extractor.extract_entities_comprehend(
    text,
    language_code='en'
)

print(f"Found {len(entities)} entities:")
for entity in entities[:10]:
    print(f"  {entity['Text']} ({entity['Type']}) - confidence: {entity['Score']:.2f}")

# Method 2: SpaCy (more control)
entities_spacy = extractor.extract_entities_spacy(
    text,
    model='en_core_web_trf',  # Transformer model
    entity_types=['PERSON', 'GPE', 'ORG', 'DATE', 'EVENT']
)

# Entity linking (resolve to Wikipedia/Wikidata)
linked_entities = extractor.link_entities(
    entities_spacy,
    knowledge_base='wikidata'
)

# Process entire corpus
entity_index = extractor.build_entity_index(
    corpus_bucket='s3://my-corpus/processed/',
    output_bucket='s3://my-corpus/entities/',
    use_comprehend=True,
    batch_size=1000
)

# Query entity co-occurrences
connections = extractor.find_entity_connections(
    entity_index,
    entity='Abraham Lincoln',
    min_cooccurrence=5
)

# Network visualization
extractor.plot_entity_network(
    connections,
    save_path='entity_network.html'
)
```

### 4. Sentiment Analysis Over Time

Track emotional tone and sentiment trends.

```python
from src.sentiment import SentimentAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze single document with AWS Comprehend
text = "It was the best of times, it was the worst of times..."
sentiment = analyzer.analyze_comprehend(text)

print(f"Sentiment: {sentiment['Sentiment']}")
print(f"Scores: {sentiment['SentimentScore']}")

# Historical sentiment analysis (newspapers)
newspapers = load_newspaper_corpus(
    's3://my-corpus/newspapers/',
    date_range=['1850-01-01', '1870-12-31']
)

# Analyze sentiment for each document
sentiments = []
for doc in newspapers:
    sent = analyzer.analyze_comprehend(doc['text'])
    sentiments.append({
        'date': doc['date'],
        'title': doc['title'],
        'sentiment': sent['Sentiment'],
        'positive': sent['SentimentScore']['Positive'],
        'negative': sent['SentimentScore']['Negative'],
        'neutral': sent['SentimentScore']['Neutral']
    })

sentiment_df = pd.DataFrame(sentiments)

# Time series analysis
monthly_sentiment = sentiment_df.groupby(
    pd.Grouper(key='date', freq='M')
)['positive'].mean()

# Plot sentiment over time (Civil War era)
analyzer.plot_sentiment_timeline(
    monthly_sentiment,
    title='Newspaper Sentiment: 1850-1870',
    events=[
        ('1861-04', 'Civil War begins'),
        ('1865-04', 'Civil War ends')
    ],
    save_path='sentiment_timeline.png'
)

# Aspect-based sentiment (opinions about specific topics)
aspects = analyzer.extract_aspect_sentiment(
    documents=newspapers,
    aspects=['slavery', 'union', 'economy'],
    aggregation='monthly'
)
```

### 5. Word Embeddings and Semantic Change

Track how word meanings evolve over time.

```python
from src.embeddings import HistoricalEmbeddings
import numpy as np

# Initialize embeddings analyzer
embeddings = HistoricalEmbeddings()

# Train Word2Vec models for different time periods
periods = {
    '1800-1850': load_texts('s3://corpus/1800-1850/'),
    '1850-1900': load_texts('s3://corpus/1850-1900/'),
    '1900-1950': load_texts('s3://corpus/1900-1950/'),
    '1950-2000': load_texts('s3://corpus/1950-2000/')
}

models = {}
for period, texts in periods.items():
    models[period] = embeddings.train_word2vec(
        texts,
        vector_size=300,
        window=5,
        min_count=10,
        epochs=5,
        instance_type='ml.c5.4xlarge'
    )

# Detect semantic change
word = 'broadcast'
change_scores = embeddings.measure_semantic_change(
    word=word,
    models=models,
    method='cosine_distance'  # or 'displacement', 'second_order'
)

print(f"Semantic change scores for '{word}':")
for period, score in change_scores.items():
    print(f"  {period}: {score:.3f}")

# Find nearest neighbors in each period
for period, model in models.items():
    neighbors = model.wv.most_similar(word, topn=10)
    print(f"\n{period} - '{word}' similar to:")
    for neighbor, similarity in neighbors:
        print(f"  {neighbor}: {similarity:.3f}")

# Analogies (king - man + woman = queen)
analogy = embeddings.compute_analogy(
    models['1900-1950'],
    positive=['king', 'woman'],
    negative=['man']
)

# Cultural bias detection
bias_score = embeddings.measure_gender_bias(
    models,
    occupation_words=['doctor', 'nurse', 'engineer', 'teacher']
)
```

### 6. Stylometry and Authorship Attribution

Identify authors through writing style.

```python
from src.stylometry import StylometricAnalyzer

# Initialize analyzer
stylo = StylometricAnalyzer()

# Extract stylometric features
features = stylo.extract_features(
    text,
    features=[
        'word_length_distribution',
        'sentence_length_distribution',
        'lexical_diversity',
        'function_word_frequencies',
        'punctuation_usage',
        'syntactic_complexity'
    ]
)

# Build author profiles from known works
known_authors = {
    'Jane Austen': load_texts('s3://corpus/austen/'),
    'Charles Dickens': load_texts('s3://corpus/dickens/'),
    'Charlotte Brontë': load_texts('s3://corpus/bronte/')
}

author_profiles = {}
for author, texts in known_authors.items():
    author_profiles[author] = stylo.build_author_profile(texts)

# Attribution for disputed text
disputed_text = load_text('disputed_novel.txt')
disputed_features = stylo.extract_features(disputed_text)

# Calculate similarity to each author
similarities = {}
for author, profile in author_profiles.items():
    sim = stylo.calculate_similarity(
        disputed_features,
        profile,
        method='burrows_delta'  # or 'cosine', 'manhattan'
    )
    similarities[author] = sim

# Predict author
predicted_author = min(similarities, key=similarities.get)
print(f"Most likely author: {predicted_author}")
print(f"Delta score: {similarities[predicted_author]:.3f}")

# Verification using multiple methods
verification = stylo.verify_authorship(
    disputed_text,
    candidate_author='Jane Austen',
    known_texts=known_authors['Jane Austen'],
    methods=['burrows_delta', 'ml_classifier', 'ngram_analysis']
)

# Visualize in feature space (PCA)
stylo.plot_author_space(
    author_profiles,
    disputed_features,
    save_path='author_space.png'
)
```

## Advanced Analyses

### Cultural Evolution and Meme Tracking

```python
from src.cultural_evolution import MemeTracker

tracker = MemeTracker()

# Define concepts/memes to track
concepts = [
    'democracy',
    'liberty',
    'equality',
    'socialism',
    'capitalism'
]

# Track concept frequency over time
corpus_timeline = load_corpus_with_dates('s3://corpus/newspapers/')

concept_trends = tracker.track_concepts(
    corpus=corpus_timeline,
    concepts=concepts,
    time_resolution='year',
    normalize=True  # Normalize by total words per period
)

# Detect concept emergence and decline
emergence = tracker.detect_emergence(
    concept_trends,
    concept='socialism',
    threshold=2.0  # 2x increase in usage
)

print(f"'Socialism' emerged around {emergence['year']}")

# Co-occurrence networks (which concepts appear together?)
cooccurrence = tracker.build_cooccurrence_network(
    corpus=corpus_timeline,
    concepts=concepts,
    window_size=50,  # words
    time_period='1900-1950'
)

tracker.visualize_network(cooccurrence, save_path='concept_network.html')
```

### Literary Genre Classification

```python
from src.classification import GenreClassifier
import sagemaker

classifier = GenreClassifier()

# Prepare training data
training_data = classifier.prepare_training_data(
    corpus='s3://corpus/labeled_novels/',
    genres=['romance', 'mystery', 'science_fiction', 'horror', 'western'],
    features='tfidf',  # or 'embeddings', 'stylometric'
    max_features=10000
)

# Train classifier
model = classifier.train(
    training_data,
    model_type='random_forest',  # or 'xgboost', 'neural_network'
    instance_type='ml.m5.4xlarge'
)

# Evaluate
metrics = classifier.evaluate(model, test_data=training_data['test'])
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_macro']:.3f}")

# Classify unlabeled novels
predictions = classifier.predict(
    model,
    texts='s3://corpus/unlabeled_novels/',
    batch_size=100
)

# Fine-grained classification with BERT
bert_classifier = classifier.train_bert(
    training_data,
    model='bert-base-uncased',
    instance_type='ml.p3.2xlarge',
    epochs=3
)
```

## Full-Text Search with OpenSearch

```python
from src.search import TextSearchEngine

# Initialize search engine
search = TextSearchEngine(
    opensearch_endpoint='https://your-domain.us-east-1.es.amazonaws.com'
)

# Index corpus
search.index_corpus(
    corpus_bucket='s3://my-corpus/processed/',
    index_name='historical-texts',
    batch_size=1000,
    metadata_fields=['author', 'year', 'title', 'genre']
)

# Full-text search
results = search.search(
    query='industrial revolution',
    index='historical-texts',
    filters={'year': {'gte': 1750, 'lte': 1850}},
    size=20
)

for hit in results:
    print(f"{hit['title']} ({hit['year']}) - Score: {hit['score']:.2f}")
    print(f"  {hit['highlight']}")

# Semantic search with embeddings
search.enable_knn(index='historical-texts', vector_field='embedding')

semantic_results = search.semantic_search(
    query='labor conditions in factories',
    k=10
)

# Aggregations (faceted search)
aggregations = search.aggregate(
    index='historical-texts',
    aggs={
        'authors': {'terms': {'field': 'author', 'size': 20}},
        'decades': {'histogram': {'field': 'year', 'interval': 10}},
        'genres': {'terms': {'field': 'genre'}}
    }
)
```

## AI-Powered Interpretation with Claude

```python
from src.ai_interpretation import LiteraryInterpreter

interpreter = LiteraryInterpreter()

# Analyze text passage
passage = """
It is a truth universally acknowledged, that a single man in
possession of a good fortune, must be in want of a wife.
"""

analysis = interpreter.analyze_passage(
    text=passage,
    analyses=[
        'literary_devices',
        'historical_context',
        'thematic_analysis',
        'irony_detection'
    ],
    model='claude-3-sonnet'
)

print(analysis['literary_devices'])
print(analysis['historical_context'])

# Compare authors
comparison = interpreter.compare_authors(
    author1='Jane Austen',
    author2='George Eliot',
    texts1=austen_corpus,
    texts2=eliot_corpus,
    dimensions=['style', 'themes', 'characterization']
)

# Generate research questions
questions = interpreter.generate_research_questions(
    corpus_summary={
        'period': '1850-1900',
        'genre': 'American novels',
        'key_themes': ['industrialization', 'westward expansion', 'slavery'],
        'major_authors': ['Melville', 'Twain', 'Hawthorne']
    }
)

print("Suggested research questions:")
for q in questions:
    print(f"- {q}")
```

## Cost Estimate

**One-time setup:** $50-100

**Storage (1 year):**
- Raw texts (100 GB): $2.30/month
- Processed data (200 GB): $4.60/month
- Indices (50 GB): $1.15/month
- **Total: ~$10/month**

**Processing costs:**
- OCR (Textract, 10K pages): $15
- Comprehend (1M documents): $300
- Topic modeling (1M documents): $100-200
- Entity extraction (1M documents): $100-200

**Research project (corpus of 100K documents):**
- Preprocessing: $50-100
- Topic modeling: $100-200
- Entity extraction: $50-100
- Sentiment analysis: $30-50
- Word embeddings: $50-100
- Search infrastructure: $50/month
- **Total: $500-1,000 + $50/month**

## Performance Benchmarks

**Text preprocessing:**
- OCR (Textract): ~1 second per page
- Cleaning and tokenization: 100-500 docs/second (CPU)
- 100K documents: ~30-60 minutes

**Topic modeling:**
- LDA (100K docs, 50 topics): 2-4 hours on c5.4xlarge
- BERTopic (100K docs): 4-8 hours on p3.2xlarge

**Entity extraction:**
- Comprehend: 1,000 docs/minute
- SpaCy (CPU): 100 docs/minute
- SpaCy (GPU): 1,000 docs/minute

## Best Practices

1. **OCR quality:** Validate with spot checks, consider re-OCR for poor quality
2. **Historical language:** Build custom dictionaries for period-specific vocabulary
3. **Metadata:** Rich metadata essential for temporal/geographic analysis
4. **Sampling:** Test on samples before processing full corpus
5. **Version control:** Track corpus versions, preprocessing steps
6. **Reproducibility:** Document all parameters, random seeds
7. **Privacy:** Be aware of copyright and privacy issues with recent texts

## References

### Resources

- **HathiTrust Research Center:** https://analytics.hathitrust.org/
- **Chronicling America:** https://chroniclingamerica.loc.gov/
- **Project Gutenberg:** https://www.gutenberg.org/
- **Internet Archive:** https://archive.org/

### Software

- **NLTK:** https://www.nltk.org/
- **spaCy:** https://spacy.io/
- **Gensim:** https://radimrehurek.com/gensim/
- **BERTopic:** https://maartengr.github.io/BERTopic/

### Key Papers

1. Michel et al. (2011). "Quantitative Analysis of Culture Using Millions of Digitized Books." *Science*
2. Hamilton et al. (2016). "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change." *ACL*
3. Underwood (2015). "The Literary Uses of High-Dimensional Space." *Big Data & Society*
4. Jockers (2013). *Macroanalysis: Digital Methods and Literary History*

## Next Steps

1. Deploy CloudFormation stack
2. Download sample corpus (Gutenberg)
3. Run preprocessing pipeline
4. Train topic model on sample
5. Build full-text search index
6. Scale to larger corpus

---

**Tier 1 Project Status:** Production-ready
**Estimated Setup Time:** 4-6 hours
**Processing time:** 100K documents in 2-4 hours
**Cost:** $500-1,000 for complete analysis + $50/month

For questions, consult NLP documentation or digital humanities communities.
