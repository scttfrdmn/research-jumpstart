# Historical Text Corpus Analysis with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $7-12 | **Platform:** AWS + Local machine

Analyze historical text corpora at scale using AWS serverless services. Upload literary texts to S3, perform NLP analysis with Lambda, store results in DynamoDB, and query with Athena for full-text search and literary analysis.

---

## What You'll Build

A cloud-native digital humanities research pipeline that demonstrates:

1. **Corpus Storage** - Upload historical texts (Project Gutenberg) to S3
2. **Serverless NLP Processing** - Lambda functions for text analysis in parallel
3. **Metadata Database** - Store linguistic features in DynamoDB
4. **Full-Text Search** - Query corpus with Athena for complex literary queries
5. **Comparative Analysis** - Analyze vocabulary evolution across authors and periods

This bridges the gap between local text analysis (Tier 1) and production digital humanities infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter Notebook                          │ │
│  │ • upload_to_s3.py - Upload text corpus                    │ │
│  │ • lambda_function.py - NLP processing function            │ │
│  │ • query_results.py - Analyze results                      │ │
│  │ • text_analysis.ipynb - Full workflow notebook            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Raw texts      │→ │ NLP processing:  │→ │ Linguistic       │
│  │ • Organized by   │  │ - Word freq      │  │ features:        │
│  │   author/period  │  │ - NER            │  │ - Vocabulary     │
│  │                  │  │ - Topics (LDA)   │  │ - Named entities │
│  │                  │  │ - Sentiment      │  │ - Themes         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│                                               ┌──────────────────┐
│                                               │  Athena (SQL)    │
│                                               │                  │
│                                               │ Full-text search │
│                                               │ Complex queries  │
│                                               └──────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  IAM Role (Permissions)                                      │
│  │  • S3 read/write                                             │
│  │  • DynamoDB read/write                                       │
│  │  • Lambda execution                                          │
│  │  • CloudWatch logging                                        │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Knowledge
- Basic Python (pandas, boto3, nltk)
- Understanding of NLP concepts (tokenization, NER, topic modeling)
- AWS fundamentals (S3, Lambda, IAM, DynamoDB)
- Digital humanities or literary analysis concepts

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - nltk (Natural Language Toolkit)
  - spacy (Advanced NLP)
  - gensim (Topic modeling)
  - pandas (Data manipulation)
  - matplotlib, wordcloud (Visualization)
  - jupyter (Analysis notebook)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB table creation
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/digital-humanities/text-analysis/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Quick Start (15 minutes)

### Step 1: Set Up AWS (15 minutes)
```bash
# Follow setup_guide.md for detailed instructions
# Creates:
# - S3 bucket: text-corpus-{your-id}
# - IAM role: lambda-text-processor
# - Lambda function: process-text-document
# - DynamoDB table: TextAnalysis
# - Athena workspace: text_corpus_db
```

### Step 2: Upload Sample Corpus (5 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Process Texts (10 minutes)
```bash
# Lambda processes each document automatically
# Or trigger manually for testing
python scripts/lambda_function.py --test
```

### Step 4: Query Results (5 minutes)
```bash
python scripts/query_results.py --author "Jane Austen"
```

### Step 5: Analyze in Jupyter (30 minutes)
```bash
jupyter notebook notebooks/text_analysis.ipynb
```

---

## Detailed Workflow

### 1. Corpus Preparation (Setup)

**What's happening:**
- Download historical texts from Project Gutenberg (~50-100 texts)
- Organize by author, period, and genre
- Upload to S3 with metadata tags
- Create S3 bucket with proper folder structure

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Organization Structure:**
```
s3://text-corpus-{your-id}/
├── raw/
│   ├── austen/
│   │   ├── pride-and-prejudice.txt
│   │   └── sense-and-sensibility.txt
│   ├── dickens/
│   │   ├── great-expectations.txt
│   │   └── oliver-twist.txt
│   ├── bronte/
│   │   ├── jane-eyre.txt
│   │   └── wuthering-heights.txt
│   └── [other-authors]/
├── processed/
│   └── [analysis-results]/
└── metadata/
    └── [document-metadata]/
```

**Time:** 20-30 minutes (includes download and upload)

### 2. Lambda NLP Processing

**What's happening:**
- Lambda triggered on S3 upload (or manually invoked)
- Performs comprehensive text analysis:
  - **Word frequency analysis**: Most common words, vocabulary size
  - **Named Entity Recognition (NER)**: Extract people, places, organizations
  - **Topic modeling**: Discover themes using Latent Dirichlet Allocation
  - **Literary features**: Sentence length, vocabulary richness (TTR), readability
  - **Sentiment analysis**: Overall emotional tone
- Stores results in DynamoDB for fast querying

**Lambda function operations:**
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'text-corpus-xxxx'},
        'object': {'key': 'raw/austen/pride-and-prejudice.txt'}
    }]
}

# Processing steps:
# 1. Download text from S3
# 2. Tokenize and clean text
# 3. Extract linguistic features
# 4. Perform NER and topic modeling
# 5. Calculate vocabulary statistics
# 6. Store results in DynamoDB
```

**NLP Features Extracted:**
- Total words, unique words, type-token ratio
- Top 50 most frequent words
- Named entities (people, places, organizations)
- Topics and keywords (LDA)
- Average sentence length
- Readability scores (Flesch-Kincaid)
- Sentiment polarity and subjectivity

**Files involved:**
- `scripts/lambda_function.py` - NLP processing code
- `setup_guide.md` - Lambda deployment steps

**Time:** 2-5 minutes per document (varies by length)

### 3. Results Storage

**What's happening:**
- Processed results stored in DynamoDB
- Original texts kept in S3 for reference
- Metadata enables fast queries across corpus

**DynamoDB Schema:**
```
Table: TextAnalysis
Primary Key: document_id (String)
Sort Key: timestamp (Number)

Attributes:
- author (String)
- title (String)
- period (String) - e.g., "Victorian", "Romantic"
- genre (String) - e.g., "Novel", "Poetry"
- word_count (Number)
- unique_words (Number)
- vocabulary_richness (Number) - Type-Token Ratio
- top_words (List) - Top 50 frequent words
- named_entities (Map) - {people: [], places: [], organizations: []}
- topics (List) - Discovered themes
- avg_sentence_length (Number)
- readability_score (Number)
- sentiment_score (Number)
- processing_timestamp (Number)
```

**S3 Structure:**
```
s3://text-corpus-{your-id}/
├── raw/                          # Original text files
│   ├── austen/
│   ├── dickens/
│   └── ...
├── processed/                     # Detailed JSON results
│   ├── pride-and-prejudice.json
│   ├── great-expectations.json
│   └── ...
└── logs/                          # Lambda execution logs
    └── processing_log.txt
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for specific authors, periods, or themes
- Use Athena for complex SQL queries across corpus
- Download results to local machine for visualization
- Generate comparative analyses and publication-quality figures

**Query Examples:**
```python
# Query by author
query_results.py --author "Jane Austen"

# Query by period
query_results.py --period "Victorian"

# Query by theme (using topics)
query_results.py --theme "romance"

# Complex Athena SQL queries
SELECT author, AVG(vocabulary_richness) as avg_vr
FROM text_corpus_db.documents
WHERE period = 'Victorian'
GROUP BY author
ORDER BY avg_vr DESC;
```

**Files involved:**
- `notebooks/text_analysis.ipynb` - Full analysis workflow
- `scripts/query_results.py` - Programmatic querying
- (Optional) Athena queries for advanced SQL analysis

**Time:** 30-60 minutes for comprehensive analysis

---

## Project Files

```
tier-2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_guide.md                     # AWS setup instructions
├── cleanup_guide.md                   # Resource deletion guide
│
├── notebooks/
│   └── text_analysis.ipynb           # Main analysis notebook
│                                     # - Download sample corpus
│                                     # - Upload to S3
│                                     # - Trigger Lambda processing
│                                     # - Query and visualize results
│                                     # - Comparative analysis
│
└── scripts/
    ├── upload_to_s3.py               # Upload texts to S3
    │                                 # - Organize by author/period
    │                                 # - Add metadata tags
    │                                 # - Progress tracking
    │
    ├── lambda_function.py            # Lambda NLP processing
    │                                 # - Word frequency analysis
    │                                 # - Named Entity Recognition
    │                                 # - Topic modeling (LDA)
    │                                 # - Literary feature extraction
    │                                 # - Store in DynamoDB
    │
    ├── query_results.py              # Query and analyze results
    │                                 # - Query by author/period/theme
    │                                 # - Aggregate statistics
    │                                 # - Export for visualization
    │
    └── __init__.py
```

---

## Cost Breakdown

**Total estimated cost: $7-12 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 500MB text files × 7 days | $0.10 |
| **S3 Requests** | ~500 PUT/GET requests | $0.02 |
| **Lambda Executions** | 100 invocations × 2 min | $1.20 |
| **Lambda Compute** | 100 × 120 sec × 512MB | $1.50 |
| **DynamoDB Storage** | 100 documents, 1 week | $0.25 |
| **DynamoDB Queries** | ~1000 read/write operations | $0.25 |
| **Athena Queries** | 10 queries × 100MB scanned | $0.50 |
| **Data Transfer** | Upload + download (~1GB) | $0.10 |
| **CloudWatch Logs** | ~50MB logs | Free |
| **Total** | | **$3.92-$8.00** |

**With additional analysis:**
- More extensive corpus (500 texts): +$5-8
- Advanced topic modeling: +$2-3
- Real-time processing: +$1-2

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.10 savings/week)
2. Use Lambda for < 2 minutes processing per document
3. Batch DynamoDB writes to reduce operations
4. Use Athena queries efficiently (scan less data)
5. Use S3 Intelligent-Tiering for long-term storage

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations free (12 months)
- **DynamoDB**: 25GB storage free (always free)
- **Athena**: First 10MB scanned free (per query)

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and organization
- Lambda function deployment and triggers
- DynamoDB NoSQL database design
- Athena serverless SQL queries
- IAM role creation with least privilege
- CloudWatch monitoring and logs

### Cloud Concepts
- Object storage for unstructured data (text files)
- Serverless computing (no servers to manage)
- Event-driven architecture (S3 triggers Lambda)
- NoSQL database design for flexible queries
- Serverless SQL (Athena)
- Cost-conscious design patterns

### Digital Humanities Skills
- Corpus linguistics methods
- Computational text analysis
- Named Entity Recognition for literary analysis
- Topic modeling for theme discovery
- Vocabulary evolution across periods
- Authorship attribution techniques
- Literary feature extraction

### NLP Techniques
- Text preprocessing and tokenization
- Word frequency analysis
- Named Entity Recognition (NER)
- Topic modeling (Latent Dirichlet Allocation)
- Sentiment analysis
- Readability metrics
- Vocabulary richness measures

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 10 minutes
- Create DynamoDB table: 3 minutes
- Configure Athena workspace: 5 minutes
- **Subtotal setup: 35 minutes**

**Corpus Preparation:**
- Download sample texts: 10 minutes
- Upload to S3: 5-10 minutes
- **Subtotal preparation: 15-20 minutes**

**Data Processing:**
- Lambda processing (100 texts): 10-15 minutes
- Verify results in DynamoDB: 5 minutes
- **Subtotal processing: 15-20 minutes**

**Analysis:**
- Query results: 5 minutes
- Jupyter analysis: 45-60 minutes
- Generate visualizations: 15-20 minutes
- **Subtotal analysis: 65-85 minutes**

**Total time: 2-3 hours** (including setup)

---

## Research Use Cases

### Literary Analysis
- **Authorship attribution**: Compare vocabulary and style across authors
- **Genre analysis**: Identify distinguishing features of novels vs. poetry
- **Theme tracking**: Discover how themes evolve across periods
- **Character analysis**: Extract and analyze character mentions and relationships

### Historical Research
- **Discourse analysis**: Track how language and concepts change over time
- **Cultural studies**: Analyze representation of gender, class, race in literature
- **Reception studies**: Compare contemporary reviews with modern interpretations

### Computational Linguistics
- **Corpus linguistics**: Large-scale language pattern analysis
- **Diachronic analysis**: Language change over time
- **Comparative literature**: Cross-cultural and cross-linguistic analysis

---

## What You'll Discover

### Literary Insights
- How vocabulary richness varies across authors and periods
- Common themes in Victorian literature vs. Romantic literature
- Named entities (characters, places) and their frequency
- Stylistic differences between authors (sentence length, complexity)
- Evolution of literary language over centuries

### AWS Insights
- Serverless computing advantages for batch text processing
- NoSQL database design for flexible literary queries
- Cost-effective cloud analysis for digital humanities
- Scale from 100 texts to 100,000 texts without infrastructure changes

### Research Insights
- Reproducibility: Same corpus, same code, same results
- Collaboration: Share workflows, data, and findings with colleagues
- Scale: Analyze entire literary collections in minutes
- Persistence: Results saved permanently for future research

---

## Next Steps

### Extend This Project
1. **Larger Corpus**: Add more authors and periods (500+ texts)
2. **Advanced NLP**: Add dependency parsing, coreference resolution
3. **Network Analysis**: Build character co-occurrence networks
4. **Comparative Analysis**: Compare translations of the same work
5. **Temporal Analysis**: Track language evolution decade-by-decade
6. **Automation**: Set up Lambda trigger for automatic processing
7. **Visualization**: Create interactive dashboards with QuickSight
8. **Machine Learning**: Train classification models for genre/author

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda layers for NLP libraries
- Multi-region deployment for global research teams
- Advanced monitoring and cost optimization
- Integration with institutional repositories

### Explore Related Projects
- Text Mining for Social Sciences (sentiment analysis at scale)
- Historical Newspaper Analysis (OCR + NLP)
- Multilingual Corpus Analysis (comparative literature)
- Digital Archives Processing (metadata extraction)

---

## Troubleshooting

### Common Issues

**"botocore.exceptions.NoCredentialsError"**
```bash
# Solution: Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Key, region (us-east-1), output format (json)
```

**"S3 bucket already exists"**
```bash
# Solution: Use a unique bucket name
# Add your username and timestamp
text-corpus-$(whoami)-$(date +%s)
```

**"Lambda timeout processing long texts"**
```python
# Solution: Increase Lambda timeout in AWS console
# Default: 3 seconds → Increase to: 300 seconds (5 minutes)
# For very long texts (novels): 600 seconds (10 minutes)
# Note: Costs increase with processing time
```

**"Lambda out of memory"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# For NLP with spaCy: 512 MB or 1024 MB
# For topic modeling: 1024 MB or 2048 MB
# Note: More memory = faster processing but slightly higher cost
```

**"NLTK data not found"**
```bash
# Solution: Download NLTK data in Lambda layer or at runtime
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
# Or include in Lambda deployment package
```

**"spaCy model not loading in Lambda"**
```bash
# Solution: Create Lambda layer with spaCy model
# Or use smaller model: en_core_web_sm (15MB vs en_core_web_lg 800MB)
# Include model in deployment package
```

**"DynamoDB queries return no results"**
```bash
# Solution: Verify Lambda is writing to correct table
# Check CloudWatch logs for errors
aws logs tail /aws/lambda/process-text-document --follow

# Scan DynamoDB table to verify data
aws dynamodb scan --table-name TextAnalysis --limit 10
```

**"Text encoding errors (UnicodeDecodeError)"**
```python
# Solution: Handle encoding explicitly
# In upload_to_s3.py and lambda_function.py:
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
```

**"High costs at the end"**
```bash
# Solution: Follow cleanup_guide.md immediately after analysis
# Delete S3 objects, DynamoDB table, and Lambda function
# Monitor costs in AWS Cost Explorer
```

See the full troubleshooting guide in `setup_guide.md` for more solutions.

---

## Sample Analysis Results

### Vocabulary Richness Comparison

```
Author              | Unique Words | Type-Token Ratio | Avg Sentence Length
--------------------|--------------|------------------|--------------------
Jane Austen         | 6,500        | 0.68             | 24.3 words
Charles Dickens     | 12,400       | 0.73             | 28.7 words
Charlotte Bronte    | 8,200        | 0.71             | 26.1 words
Emily Bronte        | 7,800        | 0.69             | 22.5 words
George Eliot        | 11,200       | 0.75             | 30.2 words
```

### Common Themes Discovered (LDA)

**Victorian Literature:**
- Topic 1: society, class, gentleman, lady, marriage
- Topic 2: london, street, house, city, night
- Topic 3: love, heart, feeling, emotion, passion

**Romantic Poetry:**
- Topic 1: nature, wind, sky, mountain, beauty
- Topic 2: soul, spirit, divine, eternal, immortal
- Topic 3: love, beloved, heart, desire, sorrow

### Named Entity Frequency

**Pride and Prejudice** (Jane Austen):
- Characters: Elizabeth (622), Darcy (418), Jane (234), Bingley (198)
- Places: Longbourn (89), Pemberley (67), London (43), Rosings (34)

---

## Jupyter Notebook Workflow

The main analysis notebook (`text_analysis.ipynb`) includes:

### 1. Setup and Data Loading
- Configure AWS credentials
- Download sample texts from Project Gutenberg
- Upload to S3 with metadata

### 2. Lambda Processing
- Deploy Lambda function
- Trigger processing for entire corpus
- Monitor progress via CloudWatch

### 3. Query and Explore
- Query DynamoDB for specific authors
- Retrieve linguistic features
- Load into pandas DataFrames

### 4. Comparative Analysis
- Compare vocabulary richness across authors
- Analyze theme evolution across periods
- Identify stylistic differences

### 5. Visualization
- Word clouds for each author
- Time series of vocabulary evolution
- Topic modeling heatmaps
- Named entity networks
- Readability score comparisons

### 6. Advanced Analysis
- Authorship attribution using features
- Genre classification
- Sentiment analysis across corpus
- Character co-occurrence networks

### 7. Export Results
- Save figures for publication
- Generate CSV reports
- Export to LaTeX tables

---

## Sample Code Snippets

### Upload Corpus to S3
```python
import boto3
from pathlib import Path

s3 = boto3.client('s3')
bucket_name = 'text-corpus-your-id'

# Upload with metadata
for text_file in Path('corpus/austen').glob('*.txt'):
    s3.upload_file(
        str(text_file),
        bucket_name,
        f'raw/austen/{text_file.name}',
        ExtraArgs={
            'Metadata': {
                'author': 'Jane Austen',
                'period': 'Romantic',
                'genre': 'Novel'
            }
        }
    )
```

### Query DynamoDB
```python
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('TextAnalysis')

# Query by author
response = table.scan(
    FilterExpression='author = :author',
    ExpressionAttributeValues={':author': 'Jane Austen'}
)

for item in response['Items']:
    print(f"{item['title']}: {item['vocabulary_richness']:.2f}")
```

### Analyze with Athena
```sql
-- Compare vocabulary across periods
SELECT
    period,
    AVG(vocabulary_richness) as avg_richness,
    AVG(avg_sentence_length) as avg_sentence_len,
    COUNT(*) as num_texts
FROM text_corpus_db.documents
GROUP BY period
ORDER BY avg_richness DESC;
```

---

## Prerequisites (Detailed)

### AWS Account Setup
1. Create AWS account: https://aws.amazon.com/
2. (Optional) Activate free tier: https://console.aws.amazon.com/billing/
3. Create IAM user for programmatic access
4. Generate access keys for CLI/SDK access

### Local Setup
```bash
# Install Python 3.8+ (if needed)
python --version

# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter: Access Key ID
# Enter: Secret Access Key
# Enter: Region (us-east-1 recommended)
# Enter: Output format (json)

# Verify AWS access
aws s3 ls
aws sts get-caller-identity
```

### Data Preparation
- Sample texts provided via download script
- Or download your own from: https://www.gutenberg.org/
- Recommended: Start with 10-50 texts for initial analysis

---

## Running the Project

### Option 1: Automated (Recommended for First Time)
```bash
# Step 1: Setup AWS services (follow prompts)
python scripts/setup_aws.py  # Optional helper script

# Step 2: Upload corpus
python scripts/upload_to_s3.py

# Step 3: Deploy Lambda (follow setup_guide.md)
# Manual: Deploy scripts/lambda_function.py to Lambda console

# Step 4: Analyze results
jupyter notebook notebooks/text_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://text-corpus-$(whoami)-$(date +%s) --region us-east-1

# 2. Upload data
python scripts/upload_to_s3.py --bucket text-corpus-your-id

# 3. Create DynamoDB table
aws dynamodb create-table --cli-input-json file://dynamodb-schema.json

# 4. Deploy Lambda (see setup_guide.md)

# 5. Run analysis notebook
jupyter notebook notebooks/text_analysis.ipynb
```

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://text-corpus-your-id --recursive
aws s3 rb s3://text-corpus-your-id

# Delete DynamoDB table
aws dynamodb delete-table --table-name TextAnalysis

# Delete Lambda function
aws lambda delete-function --function-name process-text-document

# Delete IAM role
aws iam detach-role-policy --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy --role-name lambda-text-processor \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam delete-role --role-name lambda-text-processor

# Or use: python scripts/cleanup_aws.py (automated)
```

See `cleanup_guide.md` for detailed step-by-step instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/latest/developerguide/)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

### Text Data Sources
- [Project Gutenberg](https://www.gutenberg.org/) - 70,000+ free ebooks
- [Oxford Text Archive](https://ota.bodleian.ox.ac.uk/) - Literary and linguistic data
- [HathiTrust Digital Library](https://www.hathitrust.org/) - Millions of digitized texts
- [Internet Archive](https://archive.org/details/texts) - Historical texts

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/usage)
- [Gensim Documentation](https://radimrehurek.com/gensim/) - Topic modeling
- [pandas Documentation](https://pandas.pydata.org/docs/)

### Digital Humanities Resources
- [Programming Historian](https://programminghistorian.org/) - DH tutorials
- [TAPoR](http://tapor.ca/) - Text analysis tools
- [Voyant Tools](https://voyant-tools.org/) - Web-based text analysis

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `digital-humanities`, `tier-2`, `aws`, `nlp`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `aws-lambda`

### Digital Humanities Help
- Digital Humanities Slack: https://digitalhumanities.slack.com/
- Humanist Discussion Group: https://dhhumanist.org/
- DH Questions: https://digitalhumanities.org/answers/

---

## Cost Tracking

### Monitor Your Spending

```bash
# Check current AWS charges
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost"

# Set up billing alerts in AWS console:
# https://docs.aws.amazon.com/billing/latest/userguide/budgets-create.html
```

Recommended alerts:
- $5 threshold (notification)
- $15 threshold (warning)
- $30 threshold (critical - likely misconfigured)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $7-12 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable (pay per second) |
| **Corpus Size** | Limited to ~5,000 texts | Millions of texts possible |
| **Processing** | Sequential in notebook | Parallel Lambda functions |
| **Persistence** | Session-based | Permanent S3 + DynamoDB |
| **Collaboration** | Limited | Full team access to corpus |
| **Querying** | Python only | SQL (Athena) + Python |

---

## What's Next?

After completing this project:

**Skill Building**
- Advanced NLP techniques (dependency parsing, coreference)
- Lambda layers for large NLP libraries
- DynamoDB query optimization
- Athena query performance tuning
- Cost optimization strategies

**Project Extensions**
- Real-time text analysis pipeline
- Automated corpus ingestion from archives
- Character network visualization
- Temporal language evolution analysis
- Cross-linguistic comparative analysis
- Integration with digital repositories

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Multi-region deployment for global teams
- API Gateway for public corpus access
- SageMaker for ML-based analysis

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_dh_tier2,
  title = {Historical Text Corpus Analysis with S3 and Lambda: Tier 2},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

---

## License

Apache License 2.0 - See [LICENSE](../../../../LICENSE) for details.

---

## Ready to Start?

Follow the [setup_guide.md](setup_guide.md) to configure your AWS environment and begin analyzing historical texts at scale!

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
