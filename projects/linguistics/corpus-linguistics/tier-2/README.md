# Corpus Linguistics Analysis with S3 and Lambda - Tier 2 Project

**Duration:** 2-4 hours | **Cost:** $6-12 | **Platform:** AWS + Local machine

Analyze multilingual text corpora using serverless AWS services. Upload linguistic corpora to S3, perform automated linguistic annotation and analysis with Lambda, store results in DynamoDB, and query patterns with Athena—all without managing servers.

---

## What You'll Build

A cloud-native corpus linguistics pipeline that demonstrates:

1. **Corpus Storage** - Upload multilingual text corpora to S3 (~2-5GB)
2. **Serverless Analysis** - Lambda functions to perform linguistic annotation (POS tagging, lemmatization)
3. **Metadata Storage** - Store linguistic features in DynamoDB for fast querying
4. **Pattern Discovery** - Query collocations, concordances, and lexical patterns with Athena
5. **Visualization** - Generate word clouds, frequency distributions, and collocation networks

This bridges the gap between local corpus analysis (Tier 1) and production infrastructure (Tier 3).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Local Machine                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Python Scripts & Jupyter Notebook                          │ │
│  │ • upload_to_s3.py - Upload corpus files                   │ │
│  │ • lambda_function.py - Linguistic analysis                │ │
│  │ • query_results.py - Query linguistic patterns           │ │
│  │ • corpus_analysis.ipynb - Visualization & exploration    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AWS Services                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  S3 Bucket       │  │ Lambda Function  │  │  DynamoDB        │
│  │                  │  │                  │  │                  │
│  │ • Raw corpus     │→ │ NLP Processing:  │→ │ Linguistic       │
│  │   (txt files)    │  │ • Tokenization   │  │ metadata:        │
│  │ • Organized by:  │  │ • POS tagging    │  │ • Word freq      │
│  │   - language     │  │ • Lemmatization  │  │ • Collocations   │
│  │   - genre        │  │ • N-grams        │  │ • Complexity     │
│  │   - register     │  │ • Complexity     │  │ • POS patterns   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│  ┌──────────────────────────────────────────────────────────────┐
│  │  Athena (SQL Queries)                                        │
│  │  • Query word frequencies across languages                   │
│  │  • Find collocations and n-grams                            │
│  │  • Compare lexical diversity metrics                        │
│  │  • Analyze syntactic complexity patterns                    │
│  └──────────────────────────────────────────────────────────────┘
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
- Basic Python (pandas, boto3)
- Understanding of corpus linguistics concepts
- AWS fundamentals (S3, Lambda, DynamoDB, IAM)
- Basic NLP concepts (tokenization, POS tagging, lemmatization)

### Required Tools
- **Python 3.8+** with:
  - boto3 (AWS SDK)
  - nltk (Natural Language Toolkit)
  - spacy (Advanced NLP library)
  - pandas (data manipulation)
  - matplotlib, wordcloud (visualization)
  - networkx (collocation networks)

- **AWS Account** with:
  - S3 bucket access
  - Lambda permissions
  - DynamoDB access
  - IAM role creation capability
  - (Optional) Athena query capability

### Installation

```bash
# Clone the project
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/linguistics/corpus-linguistics/tier-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Download spaCy models (optional, for advanced analysis)
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
```

---

## Quick Start (15 minutes)

### Step 1: Set Up AWS (10 minutes)
```bash
# Follow setup_guide.md for detailed instructions
# Creates:
# - S3 bucket: linguistic-corpus-{your-id}
# - IAM role: lambda-linguistic-processor
# - Lambda function: analyze-linguistic-corpus
# - DynamoDB table: LinguisticAnalysis
```

### Step 2: Upload Sample Corpus (3 minutes)
```bash
python scripts/upload_to_s3.py
```

### Step 3: Process Corpus with Lambda (2 minutes)
```bash
# Lambda automatically triggers on S3 upload
# Or invoke manually via AWS console or boto3
```

### Step 4: Query Results (2 minutes)
```bash
python scripts/query_results.py
```

### Step 5: Visualize (5 minutes)
Open `notebooks/corpus_analysis.ipynb` in Jupyter and run all cells.

---

## Detailed Workflow

### 1. Corpus Preparation (Setup)

**What's happening:**
- Download or prepare multilingual text corpus
- Organize by language, genre, and register
- Create S3 bucket with proper permissions
- Upload corpus files maintaining directory structure

**Files involved:**
- `setup_guide.md` - Step-by-step AWS setup
- `scripts/upload_to_s3.py` - Upload automation

**Corpus structure:**
```
corpus/
├── english/
│   ├── academic/
│   │   ├── article1.txt
│   │   └── article2.txt
│   ├── news/
│   │   ├── news1.txt
│   │   └── news2.txt
│   └── fiction/
│       ├── novel1.txt
│       └── novel2.txt
├── spanish/
│   ├── academic/
│   └── news/
└── french/
    ├── academic/
    └── news/
```

**Time:** 20-30 minutes (includes setup_guide steps)

### 2. Lambda Linguistic Processing

**What's happening:**
- Lambda reads text files from S3
- Performs linguistic annotation:
  - **Tokenization** - Split text into words and sentences
  - **POS Tagging** - Part-of-speech annotation
  - **Lemmatization** - Convert words to base forms
  - **N-gram Extraction** - Bigrams and trigrams
  - **Collocation Detection** - Statistically significant word pairs
  - **Lexical Diversity** - Type-token ratio (TTR), MATTR
  - **Syntactic Complexity** - Average sentence length, clause depth
- Stores results in DynamoDB with metadata

**Lambda function event**:
```python
# Event: S3 upload trigger
{
    'Records': [{
        'bucket': {'name': 'linguistic-corpus-xxxx'},
        'object': {'key': 'english/academic/article1.txt'}
    }]
}

# Processing outputs:
# - Word frequencies (lemmatized)
# - POS tag distributions
# - Top collocations (bigrams/trigrams)
# - Lexical diversity metrics
# - Syntactic complexity scores
```

**Files involved:**
- `scripts/lambda_function.py` - NLP processing code
- `setup_guide.md` - Lambda deployment steps

**Time:** 5-15 minutes execution (depends on corpus size)

### 3. Results Storage

**What's happening:**
- Linguistic annotations stored in DynamoDB
- Organized by text_id, language, genre, register
- Fast queries by any attribute
- Original texts remain in S3 for reference

**DynamoDB Schema:**
```json
{
    "text_id": "english_academic_article1",
    "language": "english",
    "genre": "academic",
    "register": "formal",
    "word_count": 5420,
    "unique_words": 1243,
    "type_token_ratio": 0.229,
    "avg_sentence_length": 23.4,
    "top_collocations": [
        {"bigram": "machine learning", "freq": 45, "pmi": 8.32},
        {"bigram": "neural network", "freq": 38, "pmi": 7.89}
    ],
    "pos_distribution": {
        "NOUN": 1245,
        "VERB": 678,
        "ADJ": 432
    },
    "lexical_diversity": {
        "ttr": 0.229,
        "mattr": 0.654,
        "hdd": 0.712
    }
}
```

**S3 Structure:**
```
s3://linguistic-corpus-{your-id}/
├── raw/                           # Original text files
│   ├── english/
│   │   ├── academic/
│   │   ├── news/
│   │   └── fiction/
│   ├── spanish/
│   └── french/
├── processed/                     # Annotated results (JSON)
│   ├── english_academic_article1.json
│   ├── english_news_news1.json
│   └── ...
└── logs/                          # Lambda execution logs
    └── processing_log.txt
```

### 4. Results Analysis

**What's happening:**
- Query DynamoDB for linguistic patterns
- Download specific texts or annotations
- Analyze with Jupyter notebook:
  - Word frequency distributions
  - Collocation networks
  - Lexical diversity comparisons
  - Cross-linguistic analysis
  - Genre/register differences
- Create publication-quality visualizations

**Analysis capabilities:**
- **Concordances** - Find words in context (KWIC format)
- **Collocations** - Statistically significant word combinations
- **Frequency Lists** - Most common words/lemmas by corpus subset
- **Lexical Diversity** - Compare vocabulary richness across texts
- **Syntactic Complexity** - Sentence structure analysis
- **Cross-linguistic** - Compare patterns across languages

**Files involved:**
- `notebooks/corpus_analysis.ipynb` - Analysis notebook
- `scripts/query_results.py` - Query and retrieve data
- (Optional) Athena SQL queries for ad-hoc analysis

**Time:** 30-60 minutes analysis

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
│   └── corpus_analysis.ipynb         # Main analysis notebook
│                                     # - Upload corpus
│                                     # - Trigger processing
│                                     # - Query patterns
│                                     # - Visualize results
│
└── scripts/
    ├── upload_to_s3.py               # Upload corpus to S3
    │                                 # - Organize by language/genre
    │                                 # - Progress tracking
    │                                 # - Error handling
    │
    ├── lambda_function.py            # Lambda NLP processing
    │                                 # - Tokenization, POS, lemmas
    │                                 # - N-grams and collocations
    │                                 # - Lexical diversity
    │                                 # - Store in DynamoDB
    │
    ├── query_results.py              # Query DynamoDB
    │                                 # - Search by language/genre
    │                                 # - Find linguistic patterns
    │                                 # - Export results
    │
    └── __init__.py
```

---

## Cost Breakdown

**Total estimated cost: $6-12 per run**

| Service | Usage | Cost |
|---------|-------|------|
| **S3 Storage** | 3GB × 7 days | $0.21 |
| **S3 Requests** | ~500 PUT/GET requests | $0.03 |
| **Lambda Executions** | 50 invocations × 30 sec | $0.50 |
| **Lambda Compute** | 50 GB-seconds (512MB) | $0.80 |
| **DynamoDB Storage** | ~100MB data | $0.25 |
| **DynamoDB Reads** | 1000 read units | $0.25 |
| **DynamoDB Writes** | 500 write units | $0.65 |
| **Data Transfer** | Upload + download (1GB) | $0.10 |
| **Athena Queries** | 5 queries × 500MB scanned | $0.01 |
| **Total** | | **$6-12** |

**Cost optimization tips:**
1. Delete S3 objects after analysis ($0.21 savings)
2. Use Lambda for < 1 minute processing (set timeout appropriately)
3. Use DynamoDB on-demand pricing (pay per request)
4. Compress text files before upload (reduce storage/transfer costs)
5. Query DynamoDB directly instead of Athena when possible

**Free Tier Usage:**
- **S3**: First 5GB free (12 months)
- **Lambda**: 1M invocations + 400,000 GB-seconds free (12 months)
- **DynamoDB**: 25GB storage + 25 read/write units free (always free)
- **Athena**: First 1TB scanned free (per month)

Most of this project can run within free tier limits!

---

## Key Learning Objectives

### AWS Services
- S3 bucket creation and organization
- Lambda function deployment with NLP libraries
- DynamoDB table design for linguistic data
- IAM role creation with least privilege
- CloudWatch monitoring and logs
- (Optional) Athena for serverless SQL queries

### Cloud Concepts
- Object storage for unstructured text data
- Serverless computing (no infrastructure management)
- Event-driven architecture (S3 triggers Lambda)
- NoSQL database design
- Cost-conscious design

### Corpus Linguistics Skills
- Automated linguistic annotation at scale
- Collocation analysis and n-gram extraction
- Lexical diversity measurement
- Syntactic complexity analysis
- Cross-linguistic comparison
- Concordance generation

---

## Time Estimates

**First Time Setup:**
- Read setup_guide.md: 10 minutes
- Create S3 bucket: 2 minutes
- Create DynamoDB table: 3 minutes
- Create IAM role: 5 minutes
- Deploy Lambda: 5-10 minutes (with dependencies)
- Configure S3 triggers: 3 minutes
- **Subtotal setup: 28-33 minutes**

**Corpus Processing:**
- Prepare corpus files: 5-10 minutes
- Upload to S3: 5-15 minutes (3GB, depends on connection)
- Lambda processing: 5-15 minutes (50 texts × 30 sec each)
- **Subtotal processing: 15-40 minutes**

**Analysis:**
- Query DynamoDB: 5 minutes
- Jupyter analysis: 30-45 minutes
- Generate visualizations: 10-15 minutes
- **Subtotal analysis: 45-65 minutes**

**Total time: 2-3 hours** (including setup)

---

## AWS Account Setup

### Prerequisites
1. Create AWS account: https://aws.amazon.com/
2. (Optional) Activate free tier: https://console.aws.amazon.com/billing/
3. Create IAM user for programmatic access

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
```

### Data Preparation
- Sample corpora provided in notebook
- Or bring your own: Plain text files (.txt)
- Supported: Any language with NLTK/spaCy support
- Recommended: 50-500 texts, 1-10MB each

---

## Running the Project

### Option 1: Automated (Recommended for First Time)
```bash
# Step 1: Setup AWS services (follow setup_guide.md)
# Manual: Create S3, DynamoDB, Lambda, IAM role

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Upload corpus
python scripts/upload_to_s3.py --bucket linguistic-corpus-YOUR_ID --corpus-dir ./sample_corpus

# Step 4: Lambda processes automatically (or invoke manually)
# Check DynamoDB for results

# Step 5: Analyze results
jupyter notebook notebooks/corpus_analysis.ipynb
```

### Option 2: Manual (Detailed Control)
```bash
# 1. Create S3 bucket manually
aws s3 mb s3://linguistic-corpus-$(date +%s) --region us-east-1

# 2. Upload corpus with metadata
aws s3 cp ./sample_corpus/ s3://linguistic-corpus-xxxx/raw/ --recursive

# 3. Deploy Lambda (see setup_guide.md)
# Package dependencies: nltk, spacy models
# Upload to Lambda console

# 4. Invoke Lambda for each text
python scripts/invoke_lambda.py

# 5. Run analysis notebook
jupyter notebook notebooks/corpus_analysis.ipynb
```

---

## Jupyter Notebook Workflow

The main analysis notebook includes:

### 1. Data Loading
- Download linguistic annotations from DynamoDB
- Load into pandas DataFrame
- Filter by language, genre, register

### 2. Frequency Analysis
- Word frequency distributions
- Lemma frequency distributions
- POS tag distributions
- Compare across languages/genres

### 3. Collocation Analysis
- Extract top bigrams/trigrams
- Calculate pointwise mutual information (PMI)
- Visualize collocation networks
- Compare collocations across corpora

### 4. Lexical Diversity
- Type-token ratio (TTR)
- Moving average TTR (MATTR)
- Hypergeometric distribution D (HD-D)
- Compare across texts and genres

### 5. Syntactic Complexity
- Average sentence length
- Average word length
- Clause complexity
- Compare formal vs. informal registers

### 6. Cross-Linguistic Analysis
- Compare word frequencies across languages
- Identify cognates and borrowings
- Analyze syntactic differences
- Genre differences across languages

### 7. Visualization
- Word clouds (by language/genre)
- Frequency distribution plots
- Collocation network graphs
- Complexity comparison charts
- Interactive visualizations

### 8. Export
- Save figures (PNG, SVG)
- Export data (CSV, JSON)
- Generate summary statistics

---

## What You'll Discover

### Linguistic Insights
- Most frequent words and collocations in your corpus
- Lexical diversity across different genres and registers
- Syntactic complexity patterns
- Cross-linguistic differences in word usage
- Genre-specific vocabulary and structures

### AWS Insights
- Serverless NLP processing at scale
- Cost-effective cloud-based corpus analysis
- Fast querying with NoSQL databases
- Scalability: Process 100 texts as easily as 10,000

### Research Insights
- **Reproducibility**: Same code, same results
- **Collaboration**: Share corpora and results in cloud
- **Scale**: Analyze large corpora without local storage limits
- **Flexibility**: Add new languages/texts incrementally

---

## Example Queries

### Query DynamoDB
```python
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('LinguisticAnalysis')

# Find all academic texts
response = table.query(
    IndexName='genre-index',
    KeyConditionExpression='genre = :genre',
    ExpressionAttributeValues={':genre': 'academic'}
)

# Find texts with high lexical diversity
response = table.scan(
    FilterExpression='lexical_diversity.ttr > :threshold',
    ExpressionAttributeValues={':threshold': 0.5}
)
```

### Query with Athena (SQL)
```sql
-- Find top collocations in academic English
SELECT
    text_id,
    language,
    collocation.bigram,
    collocation.pmi
FROM linguistic_analysis
WHERE language = 'english'
    AND genre = 'academic'
ORDER BY collocation.pmi DESC
LIMIT 20;

-- Compare lexical diversity across genres
SELECT
    genre,
    AVG(type_token_ratio) as avg_ttr,
    AVG(lexical_diversity.mattr) as avg_mattr
FROM linguistic_analysis
GROUP BY genre
ORDER BY avg_ttr DESC;
```

---

## Next Steps

### Extend This Project
1. **More Languages**: Add support for 50+ languages with spaCy
2. **More Features**: Add sentiment analysis, named entity recognition
3. **Larger Corpora**: Process millions of texts with AWS Batch
4. **Real-time Analysis**: Set up Lambda triggers for automatic processing
5. **Visualization Dashboard**: Create interactive dashboard with AWS QuickSight
6. **Machine Learning**: Train language models on your corpus with SageMaker

### Move to Tier 3 (Production)
- CloudFormation for infrastructure-as-code
- Production-grade Lambda layers with dependencies
- Auto-scaling DynamoDB for large corpora
- Multi-region deployment
- Advanced monitoring and alerting
- Cost optimization with Reserved Capacity

### Research Applications
- **Diachronic Analysis**: Track language change over time
- **Sociolinguistics**: Compare language use across demographics
- **Discourse Analysis**: Analyze conversation patterns
- **Stylometry**: Authorship attribution
- **Translation Studies**: Compare source and target texts
- **Language Learning**: Create learner corpora and analyze errors

---

## Troubleshooting

### Common Issues

**"botocore.exceptions.NoCredentialsError"**
```bash
# Solution: Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Key, region, output format
```

**"S3 bucket already exists"**
```bash
# Solution: Use a unique bucket name
s3://linguistic-corpus-$(date +%s)-yourname
# Or append your name/ID
```

**"Lambda timeout"**
```python
# Solution: Increase timeout in Lambda console
# Default: 3 seconds
# Recommended: 60 seconds for small texts
# For larger texts: 300 seconds (5 minutes)
```

**"Lambda out of memory"**
```python
# Solution: Increase Lambda memory allocation
# Default: 128 MB
# Recommended: 512 MB (with NLTK)
# Recommended: 1024 MB (with spaCy models)
# Cost increases slightly with memory
```

**"NLTK data not found"**
```python
# Solution: Include NLTK data in Lambda deployment
# Create layer with: punkt, averaged_perceptron_tagger, wordnet
# Or download in Lambda code (slower, not recommended)
```

**"spaCy model not found"**
```python
# Solution: Include spaCy models in Lambda layer
# Models are large (10-50MB), consider:
# 1. Use NLTK instead (smaller)
# 2. Create Lambda layer with models
# 3. Download from S3 on cold start (slower)
```

**"DynamoDB query returns no results"**
```python
# Check Lambda CloudWatch logs for errors
# Verify Lambda has write permissions to DynamoDB
# Check table name matches in code
```

**"Data too large for Lambda"**
```python
# Solution: Split large texts into chunks
# Lambda payload limit: 6MB (synchronous), 256KB (asynchronous)
# For large texts: Read from S3 directly (not via event)
```

See troubleshooting section in setup_guide.md for more solutions.

---

## Cleanup Guide

When finished, delete AWS resources to stop incurring charges:

```bash
# Delete S3 bucket and contents
aws s3 rm s3://linguistic-corpus-xxxx --recursive
aws s3 rb s3://linguistic-corpus-xxxx

# Delete Lambda function
aws lambda delete-function --function-name analyze-linguistic-corpus

# Delete DynamoDB table
aws dynamodb delete-table --table-name LinguisticAnalysis

# Delete IAM role (first detach policies)
aws iam detach-role-policy --role-name lambda-linguistic-processor \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role-policy --role-name lambda-linguistic-processor \
  --policy-name linguistic-processor-policy
aws iam delete-role --role-name lambda-linguistic-processor

# Or use: python scripts/cleanup.py (automated)
```

See `cleanup_guide.md` for detailed step-by-step instructions.

---

## Resources

### AWS Documentation
- [S3 Getting Started](https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [DynamoDB Developer Guide](https://docs.aws.amazon.com/dynamodb/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/)

### Corpus Linguistics Resources
- [NLTK Book](https://www.nltk.org/book/)
- [spaCy Documentation](https://spacy.io/usage)
- [Corpus Linguistics Methods](https://www.corpuslinguistics.net/)
- [Lancaster Stats Tools](http://corpora.lancs.ac.uk/stats/)

### Python Libraries
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy API](https://spacy.io/api)
- [pandas Documentation](https://pandas.pydata.org/docs/)

### Open Corpora
- [COCA (Corpus of Contemporary American English)](https://www.english-corpora.org/coca/)
- [BNC (British National Corpus)](http://www.natcorp.ox.ac.uk/)
- [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download)
- [Universal Dependencies](https://universaldependencies.org/)

---

## Getting Help

### Project Issues
- GitHub Issues: https://github.com/research-jumpstart/research-jumpstart/issues
- Tag: `linguistics`, `corpus-linguistics`, `tier-2`, `aws`

### AWS Support
- AWS Support Center: https://console.aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- Stack Overflow: Tag `amazon-aws`, `boto3`, `aws-lambda`

### Linguistics Help
- Corpora Mailing List: https://www.hit.uib.no/corpora/
- Linguistics Stack Exchange: https://linguistics.stackexchange.com/
- Reddit: r/linguistics, r/LanguageTechnology

---

## Cost Tracking

### Monitor Your Spending

```bash
# Check current AWS charges
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=SERVICE

# Set up billing alerts in AWS console:
# https://docs.aws.amazon.com/billing/latest/userguide/budgets-create.html
```

Recommended alerts:
- $5 threshold (warning)
- $15 threshold (warning)
- $30 threshold (critical)

---

## Key Differences from Tier 1

| Aspect | Tier 1 (Studio Lab) | Tier 2 (AWS) |
|--------|-------------------|------------|
| **Platform** | Free SageMaker Studio Lab | AWS account required |
| **Cost** | Free | $6-12 per run |
| **Storage** | 15GB persistent | Unlimited (pay per GB) |
| **Compute** | 4GB RAM, 12-hour sessions | Scalable Lambda (pay per second) |
| **Corpus Size** | Limited to 15GB | Petabytes possible |
| **Processing** | Single notebook | Parallel Lambda invocations |
| **Persistence** | Session-based | Permanent S3/DynamoDB storage |
| **Collaboration** | Limited | Full team access to corpus |
| **Languages** | All NLTK/spaCy languages | All NLTK/spaCy languages |
| **Automation** | Manual execution | Event-driven (S3 triggers) |

---

## What's Next?

After completing this project:

**Skill Building**
- AWS Lambda advanced features (layers, versions)
- DynamoDB advanced queries (GSI, LSI)
- Athena optimization techniques
- NLP pipeline optimization
- Cost optimization strategies

**Project Extensions**
- Real-time corpus analysis (Twitter, news feeds)
- Automated linguistic annotation pipeline
- Multi-corpus comparison dashboard
- Diachronic analysis (language change over time)
- Integration with other services (SNS for notifications)

**Tier 3 Transition**
- Production infrastructure with CloudFormation
- High-availability architecture
- Advanced monitoring and alerting
- Auto-scaling for large corpora
- Multi-region deployment for global access

---

## Citation

If you use this project in your research:

```bibtex
@software{research_jumpstart_corpus_tier2,
  title = {Corpus Linguistics Analysis with S3 and Lambda: Tier 2},
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

**Ready to start?** Follow the [setup_guide.md](setup_guide.md) to get started!

**Last updated:** 2025-11-14 | Research Jumpstart v1.0.0
