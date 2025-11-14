# Digital Humanities Text Analysis - Unified Studio

Production-ready text analysis toolkit for digital humanities research on AWS. Scale corpus analysis from thousands to millions of documents with enterprise infrastructure.

## Overview

This project provides:
- **CloudFormation Infrastructure**: S3 buckets, IAM roles, DynamoDB, Comprehend integration
- **Python Analysis Package**: Text loading, NLP analysis, sentiment, entity extraction, visualization
- **Production Notebooks**: Reproducible text analysis workflows in SageMaker
- **AWS AI Services**: Integration with Amazon Comprehend for advanced NLP

## Architecture

```
unified-studio/
├── cloudformation/
│   └── text-analysis-infrastructure.yaml  # AWS infrastructure template
├── src/
│   ├── __init__.py
│   ├── data_access.py                     # S3 and local corpus loading
│   ├── text_analysis.py                   # NLP analysis functions
│   └── visualization.py                   # Plotting functions
├── notebooks/                              # Analysis notebooks (optional)
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 1. Deploy AWS Infrastructure

```bash
# Configure AWS CLI
aws configure

# Deploy CloudFormation stack
cd cloudformation
aws cloudformation create-stack \
  --stack-name digital-humanities-text-analysis-dev \
  --template-body file://text-analysis-infrastructure.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=dev \
    ParameterKey=MonthlyBudgetLimit,ParameterValue=30 \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for stack creation
aws cloudformation wait stack-create-complete \
  --stack-name digital-humanities-text-analysis-dev

# Get outputs
aws cloudformation describe-stacks \
  --stack-name digital-humanities-text-analysis-dev \
  --query 'Stacks[0].Outputs'
```

### 2. Export Environment Variables

```bash
# From CloudFormation outputs
export CORPUS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name digital-humanities-text-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`CorpusBucketName`].OutputValue' \
  --output text)

export RESULTS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name digital-humanities-text-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ResultsBucketName`].OutputValue' \
  --output text)

export MODELS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name digital-humanities-text-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ModelsBucketName`].OutputValue' \
  --output text)

export ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name digital-humanities-text-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`AnalysisRoleArn`].OutputValue' \
  --output text)

export METADATA_TABLE=$(aws cloudformation describe-stacks \
  --stack-name digital-humanities-text-analysis-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`MetadataTableName`].OutputValue' \
  --output text)
```

### 3. Install Python Package

```bash
# Clone repository
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/digital-humanities/text-analysis/unified-studio

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 4. Upload Text Corpus

```bash
# Upload text files to the corpus bucket
aws s3 cp your_document.txt s3://${CORPUS_BUCKET}/corpus/your_document.txt

# Or sync entire directory
aws s3 sync ./texts/ s3://${CORPUS_BUCKET}/corpus/
```

### 5. Run Analysis

```python
from src.data_access import TextDataAccess
from src.text_analysis import (
    analyze_sentiment,
    extract_entities,
    extract_keywords,
    calculate_readability
)
from src.visualization import plot_word_cloud, plot_sentiment_timeline

# Initialize data access
data_access = TextDataAccess(use_anon=False, region='us-east-1')

# Load corpus from S3
corpus_df = data_access.load_corpus_from_s3(
    bucket='your-corpus-bucket',
    prefix='corpus/'
)

# Analyze sentiment
for idx, row in corpus_df.iterrows():
    sentiment = analyze_sentiment(row['text'])
    print(f"Document {row['document_id']}: {sentiment['compound']:.2f}")

# Extract entities
text = corpus_df.iloc[0]['text']
entities = extract_entities(text)
print(f"Entities: {entities}")

# Calculate readability
readability = calculate_readability(text)
print(f"Reading ease: {readability['flesch_reading_ease']:.1f}")

# Visualize
plot_word_cloud(text, save_path='wordcloud.png')
plot_sentiment_timeline(corpus_df, save_path='sentiment.png')

# Save results
data_access.save_results(
    corpus_df,
    bucket='your-results-bucket',
    key='analysis/corpus_analysis.csv'
)
```

## Infrastructure Components

### S3 Buckets

**Corpus Bucket** (`digital-humanities-text-analysis-corpus-{env}-{account}`)
- Stores text documents and corpora
- Encryption: AES256
- Versioning: Enabled
- Lifecycle: Transition to IA after 60 days, Glacier after 120 days

**Results Bucket** (`digital-humanities-text-analysis-results-{env}-{account}`)
- Stores analysis outputs (CSV, plots, reports)
- Encryption: AES256
- Lifecycle: Delete after 180 days

**Models Bucket** (`digital-humanities-text-analysis-models-{env}-{account}`)
- Stores trained NLP models
- Encryption: AES256
- Versioning: Enabled

### IAM Role

**AnalysisRole** (`digital-humanities-text-analysis-role-{env}`)
- Used by: SageMaker, Lambda, Comprehend
- Permissions:
  - Read/write to corpus, results, and models buckets
  - Read-only access to public datasets
  - Amazon Comprehend API access
  - DynamoDB read/write for metadata
  - CloudWatch Logs write access

### DynamoDB Table

**DocumentMetadataTable** (`digital-humanities-text-analysis-metadata-{env}`)
- Stores document metadata and analysis results
- Billing: Pay-per-request
- Attributes: document_id (hash key), upload_date (range key)

### Monitoring

**CloudWatch Logs** (`/aws/digital-humanities-text-analysis/{env}`)
- Log retention: 30 days
- Captures all analysis execution logs

**SNS Topic** (`digital-humanities-text-analysis-alerts-{env}`)
- Receives cost and error alerts

**Cost Alarm**
- Monitors S3 costs
- Threshold: $30/month (configurable)

## API Documentation

### TextDataAccess

```python
class TextDataAccess:
    """Handle loading and saving text documents and analysis results."""

    def __init__(self, use_anon: bool = False, region: str = 'us-east-1'):
        """Initialize data access client."""

    def load_text_from_s3(self, bucket: str, key: str, encoding: str = 'utf-8') -> str:
        """Load text file from S3."""

    def load_corpus_from_s3(
        self,
        bucket: str,
        prefix: str = '',
        file_extensions: List[str] = ['.txt', '.md']
    ) -> pd.DataFrame:
        """Load multiple text files from S3 into DataFrame."""

    def load_text_from_local(self, file_path: str, encoding: str = 'utf-8') -> str:
        """Load text file from local filesystem."""

    def load_corpus_from_local(
        self,
        directory: str,
        file_extensions: List[str] = ['.txt', '.md'],
        recursive: bool = True
    ) -> pd.DataFrame:
        """Load text files from local directory."""

    def save_results(self, df: pd.DataFrame, bucket: str, key: str):
        """Save analysis results to S3."""

    def save_model(self, model_data: bytes, bucket: str, key: str, metadata: Dict = None):
        """Save trained model to S3."""

    def save_metadata(self, table_name: str, document_id: str, metadata: Dict):
        """Save document metadata to DynamoDB."""
```

### Text Analysis Functions

```python
def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment of text."""

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text."""

def extract_keywords(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """Extract most frequent keywords."""

def calculate_readability(text: str) -> Dict[str, float]:
    """Calculate readability metrics."""

def detect_language(text: str) -> str:
    """Detect language of text."""

def perform_topic_modeling(
    corpus_df: pd.DataFrame,
    text_column: str = 'text',
    n_topics: int = 5
) -> Dict:
    """Perform topic modeling on corpus."""

def analyze_corpus_statistics(corpus_df: pd.DataFrame) -> Dict:
    """Generate comprehensive corpus statistics."""

def compare_texts(text1: str, text2: str) -> Dict[str, float]:
    """Compare two texts for similarity."""

def extract_ngrams(text: str, n: int = 2, top_k: int = 10) -> List[Tuple[str, int]]:
    """Extract most frequent n-grams."""
```

### Visualization Functions

```python
def plot_word_cloud(text: str, save_path: str = None):
    """Create word cloud visualization."""

def plot_sentiment_timeline(
    df: pd.DataFrame,
    text_column: str = 'text',
    date_column: str = None
):
    """Plot sentiment over time or documents."""

def plot_entity_network(entities_dict: Dict[str, List[str]]):
    """Visualize entity distribution by type."""

def plot_topic_distribution(topics_dict: Dict[str, List[str]]):
    """Visualize topic modeling results."""

def plot_ngram_frequency(ngrams: List[Tuple[str, int]]):
    """Plot n-gram frequency distribution."""

def plot_readability_comparison(df: pd.DataFrame, group_column: str = None):
    """Compare readability scores across documents."""

def plot_corpus_statistics(stats: Dict):
    """Visualize corpus statistics."""

def plot_text_similarity_matrix(df: pd.DataFrame):
    """Create similarity matrix heatmap."""
```

## Cost Estimates

### Development Environment (Small)
- **S3 Storage**: 10 GB text corpus → ~$0.23/month
- **S3 Requests**: 10,000 GET/PUT → ~$0.05/month
- **DynamoDB**: 100,000 reads/writes → ~$1.25/month
- **SageMaker Studio**: 10 hours ml.t3.medium → ~$4.60/month
- **Comprehend**: 10,000 text units → ~$1.00/month
- **Data Transfer**: 10 GB out → ~$0.90/month
- **CloudWatch**: Logs and monitoring → ~$2.00/month
- **Total**: ~$10-15/month

### Production Environment (Medium)
- **S3 Storage**: 500 GB corpus → ~$11.50/month
- **S3 Requests**: 100,000 operations → ~$0.50/month
- **DynamoDB**: 10M reads/writes → ~$12.50/month
- **SageMaker Training**: 50 hours ml.m5.xlarge → ~$230/month
- **Comprehend**: 1M text units → ~$100/month
- **Data Transfer**: 100 GB out → ~$9/month
- **CloudWatch**: Enhanced monitoring → ~$10/month
- **Total**: ~$375-400/month

### Enterprise Environment (Large)
- **S3 Storage**: 5 TB corpus → ~$115/month
- **S3 Glacier**: 20 TB archived → ~$80/month
- **DynamoDB**: 100M reads/writes → ~$125/month
- **SageMaker Processing**: 200 hours ml.m5.4xlarge → ~$3,680/month
- **Comprehend**: 10M text units → ~$1,000/month
- **Batch Compute**: 500 vCPU-hours → ~$150/month
- **Data Transfer**: 500 GB out → ~$45/month
- **CloudWatch**: Full monitoring → ~$30/month
- **Total**: ~$5,200-5,500/month

### Cost Optimization Tips

1. **Lifecycle Policies**: Archive old corpora to Glacier
2. **DynamoDB On-Demand**: Use provisioned capacity for predictable workloads
3. **Comprehend Batching**: Batch requests for 65% discount
4. **Spot Instances**: Use Spot for batch processing
5. **S3 Select**: Filter data server-side to reduce transfer
6. **Compression**: Compress text files (70-90% reduction)

## Usage Examples

### Example 1: Sentiment Analysis of Historical Letters

```python
from src.data_access import TextDataAccess
from src.text_analysis import analyze_sentiment
from src.visualization import plot_sentiment_timeline

# Load letters
data_access = TextDataAccess()
letters_df = data_access.load_corpus_from_s3('bucket', 'letters/')

# Analyze sentiment
sentiments = []
for text in letters_df['text']:
    sentiment = analyze_sentiment(text)
    sentiments.append(sentiment)

letters_df['sentiment'] = [s['compound'] for s in sentiments]

# Visualize timeline
plot_sentiment_timeline(letters_df, date_column='last_modified')

# Save results
data_access.save_results(letters_df, 'bucket', 'results/sentiment.csv')
```

### Example 2: Topic Modeling on Literary Corpus

```python
from src.text_analysis import perform_topic_modeling, analyze_corpus_statistics
from src.visualization import plot_topic_distribution

# Load corpus
corpus_df = data_access.load_corpus_from_local('./novels/')

# Corpus statistics
stats = analyze_corpus_statistics(corpus_df)
print(f"Total documents: {stats['total_documents']}")
print(f"Vocabulary size: {stats['vocabulary_size']:,}")

# Topic modeling
topics = perform_topic_modeling(corpus_df, n_topics=5, n_words=10)

# Visualize
plot_topic_distribution(topics, save_path='topics.png')
```

### Example 3: Comparative Text Analysis

```python
from src.text_analysis import compare_texts, extract_keywords
from src.visualization import plot_text_similarity_matrix

# Load two versions of a text
text1 = data_access.load_text_from_s3('bucket', 'version1.txt')
text2 = data_access.load_text_from_s3('bucket', 'version2.txt')

# Compare
comparison = compare_texts(text1, text2)
print(f"Jaccard similarity: {comparison['jaccard_similarity']:.2f}")
print(f"Unique to v1: {comparison['unique_to_text1']} words")
print(f"Unique to v2: {comparison['unique_to_text2']} words")

# Extract unique keywords
keywords1 = extract_keywords(text1, top_n=20)
keywords2 = extract_keywords(text2, top_n=20)

# Compare entire corpus
corpus_df = data_access.load_corpus_from_s3('bucket', 'editions/')
plot_text_similarity_matrix(corpus_df, save_path='similarity.png')
```

### Example 4: Entity Extraction and Network Analysis

```python
from src.text_analysis import extract_entities
from src.visualization import plot_entity_network

# Load text
text = data_access.load_text_from_s3('bucket', 'manuscript.txt')

# Extract entities
entities = extract_entities(text)

print(f"People: {len(entities['PERSON'])}")
print(f"Locations: {len(entities['LOCATION'])}")
print(f"Organizations: {len(entities['ORGANIZATION'])}")

# Visualize
plot_entity_network(entities, save_path='entities.png')

# Save to metadata table
data_access.save_metadata(
    table_name='metadata-table',
    document_id='manuscript_001',
    metadata={
        'upload_date': '2025-11-09',
        'entities': entities,
        'entity_count': sum(len(v) for v in entities.values())
    }
)
```

### Example 5: Readability Analysis Across Time Periods

```python
from src.text_analysis import calculate_readability
from src.visualization import plot_readability_comparison

# Load corpus with time periods
corpus_df = data_access.load_corpus_from_s3('bucket', 'historical/')

# Add time period grouping (from metadata)
corpus_df['century'] = corpus_df['document_id'].str.extract(r'(\d{2}00s)')

# Calculate readability
readability_scores = []
for text in corpus_df['text']:
    scores = calculate_readability(text)
    readability_scores.append(scores['flesch_reading_ease'])

corpus_df['readability'] = readability_scores

# Compare across centuries
plot_readability_comparison(corpus_df, group_column='century')

# Statistical summary
print(corpus_df.groupby('century')['readability'].describe())
```

## Integration with AWS Comprehend

For advanced NLP capabilities, integrate with Amazon Comprehend:

```python
import boto3

comprehend = boto3.client('comprehend', region_name='us-east-1')

# Sentiment analysis
response = comprehend.detect_sentiment(
    Text=text,
    LanguageCode='en'
)
print(response['Sentiment'])

# Entity extraction
response = comprehend.detect_entities(
    Text=text,
    LanguageCode='en'
)
for entity in response['Entities']:
    print(f"{entity['Type']}: {entity['Text']} ({entity['Score']:.2f})")

# Key phrases
response = comprehend.detect_key_phrases(
    Text=text,
    LanguageCode='en'
)
for phrase in response['KeyPhrases']:
    print(f"{phrase['Text']} ({phrase['Score']:.2f})")
```

## Deployment Checklist

- [ ] AWS account with appropriate permissions
- [ ] AWS CLI configured with credentials
- [ ] Deploy CloudFormation stack
- [ ] Export environment variables
- [ ] Subscribe to SNS alert topic
- [ ] Upload text corpus to S3
- [ ] Install Python package
- [ ] Download spaCy language models
- [ ] Run test analysis
- [ ] Set up budget alerts
- [ ] Configure DynamoDB backup
- [ ] Document data retention policy
- [ ] Configure IAM user access
- [ ] Enable CloudTrail logging (optional)

## Troubleshooting

### Issue: "Access Denied" when accessing S3
**Solution**: Verify IAM role has correct permissions. Check S3AccessPolicy in CloudFormation template.

### Issue: Text encoding errors
**Solution**: Specify correct encoding in load functions (utf-8, latin-1, cp1252). Use `encoding='utf-8'` parameter.

### Issue: High Comprehend costs
**Solution**: Batch API calls, cache results, use custom models, limit text length to 5,000 bytes.

### Issue: DynamoDB throttling
**Solution**: Switch to provisioned capacity, increase WCU/RCU, use batch operations, add exponential backoff.

### Issue: Memory errors with large corpus
**Solution**: Process in batches, use generators, increase instance memory, use Dask for distributed processing.

## Development

### Running Tests
```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## Resources

- **NLTK**: [Documentation](https://www.nltk.org/)
- **spaCy**: [Documentation](https://spacy.io/)
- **Gensim**: [Topic Modeling](https://radimrehurek.com/gensim/)
- **Amazon Comprehend**: [Developer Guide](https://docs.aws.amazon.com/comprehend/)
- **Project Gutenberg**: [Free Texts](https://www.gutenberg.org/)
- **HathiTrust**: [Digital Library](https://www.hathitrust.org/)

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [Project Issues](https://github.com/yourusername/research-jumpstart/issues)
- Email: support@example.com

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{digital_humanities_unified_studio,
  title = {Digital Humanities Text Analysis - Unified Studio},
  author = {Research Jumpstart},
  year = {2025},
  url = {https://github.com/yourusername/research-jumpstart}
}
```
