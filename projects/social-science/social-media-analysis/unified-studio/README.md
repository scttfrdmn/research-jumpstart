# Social Media Analysis & Misinformation Detection
## Production-Ready Unified Studio Implementation

A comprehensive AWS-powered solution for analyzing social media data, detecting sentiment patterns, identifying misinformation indicators, and understanding information spread through network analysis.

## 🎯 Overview

This production implementation extends the Studio Lab prototype with enterprise-grade capabilities:

- **Scalable Data Access**: Process millions of social media posts from S3
- **Advanced NLP**: AWS Comprehend integration alongside local models
- **Network Analysis**: Graph-based understanding of information spread
- **Misinformation Detection**: Multi-factor risk scoring and pattern recognition
- **Production Infrastructure**: CloudFormation-managed AWS resources
- **Cost Monitoring**: Built-in budget alerts and optimization

## 🚀 Features

### Data Processing
- Load Twitter, Reddit, and custom social media datasets from S3
- Support for JSON, JSONL, CSV, and Parquet formats
- Automatic date filtering and sampling
- Batch processing for large datasets
- Results persistence to S3

### NLP Analysis
- **Local Models**: VADER sentiment, scikit-learn topic modeling
- **AWS Comprehend**: Advanced sentiment, entity detection, key phrases
- **Text Preprocessing**: URL/mention removal, tokenization, lemmatization
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) with Gensim
- **Language Detection**: Multi-language support via Comprehend

### Misinformation Detection
- Pattern recognition (excessive caps, punctuation, urgency language)
- Risk scoring (0-10 scale) based on multiple indicators
- Vague source detection
- Conspiracy language identification
- Call-to-action analysis

### Network Analysis
- Interaction network construction
- Centrality metrics (degree, betweenness, eigenvector)
- Community detection
- Influence mapping
- Information spread visualization

### Visualization
- Sentiment distribution plots
- Topic evolution over time
- Network graphs with centrality highlighting
- Engagement analysis (likes, retweets, viral content)
- Publication-quality figures (300 DPI)

## 📋 Prerequisites

### AWS Requirements
- AWS Account with billing enabled
- IAM permissions for:
  - CloudFormation stack creation
  - S3 bucket management
  - IAM role creation
  - Comprehend API access (optional)
  - CloudWatch logs and alarms
- AWS CLI configured (`aws configure`)

### Local Requirements
- Python 3.9+
- pip or conda
- 4GB+ RAM recommended
- 10GB+ disk space for dependencies

### Knowledge Requirements
- Python programming (intermediate)
- Basic AWS concepts (S3, IAM, CloudFormation)
- Social media data formats (JSON, CSV)
- NLP fundamentals helpful

## 🔧 Installation

### Step 1: Deploy AWS Infrastructure

```bash
# Navigate to CloudFormation directory
cd cloudformation/

# Deploy stack
aws cloudformation create-stack \
  --stack-name social-media-analysis-dev \
  --template-body file://social-media-infrastructure.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=Environment,ParameterValue=dev \
    ParameterKey=EnableComprehend,ParameterValue=true \
    ParameterKey=MonthlyBudgetLimit,ParameterValue=100

# Wait for completion (5-10 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name social-media-analysis-dev

# Get outputs
aws cloudformation describe-stacks \
  --stack-name social-media-analysis-dev \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
  --output table
```

See [CloudFormation README](cloudformation/README.md) for detailed deployment options.

### Step 2: Configure Environment

```bash
# Create .env file
cat > .env << EOF
# AWS Configuration
DATA_BUCKET=social-media-analysis-dev-ACCOUNT_ID
RESULTS_BUCKET=social-media-analysis-results-dev-ACCOUNT_ID
ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/social-media-analysis-analysis-role-dev
AWS_REGION=us-east-1

# Analysis Configuration
USE_COMPREHEND=true
BATCH_SIZE=100
MAX_WORKERS=4
EOF

# Replace ACCOUNT_ID with your AWS account ID
```

### Step 3: Install Python Package

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "from social_media_analysis import SocialMediaDataAccess, ComprehendAnalyzer; print('Success!')"

# Test AWS connectivity
aws s3 ls s3://$DATA_BUCKET

# Test Comprehend (if enabled)
aws comprehend detect-sentiment \
  --text "This is a test" \
  --language-code en
```

## 🏃 Quick Start

### Example 1: Analyze Twitter Dataset

```python
import os
from dotenv import load_dotenv
from social_media_analysis import SocialMediaDataAccess, analyze_sentiment
from social_media_analysis.visualization import plot_sentiment_distribution

# Load configuration
load_dotenv()

# Initialize data access
data_client = SocialMediaDataAccess(region=os.getenv('AWS_REGION'))

# Load Twitter data
df = data_client.load_twitter_dataset(
    bucket=os.getenv('DATA_BUCKET'),
    prefix='twitter/2025/11/',
    date_range=('2025-11-01', '2025-11-07'),
    sample_size=10000
)

print(f"Loaded {len(df)} tweets")
print(df.head())

# Analyze sentiment
results = analyze_sentiment(df['text'])

# Visualize
fig = plot_sentiment_distribution(results, save_path='sentiment_dist.png')
print("Sentiment analysis complete!")
```

### Example 2: Misinformation Detection

```python
from social_media_analysis import detect_misinformation_patterns, preprocess_text

# Load data
df = data_client.load_csv_dataset(
    bucket=os.getenv('DATA_BUCKET'),
    key='datasets/reddit_sample.csv'
)

# Preprocess text
df['clean_text'] = df['text'].apply(preprocess_text)

# Detect misinformation patterns
df['misinfo'] = df['text'].apply(detect_misinformation_patterns)

# Extract risk scores
df['risk_score'] = df['misinfo'].apply(lambda x: x['risk_score'])

# Find high-risk posts
high_risk = df[df['risk_score'] >= 7].sort_values('risk_score', ascending=False)

print(f"Found {len(high_risk)} high-risk posts")
print(high_risk[['text', 'risk_score']].head())

# Save results
data_client.save_results(high_risk, 'high_risk_posts.csv')
```

### Example 3: Network Analysis

```python
from social_media_analysis.network_analysis import (
    build_interaction_network,
    calculate_centrality,
    detect_communities
)
from social_media_analysis.visualization import plot_network_graph

# Load interaction data
df = data_client.load_twitter_dataset(
    bucket=os.getenv('DATA_BUCKET'),
    prefix='twitter/interactions/',
    sample_size=5000
)

# Build network
G = build_interaction_network(df)
print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Calculate centrality
centrality = calculate_centrality(G)
top_influencers = sorted(
    centrality['degree'].items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("Top 10 Influencers:")
for user, score in top_influencers:
    print(f"  {user}: {score:.4f}")

# Detect communities
communities = detect_communities(G)
print(f"Found {len(communities)} communities")

# Visualize
plot_network_graph(G, save_path='network.png')
```

### Example 4: AWS Comprehend Integration

```python
from social_media_analysis import ComprehendAnalyzer

# Initialize Comprehend client
comprehend = ComprehendAnalyzer(region=os.getenv('AWS_REGION'))

# Analyze single post
text = "Breaking news: Major tech company announces AI breakthrough!"

sentiment = comprehend.analyze_sentiment(text)
print(f"Sentiment: {sentiment['Sentiment']} ({sentiment['SentimentScore']})")

entities = comprehend.detect_entities(text)
print(f"Entities found: {len(entities['Entities'])}")
for entity in entities['Entities']:
    print(f"  - {entity['Text']} ({entity['Type']}): {entity['Score']:.2f}")

key_phrases = comprehend.detect_key_phrases(text)
print(f"Key phrases: {len(key_phrases['KeyPhrases'])}")
for phrase in key_phrases['KeyPhrases']:
    print(f"  - {phrase['Text']}: {phrase['Score']:.2f}")
```

## 📦 Module Documentation

### `data_access.py`

**`SocialMediaDataAccess`** - Primary class for data operations

```python
class SocialMediaDataAccess:
    def __init__(self, use_anon: bool = True, region: str = 'us-east-1'):
        """Initialize S3 client."""

    def load_twitter_dataset(self, bucket: str, prefix: str,
                           date_range: Optional[Tuple[str, str]] = None,
                           sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load Twitter data from S3 JSON/JSONL files."""

    def load_reddit_dataset(self, bucket: str, prefix: str,
                          subreddits: Optional[List[str]] = None,
                          sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load Reddit data from S3."""

    def load_csv_dataset(self, bucket: str, key: str) -> pd.DataFrame:
        """Load CSV dataset from S3."""

    def save_results(self, df: pd.DataFrame, filename: str):
        """Save analysis results to S3."""

    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """Validate dataset structure and quality."""
```

### `nlp_analysis.py`

```python
def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """Clean and normalize text (URLs, mentions, lemmatization)."""

def analyze_sentiment(texts: pd.Series) -> pd.DataFrame:
    """Sentiment analysis using VADER or Comprehend."""

def extract_topics(texts: List[str], n_topics: int = 5) -> Dict:
    """Topic modeling using Gensim LDA."""

def detect_misinformation_patterns(text: str) -> Dict:
    """Detect misinformation indicators with risk scoring."""
```

### `network_analysis.py`

```python
def build_interaction_network(df: pd.DataFrame) -> nx.Graph:
    """Construct network from retweets, mentions, replies."""

def calculate_centrality(G: nx.Graph) -> Dict:
    """Degree, betweenness, eigenvector centrality."""

def detect_communities(G: nx.Graph) -> List:
    """Community detection via modularity optimization."""
```

### `comprehend_client.py`

```python
class ComprehendAnalyzer:
    def analyze_sentiment(self, text: str, language: str = 'en') -> dict:
        """AWS Comprehend sentiment (POSITIVE/NEGATIVE/NEUTRAL/MIXED)."""

    def detect_entities(self, text: str, language: str = 'en') -> dict:
        """Named entity recognition (PERSON, LOCATION, ORGANIZATION, etc.)."""

    def detect_key_phrases(self, text: str, language: str = 'en') -> dict:
        """Extract salient key phrases."""
```

### `visualization.py`

```python
def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = None):
    """Bar chart of sentiment categories."""

def plot_topic_evolution(topics_over_time: pd.DataFrame, save_path: str = None):
    """Line chart showing topic trends."""

def plot_network_graph(G: nx.Graph, save_path: str = None):
    """Network visualization with spring layout."""

def plot_engagement_analysis(df: pd.DataFrame, save_path: str = None):
    """Histogram of engagement metrics."""
```

## 💰 Cost Considerations

### Monthly Cost Estimates

**Development/Testing** (~$20-50/month):
- S3 Storage (10GB): $0.23
- CloudWatch Logs (1GB): $0.53
- Comprehend (500K units): $50
- Data Transfer: ~$1

**Active Research** (~$200-500/month):
- S3 Storage (100GB): $2.30
- CloudWatch Logs (10GB): $5.30
- Comprehend (5M units): $500
- Data Transfer: ~$10

**Production Analysis** (~$1,000-3,000/month):
- S3 Storage (1TB): $23
- CloudWatch Logs (50GB): $26.50
- Comprehend (30M units): $3,000
- Data Transfer: ~$50

### Cost Optimization

1. **Use Local Models**: Set `USE_COMPREHEND=false` to use free VADER sentiment
   - Savings: ~95% on NLP costs
   - Tradeoff: Slightly lower accuracy

2. **Batch Processing**: Process data in larger batches
   - Reduces API overhead
   - More efficient resource utilization

3. **Data Lifecycle**: Template includes automatic S3 lifecycle policies
   - Transitions to Infrequent Access after 30 days
   - Intelligent-Tiering after 90 days
   - Automatic deletion after retention period

4. **Sampling**: Use `sample_size` parameter for testing
   - Test on 1,000 posts before processing millions
   - Verify analysis logic before scaling up

5. **Budget Alerts**: Template creates CloudWatch alarms
   - Alert at threshold (default $100/month)
   - Subscribe via SNS for email notifications

## 📓 Example Notebooks

The `notebooks/` directory contains Jupyter notebooks demonstrating key workflows:

- **`01-data-exploration.ipynb`**: Loading and exploring social media datasets
- **`02-sentiment-analysis.ipynb`**: Sentiment analysis comparison (VADER vs Comprehend)
- **`03-misinformation-detection.ipynb`**: Pattern recognition and risk scoring
- **`04-network-analysis.ipynb`**: Building and analyzing interaction networks

To run notebooks:

```bash
# Start Jupyter Lab
jupyter lab notebooks/

# Or use SageMaker Studio for production
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Reinstall package
pip uninstall social-media-analysis
pip install -e .

# Verify installation
python -c "import social_media_analysis; print(social_media_analysis.__version__)"
```

### AWS Permission Denied

```bash
# Verify role ARN
aws iam get-role --role-name social-media-analysis-analysis-role-dev

# Test S3 access
aws s3 ls s3://$DATA_BUCKET

# Test Comprehend access
aws comprehend detect-sentiment --text "test" --language-code en
```

### High Comprehend Costs

```python
# Switch to local VADER
os.environ['USE_COMPREHEND'] = 'false'

# Or process in smaller batches
df_sample = df.sample(n=1000)  # Test on 1K posts first
```

### Memory Errors

```python
# Load data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)

# Or use sampling
df = data_client.load_twitter_dataset(
    bucket=bucket,
    prefix=prefix,
    sample_size=50000  # Limit to 50K posts
)
```

### NLTK Data Missing

```bash
# Download required NLTK data
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

## 🔐 Security Best Practices

1. **Never commit credentials**: Use `.env` file (gitignored)
2. **Use IAM roles**: Leverage instance/notebook roles when possible
3. **Restrict S3 access**: Buckets have public access blocked by default
4. **Enable MFA**: Protect AWS account with multi-factor authentication
5. **Monitor access**: Review CloudTrail logs regularly
6. **Rotate credentials**: Update IAM access keys every 90 days
7. **Encrypt data**: S3 encryption enabled by default in template

## 📚 Additional Resources

### Documentation
- [AWS Comprehend Developer Guide](https://docs.aws.amazon.com/comprehend/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Gensim LDA Tutorial](https://radimrehurek.com/gensim/models/ldamodel.html)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)

### Research Papers
- "VADER: A Parsimonious Rule-based Model for Sentiment Analysis" (Hutto & Gilbert, 2014)
- "Latent Dirichlet Allocation" (Blei et al., 2003)
- "The spread of true and false news online" (Vosoughi et al., 2018)

### Datasets
- [Twitter Developer Platform](https://developer.twitter.com/)
- [Pushshift Reddit Dataset](https://files.pushshift.io/reddit/)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

## 🤝 Contributing

This project is part of Research Jumpstart. See the main repository for:
- Contribution guidelines
- Code of conduct
- Issue templates
- Project template

## 📄 License

Apache License 2.0 - See repository root for full license text.

## 🆘 Support

- **Issues**: Open GitHub issue in Research Jumpstart repository
- **Questions**: Use GitHub Discussions
- **Security**: Report vulnerabilities privately via GitHub Security Advisories

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_social_media,
  title={Social Media Analysis \& Misinformation Detection - Research Jumpstart},
  author={Research Jumpstart Contributors},
  year={2025},
  url={https://github.com/yourusername/research-jumpstart}
}
```

## 🗺️ Roadmap

- [ ] Real-time streaming analysis (Twitter Streaming API)
- [ ] Multi-language support expansion
- [ ] Advanced misinformation models (BERT-based)
- [ ] Automated fact-checking integration
- [ ] Temporal network evolution analysis
- [ ] Bot detection capabilities
- [ ] Dashboard visualization (Streamlit/Plotly Dash)

---

**Version**: 1.0.0
**Last Updated**: 2025-11-09
**Maintainer**: Research Jumpstart Team
