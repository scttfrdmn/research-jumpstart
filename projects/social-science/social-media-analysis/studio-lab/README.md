# Social Media Analysis & Misinformation Detection - Studio Lab

**Start here!** Free tier version for learning social media analysis and misinformation detection patterns.

## What You'll Build

Analyze social media posts to detect patterns, sentiment, and potential misinformation:
- Sentiment analysis (positive, negative, neutral)
- Topic modeling (what are people talking about?)
- Engagement pattern analysis
- Misinformation indicators detection
- Network visualization of information spread
- Time series analysis of topics

**Time to complete**: 3-4 hours

---

## Quick Start

### Step 1: Get Studio Lab Access
1. Go to https://studiolab.sagemaker.aws
2. Request free account
3. Wait for approval email

### Step 2: Set Up Environment

```bash
# Clone repository
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd research-jumpstart/projects/social-science/social-media-analysis/studio-lab

# Create environment
conda env create -f environment.yml
conda activate social-media-analysis

# Launch notebook
jupyter notebook quickstart.ipynb
```

### Step 3: Run Analysis
Click **Run All** in the notebook!

---

## What's Included

**Sample Dataset** (`sample_data.csv`):
- 500+ synthetic social media posts
- Mix of Twitter and Reddit content
- Various topics and sentiment
- Engagement metrics (likes, shares, comments)
- Timestamps for temporal analysis

**Analysis Workflow**:
1. **Data Loading & Exploration** (5 min)
   - Load sample posts
   - Basic statistics
   - Platform distribution

2. **Text Preprocessing** (10 min)
   - Clean text (remove URLs, mentions, hashtags)
   - Tokenization
   - Stop word removal
   - Lemmatization

3. **Sentiment Analysis** (10 min)
   - VADER sentiment scoring
   - Positive/negative/neutral classification
   - Sentiment distribution visualization
   - Sentiment over time

4. **Topic Modeling** (15 min)
   - Latent Dirichlet Allocation (LDA)
   - Extract main topics
   - Visualize topic keywords
   - Assign topics to posts

5. **Engagement Analysis** (10 min)
   - Engagement rate calculation
   - Viral content identification
   - Engagement vs sentiment correlation

6. **Misinformation Patterns** (15 min)
   - ALL CAPS detection
   - Excessive punctuation (!!!)
   - Emotional language patterns
   - Vague source references
   - Call-to-action urgency
   - Conspiracy indicator keywords

7. **Network Analysis** (10 min)
   - User interaction networks
   - Information spread visualization
   - Influential users identification

8. **Temporal Analysis** (10 min)
   - Post volume over time
   - Topic trends
   - Sentiment shifts

---

## Key Analyses

### Sentiment Analysis

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner):
- **Positive**: Compound score > 0.05
- **Negative**: Compound score < -0.05
- **Neutral**: Between -0.05 and 0.05

**Example**:
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
text = "This is amazing news! So happy!"
scores = analyzer.polarity_scores(text)
# {'neg': 0.0, 'neu': 0.317, 'pos': 0.683, 'compound': 0.8633}
```

### Misinformation Indicators

**Red flags detected**:
- ALL CAPS words (> 3 in post)
- Excessive punctuation (!!!, ???)
- Vague sources ("they", "mainstream media")
- Urgency language ("BREAKING", "URGENT")
- Call to action ("SHARE", "Wake up")
- Conspiracy keywords ("truth", "cover-up", "they don't want")

**Risk scoring**: 0-10 scale based on number of indicators

### Topic Modeling

Latent Dirichlet Allocation (LDA):
- Identifies main discussion topics
- Extracts topic keywords
- Assigns topic probabilities to each post

**Example topics**:
- Topic 1: Climate, environment, science
- Topic 2: Politics, election, government
- Topic 3: Health, vaccine, medical
- Topic 4: Technology, AI, innovation

---

## Customization

### Add Your Own Data

Replace `sample_data.csv` with your data. Required columns:
```
post_id, timestamp, platform, user_id, text, retweets, likes, replies
```

### Adjust Misinformation Detection

Modify keyword lists and thresholds:
```python
# In notebook
CONSPIRACY_KEYWORDS = ['truth', 'wake up', 'they', 'cover-up']
CAPS_THRESHOLD = 3  # Number of ALL CAPS words
RISK_SCORE_HIGH = 5  # Threshold for high-risk content
```

### Change Topic Model Parameters

```python
# Number of topics to extract
n_topics = 5

# LDA parameters
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10,
    learning_method='online',
    random_state=42
)
```

---

## Understanding Results

### Engagement Rate

```
Engagement Rate = (Likes + Retweets + Replies) / Followers
```

High engagement (>5%) may indicate:
- Viral content
- Controversial topic
- Bot amplification
- Coordinated sharing

### Misinformation Risk Score

| Score | Risk Level | Action |
|-------|-----------|---------|
| 0-2 | Low | Standard content |
| 3-5 | Medium | Review manually |
| 6-8 | High | Likely misinformation |
| 9-10 | Critical | Strong indicators |

**Note**: Automated detection is not perfect. Human review required.

### Network Centrality

- **High degree**: Users with many connections
- **High betweenness**: Users bridging communities
- **High eigenvector**: Users connected to influential users

These users are key to information spread.

---

## Limitations

**Studio Lab Version**:
- Sample data (not real social media posts)
- Limited to ~1,000 posts (memory constraints)
- Basic NLP (no deep learning models)
- Simple network analysis
- No real-time streaming

**Transition to Unified Studio for**:
- Real Twitter/Reddit datasets (millions of posts)
- Amazon Comprehend (advanced NLP)
- Amazon Neptune (graph database for networks)
- Amazon Bedrock (AI content analysis)
- Real-time processing with Kinesis
- Scalable analysis (100K+ posts)

---

## Common Issues

### NLTK Data Not Found

**Error**: `Resource punkt not found`

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Memory Error

**Problem**: Kernel dies with large datasets

**Solution**:
- Reduce data size: `df = df.head(500)`
- Process in batches
- Restart kernel and clear outputs

### VADER Not Installed

**Solution**:
```bash
conda activate social-media-analysis
pip install vaderSentiment
```

---

## Next Steps

### Experiment
- Analyze different time periods
- Compare platforms (Twitter vs Reddit)
- Test different misinformation indicators
- Adjust topic model parameters

### Learn More
- Study NLP techniques
- Explore network science
- Read about misinformation research
- Try different sentiment analyzers

### Scale Up
When ready for real data:
1. Review [Unified Studio README](../unified-studio/README.md)
2. Set up AWS account
3. Access real social media datasets
4. Use advanced NLP services

---

## Resources

**NLP & Text Analysis**:
- [NLTK Documentation](https://www.nltk.org/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Gensim Topic Modeling](https://radimrehurek.com/gensim/)

**Misinformation Research**:
- [First Draft News](https://firstdraftnews.org/)
- [MIT Media Lab](https://www.media.mit.edu/)
- [Stanford Internet Observatory](https://cyber.fsi.stanford.edu/)

**Network Analysis**:
- [NetworkX Documentation](https://networkx.org/)
- [Network Science Book](http://networksciencebook.com/)

---

## Getting Help

- **Issues**: https://github.com/research-jumpstart/research-jumpstart/issues
- **Discussions**: https://github.com/research-jumpstart/research-jumpstart/discussions
- Tag: `social-science`, `social-media-analysis`

---

*Last updated: 2025-11-09 | Studio Lab Free Tier Version*
