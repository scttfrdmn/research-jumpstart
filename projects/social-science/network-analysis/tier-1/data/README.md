# Social Network Data Storage

This directory stores downloaded and processed social network datasets. Data persists between SageMaker Studio Lab sessions!

## Directory Structure

```
data/
├── raw/                    # Original downloaded files
│   ├── twitter_graph.pkl
│   ├── reddit_graph.pkl
│   ├── facebook_graph.pkl
│   ├── mastodon_graph.pkl
│   └── discord_graph.pkl
│
└── processed/              # Processed graph representations
    ├── twitter_embeddings.pt
    ├── reddit_embeddings.pt
    ├── unified_graph.pkl
    └── temporal_snapshots/
```

## Datasets

### Twitter Network
- **Source:** Stanford SNAP / Academic API
- **Nodes:** 1M users
- **Edges:** 10M interactions (retweets, mentions, replies)
- **Period:** 2023 (6 months)
- **Size:** ~3GB
- **Features:** User metadata, tweet counts, follower stats

### Reddit Network
- **Source:** Reddit API / Pushshift
- **Nodes:** 500K subreddits + users
- **Edges:** 5M hyperlinks and comments
- **Period:** 2023 (6 months)
- **Size:** ~2GB
- **Features:** Subreddit metadata, user karma, post engagement

### Facebook Network
- **Source:** Facebook Academic Graph / Synthetic
- **Nodes:** 2M users (anonymized)
- **Edges:** 20M friendships and interactions
- **Period:** 2023 (3 months)
- **Size:** ~3GB
- **Features:** Group memberships, page likes, engagement

### Mastodon Network
- **Source:** Mastodon API
- **Nodes:** 200K users
- **Edges:** 2M follows and boosts
- **Period:** 2023 (6 months)
- **Size:** ~1GB
- **Features:** Instance metadata, post counts, federated data

### Discord Network
- **Source:** Discord Academic Program / Synthetic
- **Nodes:** 500K users across servers
- **Edges:** 8M messages and reactions
- **Period:** 2023 (3 months)
- **Size:** ~1.5GB
- **Features:** Server metadata, role info, message frequency

## Usage

Data is automatically downloaded and cached on first use:

```python
from src.data_utils import load_twitter_graph, load_reddit_graph

# First run: downloads and caches
twitter_G = load_twitter_graph()  # Downloads ~3GB

# Subsequent runs: uses cache
twitter_G = load_twitter_graph()  # Instant!

# Force re-download
twitter_G = load_twitter_graph(force_download=True)
```

## Graph Format

All graphs are stored as NetworkX DiGraph objects with:

```python
# Node attributes
{
    'user_id': str,
    'platform': str,
    'features': np.ndarray,  # Learned embeddings
    'metadata': dict,        # Platform-specific info
    'timestamp_first': datetime,
    'timestamp_last': datetime
}

# Edge attributes
{
    'weight': float,         # Interaction strength
    'edge_type': str,        # 'retweet', 'reply', 'mention', etc.
    'timestamp': datetime,
    'sentiment': float       # -1 to 1
}
```

## Storage Management

Check current usage:
```bash
du -sh data/
```

Clean old files:
```bash
rm -rf data/raw/*.backup
rm -rf data/processed/old_*.pt
```

## Persistence

✅ **Persistent:** This directory survives Studio Lab session restarts
✅ **15GB Limit:** Studio Lab provides 15GB persistent storage
✅ **Shared:** All notebooks in this project share this data directory

## Privacy & Ethics

- All datasets use publicly available data or synthetic data
- User IDs are anonymized/hashed
- No personally identifiable information (PII) stored
- Compliant with platform Terms of Service
- Follow ethical research guidelines

## Notes

- Graphs stored in pickle format for fast loading
- Embeddings stored in PyTorch tensors (.pt)
- Temporal snapshots capture network evolution
- .gitignore excludes data/ from version control
