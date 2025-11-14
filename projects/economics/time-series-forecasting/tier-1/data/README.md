# Data Directory

This directory stores economic time series data downloaded from various sources. The data is cached here to enable persistence between Studio Lab sessions.

## Directory Structure

```
data/
├── raw/                    # Raw downloaded data
│   ├── fred_*.csv         # FRED data files (by series ID)
│   ├── world_bank_*.csv   # World Bank data
│   └── oecd_*.csv         # OECD data
│
└── processed/             # Cleaned and processed data
    ├── multi_country_panel.csv      # Main panel dataset
    ├── world_bank_panel.csv         # World Bank panel
    └── aligned_indicators.parquet   # Aligned indicators (optimized format)
```

## Data Sources

### FRED (Federal Reserve Economic Data)
- Source: https://fred.stlouisfed.org/
- Coverage: US and international economic indicators
- Frequency: Daily, monthly, quarterly, annual
- Access: Free API via pandas-datareader

### World Bank
- Source: https://data.worldbank.org/
- Coverage: 200+ countries, development indicators
- Frequency: Annual, some quarterly
- Access: Free API via wbdata package

### OECD
- Source: https://data.oecd.org/
- Coverage: OECD member countries
- Frequency: Monthly, quarterly, annual
- Access: Free API

## Key Indicators

### GDP and Growth
- GDP (level, real)
- GDP growth rate (YoY, QoQ)
- GDP per capita

### Prices and Inflation
- CPI (Consumer Price Index)
- Inflation rate (YoY)
- PPI (Producer Price Index)

### Labor Market
- Unemployment rate
- Employment level
- Labor force participation rate

### Monetary Policy
- Interest rates (policy rate, 10-year bond)
- Money supply (M1, M2)
- Exchange rates

### External Sector
- Trade balance
- Current account balance
- Exports, imports

### Financial Markets
- Stock market indices
- Bond yields
- Credit spreads

## Cache Management

Data is cached automatically to avoid re-downloading. To force refresh:

```python
from src.data_utils import load_fred_data

# Force re-download
data = load_fred_data('GDP', force_download=True)
```

## Storage Optimization

- Raw CSV files: Compressed for large datasets
- Processed data: Saved as Parquet for faster loading
- Old versions: Automatically cleaned after 30 days

## Data Quality

All data undergoes:
1. Missing value detection and handling
2. Outlier detection
3. Stationarity testing
4. Frequency alignment
5. Date range validation

See `src/preprocessing.py` for details.
