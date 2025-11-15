# Sample Economic Data

This directory contains sample economic time series data for testing the forecasting pipeline.

## Data Sources

### Option 1: Use Provided Sample Data

Run the upload script with `--create-sample` flag:

```bash
python scripts/upload_to_s3.py --bucket economic-data-{your-id} --create-sample
```

This creates:
- `gdp/usa_gdp_quarterly.csv` - Quarterly GDP data (2018-2023)
- `unemployment/usa_unemployment_monthly.csv` - Monthly unemployment rate (2018-2023)
- `inflation/usa_cpi_monthly.csv` - Monthly CPI data (2018-2023)

### Option 2: Download from FRED

Get real economic data from Federal Reserve Economic Data (FRED):

```bash
# Get API key from: https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY=your_api_key_here

# Download indicators
python scripts/upload_to_s3.py \
  --bucket economic-data-{your-id} \
  --download-fred GDP,UNRATE,CPIAUCSL,FEDFUNDS
```

Common FRED series codes:
- `GDP` - Gross Domestic Product
- `UNRATE` - Unemployment Rate
- `CPIAUCSL` - Consumer Price Index (All Urban Consumers)
- `FEDFUNDS` - Federal Funds Rate
- `INDPRO` - Industrial Production Index
- `RSXFS` - Retail Sales
- `HOUST` - Housing Starts
- `NETEXP` - Net Exports

### Option 3: Use Your Own Data

Place CSV files in this directory with the following format:

```csv
date,value
2018-01-01,20000.0
2018-04-01,20250.5
2018-07-01,20500.2
...
```

**Requirements:**
- Two columns: `date` and `value`
- Date format: YYYY-MM-DD
- Sorted by date (oldest to newest)
- At least 10 data points
- Consistent frequency (monthly or quarterly)

**Directory structure:**
```
sample_data/
├── gdp/
│   ├── usa_gdp_quarterly.csv
│   ├── chn_gdp_quarterly.csv
│   └── deu_gdp_quarterly.csv
├── unemployment/
│   ├── usa_unemployment_monthly.csv
│   └── ...
└── inflation/
    ├── usa_cpi_monthly.csv
    └── ...
```

## Data Guidelines

### Time Series Length
- **Minimum**: 10 observations
- **Recommended**: 20-50 observations
- **Maximum**: 1000 observations (Lambda timeout constraints)

### Frequency
- **Quarterly**: GDP, National Accounts (4 per year)
- **Monthly**: Unemployment, Inflation, Industrial Production (12 per year)

### Missing Values
- Remove rows with missing values
- Or interpolate using pandas: `df.interpolate(method='linear')`

### Outliers
- Keep outliers for economic data (they're often real events)
- Lambda will handle robust forecasting

## Example Data

### GDP (Quarterly)
```csv
date,value
2018-01-01,19900.5
2018-04-01,20180.3
2018-07-01,20420.8
2018-10-01,20650.2
2019-01-01,20901.5
```

### Unemployment Rate (Monthly)
```csv
date,value
2018-01-01,4.1
2018-02-01,4.0
2018-03-01,4.0
2018-04-01,3.9
2018-05-01,3.8
```

### CPI (Monthly)
```csv
date,value
2018-01-01,247.9
2018-02-01,248.9
2018-03-01,249.6
2018-04-01,250.5
2018-05-01,251.6
```

## Data Preparation Tips

### From Excel/Sheets
```python
import pandas as pd

# Read from Excel
df = pd.read_excel('economic_data.xlsx', sheet_name='GDP')

# Select columns
df = df[['Date', 'GDP Value']]

# Rename columns
df.columns = ['date', 'value']

# Convert date format
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date')

# Save as CSV
df.to_csv('gdp_data.csv', index=False)
```

### From API
```python
import pandas as pd
import requests

# Example: World Bank API
url = "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD"
params = {"format": "json", "date": "2018:2023"}

response = requests.get(url, params=params)
data = response.json()[1]

df = pd.DataFrame([
    {
        'date': pd.to_datetime(d['date'] + '-01-01'),
        'value': d['value']
    }
    for d in data if d['value'] is not None
])

df.to_csv('worldbank_gdp.csv', index=False)
```

## Upload to S3

After preparing data:

```bash
# Upload all CSV files in this directory
python scripts/upload_to_s3.py \
  --bucket economic-data-{your-id} \
  --data-dir sample_data/
```

This will:
1. Validate each CSV file
2. Upload to S3 with proper folder structure
3. Trigger Lambda forecasting automatically

## Troubleshooting

**"Missing required columns"**
- Ensure CSV has exactly two columns: `date` and `value`

**"Invalid date format"**
- Use ISO format: YYYY-MM-DD
- Convert with: `pd.to_datetime(df['date'])`

**"Insufficient data points"**
- Need at least 10 observations for forecasting
- Add more historical data

**"Lambda timeout"**
- Large files (>1000 rows) may timeout
- Split into smaller files or increase Lambda timeout

## Resources

- FRED API: https://fred.stlouisfed.org/docs/api/fred/
- World Bank Data: https://data.worldbank.org/
- OECD Data: https://stats.oecd.org/
- IMF Data: https://www.imf.org/en/Data

---

For questions, see main README.md or open a GitHub issue.
