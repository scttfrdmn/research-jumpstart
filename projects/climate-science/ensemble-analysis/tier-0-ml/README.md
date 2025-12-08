# Climate Temperature Forecasting with LSTM (Tier-0 ML)

**Duration:** 60-90 minutes
**Cost:** FREE
**Platform:** Google Colab or SageMaker Studio Lab

## Overview

Train LSTM (Long Short-Term Memory) neural networks to forecast global temperature anomalies using deep learning. This project demonstrates how deep learning can capture complex climate patterns and generate multi-year temperature predictions.

## What You'll Learn

- Prepare time series climate data for deep learning
- Build LSTM encoder-decoder architectures
- Train neural networks with early stopping and learning rate scheduling
- Generate 6-year temperature forecasts (2025-2030)
- Quantify forecast uncertainty using ensemble methods
- Compare LSTM performance with traditional statistical approaches

## Why LSTM for Climate?

**Advantages over traditional methods (ARIMA, linear regression):**
- Captures long-term dependencies (seasonal patterns, multi-year trends)
- Learns non-linear temperature dynamics automatically
- Handles sequential data naturally
- Better performance on complex climate patterns

## What You'll Build

- **LSTM encoder-decoder model** for multi-step forecasting
- **60-month lookback window** (5 years of history)
- **12-month forecast horizon** (1 year ahead)
- **Ensemble forecasting** (5 models) for uncertainty quantification
- **6-year projections** (2025-2030) with confidence intervals

## Dataset

**NOAA GISTEMP** global temperature anomalies:
- Monthly data from 1880-2024 (1,700+ observations)
- Temperature anomaly = difference from 1951-1980 baseline
- Source: NASA Goddard Institute for Space Studies

## Quick Start

### Option 1: Google Colab (Recommended)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/climate-science/ensemble-analysis/tier-0-ml/climate-temperature-forecasting-lstm.ipynb)

1. Click the badge above
2. Sign in with Google account
3. Click "Runtime" → "Run all"
4. Complete in 60-90 minutes

### Option 2: SageMaker Studio Lab
1. Create free account: https://studiolab.sagemaker.aws
2. Clone this repository
3. Navigate to this notebook
4. Run all cells
5. 15 GB persistent storage included

### Option 3: Local Development
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/climate-science/ensemble-analysis/tier-0-ml

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook climate-temperature-forecasting-lstm.ipynb
```

## Requirements

### Python Packages
- tensorflow >= 2.10.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0

### Compute Resources
- **RAM:** 4 GB minimum, 8 GB recommended
- **CPU:** Any modern CPU (Google Colab free tier sufficient)
- **GPU:** Optional (speeds up training by 2-3x)
- **Storage:** 500 MB

### Time
- **Setup:** 2-5 minutes
- **Data loading:** 1-2 minutes
- **Model training:** 20-40 minutes (CPU) or 5-10 minutes (GPU)
- **Forecasting:** 5-10 minutes
- **Ensemble training:** 30-45 minutes
- **Total:** 60-90 minutes

## Project Structure

```
tier-0-ml/
├── climate-temperature-forecasting-lstm.ipynb  # Main notebook
├── README.md                                    # This file
└── requirements.txt                             # Python dependencies
```

## Key Results

After completing this project, you'll have:

1. **Trained LSTM model** with MAE < 0.15°C on test data
2. **6-year forecast** (2025-2030) with monthly resolution
3. **Uncertainty estimates** via ensemble forecasting
4. **Performance comparison** showing LSTM outperforms ARIMA
5. **Publication-ready visualizations** of predictions and confidence intervals

## Model Architecture

```
Encoder-Decoder LSTM:
├── Input: 60 months of temperature history
├── Encoder LSTM Layer 1: 64 units, return sequences
├── Dropout: 0.2
├── Encoder LSTM Layer 2: 32 units
├── Dropout: 0.2
├── Repeat Vector: 12 timesteps (forecast horizon)
├── Decoder LSTM: 32 units, return sequences
├── Dropout: 0.2
└── TimeDistributed Dense: 1 output per timestep

Total parameters: ~50,000
```

## Sample Forecast Output

```
LSTM MODEL PERFORMANCE:
   • Test MAE: 0.12°C (excellent accuracy)
   • Test RMSE: 0.18°C
   • R² Score: 0.95 (strong correlation)
   • Ensemble uncertainty: ±0.08°C (95% CI)

FORECAST (2025-2030):
   • Mean projected anomaly: 1.35°C
   • Projected range: 1.15°C to 1.55°C
   • Trend: +0.25°C over 6 years
   • Warming rate: 0.42°C/decade
```

## Common Questions

**Q: How accurate are the forecasts?**
A: LSTM achieves MAE < 0.15°C on test data. For multi-year forecasts, uncertainty increases with horizon (±0.2-0.3°C at 6 years).

**Q: Why LSTM instead of ARIMA?**
A: LSTM captures non-linear patterns and long-term dependencies better than ARIMA. Typical improvement: 20-30% lower forecast error.

**Q: Can I use this for policy decisions?**
A: No. This is an educational project. Operational climate forecasts require ensemble physics-based models (CMIP6), not single LSTM models.

**Q: How long does training take?**
A: 20-40 minutes on CPU (Colab free tier), 5-10 minutes on GPU.

**Q: Do I need a GPU?**
A: No, but it speeds up training by 2-3x. Colab free tier provides T4 GPU.

## Troubleshooting

**Issue: TensorFlow import error**
```bash
pip install --upgrade tensorflow
```

**Issue: Out of memory during training**
- Reduce batch size from 32 to 16
- Reduce LSTM units from 64 to 32

**Issue: Training too slow on CPU**
- Enable GPU in Colab: Runtime → Change runtime type → GPU
- Reduce epochs from 100 to 50

**Issue: Forecasts look unrealistic**
- Check data scaling (should be [0, 1])
- Verify sequence creation (lookback=60, horizon=12)
- Ensure chronological train/val/test split

## Next Steps

### Tier 1: Multi-Variable Climate Forecasting (4-8 hours, FREE)
- Add precipitation, sea level, CO2 as input features
- Multi-variate LSTM with attention mechanisms
- Regional climate forecasts using CMIP6 data
- Persistent storage for large models (SageMaker Studio Lab)

[View Tier 1 →](../tier-1/README.md)

### Tier 2: Production Climate ML Platform (2-3 days, $400-800/month)
- 100GB+ CMIP6 ensemble data on S3 (20+ climate models)
- Distributed training with SageMaker (multi-GPU)
- Real-time forecasting API with Lambda
- Automated retraining with new climate data
- CloudFormation one-click deployment

[View Tier 2 →](../tier-2/README.md)

### Tier 3: Enterprise Climate Intelligence (Ongoing, $3K-12K/month)
- Global climate modeling at 1km resolution
- Multi-model ensemble forecasting (LSTM + Transformers + Physics-based)
- Climate impact scenarios for agriculture, infrastructure, health
- AI-assisted interpretation (Amazon Bedrock)
- Integration with Earth observation data

[View Tier 3 →](../tier-3/README.md)

## Related Projects

- [Climate Ensemble Analysis (Statistical)](../tier-0/) - Traditional statistical climate analysis
- [Economics - Time Series Forecasting](../../../economics/time-series-forecasting/) - LSTM for economic data
- [Urban Planning - Traffic Prediction](../../../urban-planning/transportation-optimization/) - GCN+LSTM for traffic

## Resources

- **LSTM Paper:** [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf)
- **Climate Data:** [NOAA GISTEMP](https://data.giss.nasa.gov/gistemp/)
- **TensorFlow Tutorials:** [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- **Deep Learning for Climate:** [Reichstein et al. (2019) Nature](https://www.nature.com/articles/s41586-019-0912-1)

## Citation

If you use this project, please cite:

```bibtex
@software{research_jumpstart_climate_lstm,
  title = {Climate Temperature Forecasting with LSTM},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the data source:
```bibtex
@misc{gistemp2024,
  title = {GISS Surface Temperature Analysis (GISTEMP), version 4},
  author = {{NASA Goddard Institute for Space Studies}},
  year = {2024},
  url = {https://data.giss.nasa.gov/gistemp/}
}
```

## License

Apache 2.0 - See [LICENSE](../../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/YOUR_USERNAME/research-jumpstart) - Pre-built research workflows for cloud computing*
