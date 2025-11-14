# Land Cover Classification from Satellite Imagery

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB Landsat/Sentinel imagery

## Research Goal

Train a deep learning model to classify land cover types (forest, agriculture, urban, water, bare soil) from satellite imagery. Learn the fundamentals of remote sensing analysis and cloud-based geospatial computing.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/environmental/ecosystem-monitoring/tier-0/land-cover-classification.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/environmental/ecosystem-monitoring/tier-0/land-cover-classification.ipynb)

## What You'll Build

1. **Download satellite imagery** (~1.5GB Landsat 8 scene, takes 15-20 min)
2. **Preprocess geospatial data** (band stacking, cloud masking, normalization)
3. **Extract training samples** (labeled pixels for supervised learning)
4. **Train CNN classifier** (60-75 minutes on GPU)
5. **Generate classification map** (predict land cover for entire scene)
6. **Assess accuracy** (confusion matrix, per-class metrics)

## Dataset

**Landsat 8 Multispectral Imagery**
- Scene: Path 37, Row 37 (Arizona, USA)
- Date: Summer 2023 (cloud-free conditions)
- Bands: 7 multispectral + 1 panchromatic
- Resolution: 30m (multispectral)
- Size: ~1.5GB GeoTIFF files
- Source: USGS EarthExplorer / AWS Open Data

**Land Cover Classes:**
1. Forest (coniferous, deciduous, mixed)
2. Agriculture (crops, pasture)
3. Urban/Built-up
4. Water (lakes, rivers, reservoirs)
5. Bare soil/Rock

**Training Data:**
- 500 labeled pixels per class
- Stratified random sampling
- Validation split: 80/20

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`land-cover-classification.ipynb`)
- Satellite imagery download utilities
- CNN architecture for multispectral classification
- Training and validation pipeline
- Accuracy assessment tools
- Visualization functions

## Key Methods

- **Supervised classification:** Machine learning from labeled training data
- **Convolutional Neural Networks:** Spatial feature extraction
- **Multispectral indices:** NDVI, NDWI, NDBI for feature engineering
- **Cloud masking:** Quality assessment band filtering
- **Accuracy assessment:** Confusion matrix, F1-scores, kappa coefficient

## Technical Details

### CNN Architecture

```python
Input: 7 bands (Blue, Green, Red, NIR, SWIR1, SWIR2, TIR)
       + 3 indices (NDVI, NDWI, NDBI)
       = 10 input channels

Conv2D(32, 3x3) → ReLU → BatchNorm
Conv2D(64, 3x3) → ReLU → BatchNorm → MaxPool
Conv2D(128, 3x3) → ReLU → BatchNorm → MaxPool
Flatten → Dense(256) → Dropout(0.5)
Dense(5, softmax)

Output: 5 land cover classes
```

### Preprocessing Pipeline

```python
1. Load GeoTIFF bands
2. Stack into multispectral array
3. Apply cloud mask (QA band)
4. Calculate spectral indices
5. Normalize to [0, 1]
6. Extract training patches
7. Data augmentation (rotation, flip)
```

### Training Configuration

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical cross-entropy
- **Batch size:** 32
- **Epochs:** 50 (with early stopping)
- **Augmentation:** Random rotation, flip, brightness
- **Hardware:** GPU recommended (60-75 min), CPU possible (4-5 hours)

## Expected Results

**Classification Accuracy:**
- Overall accuracy: 88-92%
- Kappa coefficient: 0.85-0.90

**Per-Class F1-Scores:**
- Forest: 0.90-0.95 (high spectral separability)
- Agriculture: 0.85-0.90 (seasonal variability)
- Urban: 0.88-0.92 (distinct spectral signature)
- Water: 0.95-0.98 (excellent NDWI discrimination)
- Bare soil: 0.80-0.85 (confusion with urban/agriculture)

**Common Challenges:**
- Agriculture/bare soil confusion (similar spectral properties)
- Urban/bare soil confusion (impervious surfaces)
- Shadow effects near mountains
- Mixed pixels at class boundaries

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-sensor environmental monitoring ensemble](../tier-1/) on Studio Lab
  - Cache 10GB multi-sensor data (optical + radar + LiDAR)
  - Train ensemble models (5-6 hours continuous)
  - Temporal change detection requiring persistence
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ satellite archives on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs with hyperparameter tuning

- **Tier 3:** [Production-scale monitoring](../tier-3/) with full CloudFormation
  - Continental-scale land cover mapping
  - Multi-temporal analysis (5+ years)
  - Automated change detection alerts

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- rasterio, GDAL, geopandas
- scikit-learn, scipy
- matplotlib, cartopy

**Note:** First run downloads 1.5GB of data (15-20 minutes)

## Troubleshooting

### Download Issues

**Problem:** "Slow download speeds"
```
Solution: Use AWS S3 data source instead of USGS:
imagery_url = 's3://landsat-pds/c1/L8/037/037/...'
# Faster access, no authentication required
```

**Problem:** "Download fails partway"
```
Solution: Use resume-capable download:
import requests
response = requests.get(url, stream=True)
# Implement chunked download with retry logic
```

### Memory Issues

**Problem:** "Kernel crashes during training"
```
Cause: 10GB imagery + 4GB model = exceeds Colab RAM

Solution:
1. Reduce batch size: batch_size = 16 (instead of 32)
2. Process imagery in tiles (512x512 patches)
3. Use mixed precision training: tf.keras.mixed_precision
```

### Training Issues

**Problem:** "Training takes too long"
```
Solution:
1. Use GPU runtime: Runtime → Change runtime type → GPU
2. Reduce epochs: epochs = 30 (instead of 50)
3. Use transfer learning: Start from pre-trained ResNet
```

**Problem:** "Model overfitting"
```
Symptoms: Training accuracy 95%+, validation accuracy 80%

Solution:
1. Increase dropout: Dropout(0.6)
2. Add more training samples (500 → 1000 per class)
3. Stronger data augmentation
4. Reduce model complexity (fewer Conv layers)
```

### Accuracy Issues

**Problem:** "Low accuracy for specific classes"
```
Common issues:
- Agriculture/bare soil confusion:
  → Add temporal information (multi-date imagery)
  → Include texture features

- Urban/bare soil confusion:
  → Add thermal band (surface temperature)
  → Include object-based features (shape, size)

- Shadow effects:
  → Apply topographic correction
  → Use shadow removal preprocessing
```

## Extension Ideas

### Beginner (2-4 hours)

1. **Additional indices**
   - EVI (Enhanced Vegetation Index)
   - SAVI (Soil-Adjusted Vegetation Index)
   - UI (Urban Index)

2. **Different regions**
   - Tropical rainforest (high biomass)
   - Arid desert (sparse vegetation)
   - Coastal zones (water boundaries)

3. **Seasonal comparison**
   - Winter vs summer imagery
   - Phenological patterns
   - Seasonal crop classification

### Intermediate (4-8 hours)

4. **Multi-temporal analysis**
   - Classify 4 seasons
   - Temporal feature stacking
   - Change detection between dates

5. **Transfer learning**
   - Fine-tune ImageNet pre-trained models
   - Domain adaptation techniques
   - Few-shot learning for rare classes

6. **Advanced architectures**
   - U-Net for semantic segmentation
   - ResNet for deeper feature extraction
   - Attention mechanisms

### Advanced (8+ hours)

7. **Multi-sensor fusion**
   - Combine Landsat + Sentinel-2 (higher resolution)
   - Add SAR data (all-weather capability)
   - LiDAR for 3D structure

8. **Active learning**
   - Iterative labeling of uncertain pixels
   - Semi-supervised learning
   - Pseudo-labeling strategies

9. **Uncertainty quantification**
   - Bayesian neural networks
   - Monte Carlo dropout
   - Confidence-aware predictions

## Resources

### Satellite Data Sources

- **Landsat on AWS**: https://registry.opendata.aws/landsat-8/
- **USGS EarthExplorer**: https://earthexplorer.usgs.gov/
- **Copernicus Open Access Hub**: https://scihub.copernicus.eu/
- **Google Earth Engine**: https://earthengine.google.com/

### Remote Sensing Tutorials

- **rasterio documentation**: https://rasterio.readthedocs.io/
- **GDAL tutorials**: https://gdal.org/tutorials/
- **Awesome Satellite Imagery**: https://github.com/chrieke/awesome-satellite-imagery-datasets

### Machine Learning for Remote Sensing

- **Deep Learning for Remote Sensing**: https://github.com/robmarkcole/satellite-image-deep-learning
- **TorchGeo**: https://github.com/microsoft/torchgeo
- **EuroSAT dataset**: https://github.com/phelber/EuroSAT

### Papers

- **Deep learning for land cover**: Zhang et al. (2018), "Deep learning for remote sensing data"
- **CNNs for satellite imagery**: Kussul et al. (2017), "Deep learning classification of land cover"
- **Transfer learning**: Xie et al. (2016), "Transfer learning from deep features for remote sensing"

## Dataset Citations

```bibtex
@misc{usgs_landsat,
  title = {Landsat 8 OLI/TIRS Collection 2 Level-1},
  author = {{U.S. Geological Survey}},
  year = {2023},
  url = {https://www.usgs.gov/landsat-missions/landsat-8}
}
```

## Support

- **GitHub Issues**: Tag with `environmental` and `tier-0`
- **GIS Stack Exchange**: https://gis.stackexchange.com/
- **Remote Sensing Stack Exchange**: https://gis.stackexchange.com/questions/tagged/remote-sensing

---

Built for Research Jumpstart with [Claude Code](https://claude.com/claude-code)
