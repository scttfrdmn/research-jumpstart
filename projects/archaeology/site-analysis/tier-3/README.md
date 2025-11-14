# Archaeological Site Analysis at Scale - AWS Research Jumpstart

**Tier 1 Flagship Project**

Analyze archaeological sites, artifacts, and landscapes using remote sensing, computer vision, GIS, and machine learning on AWS. Discover hidden sites with LiDAR, classify artifacts with deep learning, model settlement patterns, reconstruct chronologies, and create 3D models from photogrammetry across millions of hectares.

## Overview

This flagship project demonstrates how to build archaeological analysis systems at scale using AWS services. We'll work with LiDAR point clouds, satellite imagery, artifact photographs, excavation data, and radiocarbon dates to discover sites, classify artifacts, analyze spatial patterns, build chronological models, and reconstruct ancient landscapes.

### Key Features

- **Remote Sensing:** LiDAR point cloud processing, satellite imagery analysis for site detection
- **Artifact Classification:** CNN models for pottery, lithics, architecture classification
- **Spatial Analysis:** GIS-based settlement pattern analysis, visibility analysis, least-cost paths
- **Chronology:** Bayesian radiocarbon calibration, seriation, temporal modeling
- **3D Reconstruction:** Photogrammetry from drone/ground photos, Structure from Motion (SfM)
- **Network Analysis:** Trade route modeling, material sourcing, cultural diffusion
- **Database Integration:** Open Context, tDAR, DINAA integration

### Scientific Applications

1. **Site Discovery:** Detect archaeological features hidden by vegetation using LiDAR
2. **Artifact Analysis:** Automated classification and morphometric analysis of artifacts
3. **Settlement Patterns:** Analyze spatial distribution, clustering, and environmental correlates
4. **Chronological Modeling:** Build temporal frameworks with Bayesian statistics
5. **Landscape Archaeology:** Understand ancient movement, visibility, resource access
6. **Cultural Heritage:** Document and preserve sites with 3D models and databases

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           Archaeological Analysis Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Airborne     │      │ Satellite    │      │ Drone        │
│ LiDAR        │─────▶│ Imagery      │─────▶│ Photos       │
│ (ALS, UAV)   │      │ (Sentinel-2) │      │ (Structure)  │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   S3 Data Lake    │
                    │  (LiDAR, images   │
                    │   artifacts, GIS) │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ PDAL Pipeline │   │ SageMaker         │   │ AWS Batch  │
│ (LiDAR)       │   │ (CNN Models)      │   │ (Process)  │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  PostGIS (RDS)    │
                    │  (Spatial DB)     │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Site         │   │ Artifact          │   │ Settlement    │
│ Discovery    │   │ Classification    │   │ Analysis      │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ QuickSight/Bedrock│
                    │ Archaeological    │
                    │ Dashboards & Maps │
                    └───────────────────┘
```

## Major Data Sources

### 1. LiDAR Data (Free & Commercial)

**OpenTopography:**
- **Coverage:** US, parts of Europe, select global sites
- **Resolution:** 1-8 points/m² (airborne), 10-100 points/m² (terrestrial)
- **Access:** Free academic/research access
- **URL:** https://opentopography.org/
- **Applications:** Site detection, terrain analysis, feature extraction

**USGS 3DEP:**
- **Coverage:** United States
- **Resolution:** Variable (0.7-8 points/m²)
- **Format:** LAZ (compressed LAS)
- **Access:** Free via The National Map
- **S3:** Available through AWS Open Data

**Environment Agency (UK):**
- **Coverage:** England
- **Resolution:** 0.25-2m DTM
- **Access:** Free via data.gov.uk
- **Applications:** Landscape archaeology, earthwork detection

**Commercial LiDAR:**
- **Vendors:** Fugro, Quantum Spatial, Woolpert
- **Resolution:** 10-30 points/m² (standard), 100+ points/m² (UAV)
- **Cost:** $100-500/km² for new acquisitions

### 2. Satellite Imagery (Free)

**Sentinel-2 (ESA):**
- **Resolution:** 10m (RGB, NIR), 20m (red edge, SWIR)
- **Revisit:** 5 days
- **Bands:** 13 multispectral bands
- **S3:** `s3://sentinel-cogs/`
- **Applications:** Crop mark detection, site monitoring, landscape context

**Landsat 8/9 (USGS):**
- **Resolution:** 30m multispectral, 15m panchromatic
- **History:** 50+ years (Landsat program)
- **S3:** `s3://usgs-landsat/`
- **Applications:** Long-term monitoring, historical comparison

**CORONA Declassified:**
- **Period:** 1960s-1970s
- **Resolution:** 2-3m
- **Applications:** Historical landscape analysis, destroyed sites

### 3. Archaeological Databases

**Open Context:**
- **Records:** 1M+ artifacts, features, contexts
- **Coverage:** Global
- **API:** RESTful JSON API
- **Access:** Free, open license
- **URL:** https://opencontext.org/

**tDAR (Digital Archaeological Record):**
- **Focus:** North America, especially Southwest US
- **Content:** Reports, datasets, GIS files, images
- **API:** Available for partners
- **URL:** https://www.tdar.org/

**DINAA (Digital Index of North American Archaeology):**
- **Records:** 500K+ sites (US state databases)
- **Coverage:** Eastern United States
- **Access:** Research access with agreements
- **URL:** http://ux.opencontext.org/archaeology-site-data/

### 4. Radiocarbon Databases

**IntChron:**
- **Dates:** 100K+ radiocarbon dates
- **Coverage:** Global
- **Access:** Free
- **URL:** https://intchron.org/

**CARD (Canadian Archaeological Radiocarbon Database):**
- **Dates:** 40K+ Canadian dates
- **Access:** Free
- **URL:** https://www.canadianarchaeology.ca/

**EUROEVOL:**
- **Coverage:** Neolithic Europe
- **Dates:** 14K+ dates
- **Access:** Open dataset

### 5. Heritage Agency Data

**Historic England:**
- **Records:** 400K+ sites (England)
- **Data:** Locations, descriptions, designations
- **Access:** Open via API

**USGS Archaeological Sites Database:**
- **Coverage:** US federal lands
- **Access:** Restricted (site protection)

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# Python dependencies
pip install -r requirements.txt

# Key libraries:
# - pdal: LiDAR processing
# - rasterio: Raster data
# - geopandas: Spatial analysis
# - opencv-cv2: Computer vision
# - torch/tensorflow: Deep learning
# - laspy: LAS/LAZ file I/O
# - pyproj: Coordinate transformations
# - matplotlib, seaborn: Visualization
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name archaeology-analysis-stack \
  --template-body file://cloudformation/archaeology-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name archaeology-analysis-stack

# Get outputs
aws cloudformation describe-stacks \
  --stack-name archaeology-analysis-stack \
  --query 'Stacks[0].Outputs'
```

### Access LiDAR Data

```python
from src.lidar_processor import LidarProcessor
import pdal

# Initialize processor
processor = LidarProcessor()

# Download LiDAR from OpenTopography
site_bbox = [-118.85, 34.02, -118.80, 34.06]  # Los Angeles area

lidar_data = processor.download_opentopography(
    bbox=site_bbox,
    dataset_id='CA_LosAngeles_2016',
    output_format='laz'
)

# Load into memory
pipeline = processor.load_laz(
    's3://archaeology-data/lidar/site_001.laz'
)

print(f"Loaded {pipeline.arrays[0].shape[0]:,} points")
print(f"Bounds: {pipeline.metadata['readers.las']['bounds']}")
```

## Core Analyses

### 1. LiDAR-Based Site Discovery

Process airborne LiDAR to detect archaeological features.

```python
from src.lidar_archaeology import SiteDetector
import numpy as np
import pdal
import rasterio
from rasterio.transform import from_bounds
import cv2

detector = SiteDetector()

# Load LiDAR point cloud
lidar_file = 's3://archaeology-data/lidar/maya_forest_region.laz'

# PDAL pipeline for ground classification
pipeline_json = {
    "pipeline": [
        {
            "type": "readers.las",
            "filename": lidar_file
        },
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 12,
            "multiplier": 2.0
        },
        {
            "type": "filters.smrf",  # Simple Morphological Filter (ground points)
            "ignore": "Classification[7:7]",
            "slope": 0.15,
            "window": 18,
            "threshold": 0.5,
            "scalar": 1.25
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"  # Ground points only
        }
    ]
}

# Execute pipeline
pipeline = pdal.Pipeline(json.dumps(pipeline_json))
pipeline.execute()
ground_points = pipeline.arrays[0]

print(f"Classified {len(ground_points):,} ground points")

# Create Digital Terrain Model (DTM)
dtm = detector.create_dtm(
    ground_points,
    resolution=0.5,  # 0.5m pixels
    method='idw',  # Inverse distance weighting
    radius=2.0
)

# Save DTM to S3
with rasterio.open(
    's3://archaeology-results/dtm_site_001.tif',
    'w',
    driver='GTiff',
    height=dtm.shape[0],
    width=dtm.shape[1],
    count=1,
    dtype=dtm.dtype,
    crs='EPSG:32615',  # UTM Zone 15N
    transform=from_bounds(*site_bbox, dtm.shape[1], dtm.shape[0])
) as dst:
    dst.write(dtm, 1)

# Generate hillshade for visualization
hillshade = detector.calculate_hillshade(
    dtm,
    azimuth=315,  # NW light source
    altitude=45,
    z_factor=1.0
)

# Multi-directional hillshade (reveals features from all angles)
multidirectional_hillshade = detector.multidirectional_hillshade(
    dtm,
    azimuths=[0, 45, 90, 135, 180, 225, 270, 315],
    altitude=45
)

# Calculate slope (archaeological features often visible in slope)
slope = detector.calculate_slope(dtm, units='degrees')

# Local Relief Model (LRM) - removes large-scale topography
lrm = detector.local_relief_model(
    dtm,
    window_size=20  # meters
)

# Sky View Factor (SVF) - excellent for archaeological features
svf = detector.sky_view_factor(
    dtm,
    n_directions=16,
    max_distance=10
)

# Detect potential archaeological features
# Look for: circular mounds, rectangular structures, linear features (walls, roads)

# Mound detection (positive relief features)
mounds = detector.detect_mounds(
    dtm,
    min_height=0.5,  # meters above surrounding
    min_diameter=5,  # meters
    max_diameter=50,
    circularity_threshold=0.6
)

print(f"Detected {len(mounds)} potential mounds:")
for i, mound in enumerate(mounds[:5], 1):
    print(f"  {i}. Height: {mound['height']:.2f}m, "
          f"Diameter: {mound['diameter']:.1f}m, "
          f"Location: {mound['centroid']}")

# Structure detection (rectangular features)
structures = detector.detect_rectangular_features(
    lrm,
    min_area=25,  # m²
    max_area=500,
    aspect_ratio_range=(0.5, 2.5),
    edge_detection_method='canny'
)

print(f"\nDetected {len(structures)} potential structures")

# Linear feature detection (walls, roads, terraces)
linear_features = detector.detect_linear_features(
    lrm,
    method='hough_transform',
    min_length=10,  # meters
    max_gap=2,
    orientation_bins=8  # Detect preferred orientations
)

print(f"\nDetected {len(linear_features)} linear features")
if len(linear_features) > 0:
    orientations = [f['orientation'] for f in linear_features]
    print(f"Preferred orientations: {np.histogram(orientations, bins=8)[0]}")

# Terrace detection (for agricultural features)
terraces = detector.detect_terraces(
    dtm,
    slope_map=slope,
    min_slope=5,  # degrees
    max_slope=30,
    min_length=20,
    vertical_spacing_range=(1, 5)  # meters between terraces
)

# Create composite detection map
detection_map = detector.create_detection_map(
    mounds=mounds,
    structures=structures,
    linear_features=linear_features,
    terraces=terraces,
    confidence_threshold=0.7
)

# Visualize results
detector.visualize_detections(
    dtm=dtm,
    hillshade=multidirectional_hillshade,
    detections=detection_map,
    output_path='s3://archaeology-results/site_detections.png'
)

# Export to GIS format for field verification
detector.export_to_shapefile(
    detections=detection_map,
    output_path='s3://archaeology-results/site_detections.shp',
    crs='EPSG:32615'
)

# Calculate detection statistics
stats = detector.calculate_detection_stats(
    detection_map,
    area_ha=site_bbox_area
)

print(f"\nDetection Statistics:")
print(f"  Total features detected: {stats['total_features']}")
print(f"  Feature density: {stats['density_per_ha']:.2f} features/ha")
print(f"  Mounds: {stats['mounds']}")
print(f"  Structures: {stats['structures']}")
print(f"  Linear features: {stats['linear_features']}")
print(f"  Terraces: {stats['terraces']}")
```

### 2. Artifact Classification with Deep Learning

Train CNN models to classify pottery, lithics, and architecture.

```python
from src.artifact_classification import ArtifactClassifier
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sagemaker
from sagemaker.pytorch import PyTorch

classifier = ArtifactClassifier()

# Prepare training dataset
# Assumes artifact images in S3 with directory structure:
# s3://archaeology-data/artifacts/
#   pottery/
#     plainware/
#     decorated/
#     amphorae/
#   lithics/
#     projectile_points/
#     scrapers/
#     cores/
#   architecture/
#     columns/
#     capitals/
#     friezes/

training_data = classifier.prepare_dataset(
    s3_path='s3://archaeology-data/artifacts/',
    artifact_types=['pottery', 'lithics', 'architecture'],
    min_samples_per_class=100,
    test_split=0.2,
    val_split=0.1
)

print(f"Training samples: {len(training_data['train'])}")
print(f"Validation samples: {len(training_data['val'])}")
print(f"Test samples: {len(training_data['test'])}")
print(f"\nClass distribution:")
for class_name, count in training_data['class_counts'].items():
    print(f"  {class_name}: {count}")

# Data augmentation (critical for archaeological images)
augmentation = transforms.Compose([
    transforms.RandomRotation(180),  # Artifacts can be in any orientation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Train classifier using SageMaker
estimator = PyTorch(
    entry_point='train_artifact_classifier.py',
    source_dir='src/training',
    role=sagemaker.get_execution_role(),
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'architecture': 'resnet50',  # or 'efficientnet_b3', 'vit_base'
        'pretrained': True,  # Transfer learning
        'num_classes': len(training_data['classes']),
        'early_stopping_patience': 10,
        'class_weights': 'balanced'  # Handle class imbalance
    }
)

estimator.fit({
    'train': 's3://archaeology-data/artifacts/train/',
    'val': 's3://archaeology-data/artifacts/val/'
})

# Evaluate on test set
test_results = classifier.evaluate(
    model_data=estimator.model_data,
    test_data='s3://archaeology-data/artifacts/test/'
)

print(f"\nTest Results:")
print(f"  Overall Accuracy: {test_results['accuracy']:.3f}")
print(f"  Top-5 Accuracy: {test_results['top5_accuracy']:.3f}")
print(f"  F1 Score (weighted): {test_results['f1_weighted']:.3f}")
print(f"  Cohen's Kappa: {test_results['kappa']:.3f}")

print(f"\nPer-class metrics:")
for class_name in training_data['classes']:
    metrics = test_results['per_class'][class_name]
    print(f"  {class_name}:")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall: {metrics['recall']:.3f}")
    print(f"    F1: {metrics['f1']:.3f}")

# Confusion matrix
classifier.plot_confusion_matrix(
    test_results['confusion_matrix'],
    class_names=training_data['classes'],
    output_path='s3://archaeology-results/confusion_matrix.png'
)

# Deploy model for inference
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    endpoint_name='artifact-classifier-prod'
)

# Classify new artifacts
new_artifact_image = 's3://archaeology-data/new-finds/artifact_12345.jpg'

prediction = classifier.predict(
    predictor,
    image_path=new_artifact_image,
    return_probabilities=True,
    top_k=5
)

print(f"\nClassification for {new_artifact_image}:")
for i, (class_name, prob) in enumerate(prediction['top_k'], 1):
    print(f"  {i}. {class_name}: {prob:.3f}")

# Uncertainty quantification (important for borderline cases)
uncertainty = classifier.estimate_uncertainty(
    predictor,
    image_path=new_artifact_image,
    method='monte_carlo_dropout',
    n_samples=30
)

print(f"\nUncertainty: {uncertainty['entropy']:.3f}")
print(f"Confidence: {uncertainty['confidence']:.3f}")
if uncertainty['entropy'] > 1.5:
    print("⚠️  High uncertainty - recommend expert review")

# Batch classification for excavation finds
excavation_images = 's3://archaeology-data/excavation-2024/artifacts/'

batch_predictions = classifier.batch_classify(
    predictor,
    image_folder=excavation_images,
    batch_size=32,
    output_csv='s3://archaeology-results/excavation-2024-classifications.csv'
)

print(f"\nBatch classified {len(batch_predictions)} artifacts")

# Generate classification report for excavation
report = classifier.generate_excavation_report(
    batch_predictions,
    excavation_id='EXC-2024-001',
    include_images=True,
    output_path='s3://archaeology-results/excavation-report.pdf'
)

# Morphometric analysis
# Extract quantitative features from artifact images
morphometrics = classifier.extract_morphometrics(
    artifact_images=batch_predictions['images'],
    features=[
        'area',
        'perimeter',
        'length',
        'width',
        'aspect_ratio',
        'circularity',
        'solidity',
        'rim_diameter',  # For pottery
        'thickness',
        'symmetry'
    ]
)

# Cluster artifacts by morphology
clusters = classifier.cluster_by_morphology(
    morphometrics,
    method='kmeans',
    n_clusters=5
)

classifier.visualize_morphometric_clusters(
    morphometrics,
    clusters,
    output_path='s3://archaeology-results/morphometric-clusters.png'
)
```

### 3. Settlement Pattern Analysis

Analyze spatial distribution of archaeological sites.

```python
from src.spatial_analysis import SettlementAnalyzer
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import kstest
import matplotlib.pyplot as plt

analyzer = SettlementAnalyzer()

# Load site data
# Can come from archaeological databases or field surveys
sites = analyzer.load_sites(
    source='open_context',  # or 'tdar', 'dinaa', 'csv', 'shapefile'
    region='mediterranean',
    time_period='roman',  # Filter by period
    site_types=['settlement', 'villa', 'fort']
)

print(f"Loaded {len(sites)} sites")
print(f"Time range: {sites['date_min'].min()} to {sites['date_max'].max()}")

# Spatial point pattern analysis
# Test whether sites are randomly distributed, clustered, or dispersed

# Ripley's K function
k_function = analyzer.ripleys_k(
    sites,
    radii=np.arange(1000, 50000, 1000),  # meters
    edge_correction='border',
    n_simulations=99  # For Monte Carlo envelope
)

# Plot K function with confidence envelope
analyzer.plot_k_function(
    k_function,
    output_path='s3://archaeology-results/ripleys_k.png'
)

if k_function['clustered']:
    print("Sites show significant clustering")
    print(f"Peak clustering at {k_function['peak_distance']:.0f}m")
else:
    print("Sites do not show significant clustering (random or dispersed)")

# Nearest neighbor analysis
nn_results = analyzer.nearest_neighbor_analysis(
    sites,
    n_neighbors=5
)

print(f"\nNearest Neighbor Analysis:")
print(f"  Mean distance to nearest neighbor: {nn_results['mean_nn_dist']:.0f}m")
print(f"  Clark-Evans R: {nn_results['clark_evans_r']:.3f}")
if nn_results['clark_evans_r'] < 1:
    print(f"  Pattern: Clustered (p={nn_results['p_value']:.4f})")
elif nn_results['clark_evans_r'] > 1:
    print(f"  Pattern: Dispersed (p={nn_results['p_value']:.4f})")
else:
    print(f"  Pattern: Random")

# Voronoi tessellation (territories)
voronoi = analyzer.create_voronoi_polygons(
    sites,
    boundary=study_area_polygon
)

# Calculate territory sizes
territory_areas = [poly.area / 10000 for poly in voronoi.geometry]  # hectares
print(f"\nTerritory sizes:")
print(f"  Mean: {np.mean(territory_areas):.1f} ha")
print(f"  Median: {np.median(territory_areas):.1f} ha")
print(f"  Std: {np.std(territory_areas):.1f} ha")

# Visualize territories
analyzer.plot_voronoi(
    voronoi,
    sites,
    output_path='s3://archaeology-results/voronoi_territories.png'
)

# Environmental correlates
# What environmental factors influenced site locations?

# Load environmental data
env_data = analyzer.load_environmental_data(
    variables=[
        'elevation',
        'slope',
        'aspect',
        'distance_to_water',
        'soil_type',
        'vegetation',
        'viewshed_area'
    ],
    resolution=30  # meters
)

# Extract environmental values at site locations
site_environments = analyzer.extract_environmental_values(
    sites,
    env_data
)

# Compare to random locations (null hypothesis: no preference)
random_points = analyzer.generate_random_points(
    n=len(sites) * 10,
    boundary=study_area_polygon
)

random_environments = analyzer.extract_environmental_values(
    random_points,
    env_data
)

# Statistical tests
env_preferences = analyzer.test_environmental_preferences(
    site_environments,
    random_environments,
    variables=env_data.keys()
)

print(f"\nEnvironmental Preferences:")
for var, result in env_preferences.items():
    if result['significant']:
        direction = "higher" if result['effect'] > 0 else "lower"
        print(f"  {var}: {direction} than random (p={result['p_value']:.4f})")

# Elevation preference
analyzer.plot_environmental_comparison(
    site_environments['elevation'],
    random_environments['elevation'],
    variable_name='Elevation (m)',
    output_path='s3://archaeology-results/elevation_preference.png'
)

# Distance to water (critical for settlements)
analyzer.plot_environmental_comparison(
    site_environments['distance_to_water'],
    random_environments['distance_to_water'],
    variable_name='Distance to Water (m)',
    output_path='s3://archaeology-results/water_distance.png'
)

# Viewshed analysis
# What can be seen from each site?

dem = env_data['elevation']

viewsheds = analyzer.calculate_viewsheds(
    sites,
    dem,
    observer_height=1.7,  # meters (human eye height)
    target_height=0.0,
    max_distance=10000  # meters
)

# Intervisibility matrix (which sites can see each other?)
intervisibility = analyzer.calculate_intervisibility(
    sites,
    viewsheds
)

print(f"\nIntervisibility:")
print(f"  {np.sum(intervisibility) / 2:.0f} site pairs can see each other")
print(f"  Mean sites visible from each site: {np.mean(np.sum(intervisibility, axis=1)):.1f}")

# Visualize intervisibility network
analyzer.plot_intervisibility_network(
    sites,
    intervisibility,
    output_path='s3://archaeology-results/intervisibility_network.png'
)

# Least-cost path analysis
# Model ancient movement (roads, trade routes)

# Create cost surface (travel difficulty)
cost_surface = analyzer.create_cost_surface(
    dem,
    slope_weight=1.0,
    water_bonus=-0.5,  # Easier to travel near water
    vegetation_penalty=0.3
)

# Calculate least-cost paths between major sites
major_sites = sites[sites['size'] > sites['size'].quantile(0.8)]

paths = analyzer.calculate_least_cost_paths(
    major_sites,
    cost_surface,
    max_paths=50  # Connect 50 most important pairs
)

# Find route corridors (areas traversed by multiple paths)
route_corridors = analyzer.identify_route_corridors(
    paths,
    corridor_width=500,  # meters
    min_paths=3
)

print(f"\nRoute Analysis:")
print(f"  Calculated {len(paths)} least-cost paths")
print(f"  Identified {len(route_corridors)} route corridors")
print(f"  Total corridor length: {sum(c['length'] for c in route_corridors)/1000:.1f} km")

# Visualize route network
analyzer.plot_route_network(
    sites,
    paths,
    route_corridors,
    dem,
    output_path='s3://archaeology-results/route_network.png'
)

# Predictive modeling
# Where are undiscovered sites likely to be?

# Train model to predict site locations
predictive_model = analyzer.train_site_prediction_model(
    known_sites=sites,
    environmental_layers=env_data,
    model_type='random_forest',  # or 'maxent', 'logistic'
    training_split=0.8
)

# Evaluate model
eval_metrics = analyzer.evaluate_predictive_model(
    predictive_model,
    test_sites=sites_test
)

print(f"\nPredictive Model Performance:")
print(f"  AUC: {eval_metrics['auc']:.3f}")
print(f"  Accuracy: {eval_metrics['accuracy']:.3f}")
print(f"  Sensitivity: {eval_metrics['sensitivity']:.3f}")
print(f"  Specificity: {eval_metrics['specificity']:.3f}")

# Generate probability map
probability_map = analyzer.predict_site_probability(
    predictive_model,
    env_data,
    output_path='s3://archaeology-results/site_probability_map.tif'
)

# Identify high-probability areas for survey
survey_targets = analyzer.identify_survey_targets(
    probability_map,
    threshold=0.7,
    min_area=1000,  # m²
    exclude_known_sites=True,
    known_sites=sites
)

print(f"\nSurvey Recommendations:")
print(f"  {len(survey_targets)} high-probability areas identified")
print(f"  Total area: {sum(t['area'] for t in survey_targets)/10000:.1f} ha")

analyzer.export_survey_targets(
    survey_targets,
    output_path='s3://archaeology-results/survey_targets.shp'
)
```

### 4. Bayesian Radiocarbon Dating

Build chronological models with Bayesian statistics.

```python
from src.chronology import ChronologyAnalyzer
import pandas as pd
import numpy as np
import subprocess
import json

analyzer = ChronologyAnalyzer()

# Load radiocarbon dates
# Typical format: site, context, lab_code, c14_age, c14_error
dates_df = pd.read_csv('s3://archaeology-data/radiocarbon-dates.csv')

print(f"Loaded {len(dates_df)} radiocarbon dates")
print(f"Date range: {dates_df['c14_age'].min()} - {dates_df['c14_age'].max()} BP")
print(f"Sites: {dates_df['site'].nunique()}")

# Calibrate individual dates
# Uses IntCal20 calibration curve
calibrated = analyzer.calibrate_dates(
    c14_ages=dates_df['c14_age'],
    c14_errors=dates_df['c14_error'],
    curve='intcal20',  # or 'shcal20', 'marine20'
    resolution=1  # year
)

# Plot calibration example
analyzer.plot_calibration(
    c14_age=3450,
    c14_error=30,
    lab_code='Beta-123456',
    output_path='s3://archaeology-results/calibration_example.png'
)

# Bayesian chronological modeling with OxCal
# Build a sequence model for a stratigraphic sequence

# Example: multi-phase site
oxcal_model = """
Plot()
{
 Sequence("Site A")
 {
  Boundary("Start");
  Phase("Early Occupation")
  {
   R_Date("Beta-11111", 3450, 30);
   R_Date("Beta-22222", 3420, 35);
   R_Date("Beta-33333", 3380, 40);
  };
  Boundary("Transition");
  Phase("Middle Occupation")
  {
   R_Date("Beta-44444", 3250, 30);
   R_Date("Beta-55555", 3200, 35);
  };
  Boundary("Transition");
  Phase("Late Occupation")
  {
   R_Date("Beta-66666", 3050, 40);
   R_Date("Beta-77777", 3010, 30);
  };
  Boundary("End");
 };
};
"""

# Run OxCal (requires OxCal installation)
results = analyzer.run_oxcal_model(
    model_code=oxcal_model,
    output_format='json'
)

# Extract modeled dates
modeled_dates = analyzer.extract_oxcal_results(results)

print(f"\nBayesian Modeled Date Ranges (95.4% probability):")
for item in modeled_dates:
    if item['type'] == 'boundary':
        print(f"  {item['name']}: {item['start_date']:.0f} - {item['end_date']:.0f} cal BP")
        print(f"    (95.4% range: {item['range_years']:.0f} years)")

# Phase durations
phase_durations = analyzer.calculate_phase_durations(modeled_dates)

print(f"\nPhase Durations:")
for phase, duration in phase_durations.items():
    print(f"  {phase}: {duration['mean']:.0f} years "
          f"(95.4%: {duration['lower']:.0f}-{duration['upper']:.0f})")

# Compare chronologies between sites
site_a_dates = dates_df[dates_df['site'] == 'Site A']
site_b_dates = dates_df[dates_df['site'] == 'Site B']

# Test if occupations are synchronous
synchrony_test = analyzer.test_synchrony(
    site_a_dates,
    site_b_dates,
    method='bayesian'
)

print(f"\nSynchrony Test (Site A vs Site B):")
print(f"  Overlap probability: {synchrony_test['overlap_prob']:.3f}")
if synchrony_test['overlap_prob'] > 0.7:
    print(f"  Result: Sites likely contemporaneous")
else:
    print(f"  Result: Sites likely not contemporaneous")

# Sum calibration (for regional trends)
# Sum all dates from a region to see population/activity trends
regional_sum = analyzer.sum_calibration(
    dates_df,
    normalize=True,
    bootstrap_ci=True,
    n_bootstrap=1000
)

analyzer.plot_summed_probability(
    regional_sum,
    output_path='s3://archaeology-results/summed_probability.png'
)

# Aoristic analysis
# Distribute sites probabilistically across their possible date ranges
sites_with_dates = gpd.read_file('s3://archaeology-data/dated-sites.shp')

aoristic = analyzer.aoristic_analysis(
    sites_with_dates,
    time_blocks=[(1000, 500), (500, 0), (0, 500), (500, 1000)],  # cal BC/AD
    weight_by='site_size'
)

print(f"\nAoristic Analysis:")
for period, weight in aoristic.items():
    print(f"  {period[0]}-{period[1]}: {weight:.1f} sites")

analyzer.plot_aoristic(
    aoristic,
    output_path='s3://archaeology-results/aoristic_analysis.png'
)

# Seriation (ordering based on similarity)
# Useful for pottery assemblages, artifact types
assemblages = pd.read_csv('s3://archaeology-data/ceramic-assemblages.csv')

seriation_order = analyzer.seriate(
    assemblages,
    method='correspondence_analysis',
    n_dimensions=2
)

analyzer.plot_seriation(
    assemblages,
    seriation_order,
    output_path='s3://archaeology-results/seriation.png'
)

# Tempo analysis (rate of change)
# Identify periods of rapid vs slow cultural change
tempo = analyzer.calculate_tempo(
    dates_df,
    window_size=200,  # years
    metric='date_density'
)

print(f"\nTempo Analysis:")
print(f"  Periods of rapid change:")
for period in tempo['rapid_periods']:
    print(f"    {period['start']:.0f}-{period['end']:.0f} BP")
```

### 5. 3D Photogrammetry and Reconstruction

Create 3D models from photographs.

```python
from src.photogrammetry import PhotogrammetryPipeline
import subprocess
import os

pipeline = PhotogrammetryPipeline()

# Structure from Motion (SfM) using OpenDroneMap
# Assumes drone or ground photos uploaded to S3

photos_folder = 's3://archaeology-data/photos/structure_001/'

# Download photos locally for processing (or use EC2 with EBS)
local_photos = pipeline.download_photos(
    photos_folder,
    local_dir='/tmp/structure_001_photos/'
)

print(f"Downloaded {len(local_photos)} photos")

# Check photo quality
quality_report = pipeline.assess_photo_quality(
    local_photos,
    checks=['focus', 'exposure', 'overlap', 'gps']
)

print(f"\nPhoto Quality Report:")
print(f"  Sharp: {quality_report['sharp_pct']:.1f}%")
print(f"  Properly exposed: {quality_report['exposed_pct']:.1f}%")
print(f"  With GPS: {quality_report['gps_pct']:.1f}%")
print(f"  Estimated overlap: {quality_report['overlap_pct']:.1f}%")

if quality_report['overlap_pct'] < 60:
    print("⚠️  Warning: Low overlap may result in incomplete model")

# Run OpenDroneMap
# Can be deployed on EC2 with GPU (g4dn.xlarge or larger)
odm_results = pipeline.run_opendronemap(
    image_dir='/tmp/structure_001_photos/',
    output_dir='/tmp/structure_001_output/',
    options={
        'feature-quality': 'high',
        'min-num-features': 10000,
        'matcher-neighbors': 12,
        'mesh-size': 300000,  # triangles
        'mesh-octree-depth': 11,
        'texturing-data-term': 'gmi',
        'texturing-skip-global-seam-leveling': False,
        'texturing-skip-local-seam-leveling': False,
        'use-3dmesh': True,
        'pc-quality': 'high'
    }
)

print(f"\nPhotogrammetry Results:")
print(f"  Photos processed: {odm_results['photos_processed']}")
print(f"  Point cloud points: {odm_results['point_cloud_size']:,}")
print(f"  Mesh triangles: {odm_results['mesh_triangles']:,}")
print(f"  Texture resolution: {odm_results['texture_resolution']}")
print(f"  Processing time: {odm_results['processing_time']:.1f} minutes")

# Upload results to S3
point_cloud = '/tmp/structure_001_output/odm_georeferencing/odm_georeferenced_model.laz'
mesh = '/tmp/structure_001_output/odm_texturing/odm_textured_model_geo.obj'
orthophoto = '/tmp/structure_001_output/odm_orthophoto/odm_orthophoto.tif'

pipeline.upload_results(
    files=[point_cloud, mesh, orthophoto],
    s3_prefix='s3://archaeology-results/3d-models/structure_001/'
)

# Post-processing with MeshLab/Open3D
import open3d as o3d

# Load mesh
mesh_obj = o3d.io.read_triangle_mesh(mesh)
print(f"\nMesh statistics:")
print(f"  Vertices: {len(mesh_obj.vertices):,}")
print(f"  Triangles: {len(mesh_obj.triangles):,}")
print(f"  Has vertex normals: {mesh_obj.has_vertex_normals()}")
print(f"  Has textures: {mesh_obj.has_textures()}")

# Clean mesh
mesh_obj.remove_duplicated_vertices()
mesh_obj.remove_unreferenced_vertices()
mesh_obj.remove_degenerate_triangles()
mesh_obj.remove_non_manifold_edges()

# Smooth mesh
mesh_obj = mesh_obj.filter_smooth_simple(number_of_iterations=5)

# Compute normals
mesh_obj.compute_vertex_normals()

# Simplify (reduce file size)
mesh_simplified = mesh_obj.simplify_quadric_decimation(
    target_number_of_triangles=100000
)

print(f"\nSimplified mesh: {len(mesh_simplified.triangles):,} triangles")

# Save simplified mesh
o3d.io.write_triangle_mesh(
    '/tmp/structure_001_simplified.obj',
    mesh_simplified
)

# Upload to S3
pipeline.upload_to_s3(
    '/tmp/structure_001_simplified.obj',
    's3://archaeology-results/3d-models/structure_001/simplified.obj'
)

# Create web-friendly format (GLB for 3D viewers)
mesh_glb = pipeline.convert_to_glb(
    mesh_obj,
    output_path='/tmp/structure_001.glb',
    draco_compression=True
)

pipeline.upload_to_s3(
    mesh_glb,
    's3://archaeology-results/3d-models/structure_001/model.glb'
)

# Generate 2D outputs from 3D model
# Plan view (top-down orthophoto)
plan_view = pipeline.render_plan_view(
    mesh_obj,
    resolution=0.001,  # meters/pixel
    output_path='/tmp/plan_view.png'
)

# Elevation views
elevations = pipeline.render_elevations(
    mesh_obj,
    directions=['north', 'south', 'east', 'west'],
    output_dir='/tmp/elevations/'
)

# Cross-sections
cross_sections = pipeline.create_cross_sections(
    mesh_obj,
    n_sections=10,
    direction='x',
    output_dir='/tmp/cross_sections/'
)

# Measurements from 3D model
measurements = pipeline.measure_from_mesh(
    mesh_obj,
    measurements=[
        {'type': 'distance', 'points': [(0, 0, 0), (10, 0, 0)]},
        {'type': 'area', 'polygon': [(0, 0), (10, 0), (10, 10), (0, 10)]},
        {'type': 'volume', 'base_height': 0}
    ]
)

print(f"\nMeasurements:")
for m in measurements:
    print(f"  {m['type']}: {m['value']:.3f} {m['unit']}")

# Generate technical documentation
documentation = pipeline.generate_documentation(
    photos=local_photos,
    model_stats=odm_results,
    measurements=measurements,
    output_path='s3://archaeology-results/3d-models/structure_001/documentation.pdf'
)

# Virtual museum integration
# Create Sketchfab-ready package
sketchfab_package = pipeline.create_sketchfab_package(
    mesh_glb,
    title='Archaeological Structure 001',
    description='Roman villa excavated 2024',
    tags=['archaeology', 'roman', 'villa', '3d-scan'],
    categories=['Cultural Heritage & History']
)

print(f"\n3D model ready for virtual museum display")
print(f"View online: https://archaeology-museum.example.com/models/structure_001")
```

### 6. Trade Network Analysis

Model ancient trade routes and cultural connections.

```python
from src.network_analysis import TradeNetworkAnalyzer
import networkx as nx
import pandas as pd
import numpy as np

analyzer = TradeNetworkAnalyzer()

# Load material sourcing data
# Example: obsidian sources and artifact findspots
obsidian_artifacts = pd.read_csv('s3://archaeology-data/obsidian-artifacts.csv')
obsidian_sources = pd.read_csv('s3://archaeology-data/obsidian-sources.csv')

# Geochemical matching (link artifacts to sources)
matches = analyzer.match_artifacts_to_sources(
    artifacts=obsidian_artifacts,
    sources=obsidian_sources,
    method='mahalanobis',  # Multivariate distance
    elements=['SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O', 'TiO2', 'MnO'],
    threshold=3.0  # Mahalanobis distance
)

print(f"Matched {len(matches)} artifacts to sources")
print(f"\nSource distribution:")
for source, count in matches['source'].value_counts().items():
    print(f"  {source}: {count} artifacts")

# Build trade network
# Nodes: sites
# Edges: material flows (weighted by quantity/frequency)

network = analyzer.build_trade_network(
    matches,
    site_column='findspot',
    source_column='source',
    weight_by='count'
)

print(f"\nNetwork Statistics:")
print(f"  Nodes (sites): {network.number_of_nodes()}")
print(f"  Edges (connections): {network.number_of_edges()}")
print(f"  Density: {nx.density(network):.3f}")
print(f"  Average clustering: {nx.average_clustering(network):.3f}")

# Network metrics
centrality_metrics = analyzer.calculate_centrality(
    network,
    metrics=['degree', 'betweenness', 'closeness', 'eigenvector']
)

print(f"\nMost central sites (betweenness):")
top_sites = centrality_metrics['betweenness'].nlargest(5)
for site, score in top_sites.items():
    print(f"  {site}: {score:.3f}")

# Community detection (trade zones)
communities = analyzer.detect_communities(
    network,
    method='louvain'
)

print(f"\nDetected {len(set(communities.values()))} trade communities")
for comm_id in set(communities.values()):
    sites = [s for s, c in communities.items() if c == comm_id]
    print(f"  Community {comm_id}: {len(sites)} sites")

# Visualize network
analyzer.visualize_network(
    network,
    communities=communities,
    centrality=centrality_metrics['betweenness'],
    site_locations=sites_gdf,  # GeoDataFrame with coordinates
    output_path='s3://archaeology-results/trade_network.png'
)

# Geographic visualization
analyzer.plot_trade_flows_on_map(
    network,
    site_locations=sites_gdf,
    edge_width_by='weight',
    background='terrain',
    output_path='s3://archaeology-results/trade_flows_map.png'
)

# Temporal evolution of network
# How did trade networks change over time?

time_periods = [
    ('Early', -500, -300),
    ('Middle', -300, -100),
    ('Late', -100, 100)
]

temporal_networks = {}
for period_name, start, end in time_periods:
    period_data = matches[
        (matches['date_start'] >= start) &
        (matches['date_end'] <= end)
    ]
    temporal_networks[period_name] = analyzer.build_trade_network(
        period_data,
        site_column='findspot',
        source_column='source',
        weight_by='count'
    )

# Compare networks
network_comparison = analyzer.compare_networks(
    temporal_networks,
    metrics=['nodes', 'edges', 'density', 'clustering', 'centralization']
)

print(f"\nNetwork Evolution:")
for period, metrics in network_comparison.items():
    print(f"  {period}:")
    print(f"    Sites: {metrics['nodes']}")
    print(f"    Connections: {metrics['edges']}")
    print(f"    Density: {metrics['density']:.3f}")

# Analyze disruptions/changes
change_points = analyzer.detect_network_disruptions(
    temporal_networks,
    threshold=0.3  # 30% change
)

for change in change_points:
    print(f"\nNetwork disruption between {change['period1']} and {change['period2']}:")
    print(f"  Sites lost: {change['nodes_lost']}")
    print(f"  Sites gained: {change['nodes_gained']}")
    print(f"  Connections lost: {change['edges_lost']}")

# Distance decay analysis
# How does material frequency decline with distance from source?

distance_decay = analyzer.analyze_distance_decay(
    matches,
    sites_gdf,
    obsidian_sources,
    distance_bins=np.arange(0, 500, 50),  # km
    plot=True,
    output_path='s3://archaeology-results/distance_decay.png'
)

print(f"\nDistance Decay:")
print(f"  Decay exponent: {distance_decay['exponent']:.3f}")
print(f"  R²: {distance_decay['r2']:.3f}")
print(f"  Half-distance: {distance_decay['half_distance']:.0f} km")

# Fall-off in material frequency suggests exchange mode
if distance_decay['exponent'] < -0.5:
    print(f"  Interpretation: Down-the-line exchange")
elif distance_decay['exponent'] < -1.5:
    print(f"  Interpretation: Direct procurement or directed trade")
else:
    print(f"  Interpretation: Complex/redistributive exchange")
```

## Database Integration

Access archaeological data from major repositories.

```python
from src.database_connectors import OpenContextConnector, TDARConnector
import pandas as pd

# Open Context
oc = OpenContextConnector()

# Search for artifacts
results = oc.search(
    query='pottery',
    project='Murlo',
    item_type='artifact',
    properties={'ware': 'bucchero'},
    date_range=(-700, -500),
    bbox=[11.0, 43.0, 11.5, 43.5],  # Tuscany
    limit=1000
)

print(f"Found {len(results)} artifacts")

# Download full records
artifacts = oc.get_items(
    results['uuid'],
    include_context=True,
    include_images=True
)

# Convert to DataFrame
artifacts_df = pd.DataFrame(artifacts)

# Download artifact images
for uuid, image_urls in artifacts['images'].items():
    oc.download_images(
        image_urls,
        output_dir=f's3://archaeology-data/open-context-images/{uuid}/'
    )

# tDAR
tdar = TDARConnector(api_key='your-api-key')

# Search for datasets
datasets = tdar.search(
    query='Southwest ceramics',
    resource_type='dataset',
    spatial_coverage='Arizona',
    temporal_coverage=(1000, 1300)
)

print(f"Found {len(datasets)} datasets")

# Download dataset
dataset_id = datasets[0]['id']
dataset_files = tdar.download_dataset(
    dataset_id,
    output_dir='s3://archaeology-data/tdar/'
)

# DINAA
# Access through Open Context's DINAA interface
dinaa_sites = oc.search_dinaa(
    state=['Ohio', 'Indiana'],
    site_type='mound',
    time_period='Woodland',
    limit=5000
)

print(f"Found {len(dinaa_sites)} mound sites")

# Export to GeoDataFrame
sites_gdf = gpd.GeoDataFrame.from_features(dinaa_sites)
sites_gdf.to_file('s3://archaeology-data/dinaa-mounds.geojson')
```

## Cost Estimate

**One-time setup:** $100-200
- CloudFormation stack deployment
- Initial data transfer
- Model training infrastructure

**Per project costs:**

**Single Site Analysis (excavation/survey):**
- Storage (10 GB LiDAR + photos): $0.25/month
- Processing (10 hours EC2 GPU): $15-30
- SageMaker inference: $5-10
- **Total: $500-800 one-time, $5-10/month ongoing**

**Regional Survey (100 sites, 1000 km²):**
- Storage (100 GB): $2.50/month
- LiDAR processing (Batch): $100-200
- Artifact classification: $50-100
- GIS analysis: $20-50
- **Total: $2,000-4,000 one-time, $50-100/month ongoing**

**Large-Scale Landscape Study (1,000+ sites, 10,000 km²):**
- Storage (1 TB): $25/month
- Distributed processing: $500-1,000
- ML training: $200-500
- PostGIS database: $50-100/month
- **Total: $8,000-15,000 setup, $500-1,000/month ongoing**

**Continental-Scale Analysis:**
- Storage (10+ TB): $250+/month
- Massive parallel processing: $2,000-5,000
- Advanced ML models: $1,000-2,000
- High-performance databases: $200-500/month
- **Total: $25,000-50,000 setup, $2,000-5,000/month ongoing**

**Cost Optimization:**
- Use Spot instances for Batch (60-70% savings)
- S3 Intelligent-Tiering for archival data
- Reserved instances for long-running databases
- Glacier for long-term photo storage

## Performance Benchmarks

**LiDAR Processing:**
- Point cloud loading (100M points): 30-60 seconds
- Ground classification (SMRF): 2-5 minutes per km²
- DTM generation: 1-3 minutes per km²
- Feature detection: 5-10 minutes per km²
- Full pipeline (1 km²): 10-20 minutes on c5.4xlarge

**Artifact Classification:**
- Training (10K images, 50 epochs): 2-4 hours on ml.p3.2xlarge
- Inference (single image): 50-100ms
- Batch inference (1000 images): 1-2 minutes
- Real-time classification: 10-20 images/second

**Photogrammetry:**
- 100 photos, 20MP: 1-2 hours on g4dn.2xlarge
- 500 photos, 20MP: 4-8 hours on g4dn.4xlarge
- Point cloud generation: 30-50% of total time
- Meshing: 20-30% of total time
- Texturing: 20-30% of total time

**GIS Analysis:**
- Ripley's K (1000 sites): 1-2 minutes
- Viewshed (single site): 5-10 seconds
- Least-cost paths (100 paths): 5-10 minutes
- Predictive model training: 10-30 minutes
- PostGIS spatial queries: <1 second (indexed)

**Database Queries:**
- Open Context search: 1-3 seconds
- Full record retrieval (100 items): 5-10 seconds
- Image download (1000 images): 10-20 minutes (network dependent)

## Best Practices

### Data Management

1. **Metadata standards:** Use Dublin Core, MIDAS Heritage, or ArchaeoML
2. **File organization:** Structured S3 prefixes (project/site/year/type)
3. **Version control:** Git for code, S3 versioning for data
4. **Documentation:** README files, data dictionaries, processing logs
5. **Backups:** Cross-region S3 replication for critical data

### Ethical Considerations

1. **Site protection:** Blur precise coordinates in public data (±1km)
2. **Indigenous data sovereignty:** Consult with descendant communities
3. **Cultural sensitivity:** Respect cultural restrictions on data/images
4. **Repatriation:** Support return of artifacts to source communities
5. **Open science:** Share data with appropriate protections
6. **Stakeholder engagement:** Include local communities in research

### Analysis Quality

1. **Ground truthing:** Always validate remote sensing with field checks
2. **Temporal calibration:** Update radiocarbon dates with latest curves
3. **Spatial accuracy:** Use high-quality DEMs, validate GPS coordinates
4. **Model validation:** Split data, cross-validation, external testing
5. **Uncertainty:** Report confidence intervals, classification probabilities
6. **Reproducibility:** Document parameters, random seeds, software versions

### Collaboration

1. **Interdisciplinary teams:** Archaeologists, data scientists, local experts
2. **Data sharing:** Use standard formats (GeoJSON, Shapefile, LAZ, CSV)
3. **APIs:** Build interfaces for others to access your data
4. **Publications:** Cite data sources, acknowledge contributors
5. **Training:** Teach archaeological applications of AWS/ML

### Cost Management

1. **Spot instances:** Use for non-critical batch processing
2. **Lifecycle policies:** Move old data to Glacier after 1 year
3. **Monitoring:** CloudWatch alerts for unexpected costs
4. **Rightsizing:** Match instance types to workload
5. **Serverless:** Use Lambda for simple tasks instead of EC2

## Troubleshooting

### LiDAR Issues

**Problem:** Ground classification fails in dense vegetation
- **Solution:** Adjust SMRF parameters (smaller window, lower threshold)
- **Alternative:** Use cloth simulation filter (CSF) instead

**Problem:** LAZ files won't open
- **Solution:** Update PDAL/laspy libraries, check file corruption

**Problem:** Coordinate system errors
- **Solution:** Verify CRS with `pdal info`, reproject if needed

### Artifact Classification Issues

**Problem:** Poor accuracy on test set
- **Solution:** More training data, better augmentation, check for label errors

**Problem:** Model bias toward common classes
- **Solution:** Use class weights, oversampling, focal loss

**Problem:** High uncertainty on predictions
- **Solution:** Collect more training examples, ensemble models, expert review

### Photogrammetry Issues

**Problem:** Poor photo overlap
- **Solution:** Retake photos with 70-80% overlap, use flight planning software

**Problem:** Mesh has holes
- **Solution:** Add more photos, increase mesh density, manual hole filling

**Problem:** Texture artifacts
- **Solution:** Better lighting, color correction, texture blending algorithms

### GIS Issues

**Problem:** Slow PostGIS queries
- **Solution:** Add spatial index (GIST), increase work_mem, use simpler geometries

**Problem:** Viewshed calculations fail
- **Solution:** Check DEM for NoData values, reduce max distance, use lower resolution

**Problem:** Coordinate system mismatches
- **Solution:** Standardize to single CRS, use PostGIS `ST_Transform`

## Additional Resources

### Learning Resources

**Remote Sensing:**
- NASA ARSET Training: https://appliedsciences.nasa.gov/what-we-do/capacity-building/arset
- Archaeo-Geophysics Course: https://www.earthsciencedataanalytics.com/

**GIS for Archaeology:**
- "GIS for Archaeologists" by Conolly & Lake
- QGIS tutorials: https://www.qgistutorials.com/

**Radiocarbon Dating:**
- OxCal manual: https://c14.arch.ox.ac.uk/oxcalhelp/hlp_contents.html
- Bayesian Radiocarbon Dating course: https://c14.arch.ox.ac.uk/

**Machine Learning:**
- Fast.ai course: https://www.fast.ai/
- PyTorch tutorials: https://pytorch.org/tutorials/

### Software Tools

**LiDAR:**
- CloudCompare: https://www.cloudcompare.org/
- PDAL: https://pdal.io/
- LAStools: https://rapidlasso.com/lastools/

**GIS:**
- QGIS: https://qgis.org/
- PostGIS: https://postgis.net/
- GeoPandas: https://geopandas.org/

**Photogrammetry:**
- OpenDroneMap: https://www.opendronemap.org/
- MeshLab: https://www.meshlab.net/
- Agisoft Metashape: https://www.agisoft.com/

**3D Visualization:**
- Sketchfab: https://sketchfab.com/
- Potree: https://github.com/potree/potree
- Three.js: https://threejs.org/

### Data Repositories

- **Open Context:** https://opencontext.org/
- **tDAR:** https://www.tdar.org/
- **OpenTopography:** https://opentopography.org/
- **IntChron:** https://intchron.org/
- **ADS (UK):** https://archaeologydataservice.ac.uk/

### Key Papers

1. Chase et al. (2012). "Airborne LiDAR, archaeology, and the ancient Maya landscape." *Journal of Archaeological Science*
2. Traviglia & Cottica (2011). "Remote sensing applications and archaeological research in the Mediterranean." *Layers of Perception*
3. Bevan & Conolly (2013). "Mediterranean Islands, Fragile Communities and Persistent Landscapes." *Cambridge*
4. Richards et al. (2015). "Bayesian chronological models." *Radiocarbon*
5. Gravel-Miguel (2016). "Using species distribution models for archaeological site detection." *Advances in Archaeological Practice*

### Professional Organizations

- Society for American Archaeology (SAA): https://www.saa.org/
- European Association of Archaeologists (EAA): https://www.e-a-a.org/
- Archaeological Institute of America (AIA): https://www.archaeological.org/
- Computer Applications in Archaeology (CAA): https://caa-international.org/

## Next Steps

1. **Deploy Infrastructure:** Run CloudFormation stack, set up S3 buckets
2. **Test LiDAR Pipeline:** Download sample data from OpenTopography, detect features
3. **Train Artifact Classifier:** Collect training images, train CNN, evaluate
4. **Spatial Analysis:** Load site data, run Ripley's K, viewshed analysis
5. **Chronology:** Compile radiocarbon dates, calibrate, build Bayesian model
6. **3D Reconstruction:** Process drone photos with OpenDroneMap
7. **Integration:** Combine analyses into comprehensive site/regional study

---

**Tier 1 Project Status:** Production-ready
**Estimated Setup Time:** 6-8 hours
**Cost:** $500-800 per site, $2,000-15,000 for regional studies
**Impact:** Discover hidden sites, automate artifact analysis, understand ancient landscapes

For questions, consult archaeological computing resources, AWS documentation, or the CAA community.
