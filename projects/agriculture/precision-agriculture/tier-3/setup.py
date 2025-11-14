from setuptools import setup, find_packages

setup(
    name="precision-agriculture-aws",
    version="1.0.0",
    description="Large-scale precision agriculture with satellite data and IoT on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "geopandas>=0.13.0",
        "rasterio>=1.3.0",
        "xarray>=2023.6.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "tensorflow>=2.13.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
        "sentinelhub>=3.9.0",
        "planetary-computer>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Agriculture",
        "Programming Language :: Python :: 3.9",
    ],
)
