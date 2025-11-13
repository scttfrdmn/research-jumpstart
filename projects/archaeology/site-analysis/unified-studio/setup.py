from setuptools import setup, find_packages

setup(
    name="archaeological-site-analysis-aws",
    version="1.0.0",
    description="Large-scale archaeological site analysis with LiDAR, ML artifact classification, and spatial analysis on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "geopandas>=0.13.0",
        "rasterio>=1.3.0",
        "pdal>=3.2.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "networkx>=3.1",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Sociology :: History",
        "Programming Language :: Python :: 3.9",
    ],
)
