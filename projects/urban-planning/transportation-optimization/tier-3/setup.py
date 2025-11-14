from setuptools import setup, find_packages

setup(
    name="transportation-optimization-aws",
    version="1.0.0",
    description="Multi-city transportation network optimization with real-time traffic prediction on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "geopandas>=0.13.0",
        "networkx>=3.1",
        "osmnx>=1.6.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
        "sqlalchemy>=2.0.0",
        "geoalchemy2>=0.14.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3.9",
    ],
)
