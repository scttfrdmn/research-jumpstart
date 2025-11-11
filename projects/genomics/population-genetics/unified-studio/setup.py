from setuptools import setup, find_packages

setup(
    name="population-genetics-analysis",
    version="1.0.0",
    description="Scalable population genetics analysis on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "pysam>=0.21.0",
        "cyvcf2>=0.30.0",
        "scikit-allel>=1.3.5",
        "boto3>=1.28.0",
        "awswrangler>=3.2.0",
        "s3fs>=2023.6.0",
        "dask[complete]>=2023.7.0",
        "pyarrow>=12.0.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "anthropic>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "analyze-population-structure=src.population_structure:main",
            "run-selection-scan=src.selection:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
