"""Setup script for Genomic Variant Analysis - Unified Studio."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genomics-variant-analysis",
    version="1.0.0",
    author="Research Jumpstart",
    description="Production genomic variant analysis toolkit for AWS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/research-jumpstart",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.11",
    install_requires=[
        "boto3>=1.35.0",
        "pandas>=2.2.0",
        "numpy>=2.1.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "scipy>=1.14.0",
        "pysam>=0.22.0",
        "cyvcf2>=0.31.0",
        "scikit-allel>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-cov>=5.0.0",
            "black>=24.8.0",
            "flake8>=7.1.0",
            "mypy>=1.11.0",
        ],
        "notebooks": [
            "jupyter>=1.1.0",
            "jupyterlab>=4.2.0",
            "ipywidgets>=8.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "genomics-analysis=src.cli:main",
        ],
    },
)
