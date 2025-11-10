"""Setup script for Digital Humanities Text Analysis - Unified Studio."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="digital-humanities-text-analysis",
    version="1.0.0",
    author="Research Jumpstart",
    description="Production text analysis toolkit for digital humanities research on AWS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/research-jumpstart",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "boto3>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "nltk>=3.8",
        "spacy>=3.5.0",
        "scikit-learn>=1.3.0",
        "gensim>=4.3.0",
        "textstat>=0.7.0",
        "wordcloud>=1.9.0",
        "networkx>=3.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
        ],
        "aws": [
            "awscli>=1.29.0",
            "sagemaker>=2.150.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "text-analysis=src.cli:main",
        ],
    },
)
