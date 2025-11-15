"""Setup script for Digital Humanities Text Analysis - Unified Studio."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
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
        "nltk>=3.9.0",
        "spacy>=3.8.0",
        "scikit-learn>=1.5.0",
        "gensim>=4.3.0",
        "textstat>=0.7.0",
        "wordcloud>=1.9.0",
        "networkx>=3.3",
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
        "aws": [
            "awscli>=2.15.0",
            "sagemaker>=2.230.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "text-analysis=src.cli:main",
        ],
    },
)
