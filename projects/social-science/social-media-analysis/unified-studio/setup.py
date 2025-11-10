"""Setup script for social-media-analysis package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="social-media-analysis",
    version="1.0.0",
    author="Research Jumpstart Contributors",
    description="Production-ready social media analysis and misinformation detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/research-jumpstart",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "nltk>=3.8.0",
        "scikit-learn>=1.3.0",
        "gensim>=4.3.0",
        "vaderSentiment>=3.3.2",
        "boto3>=1.28.0",
        "s3fs>=2023.9.0",
        "networkx>=3.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "wordcloud>=1.9.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "notebooks": [
            "ipykernel>=6.25.0",
            "ipywidgets>=8.1.0",
            "jupyterlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "social-media-analysis=social_media_analysis.cli:main",
        ],
    },
)
