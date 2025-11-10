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
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.1.0",
        "pandas>=2.2.0",
        "scipy>=1.14.0",
        "nltk>=3.9.0",
        "scikit-learn>=1.5.0",
        "gensim>=4.3.0",
        "vaderSentiment>=3.3.2",
        "boto3>=1.35.0",
        "s3fs>=2024.9.0",
        "networkx>=3.3",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "wordcloud>=1.9.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
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
            "ipykernel>=6.29.0",
            "ipywidgets>=8.1.0",
            "jupyterlab>=4.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "social-media-analysis=social_media_analysis.cli:main",
        ],
    },
)
