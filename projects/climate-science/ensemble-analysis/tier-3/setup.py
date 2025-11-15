"""
Setup configuration for Climate Ensemble Analysis package.
"""

import os

from setuptools import find_packages, setup

# Read version from src/__init__.py
version = {}
with open(os.path.join("src", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)

# Read long description from README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="climate-ensemble-analysis",
    version=version.get("__version__", "1.0.0"),
    author="Research Jumpstart Community",
    author_email="research-jumpstart@example.com",
    description="Production tools for CMIP6 climate model ensemble analysis on AWS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research-jumpstart/research-jumpstart",
    project_urls={
        "Documentation": "https://github.com/research-jumpstart/research-jumpstart/tree/main/projects/climate-science/ensemble-analysis",
        "Source": "https://github.com/research-jumpstart/research-jumpstart",
        "Issues": "https://github.com/research-jumpstart/research-jumpstart/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-cov>=5.0.0",
            "black>=24.8.0",
            "flake8>=7.1.0",
            "mypy>=1.11.0",
        ],
        "docs": [
            "sphinx>=8.0.0",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "climate-analysis=src.cli:main",
        ],
    },
    package_data={
        "src": ["*.yml", "*.yaml", "*.json"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "climate",
        "cmip6",
        "ensemble",
        "aws",
        "s3",
        "climate-models",
        "climate-science",
        "earth-science",
    ],
)
