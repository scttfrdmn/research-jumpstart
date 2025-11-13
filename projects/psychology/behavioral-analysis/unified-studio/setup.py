from setuptools import setup, find_packages

setup(
    name="behavioral-analysis-aws",
    version="1.0.0",
    description="Large-scale behavioral data analysis on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Psychology",
        "Programming Language :: Python :: 3.9",
    ],
)
