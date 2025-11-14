from setuptools import setup, find_packages

setup(
    name="macroeconomic-forecasting",
    version="1.0.0",
    description="Scalable macroeconomic forecasting on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "statsmodels>=0.14.0",
        "pmdarima>=2.0.3",
        "prophet>=1.1.4",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "boto3>=1.28.0",
        "awswrangler>=3.2.0",
        "fredapi>=0.5.0",
        "pandas-datareader>=0.10.0",
        "wbgapi>=1.0.12",
        "anthropic>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "forecast-gdp=src.forecasting:main",
            "predict-recession=src.recession:main",
            "ingest-fred=src.data_ingestion:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Economics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
