from setuptools import setup, find_packages

setup(
    name="smart-grid-optimization-aws",
    version="1.0.0",
    description="Smart grid optimization with load forecasting, renewable integration, and energy storage on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "tensorflow>=2.13.0",
        "statsmodels>=0.14.0",
        "prophet>=1.1.4",
        "pyomo>=6.6.0",
        "cvxpy>=1.3.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.9",
    ],
)
