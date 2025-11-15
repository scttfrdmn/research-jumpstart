from setuptools import find_packages, setup

setup(
    name="learning-analytics-platform-aws",
    version="1.0.0",
    description="District-wide learning analytics platform with predictive models and real-time dashboards on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "statsmodels>=0.14.0",
        "pymer4>=0.8.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
        "sqlalchemy>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)
