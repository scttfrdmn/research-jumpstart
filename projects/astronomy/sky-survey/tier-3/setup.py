from setuptools import find_packages, setup

setup(
    name="sky-survey-analysis",
    version="1.0.0",
    description="Large-scale astronomical sky survey analysis on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "astropy>=5.3.0",
        "astroquery>=0.4.6",
        "photutils>=1.8.0",
        "sep>=1.2.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "boto3>=1.28.0",
        "awswrangler>=3.2.0",
        "sagemaker>=2.180.0",
        "anthropic>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "extract-sources=src.photometry:main",
            "classify-galaxies=src.morphology:main",
            "compute-photoz=src.photoz:main",
            "detect-transients=src.transients:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
