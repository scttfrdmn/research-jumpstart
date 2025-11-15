from setuptools import find_packages, setup

setup(
    name="ocean-marine-analysis-aws",
    version="1.0.0",
    description="Large-scale ocean and marine ecosystem analysis with satellite data, Argo floats, and species tracking on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "xarray>=2023.6.0",
        "netCDF4>=1.6.0",
        "scikit-learn>=1.3.0",
        "rasterio>=1.3.0",
        "geopandas>=0.13.0",
        "cartopy>=0.21.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
        "gsw>=3.6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Oceanography",
        "Programming Language :: Python :: 3.9",
    ],
)
