from setuptools import find_packages, setup

setup(
    name="computational-materials-aws",
    version="1.0.0",
    description="Large-scale computational materials science with DFT and ML on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pymatgen>=2023.8.0",
        "ase>=3.22.0",
        "mp-api>=0.34.0",
        "aiida-core>=2.4.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.9",
    ],
)
