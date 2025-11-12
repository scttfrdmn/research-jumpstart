from setuptools import setup, find_packages

setup(
    name="molecular-dynamics-aws",
    version="1.0.0",
    description="Large-scale molecular dynamics simulations on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "MDAnalysis>=2.5.0",
        "mdtraj>=1.9.0",
        "openmm>=8.0.0",
        "boto3>=1.28.0",
        "awswrangler>=3.2.0",
        "sagemaker>=2.180.0",
        "rdkit>=2023.3.2",
        "anthropic>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "run-md=src.md_simulations:main",
            "analyze-trajectory=src.analysis:main",
            "calculate-fep=src.free_energy:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
