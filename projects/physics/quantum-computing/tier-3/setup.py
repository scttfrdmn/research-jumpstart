from setuptools import find_packages, setup

setup(
    name="quantum-computing-aws",
    version="1.0.0",
    description="Scalable quantum computing on AWS Braket",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "amazon-braket-sdk>=1.50.0",
        "amazon-braket-pennylane-plugin>=1.17.0",
        "pennylane>=0.32.0",
        "boto3>=1.28.0",
        "scikit-learn>=1.3.0",
        "anthropic>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "run-vqe=src.vqe:main",
            "run-qaoa=src.qaoa:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
