from setuptools import find_packages, setup

setup(
    name="brain-imaging-aws",
    version="1.0.0",
    description="Large-scale neuroimaging analysis on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "nibabel>=5.1.0",
        "nilearn>=0.10.0",
        "nipype>=1.8.0",
        "dipy>=1.7.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "boto3>=1.28.0",
        "awswrangler>=3.2.0",
        "sagemaker>=2.180.0",
        "networkx>=3.1",
        "anthropic>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "process-brain-scan=src.batch_processing:main",
            "analyze-connectivity=src.functional_connectivity:main",
            "run-freesurfer=src.structural_analysis:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
