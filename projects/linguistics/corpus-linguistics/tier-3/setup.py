from setuptools import setup, find_packages

setup(
    name="corpus-linguistics-aws",
    version="1.0.0",
    description="Large-scale computational corpus linguistics with diachronic analysis and multilingual comparison on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "spacy>=3.6.0",
        "nltk>=3.8.0",
        "gensim>=4.3.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "pyspark>=3.4.0",
        "elasticsearch>=8.8.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3.9",
    ],
)
