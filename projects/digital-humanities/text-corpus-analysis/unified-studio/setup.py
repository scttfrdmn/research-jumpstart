from setuptools import setup, find_packages

setup(
    name="text-corpus-analysis",
    version="1.0.0",
    description="Large-scale historical text analysis on AWS",
    author="AWS Research Jumpstart",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "gensim>=4.3.0",
        "transformers>=4.30.0",
        "boto3>=1.28.0",
        "anthropic>=0.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3.9",
    ],
)
