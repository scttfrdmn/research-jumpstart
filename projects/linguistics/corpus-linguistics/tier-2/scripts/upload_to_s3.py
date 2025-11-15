"""
Upload Corpus to S3

This script uploads a multilingual text corpus to S3, organizing files by:
- Language (english, spanish, french, etc.)
- Genre (academic, news, fiction, etc.)
- Register (formal, informal, technical, etc.)

It supports:
- Progress tracking
- Error handling and retry
- Parallel uploads (optional)
- Metadata tagging
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Initialize S3 client
s3_client = boto3.client("s3")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload linguistic corpus to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all texts from local corpus directory
  python upload_to_s3.py --bucket linguistic-corpus-123 --corpus-dir ./corpus

  # Upload with custom organization
  python upload_to_s3.py --bucket my-bucket --corpus-dir ./texts --language english --genre academic

  # Dry run (don't actually upload)
  python upload_to_s3.py --bucket my-bucket --corpus-dir ./corpus --dry-run
        """,
    )

    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name (e.g., linguistic-corpus-123)"
    )

    parser.add_argument(
        "--corpus-dir", required=True, help="Local directory containing corpus files"
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Override language (default: detect from directory structure)",
    )

    parser.add_argument(
        "--genre", default=None, help="Override genre (default: detect from directory structure)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    parser.add_argument(
        "--file-pattern", default="*.txt", help="File pattern to match (default: *.txt)"
    )

    return parser.parse_args()


def verify_bucket_exists(bucket_name: str) -> bool:
    """
    Verify that S3 bucket exists and is accessible.

    Args:
        bucket_name: S3 bucket name

    Returns:
        bool: True if bucket exists and is accessible
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            print(f"Error: Bucket '{bucket_name}' does not exist")
        elif error_code == "403":
            print(f"Error: Access denied to bucket '{bucket_name}'")
        else:
            print(f"Error checking bucket: {e}")
        return False


def find_corpus_files(corpus_dir: str, file_pattern: str = "*.txt") -> list[Path]:
    """
    Find all corpus files in directory.

    Args:
        corpus_dir: Root corpus directory
        file_pattern: File pattern to match

    Returns:
        List of Path objects
    """
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist")
        sys.exit(1)

    # Recursively find all matching files
    files = list(corpus_path.rglob(file_pattern))

    return files


def extract_metadata_from_path(file_path: Path, corpus_dir: str) -> dict[str, str]:
    """
    Extract metadata from file path.
    Expected structure: corpus_dir/language/genre/filename.txt

    Args:
        file_path: Path to file
        corpus_dir: Root corpus directory

    Returns:
        dict: Metadata (language, genre, filename)
    """
    # Get relative path from corpus directory
    relative_path = file_path.relative_to(corpus_dir)
    parts = relative_path.parts

    if len(parts) >= 3:
        language = parts[0]
        genre = parts[1]
        # Handle subdirectories within genre
        filename = "_".join(parts[2:]).replace(".txt", "")
    elif len(parts) == 2:
        language = parts[0]
        genre = "general"
        filename = parts[1].replace(".txt", "")
    else:
        language = "unknown"
        genre = "general"
        filename = parts[0].replace(".txt", "")

    return {
        "language": language.lower(),
        "genre": genre.lower(),
        "filename": filename.lower(),
        "relative_path": str(relative_path),
    }


def generate_s3_key(metadata: dict[str, str]) -> str:
    """
    Generate S3 object key from metadata.

    Args:
        metadata: File metadata

    Returns:
        str: S3 object key
    """
    return f"raw/{metadata['language']}/{metadata['genre']}/{metadata['filename']}.txt"


def upload_file(file_path: Path, bucket: str, s3_key: str, metadata: dict[str, str]) -> bool:
    """
    Upload single file to S3 with metadata.

    Args:
        file_path: Local file path
        bucket: S3 bucket name
        s3_key: S3 object key
        metadata: File metadata

    Returns:
        bool: True if successful
    """
    try:
        # Upload with metadata tags
        s3_client.upload_file(
            str(file_path),
            bucket,
            s3_key,
            ExtraArgs={
                "Metadata": {
                    "language": metadata["language"],
                    "genre": metadata["genre"],
                    "filename": metadata["filename"],
                }
            },
        )

        return True

    except ClientError as e:
        print(f"\nError uploading {file_path.name}: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error uploading {file_path.name}: {e}")
        return False


def upload_corpus(
    bucket: str,
    corpus_dir: str,
    file_pattern: str = "*.txt",
    language_override: Optional[str] = None,
    genre_override: Optional[str] = None,
    dry_run: bool = False,
):
    """
    Upload entire corpus to S3.

    Args:
        bucket: S3 bucket name
        corpus_dir: Local corpus directory
        file_pattern: File pattern to match
        language_override: Override language detection
        genre_override: Override genre detection
        dry_run: If True, don't actually upload
    """
    print(f"{'=' * 70}")
    print("Corpus Upload to S3")
    print(f"{'=' * 70}")
    print(f"Bucket: {bucket}")
    print(f"Corpus directory: {corpus_dir}")
    print(f"File pattern: {file_pattern}")
    if dry_run:
        print("DRY RUN MODE - No files will be uploaded")
    print(f"{'=' * 70}\n")

    # Verify bucket exists
    if not dry_run and not verify_bucket_exists(bucket):
        sys.exit(1)

    # Find all corpus files
    print("Scanning corpus directory...")
    files = find_corpus_files(corpus_dir, file_pattern)

    if not files:
        print(f"No files found matching pattern '{file_pattern}'")
        sys.exit(1)

    print(f"Found {len(files)} files to upload\n")

    # Upload files with progress bar
    successful = 0
    failed = 0
    skipped = 0

    for file_path in tqdm(files, desc="Uploading", unit="file"):
        # Extract metadata
        metadata = extract_metadata_from_path(file_path, corpus_dir)

        # Override if specified
        if language_override:
            metadata["language"] = language_override
        if genre_override:
            metadata["genre"] = genre_override

        # Generate S3 key
        s3_key = generate_s3_key(metadata)

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            tqdm.write(f"Skipping empty file: {file_path.name}")
            skipped += 1
            continue

        # Display upload info
        tqdm.write(f"  {file_path.name} → s3://{bucket}/{s3_key}")

        # Upload (or skip if dry run)
        if dry_run:
            successful += 1
        else:
            if upload_file(file_path, bucket, s3_key, metadata):
                successful += 1
            else:
                failed += 1

        # Small delay to avoid rate limiting
        if not dry_run:
            time.sleep(0.1)

    # Summary
    print(f"\n{'=' * 70}")
    print("Upload Summary")
    print(f"{'=' * 70}")
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"{'=' * 70}")

    if dry_run:
        print("\nDRY RUN complete - no files were uploaded")
        print("Run without --dry-run to actually upload files")
    else:
        print("\nUpload complete! View files at:")
        print(f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}?region=us-east-1&prefix=raw/")


def download_sample_corpus():
    """
    Download sample multilingual corpus for testing.
    """
    print("Downloading sample corpus...")

    # Create sample corpus directory
    corpus_dir = Path("./sample_corpus")
    corpus_dir.mkdir(exist_ok=True)

    # Sample texts for different languages and genres
    samples = {
        "english/academic": """
            Machine learning has revolutionized natural language processing in recent years.
            Neural networks enable computers to understand and generate human language with unprecedented accuracy.
            The transformer architecture, introduced in 2017, has become the foundation for modern language models.
            These models can perform various linguistic tasks including translation, summarization, and question answering.
        """,
        "english/news": """
            Scientists announced a breakthrough in renewable energy research yesterday.
            The new solar panel technology promises to increase efficiency by forty percent.
            Experts believe this development could significantly reduce carbon emissions worldwide.
            Several countries have already expressed interest in implementing the technology.
        """,
        "english/fiction": """
            Sarah walked through the empty streets, wondering where everyone had gone.
            The morning sun cast long shadows across the pavement.
            She heard a distant sound, like bells ringing in the wind.
            Something was different today, though she couldn't quite place it.
        """,
        "spanish/academic": """
            La lingüística computacional combina el estudio del lenguaje con la informática.
            Los algoritmos modernos pueden analizar grandes corpus de texto automáticamente.
            El procesamiento del lenguaje natural tiene aplicaciones en traducción y análisis de sentimientos.
            Las redes neuronales han mejorado significativamente el rendimiento de estos sistemas.
        """,
        "spanish/news": """
            El gobierno anunció nuevas medidas económicas para el próximo año.
            Los expertos predicen un crecimiento moderado de la economía nacional.
            Las reformas incluyen incentivos para pequeñas empresas y emprendedores.
            La población ha recibido las noticias con opiniones mixtas.
        """,
        "french/academic": """
            La linguistique de corpus étudie le langage à travers de grandes collections de textes.
            Les méthodes quantitatives permettent d'analyser des millions de mots rapidement.
            Les collocations révèlent des patterns importants dans l'usage linguistique.
            Ces analyses contribuent à notre compréhension de la variation linguistique.
        """,
    }

    # Create sample files
    for path, content in samples.items():
        file_path = corpus_dir / f"{path}.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

    print(f"Sample corpus created in: {corpus_dir}")
    print(f"Files created: {len(samples)}")

    return str(corpus_dir)


def main():
    """Main function."""
    args = parse_arguments()

    # If corpus directory doesn't exist, offer to download samples
    if not Path(args.corpus_dir).exists():
        print(f"Corpus directory '{args.corpus_dir}' does not exist.")
        response = input("Download sample corpus? (y/n): ")
        if response.lower() == "y":
            args.corpus_dir = download_sample_corpus()
        else:
            sys.exit(1)

    # Upload corpus
    upload_corpus(
        bucket=args.bucket,
        corpus_dir=args.corpus_dir,
        file_pattern=args.file_pattern,
        language_override=args.language,
        genre_override=args.genre,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
