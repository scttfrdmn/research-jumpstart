#!/usr/bin/env python3
"""
Upload historical text corpus to S3 with metadata organization.

This script downloads sample texts from Project Gutenberg and uploads them
to S3 organized by author, period, and genre with appropriate metadata tags.

Usage:
    python upload_to_s3.py --bucket text-corpus-{your-id}
    python upload_to_s3.py --bucket text-corpus-{your-id} --local-corpus /path/to/texts
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import boto3
import requests
from tqdm import tqdm

# Sample corpus metadata: (gutenberg_id, author, title, period, genre)
SAMPLE_CORPUS = [
    # Jane Austen - Romantic Period
    (1342, "Jane Austen", "Pride and Prejudice", "Romantic", "Novel"),
    (161, "Jane Austen", "Sense and Sensibility", "Romantic", "Novel"),
    (158, "Jane Austen", "Emma", "Romantic", "Novel"),
    # Charles Dickens - Victorian Period
    (1400, "Charles Dickens", "Great Expectations", "Victorian", "Novel"),
    (730, "Charles Dickens", "Oliver Twist", "Victorian", "Novel"),
    (98, "Charles Dickens", "A Tale of Two Cities", "Victorian", "Novel"),
    # Brontë Sisters - Victorian Period
    (1260, "Charlotte Bronte", "Jane Eyre", "Victorian", "Novel"),
    (768, "Emily Bronte", "Wuthering Heights", "Victorian", "Novel"),
    # George Eliot - Victorian Period
    (145, "George Eliot", "Middlemarch", "Victorian", "Novel"),
    (550, "George Eliot", "The Mill on the Floss", "Victorian", "Novel"),
    # Mary Shelley - Romantic Period
    (84, "Mary Shelley", "Frankenstein", "Romantic", "Novel"),
    # Oscar Wilde - Victorian Period
    (174, "Oscar Wilde", "The Picture of Dorian Gray", "Victorian", "Novel"),
    # Thomas Hardy - Victorian Period
    (110, "Thomas Hardy", "Tess of the d'Urbervilles", "Victorian", "Novel"),
    (153, "Thomas Hardy", "Jude the Obscure", "Victorian", "Novel"),
]


class TextCorpusUploader:
    """Upload text corpus to S3 with metadata and organization."""

    def __init__(self, bucket_name: str, local_dir: str = "./corpus"):
        """
        Initialize uploader.

        Args:
            bucket_name: S3 bucket name
            local_dir: Local directory for downloaded texts
        """
        self.bucket_name = bucket_name
        self.local_dir = Path(local_dir)
        self.s3_client = boto3.client("s3")

        # Create local directory if needed
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def download_gutenberg_text(self, gutenberg_id: int, title: str) -> Optional[Path]:
        """
        Download text from Project Gutenberg.

        Args:
            gutenberg_id: Project Gutenberg ID
            title: Book title (for filename)

        Returns:
            Path to downloaded file, or None if failed
        """
        # Create safe filename
        safe_title = (
            "".join(c if c.isalnum() or c in (" ", "-") else "" for c in title)
            .strip()
            .replace(" ", "-")
            .lower()
        )
        filename = f"{safe_title}.txt"
        filepath = self.local_dir / filename

        # Skip if already downloaded
        if filepath.exists():
            print(f"  Already downloaded: {filename}")
            return filepath

        # Try different Gutenberg URL formats
        urls = [
            f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",  # UTF-8
            f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",  # ASCII
            f"https://www.gutenberg.org/ebooks/{gutenberg_id}.txt.utf-8",  # Alternative
        ]

        for url in urls:
            try:
                print(f"  Downloading: {title} from {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Save to file
                with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(response.text)

                print(f"  ✓ Downloaded: {filename} ({len(response.text)} chars)")
                return filepath

            except requests.RequestException as e:
                print(f"  Failed URL {url}: {e}")
                continue

        print(f"  ✗ Failed to download: {title}")
        return None

    def upload_text_to_s3(
        self, filepath: Path, author: str, period: str, genre: str, title: str
    ) -> bool:
        """
        Upload text file to S3 with metadata.

        Args:
            filepath: Local file path
            author: Author name
            period: Literary period (Romantic, Victorian, etc.)
            genre: Genre (Novel, Poetry, etc.)
            title: Work title

        Returns:
            True if successful, False otherwise
        """
        # Create S3 key with author organization
        safe_author = author.lower().replace(" ", "-")
        s3_key = f"raw/{safe_author}/{filepath.name}"

        try:
            # Upload with metadata
            self.s3_client.upload_file(
                str(filepath),
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    "Metadata": {
                        "author": author,
                        "title": title,
                        "period": period,
                        "genre": genre,
                        "source": "Project Gutenberg",
                    },
                    "ContentType": "text/plain",
                },
            )

            print(f"  ✓ Uploaded: s3://{self.bucket_name}/{s3_key}")
            return True

        except Exception as e:
            print(f"  ✗ Upload failed for {filepath.name}: {e}")
            return False

    def verify_bucket_exists(self) -> bool:
        """Verify S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ S3 bucket verified: {self.bucket_name}")
            return True
        except Exception as e:
            print(f"✗ Cannot access bucket {self.bucket_name}: {e}")
            print(f"  Please create bucket with: aws s3 mb s3://{self.bucket_name}")
            return False

    def create_folder_structure(self):
        """Create S3 folder structure."""
        folders = ["raw/", "processed/", "metadata/", "logs/"]

        print("Creating S3 folder structure...")
        for folder in folders:
            try:
                self.s3_client.put_object(Bucket=self.bucket_name, Key=folder)
                print(f"  ✓ Created: {folder}")
            except Exception as e:
                print(f"  ⚠ Folder {folder} may already exist: {e}")

    def upload_corpus(self, corpus_metadata: list[tuple]) -> dict[str, int]:
        """
        Upload entire corpus to S3.

        Args:
            corpus_metadata: List of (gutenberg_id, author, title, period, genre)

        Returns:
            Dict with upload statistics
        """
        stats = {"total": len(corpus_metadata), "downloaded": 0, "uploaded": 0, "failed": 0}

        print(f"\n{'=' * 70}")
        print(f"Uploading corpus of {stats['total']} texts to S3")
        print(f"Bucket: {self.bucket_name}")
        print(f"{'=' * 70}\n")

        for gutenberg_id, author, title, period, genre in tqdm(
            corpus_metadata, desc="Processing texts"
        ):
            print(f"\n{author}: {title}")

            # Download from Gutenberg
            filepath = self.download_gutenberg_text(gutenberg_id, title)

            if filepath:
                stats["downloaded"] += 1

                # Upload to S3
                if self.upload_text_to_s3(filepath, author, period, genre, title):
                    stats["uploaded"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["failed"] += 1

            # Rate limiting for Gutenberg
            time.sleep(2)

        return stats

    def upload_local_corpus(
        self, corpus_dir: Path, default_metadata: dict[str, str]
    ) -> dict[str, int]:
        """
        Upload local corpus directory to S3.

        Args:
            corpus_dir: Local directory containing text files
            default_metadata: Default metadata for all files

        Returns:
            Dict with upload statistics
        """
        stats = {"total": 0, "uploaded": 0, "failed": 0}

        # Find all text files
        text_files = list(corpus_dir.glob("**/*.txt"))
        stats["total"] = len(text_files)

        print(f"\n{'=' * 70}")
        print(f"Uploading {stats['total']} local texts to S3")
        print(f"Source: {corpus_dir}")
        print(f"Bucket: {self.bucket_name}")
        print(f"{'=' * 70}\n")

        for filepath in tqdm(text_files, desc="Uploading texts"):
            # Extract metadata from file path or use defaults
            author = default_metadata.get("author", "Unknown")
            period = default_metadata.get("period", "Unknown")
            genre = default_metadata.get("genre", "Text")
            title = filepath.stem.replace("-", " ").title()

            # If organized by author subdirectory
            if filepath.parent != corpus_dir:
                author = filepath.parent.name.replace("-", " ").title()

            if self.upload_text_to_s3(filepath, author, period, genre, title):
                stats["uploaded"] += 1
            else:
                stats["failed"] += 1

        return stats

    def list_uploaded_texts(self) -> list[dict]:
        """List all texts currently in S3 bucket."""
        texts = []

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix="raw/")

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    if obj["Key"].endswith(".txt"):
                        # Get metadata
                        try:
                            metadata = self.s3_client.head_object(
                                Bucket=self.bucket_name, Key=obj["Key"]
                            )

                            texts.append(
                                {
                                    "key": obj["Key"],
                                    "size": obj["Size"],
                                    "last_modified": obj["LastModified"],
                                    "metadata": metadata.get("Metadata", {}),
                                }
                            )
                        except Exception:
                            continue

            return texts

        except Exception as e:
            print(f"Error listing texts: {e}")
            return []


def print_summary(stats: dict[str, int], uploader: TextCorpusUploader):
    """Print upload summary."""
    print(f"\n{'=' * 70}")
    print("UPLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total texts:      {stats['total']}")
    print(f"Downloaded:       {stats.get('downloaded', 'N/A')}")
    print(f"Uploaded:         {stats['uploaded']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Success rate:     {stats['uploaded'] / stats['total'] * 100:.1f}%")
    print(f"{'=' * 70}")

    # List uploaded texts
    print("\nVerifying uploaded texts in S3...")
    texts = uploader.list_uploaded_texts()

    if texts:
        print(f"\n✓ Found {len(texts)} texts in S3:\n")

        # Group by author
        by_author = {}
        for text in texts:
            author = text["metadata"].get("author", "Unknown")
            if author not in by_author:
                by_author[author] = []
            by_author[author].append(text)

        for author, author_texts in sorted(by_author.items()):
            print(f"  {author}: {len(author_texts)} texts")
            for text in sorted(author_texts, key=lambda x: x["key"]):
                size_kb = text["size"] / 1024
                print(f"    - {text['key'].split('/')[-1]} ({size_kb:.1f} KB)")
    else:
        print("⚠ No texts found in S3. Upload may have failed.")

    print(f"\n{'=' * 70}")
    print("Next steps:")
    print("1. Process texts with Lambda: Check AWS Lambda console")
    print("2. Query results: python scripts/query_results.py")
    print("3. Analyze in Jupyter: jupyter notebook notebooks/text_analysis.ipynb")
    print(f"{'=' * 70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Upload historical text corpus to S3")
    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name (e.g., text-corpus-alice-12345)"
    )
    parser.add_argument(
        "--local-corpus", help="Path to local corpus directory (alternative to downloading)"
    )
    parser.add_argument("--author", help="Default author for local corpus files")
    parser.add_argument(
        "--period", help="Default period for local corpus files (Romantic, Victorian, etc.)"
    )
    parser.add_argument(
        "--genre", default="Novel", help="Default genre for local corpus files (default: Novel)"
    )

    args = parser.parse_args()

    # Initialize uploader
    uploader = TextCorpusUploader(bucket_name=args.bucket, local_dir="./corpus")

    # Verify bucket exists
    if not uploader.verify_bucket_exists():
        sys.exit(1)

    # Create folder structure
    uploader.create_folder_structure()

    # Upload corpus
    if args.local_corpus:
        # Upload local corpus
        corpus_dir = Path(args.local_corpus)
        if not corpus_dir.exists():
            print(f"Error: Local corpus directory not found: {corpus_dir}")
            sys.exit(1)

        default_metadata = {
            "author": args.author or "Unknown",
            "period": args.period or "Unknown",
            "genre": args.genre,
        }

        stats = uploader.upload_local_corpus(corpus_dir, default_metadata)
    else:
        # Download and upload from Gutenberg
        stats = uploader.upload_corpus(SAMPLE_CORPUS)

    # Print summary
    print_summary(stats, uploader)


if __name__ == "__main__":
    main()
