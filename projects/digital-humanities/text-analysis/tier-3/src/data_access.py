"""Data access for text corpora from S3 and local sources."""

import boto3
import pandas as pd
import logging
from typing import Optional, List, Dict
import io
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TextDataAccess:
    """Handle loading and saving text documents and analysis results."""

    def __init__(self, use_anon: bool = False, region: str = 'us-east-1'):
        """
        Initialize data access client.

        Args:
            use_anon: Use anonymous access (for public datasets)
            region: AWS region
        """
        if use_anon:
            self.s3_client = boto3.client('s3', region_name=region,
                                        config=boto3.session.Config(signature_version=boto3.session.UNSIGNED))
        else:
            self.s3_client = boto3.client('s3', region_name=region)

        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.use_anon = use_anon
        logger.info(f"Initialized TextDataAccess (anon={use_anon})")

    def load_text_from_s3(self, bucket: str, key: str, encoding: str = 'utf-8') -> str:
        """
        Load text file from S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key (path to text file)
            encoding: Text encoding (default: utf-8)

        Returns:
            Text content as string
        """
        logger.info(f"Loading text from s3://{bucket}/{key}")

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            text_content = response['Body'].read().decode(encoding)
            logger.info(f"Loaded {len(text_content)} characters")
            return text_content

        except Exception as e:
            logger.error(f"Error loading text: {e}")
            raise

    def load_corpus_from_s3(
        self,
        bucket: str,
        prefix: str = '',
        file_extensions: List[str] = ['.txt', '.md']
    ) -> pd.DataFrame:
        """
        Load multiple text files from S3 into DataFrame.

        Args:
            bucket: S3 bucket name
            prefix: S3 prefix to filter files
            file_extensions: List of file extensions to include

        Returns:
            DataFrame with columns: document_id, text, s3_key
        """
        logger.info(f"Loading corpus from s3://{bucket}/{prefix}")

        documents = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                # Filter by extension
                if not any(key.endswith(ext) for ext in file_extensions):
                    continue

                try:
                    text = self.load_text_from_s3(bucket, key)
                    documents.append({
                        'document_id': Path(key).stem,
                        'text': text,
                        's3_key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {key}: {e}")
                    continue

        df = pd.DataFrame(documents)
        logger.info(f"Loaded {len(df)} documents")
        return df

    def load_text_from_local(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Load text file from local filesystem.

        Args:
            file_path: Path to local text file
            encoding: Text encoding (default: utf-8)

        Returns:
            Text content as string
        """
        logger.info(f"Loading text from {file_path}")

        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()

        logger.info(f"Loaded {len(text)} characters")
        return text

    def load_corpus_from_local(
        self,
        directory: str,
        file_extensions: List[str] = ['.txt', '.md'],
        recursive: bool = True
    ) -> pd.DataFrame:
        """
        Load text files from local directory into DataFrame.

        Args:
            directory: Local directory path
            file_extensions: List of file extensions to include
            recursive: Search subdirectories

        Returns:
            DataFrame with columns: document_id, text, file_path
        """
        logger.info(f"Loading corpus from {directory}")

        documents = []
        path = Path(directory)

        # Get files
        if recursive:
            files = [f for ext in file_extensions for f in path.rglob(f'*{ext}')]
        else:
            files = [f for ext in file_extensions for f in path.glob(f'*{ext}')]

        for file_path in files:
            try:
                text = self.load_text_from_local(str(file_path))
                documents.append({
                    'document_id': file_path.stem,
                    'text': text,
                    'file_path': str(file_path),
                    'size': file_path.stat().st_size
                })
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        df = pd.DataFrame(documents)
        logger.info(f"Loaded {len(df)} documents")
        return df

    def save_results(self, df: pd.DataFrame, bucket: str, key: str):
        """
        Save analysis results to S3.

        Args:
            df: DataFrame to save
            bucket: S3 bucket name
            key: S3 object key
        """
        logger.info(f"Saving results to s3://{bucket}/{key}")

        # Determine format from key extension
        if key.endswith('.csv'):
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            content = csv_buffer.getvalue()
        elif key.endswith('.json'):
            json_buffer = io.StringIO()
            df.to_json(json_buffer, orient='records', indent=2)
            content = json_buffer.getvalue()
        else:
            # Default to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            content = csv_buffer.getvalue()

        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content
        )

        logger.info("Results saved successfully")

    def save_model(self, model_data: bytes, bucket: str, key: str, metadata: Optional[Dict] = None):
        """
        Save trained model to S3.

        Args:
            model_data: Serialized model bytes
            bucket: S3 bucket name
            key: S3 object key
            metadata: Optional metadata dictionary
        """
        logger.info(f"Saving model to s3://{bucket}/{key}")

        put_args = {
            'Bucket': bucket,
            'Key': key,
            'Body': model_data
        }

        if metadata:
            put_args['Metadata'] = {k: str(v) for k, v in metadata.items()}

        self.s3_client.put_object(**put_args)
        logger.info("Model saved successfully")

    def load_model(self, bucket: str, key: str) -> bytes:
        """
        Load trained model from S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            Model data as bytes
        """
        logger.info(f"Loading model from s3://{bucket}/{key}")

        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        model_data = response['Body'].read()

        logger.info("Model loaded successfully")
        return model_data

    def save_metadata(
        self,
        table_name: str,
        document_id: str,
        metadata: Dict
    ):
        """
        Save document metadata to DynamoDB.

        Args:
            table_name: DynamoDB table name
            document_id: Unique document identifier
            metadata: Metadata dictionary
        """
        logger.info(f"Saving metadata for document {document_id}")

        table = self.dynamodb.Table(table_name)

        item = {
            'document_id': document_id,
            **metadata
        }

        table.put_item(Item=item)
        logger.info("Metadata saved successfully")

    def get_metadata(
        self,
        table_name: str,
        document_id: str,
        upload_date: str
    ) -> Dict:
        """
        Retrieve document metadata from DynamoDB.

        Args:
            table_name: DynamoDB table name
            document_id: Unique document identifier
            upload_date: Upload date (sort key)

        Returns:
            Metadata dictionary
        """
        logger.info(f"Retrieving metadata for document {document_id}")

        table = self.dynamodb.Table(table_name)

        response = table.get_item(
            Key={
                'document_id': document_id,
                'upload_date': upload_date
            }
        )

        return response.get('Item', {})

    def list_text_files(self, bucket: str, prefix: str = '') -> List[str]:
        """
        List text files in S3 bucket.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix to filter

        Returns:
            List of S3 keys for text files
        """
        logger.info(f"Listing text files in s3://{bucket}/{prefix}")

        text_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        text_extensions = ['.txt', '.md', '.csv', '.json', '.xml', '.html']

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if any(obj['Key'].endswith(ext) for ext in text_extensions):
                        text_files.append(obj['Key'])

        logger.info(f"Found {len(text_files)} text files")
        return text_files
