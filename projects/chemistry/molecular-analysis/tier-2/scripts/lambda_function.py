#!/usr/bin/env python3
"""
AWS Lambda function for molecular property calculation.

This function:
1. Triggered by S3 upload of molecular structure files
2. Parses SMILES strings and calculates molecular properties
3. Stores results in DynamoDB

Deploy to AWS Lambda:
- Runtime: Python 3.11
- Handler: lambda_function.lambda_handler
- Timeout: 30 seconds
- Memory: 256 MB
- Environment: BUCKET_NAME, DYNAMODB_TABLE, AWS_REGION
"""

import json
import boto3
import logging
import os
import re
import traceback
from datetime import datetime
from decimal import Decimal

# Initialize clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Try to import RDKit (may not be available in Lambda without layer)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    RDKIT_AVAILABLE = True
    logger.info("RDKit is available")
except ImportError:
    logger.warning("RDKit not available - using pure Python calculations")
    RDKIT_AVAILABLE = False


def lambda_handler(event, context):
    """
    AWS Lambda handler for molecular property calculation.

    Args:
        event (dict): S3 event trigger or direct invocation
        context (LambdaContext): Lambda context

    Returns:
        dict: Response with status and message
    """
    try:
        logger.info("Molecular property analysis Lambda started")
        logger.info(f"Event: {json.dumps(event)}")

        # Get configuration from environment
        bucket_name = os.environ.get('BUCKET_NAME', 'molecular-data')
        table_name = os.environ.get('DYNAMODB_TABLE', 'MolecularProperties')
        region = os.environ.get('AWS_REGION', 'us-east-1')

        # Parse S3 event
        if 'Records' in event:
            record = event['Records'][0]
            s3_bucket = record['s3']['bucket']['name']
            s3_key = record['s3']['object']['key']
        else:
            # Direct invocation
            s3_bucket = event.get('bucket', bucket_name)
            s3_key = event.get('key', 'molecules/test.smi')

        logger.info(f"Processing file: s3://{s3_bucket}/{s3_key}")

        # Download file from S3
        try:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            file_data = obj['Body'].read().decode('utf-8')
            logger.info(f"Downloaded {len(file_data)} bytes from S3")
        except Exception as e:
            error_msg = f"Failed to download from S3: {str(e)}"
            logger.error(error_msg)
            return error_response(error_msg, s3_key)

        # Process molecules
        try:
            molecules = parse_molecules(file_data, s3_key)
            logger.info(f"Parsed {len(molecules)} molecules")

            # Calculate properties and store in DynamoDB
            table = dynamodb.Table(table_name)
            processed_count = 0
            error_count = 0

            for mol_data in molecules:
                try:
                    properties = calculate_properties(mol_data)
                    store_properties(table, properties)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing molecule {mol_data.get('name')}: {e}")
                    error_count += 1

            logger.info(f"Processed {processed_count} molecules, {error_count} errors")

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_response(error_msg, s3_key)

        # Return success
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing completed successfully',
                'input_file': s3_key,
                'molecules_processed': processed_count,
                'molecules_failed': error_count,
                'timestamp': datetime.utcnow().isoformat()
            })
        }

        logger.info(f"Success: {response['body']}")
        return response

    except Exception as e:
        error_msg = f"Unhandled error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_response(error_msg)


def parse_molecules(file_data, filename):
    """
    Parse molecular structure file.

    Args:
        file_data (str): File content
        filename (str): Original filename

    Returns:
        list: List of molecule dictionaries
    """
    molecules = []

    # Determine file type from extension
    if filename.endswith(('.smi', '.smiles')):
        molecules = parse_smiles_file(file_data, filename)
    elif filename.endswith('.sdf'):
        molecules = parse_sdf_file(file_data, filename)
    else:
        logger.warning(f"Unsupported file type: {filename}")
        # Try to parse as SMILES anyway
        molecules = parse_smiles_file(file_data, filename)

    return molecules


def parse_smiles_file(file_data, filename):
    """
    Parse SMILES file format.

    Format: SMILES [name] [other fields...]

    Args:
        file_data (str): File content
        filename (str): Original filename

    Returns:
        list: List of molecule dictionaries
    """
    molecules = []

    # Extract compound class from filename/path
    compound_class = extract_compound_class(filename)

    for line_num, line in enumerate(file_data.split('\n'), 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Parse SMILES line
        parts = line.split()
        if len(parts) < 1:
            logger.warning(f"Line {line_num}: Empty SMILES")
            continue

        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"MOL{line_num:05d}"

        molecules.append({
            'smiles': smiles,
            'name': name,
            'compound_class': compound_class,
            'source_file': filename
        })

    return molecules


def parse_sdf_file(file_data, filename):
    """
    Parse SDF file format (simplified, without RDKit).

    Args:
        file_data (str): File content
        filename (str): Original filename

    Returns:
        list: List of molecule dictionaries
    """
    logger.warning("SDF parsing without RDKit - limited functionality")

    # For now, just log that we received an SDF file
    # Full SDF parsing would require RDKit or complex string parsing
    molecules = []

    compound_class = extract_compound_class(filename)

    # Basic SDF parsing - look for molecule names
    mol_count = file_data.count('$$$$')
    logger.info(f"SDF file contains approximately {mol_count} molecules")

    # Create placeholder entries
    for i in range(mol_count):
        molecules.append({
            'smiles': 'C',  # Placeholder
            'name': f"SDF_MOL{i+1:05d}",
            'compound_class': compound_class,
            'source_file': filename,
            'note': 'SDF parsing requires RDKit'
        })

    return molecules


def extract_compound_class(filename):
    """
    Extract compound class from filename/path.

    Args:
        filename (str): File path

    Returns:
        str: Compound class
    """
    # Extract from path: molecules/drugs/file.smi -> drugs
    parts = filename.split('/')
    if len(parts) >= 3 and parts[0] == 'molecules':
        return parts[1]

    # Check filename for keywords
    filename_lower = filename.lower()
    if 'drug' in filename_lower:
        return 'drug'
    elif 'natural' in filename_lower:
        return 'natural_product'
    elif 'screen' in filename_lower or 'library' in filename_lower:
        return 'screening_library'
    else:
        return 'unknown'


def calculate_properties(mol_data):
    """
    Calculate molecular properties.

    Args:
        mol_data (dict): Molecule data with SMILES

    Returns:
        dict: Molecular properties
    """
    smiles = mol_data['smiles']

    if RDKIT_AVAILABLE:
        # Use RDKit for accurate calculations
        properties = calculate_properties_rdkit(smiles, mol_data)
    else:
        # Use pure Python approximations
        properties = calculate_properties_python(smiles, mol_data)

    return properties


def calculate_properties_rdkit(smiles, mol_data):
    """
    Calculate molecular properties using RDKit.

    Args:
        smiles (str): SMILES string
        mol_data (dict): Molecule metadata

    Returns:
        dict: Molecular properties
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Calculate descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)

        # Lipinski's Rule of Five
        lipinski_compliant = (
            mw <= 500 and
            logp <= 5 and
            hbd <= 5 and
            hba <= 10
        )

        properties = {
            'molecule_id': mol_data['name'],
            'compound_class': mol_data['compound_class'],
            'smiles': smiles,
            'name': mol_data['name'],
            'molecular_weight': Decimal(str(round(mw, 2))),
            'logp': Decimal(str(round(logp, 2))),
            'tpsa': Decimal(str(round(tpsa, 2))),
            'hbd': int(hbd),
            'hba': int(hba),
            'rotatable_bonds': int(rotatable_bonds),
            'aromatic_rings': int(aromatic_rings),
            'lipinski_compliant': lipinski_compliant,
            'source_file': mol_data['source_file'],
            'calculation_method': 'rdkit',
            'timestamp': datetime.utcnow().isoformat()
        }

        return properties

    except Exception as e:
        logger.error(f"RDKit calculation failed: {e}")
        raise


def calculate_properties_python(smiles, mol_data):
    """
    Calculate molecular properties using pure Python (approximations).

    These are rough estimates without RDKit. Use RDKit for accurate values.

    Args:
        smiles (str): SMILES string
        mol_data (dict): Molecule metadata

    Returns:
        dict: Approximate molecular properties
    """
    logger.info("Using pure Python approximations (less accurate)")

    # Validate SMILES syntax
    if not validate_smiles(smiles):
        raise ValueError(f"Invalid SMILES syntax: {smiles}")

    # Count atoms and estimate properties
    atom_counts = count_atoms(smiles)

    # Rough molecular weight estimate
    atomic_weights = {'C': 12, 'N': 14, 'O': 16, 'S': 32, 'P': 31, 'F': 19, 'Cl': 35.5, 'Br': 80, 'I': 127}
    mw = sum(atomic_weights.get(atom, 12) * count for atom, count in atom_counts.items())

    # Rough LogP estimate (carbon count - heteroatom count)
    logp = atom_counts.get('C', 0) * 0.5 - sum(count for atom, count in atom_counts.items() if atom in ['N', 'O']) * 0.5

    # Rough TPSA estimate (polar atoms * 20)
    tpsa = sum(count for atom, count in atom_counts.items() if atom in ['N', 'O']) * 20

    # Hydrogen bond donors/acceptors
    hbd = smiles.count('O') + smiles.count('N')  # Simplified
    hba = smiles.count('O') + smiles.count('N')

    # Rotatable bonds (approximate from single bonds)
    rotatable_bonds = smiles.count('-') + max(0, smiles.count('C') - 5)

    # Aromatic rings (count lowercase 'c' in SMILES)
    aromatic_rings = smiles.count('c') // 6  # Rough estimate

    # Lipinski's Rule of Five
    lipinski_compliant = (
        mw <= 500 and
        logp <= 5 and
        hbd <= 5 and
        hba <= 10
    )

    properties = {
        'molecule_id': mol_data['name'],
        'compound_class': mol_data['compound_class'],
        'smiles': smiles,
        'name': mol_data['name'],
        'molecular_weight': Decimal(str(round(mw, 2))),
        'logp': Decimal(str(round(logp, 2))),
        'tpsa': Decimal(str(round(tpsa, 2))),
        'hbd': int(hbd),
        'hba': int(hba),
        'rotatable_bonds': int(rotatable_bonds),
        'aromatic_rings': int(aromatic_rings),
        'lipinski_compliant': lipinski_compliant,
        'source_file': mol_data['source_file'],
        'calculation_method': 'python_approximation',
        'note': 'Approximate values - use RDKit for accurate calculations',
        'timestamp': datetime.utcnow().isoformat()
    }

    return properties


def validate_smiles(smiles):
    """
    Validate SMILES syntax (basic check).

    Args:
        smiles (str): SMILES string

    Returns:
        bool: Valid syntax
    """
    # Check for balanced parentheses and brackets
    paren_count = 0
    bracket_count = 0

    for char in smiles:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1

        if paren_count < 0 or bracket_count < 0:
            return False

    return paren_count == 0 and bracket_count == 0


def count_atoms(smiles):
    """
    Count atoms in SMILES string (simplified).

    Args:
        smiles (str): SMILES string

    Returns:
        dict: Atom counts
    """
    atom_counts = {}

    # Simple regex to find atoms
    # This is simplified and won't handle all SMILES features
    atom_pattern = r'([A-Z][a-z]?|\[.+?\])'
    atoms = re.findall(atom_pattern, smiles)

    for atom in atoms:
        # Remove brackets
        atom = atom.strip('[]')

        # Extract element symbol
        element = re.match(r'([A-Z][a-z]?)', atom)
        if element:
            element = element.group(1)
            atom_counts[element] = atom_counts.get(element, 0) + 1

    return atom_counts


def store_properties(table, properties):
    """
    Store molecular properties in DynamoDB.

    Args:
        table: DynamoDB table resource
        properties (dict): Molecular properties
    """
    try:
        table.put_item(Item=properties)
        logger.info(f"Stored properties for {properties['molecule_id']}")
    except Exception as e:
        logger.error(f"Failed to store properties: {e}")
        raise


def error_response(error_msg, filename=None):
    """
    Generate error response.

    Args:
        error_msg (str): Error message
        filename (str): Optional filename being processed

    Returns:
        dict: Error response
    """
    response = {
        'statusCode': 500,
        'body': json.dumps({
            'message': 'Processing failed',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        })
    }

    if filename:
        body = json.loads(response['body'])
        body['file'] = filename
        response['body'] = json.dumps(body)

    return response


# Local testing
if __name__ == '__main__':
    # Set environment variables for testing
    os.environ['BUCKET_NAME'] = 'molecular-data-test'
    os.environ['DYNAMODB_TABLE'] = 'MolecularProperties'
    os.environ['AWS_REGION'] = 'us-east-1'

    # Create test event
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'molecular-data-test'},
                    'object': {'key': 'molecules/drugs/test.smi'}
                }
            }
        ]
    }

    class MockContext:
        pass

    # Test with sample SMILES
    print("Testing lambda_handler with sample data...")
    print("Note: This will fail without valid S3 bucket and DynamoDB table")

    # Test pure Python calculations
    test_molecules = [
        {
            'smiles': 'CC(=O)Oc1ccccc1C(=O)O',
            'name': 'aspirin',
            'compound_class': 'drug',
            'source_file': 'test.smi'
        }
    ]

    for mol_data in test_molecules:
        try:
            props = calculate_properties(mol_data)
            print(f"\nCalculated properties for {mol_data['name']}:")
            print(f"  Molecular Weight: {props['molecular_weight']}")
            print(f"  LogP: {props['logp']}")
            print(f"  TPSA: {props['tpsa']}")
            print(f"  Lipinski Compliant: {props['lipinski_compliant']}")
        except Exception as e:
            print(f"Error: {e}")
