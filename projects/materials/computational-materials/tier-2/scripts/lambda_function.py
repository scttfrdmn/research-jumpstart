#!/usr/bin/env python3
"""
AWS Lambda function for processing crystal structure files.

This function:
1. Triggered by S3 upload of CIF/POSCAR files
2. Parses crystal structure and calculates properties
3. Stores results in DynamoDB

Deploy to AWS Lambda:
- Runtime: Python 3.11
- Handler: lambda_function.lambda_handler
- Timeout: 60 seconds
- Memory: 512 MB
- Environment: BUCKET_NAME, DYNAMODB_TABLE, AWS_REGION
"""

import json
import boto3
import logging
import os
import re
import traceback
from datetime import datetime
from io import StringIO
from decimal import Decimal

# Initialize clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    AWS Lambda handler for processing crystal structures.

    Args:
        event (dict): S3 event trigger
        context (LambdaContext): Lambda context

    Returns:
        dict: Response with status and message
    """
    try:
        logger.info("Crystal structure processing Lambda started")
        logger.info(f"Event: {json.dumps(event)}")

        # Get configuration from environment
        bucket_name = os.environ.get('BUCKET_NAME', 'materials-data')
        table_name = os.environ.get('DYNAMODB_TABLE', 'MaterialsProperties')
        region = os.environ.get('AWS_REGION', 'us-east-1')

        # Parse S3 event
        if 'Records' in event:
            record = event['Records'][0]
            s3_bucket = record['s3']['bucket']['name']
            s3_key = record['s3']['object']['key']
        else:
            # Direct invocation for testing
            s3_bucket = event.get('bucket', bucket_name)
            s3_key = event.get('key', 'structures/test.cif')

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

        # Determine file type and process
        try:
            if s3_key.lower().endswith('.cif'):
                properties = process_cif_file(file_data, s3_key)
            elif 'POSCAR' in s3_key or 'CONTCAR' in s3_key or s3_key.lower().endswith('.vasp'):
                properties = process_poscar_file(file_data, s3_key)
            else:
                error_msg = f"Unsupported file type: {s3_key}"
                logger.error(error_msg)
                return error_response(error_msg, s3_key)
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_response(error_msg, s3_key)

        # Store results in DynamoDB
        try:
            store_in_dynamodb(properties, table_name)
            logger.info(f"Results stored in DynamoDB table: {table_name}")
        except Exception as e:
            error_msg = f"Failed to store in DynamoDB: {str(e)}"
            logger.error(error_msg)
            return error_response(error_msg, s3_key)

        # Return success
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing completed successfully',
                'material_id': properties['material_id'],
                'formula': properties['formula'],
                'density': properties['density'],
                'space_group': properties.get('space_group', 'Unknown'),
                'input_file': s3_key,
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


def process_cif_file(file_data, filename):
    """
    Process CIF file and extract crystal properties.

    Args:
        file_data (str): CIF file content
        filename (str): Original filename

    Returns:
        dict: Computed properties
    """
    logger.info(f"Processing CIF file: {filename}")

    # Parse CIF file (basic parser - doesn't require pymatgen)
    lines = file_data.split('\n')

    # Initialize properties
    properties = {
        'material_id': extract_material_id(filename),
        's3_key': filename,
        'file_type': 'cif',
        'processed_at': datetime.utcnow().isoformat()
    }

    # Extract cell parameters
    cell_params = {}
    for line in lines:
        line = line.strip()
        if line.startswith('_cell_length_a'):
            cell_params['a'] = extract_number(line)
        elif line.startswith('_cell_length_b'):
            cell_params['b'] = extract_number(line)
        elif line.startswith('_cell_length_c'):
            cell_params['c'] = extract_number(line)
        elif line.startswith('_cell_angle_alpha'):
            cell_params['alpha'] = extract_number(line)
        elif line.startswith('_cell_angle_beta'):
            cell_params['beta'] = extract_number(line)
        elif line.startswith('_cell_angle_gamma'):
            cell_params['gamma'] = extract_number(line)
        elif line.startswith('_symmetry_space_group_name_H-M'):
            properties['space_group'] = line.split("'")[1] if "'" in line else "Unknown"
        elif line.startswith('_symmetry_Int_Tables_number'):
            properties['space_group_number'] = int(extract_number(line))

    # Calculate volume (for orthogonal cells only - simplified)
    if all(k in cell_params for k in ['a', 'b', 'c']):
        properties['lattice_a'] = cell_params['a']
        properties['lattice_b'] = cell_params['b']
        properties['lattice_c'] = cell_params['c']

        # Simple volume calculation (assumes orthogonal for simplicity)
        properties['volume'] = cell_params['a'] * cell_params['b'] * cell_params['c']
    else:
        properties['volume'] = 0.0
        logger.warning("Incomplete cell parameters")

    # Extract atomic positions and count atoms
    in_loop = False
    atom_count = 0
    elements = []

    for line in lines:
        line = line.strip()
        if line.startswith('loop_'):
            in_loop = True
        elif in_loop and line.startswith('_atom_site_'):
            continue
        elif in_loop and line and not line.startswith('_') and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                element = re.sub(r'[0-9]+', '', parts[1])
                elements.append(element)
                atom_count += 1
        elif in_loop and (not line or line.startswith('_') or line.startswith('loop_')):
            in_loop = False

    properties['num_atoms'] = atom_count
    properties['formula'] = generate_formula(elements)

    # Estimate density (simplified - assumes average atomic mass)
    if properties['volume'] > 0 and atom_count > 0:
        avg_mass = estimate_mass(elements)
        # Density = (mass / volume) * (1.66054 / 1e-24) conversion factor
        properties['density'] = round((avg_mass * atom_count) / (properties['volume'] * 1e-24) * 1.66054, 2)
    else:
        properties['density'] = 0.0

    # Determine crystal system
    properties['crystal_system'] = determine_crystal_system(
        cell_params.get('a', 0),
        cell_params.get('b', 0),
        cell_params.get('c', 0),
        cell_params.get('alpha', 90),
        cell_params.get('beta', 90),
        cell_params.get('gamma', 90)
    )

    logger.info(f"CIF properties: {properties}")
    return properties


def process_poscar_file(file_data, filename):
    """
    Process POSCAR/VASP file and extract crystal properties.

    Args:
        file_data (str): POSCAR file content
        filename (str): Original filename

    Returns:
        dict: Computed properties
    """
    logger.info(f"Processing POSCAR file: {filename}")

    lines = file_data.strip().split('\n')

    properties = {
        'material_id': extract_material_id(filename),
        's3_key': filename,
        'file_type': 'poscar',
        'processed_at': datetime.utcnow().isoformat()
    }

    try:
        # Line 0: comment (often contains formula)
        properties['comment'] = lines[0].strip()

        # Line 1: universal scaling factor
        scale = float(lines[1].strip())

        # Lines 2-4: lattice vectors
        a = [float(x) * scale for x in lines[2].split()]
        b = [float(x) * scale for x in lines[3].split()]
        c = [float(x) * scale for x in lines[4].split()]

        # Calculate lattice parameters
        import math
        properties['lattice_a'] = math.sqrt(sum(x**2 for x in a))
        properties['lattice_b'] = math.sqrt(sum(x**2 for x in b))
        properties['lattice_c'] = math.sqrt(sum(x**2 for x in c))

        # Calculate volume (cross product magnitude)
        cross = [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]
        properties['volume'] = abs(sum(cross[i] * c[i] for i in range(3)))

        # Line 5: element names
        elements = lines[5].split()

        # Line 6: number of atoms per element
        atom_counts = [int(x) for x in lines[6].split()]

        properties['num_atoms'] = sum(atom_counts)
        properties['formula'] = ''.join(f"{el}{cnt}" for el, cnt in zip(elements, atom_counts))

        # Estimate density
        all_elements = []
        for el, cnt in zip(elements, atom_counts):
            all_elements.extend([el] * cnt)

        if properties['volume'] > 0:
            avg_mass = estimate_mass(all_elements)
            properties['density'] = round((avg_mass * properties['num_atoms']) /
                                         (properties['volume'] * 1e-24) * 1.66054, 2)
        else:
            properties['density'] = 0.0

        # Determine crystal system (simplified)
        properties['crystal_system'] = determine_crystal_system(
            properties['lattice_a'],
            properties['lattice_b'],
            properties['lattice_c'],
            90, 90, 90  # Simplified - assumes orthogonal
        )

        logger.info(f"POSCAR properties: {properties}")
        return properties

    except Exception as e:
        logger.error(f"Error parsing POSCAR: {e}")
        raise


def extract_material_id(filename):
    """Extract material ID from filename."""
    # Remove path and extension
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]

    # Look for mp-XXXX pattern (Materials Project)
    match = re.search(r'mp-\d+', name)
    if match:
        return match.group()

    # Otherwise use filename
    return name


def extract_number(line):
    """Extract floating point number from CIF line."""
    parts = line.split()
    if len(parts) >= 2:
        # Remove uncertainty notation e.g., 3.867(2) -> 3.867
        value = parts[1].split('(')[0]
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def generate_formula(elements):
    """Generate chemical formula from list of elements."""
    from collections import Counter
    counts = Counter(elements)
    formula = ''.join(f"{el}{cnt if cnt > 1 else ''}" for el, cnt in sorted(counts.items()))
    return formula or "Unknown"


def estimate_mass(elements):
    """Estimate average atomic mass."""
    # Simplified atomic masses (g/mol)
    masses = {
        'H': 1, 'C': 12, 'N': 14, 'O': 16, 'F': 19,
        'Na': 23, 'Mg': 24, 'Al': 27, 'Si': 28, 'P': 31,
        'S': 32, 'Cl': 35, 'K': 39, 'Ca': 40, 'Fe': 56,
        'Cu': 64, 'Zn': 65, 'Ga': 70, 'Ge': 73, 'As': 75,
        'Se': 79, 'Br': 80, 'Sr': 88, 'Y': 89, 'Zr': 91,
        'Mo': 96, 'Ag': 108, 'Cd': 112, 'In': 115, 'Sn': 119,
        'Sb': 122, 'Te': 128, 'I': 127, 'Ba': 137, 'La': 139,
        'Ce': 140, 'Nd': 144, 'Gd': 157, 'Dy': 163, 'Yb': 173,
        'Lu': 175, 'Hf': 178, 'Ta': 181, 'W': 184, 'Pt': 195,
        'Au': 197, 'Hg': 201, 'Pb': 207, 'Bi': 209
    }

    if not elements:
        return 28.0  # Default to Si

    total_mass = sum(masses.get(el, 28.0) for el in elements)
    return total_mass / len(elements)


def determine_crystal_system(a, b, c, alpha, beta, gamma, tol=0.1):
    """Determine crystal system from lattice parameters."""
    def approx_equal(x, y, tol=tol):
        return abs(x - y) < tol

    # Cubic
    if approx_equal(a, b) and approx_equal(b, c) and \
       approx_equal(alpha, 90) and approx_equal(beta, 90) and approx_equal(gamma, 90):
        return "cubic"

    # Hexagonal/Trigonal
    if approx_equal(a, b) and approx_equal(alpha, 90) and \
       approx_equal(beta, 90) and approx_equal(gamma, 120):
        return "hexagonal"

    # Tetragonal
    if approx_equal(a, b) and approx_equal(alpha, 90) and \
       approx_equal(beta, 90) and approx_equal(gamma, 90):
        return "tetragonal"

    # Orthorhombic
    if approx_equal(alpha, 90) and approx_equal(beta, 90) and approx_equal(gamma, 90):
        return "orthorhombic"

    # Monoclinic
    if approx_equal(alpha, 90) and approx_equal(gamma, 90):
        return "monoclinic"

    # Triclinic (most general)
    return "triclinic"


def store_in_dynamodb(properties, table_name):
    """
    Store computed properties in DynamoDB.

    Args:
        properties (dict): Material properties
        table_name (str): DynamoDB table name
    """
    table = dynamodb.Table(table_name)

    # Convert floats to Decimal for DynamoDB
    item = {}
    for key, value in properties.items():
        if isinstance(value, float):
            item[key] = Decimal(str(value))
        else:
            item[key] = value

    table.put_item(Item=item)
    logger.info(f"Stored item with material_id: {properties['material_id']}")


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
    # Set environment variables
    os.environ['BUCKET_NAME'] = 'materials-data-test'
    os.environ['DYNAMODB_TABLE'] = 'MaterialsProperties'
    os.environ['AWS_REGION'] = 'us-east-1'

    # Create test event
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'materials-data-test'},
                    'object': {'key': 'structures/Si.cif'}
                }
            }
        ]
    }

    class MockContext:
        pass

    # Test handler (will fail without actual S3 bucket)
    print("Testing lambda_handler...")
    print("Note: This will fail without a valid S3 bucket and DynamoDB table")

    try:
        result = lambda_handler(test_event, MockContext())
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
