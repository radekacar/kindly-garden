from minio import Minio
import json
from pathlib import Path
import re
import os
import time

# Load credentials from 'credentials.json'
with open('credentials.json') as f:
    creds = json.load(f)

client = Minio(
    creds['url'].replace('https://', ''),  # Remove protocol for Minio client
    access_key=creds['accessKey'],
    secret_key=creds['secretKey'],
    secure=True  # Use HTTPS
)

team_name = 'kindly-garden'

# Find the bucket containing the team_name
bucket_name = None
for bucket in client.list_buckets():
    if team_name in bucket.name:
        bucket_name = bucket.name
        print(f"Selected bucket: {bucket_name}")
        break

if not bucket_name:
    raise ValueError(f"No bucket found containing '{team_name}'")

# Upload final result, choose one best file
file_path = 'models/lgbm_fuel_burn_model_04_002_final_prediction.parquet'
client.fput_object(bucket_name, f'{team_name}_final.parquet', file_path)
print(f'Uploaded {team_name}_final.parquet to bucket {bucket_name}')

import time
time.sleep(5)

# Download the specific result JSON file
result_json_name = f'{team_name}_final.parquet_result.json'
result_json_path = os.path.join('models', result_json_name)
try:
    client.fget_object(bucket_name, result_json_name, result_json_path)
    with open(result_json_path) as f:
        result = json.load(f)
    if result.get('status') == 'Succeeded':
        print(result['status'])
except Exception as e:
    print(f'Error downloading {result_json_name}: {e}')