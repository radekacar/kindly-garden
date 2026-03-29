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

# Find largest index in existing parquet files
max_index = 0
pattern = re.compile(rf"{team_name}_v(\d+)\.parquet$")
for obj in client.list_objects(bucket_name):
    if obj.object_name.endswith('.parquet'):
        match = pattern.match(obj.object_name)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx

print(f"Largest index in bucket parquet files: {max_index}")

# Prepare paths
models_dir = 'models'
log_path = os.path.join(models_dir, 'uploaded_so_far_rank.txt')

# Read already uploaded files from log
uploaded_files = set()
if os.path.exists(log_path):
    with open(log_path, 'r') as log_file:
        for line in log_file:
            parts = line.strip().split(',')
            if parts:
                uploaded_files.add(parts[0])

# List all rank prediction parquet files not in log
rank_pattern = re.compile(r'(xgb|lgbm)_fuel_burn_model_\d{2}_\d{3}_rank_prediction\.parquet$')
rank_files = [f for f in os.listdir(models_dir) if rank_pattern.match(f) and f not in uploaded_files]

# Prepare upload mapping: {original: versioned}
upload_mapping = {}
version = max_index + 1
for rank_file in rank_files:
    versioned_name = f'{team_name}_v{version}.parquet'
    src = os.path.join(models_dir, rank_file)
    dst = os.path.join(models_dir, versioned_name)
    Path(dst).write_bytes(Path(src).read_bytes())
    upload_mapping[rank_file] = versioned_name
    version += 1

# Upload all versioned files
for original, versioned in upload_mapping.items():
    file_path = os.path.join(models_dir, versioned)
    client.fput_object(bucket_name, versioned, file_path)
    print(f'Uploaded {versioned} to bucket {bucket_name}')

# Pause for 30 seconds
import time
time.sleep(30)

# Download all result JSON files and log results
with open(log_path, 'a') as log_file:
    for original, versioned in upload_mapping.items():
        result_json_name = f'{versioned}_result.json'
        result_json_path = os.path.join(models_dir, result_json_name)
        try:
            client.fget_object(bucket_name, result_json_name, result_json_path)
            with open(result_json_path) as f:
                result = json.load(f)
            if result.get('status') == 'Succeeded':
                score = result.get('score')
                log_file.write(f'{original},{versioned},{score}\n')
                print(f'Logged: {original}, {versioned}, {score}')
            else:
                log_file.write(f'{original},{versioned},FAILED\n')
        except Exception as e:
            log_file.write(f'{original},{versioned},ERROR\n')
            print(f'Error downloading {result_json_name}: {e}')
