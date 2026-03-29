from minio import Minio
import json
from pathlib import Path

# Load credentials from 'credentials.json'
with open('credentials.json') as f:
    creds = json.load(f)

client = Minio(
    creds['url'].replace('https://', ''),  # Remove protocol for Minio client
    access_key=creds['accessKey'],
    secret_key=creds['secretKey'],
    secure=True  # Use HTTPS
)

buckets = client.list_buckets()
for bucket in buckets:
    print(bucket.name, bucket.creation_date)

# list bucket objects
obj_list = []
for obj in client.list_objects("prc-2025-datasets", recursive=True):
    print(f"{obj.bucket_name=}, {obj.object_name=}, {obj.size}")
    obj_list.append(obj)

print(len(obj_list))

download_folder = Path("./dataset")
download_folder.mkdir(exist_ok=True)

# download all objects
for i in range(0, len(obj_list)):
    obj = obj_list[i]
    file_path = download_folder / obj.object_name
    client.fget_object(obj.bucket_name, obj.object_name, str(file_path))
    print(f"Object #{i} - {obj.object_name} downloaded.")

# Test health of downloaded parquet files
from pyarrow.parquet import ParquetFile

# Get list of all parquet files in the download folder
parquet_files = [str(p.relative_to(download_folder)) for p in download_folder.rglob('*.parquet')]

# Open the health_check.txt file in append mode
with open('health_check.txt', 'a') as f:
    # Test health of each parquet file
    for index, file in enumerate(parquet_files, start=1):
        try:
            # Try to read the file
            pf = ParquetFile(str(download_folder / file))
            # Write the result to the file
            print(f"{index}. File {file} is healthy.")
            f.write(f"{index}. File {file} is healthy.\n")
        except Exception as e:
            # Write the error to the file
            print(f"{index}. File {file} is not healthy. Error: {str(e)}")
            f.write(f"{index}. File {file} is not healthy. Error: {str(e)}\n")
