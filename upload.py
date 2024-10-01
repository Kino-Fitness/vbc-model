import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('./.env')

base_directory_path = "./saved/models/"
bucket_name = "ml-models-kino"
s3_subdirectory = "vbc/"

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)

# Walk through the base directory to upload files
for root, dirs, files in os.walk(base_directory_path):
    for file_name in files:
        local_path = os.path.join(root, file_name)

        # Preserve the directory structure after base_directory_path and prepend s3_subdirectory
        s3_key = os.path.join(s3_subdirectory, os.path.relpath(local_path, base_directory_path))
        
        try:
            # Upload file to S3
            s3_client.upload_file(local_path, bucket_name, s3_key)
            print(f"File '{local_path}' uploaded to '{bucket_name}/{s3_key}'")
        except FileNotFoundError:
            print(f"The file '{local_path}' was not found.")
        except NoCredentialsError:
            print("Credentials not available.")
