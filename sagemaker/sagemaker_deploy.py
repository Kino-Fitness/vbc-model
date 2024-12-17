import sagemaker
from sagemaker.pytorch import PyTorchModel
import boto3

# Role and session setup
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()

# Upload model to S3
s3 = boto3.client('s3')
bucket = sagemaker_session.default_bucket()
s3.upload_file('saved/model.pth', bucket, 'model/model.pth')

# Define SageMaker PyTorch model
pytorch_model = PyTorchModel(
    model_data=f"s3://{bucket}/model/model.pth",
    role=role,
    framework_version="1.12.0",
    py_version="py38",
    entry_point="inference.py"
)

# Deploy model to endpoint
predictor = pytorch_model.deploy(instance_type="ml.m5.large", initial_instance_count=1)

# Test inference
test_input = {'data': [1.0, 2.0, 3.0]} 
prediction = predictor.predict(test_input)
print("Prediction:", prediction)

# Delete endpoint
predictor.delete_endpoint()
