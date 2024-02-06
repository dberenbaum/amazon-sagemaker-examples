import os
import sagemaker
from sagemaker.pytorch import PyTorch
from torchvision.datasets import MNIST
from torchvision import transforms

sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name

bucket = sagemaker_session.default_bucket()
prefix = "sagemaker/DEMO-pytorch-mnist"

role = sagemaker.get_execution_role()

MNIST.mirrors = [
    f"https://sagemaker-example-files-prod-{region}.s3.amazonaws.com/datasets/image/MNIST/"
]

MNIST(
    "data",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

inputs = sagemaker_session.upload_data(path="data", bucket=bucket, key_prefix=prefix)
print("input spec (in this case, just an S3 path): {}".format(inputs))

env = {name: value for name, value in os.environ.items() if name.startswith("DVC")}

estimator = PyTorch(
    entry_point="mnist.py",
    role=role,
    py_version="py38",
    framework_version="1.11.0",
    instance_count=2,
    instance_type="ml.c5.2xlarge",
    hyperparameters={"epochs": 1, "backend": "gloo"},
    source_dir=".",
    environment=env,
)

estimator.fit({"training": inputs})