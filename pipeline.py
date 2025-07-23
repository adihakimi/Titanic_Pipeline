import boto3
import sagemaker
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from steps.preprocessing_step import get_preprocessing_step

# Init
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket() ##consider changing this to a specific bucket
prefix = "mlops-pipeline"

# Params
input_data_param = ParameterString(
    name="InputDataUrl",
    default_value=f"s3://{bucket}/{prefix}/data/raw/titanic.csv"
)

# Get steps
preprocessing_step = get_preprocessing_step(
    role=role,
    input_data_param=input_data_param,
    sagemaker_session=session,
    bucket=bucket,
    prefix=prefix
)

# Create the pipeline
pipeline = Pipeline(
    name="TitanicPipeline",
    parameters=[input_data_param],
    steps=[preprocessing_step],
    sagemaker_session=session
)
