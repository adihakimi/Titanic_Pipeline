from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
import sagemaker

bucket = "titanicdataset3"
s3_input_path = "s3://titanicdataset3/titanic.csv"
role = get_execution_role()

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="titanic-preprocessing"
)

sklearn_processor.run(
    code="processing/preprocessing.py",
    arguments=[
        "--input-data", "/opt/ml/processing/input/titanic.csv",
        "--output-data", "/opt/ml/processing/train",
        "--test-data", "/opt/ml/processing/test"
    ],
    inputs=[
        ProcessingInput(
            source=s3_input_path,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train",
            destination=f"s3://{bucket}/train"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test",
            destination=f"s3://{bucket}/test"
        )
    ]
)
