import os
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep


def get_preprocessing_step(role, input_data_param, sagemaker_session, bucket, prefix):
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="pipeline-preprocessing",
        sagemaker_session=sagemaker_session
    )

    step = ProcessingStep(
        name="TitanicPreprocessing",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=input_data_param,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{bucket}/{prefix}/data/processed/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{bucket}/{prefix}/data/processed/test"
            )
        ],
        code=os.path.join("preprocessing", "preprocessing.py"),
        job_arguments=[
            "--input-data", "/opt/ml/processing/input/titanic.csv", #might be removed in the future
            "--train-output", "/opt/ml/processing/train", #might be removed in the future
            "--test-output", "/opt/ml/processing/test" #might be removed in the future
        ]
    )

    return step
