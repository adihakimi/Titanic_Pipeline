import os
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

def get_training_step(role, sagemaker_session, bucket, prefix):

    estimator = SKLearn(
        entry_point=os.path.join("training", "train.py"),
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="titanic-training",
        sagemaker_session=sagemaker_session
    )

    train_input = TrainingInput(
        s3_data=f"s3://{bucket}/{prefix}/data/processed/train/",
        content_type="csv"
    )
    test_input = TrainingInput(
        s3_data=f"s3://{bucket}/{prefix}/data/processed/test/",
        content_type="csv"
    )

    step = TrainingStep(
        name="TitanicTraining",
        estimator=estimator,
        inputs={
            "train": train_input,
            "test": test_input
        }
    )

    return step
