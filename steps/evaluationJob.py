from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

def get_evaluation_step(role, sagemaker_session, model_uri, test_data_uri, destination):

    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role,
        instance_type='ml.m5.large',
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    step = ProcessingStep(
        name="EvaluationStep",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=model_uri, destination="/opt/ml/processing/model"),
            ProcessingInput(source=test_data_uri, destination="/opt/ml/processing/input/test"),
            ProcessingInput(source=destination, source="/opt/ml/processing/output"),
        
            
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation_report", source="/opt/ml/processing/output"),
        ],
        code="evaluation/evaluate.py",
    )

    return step