from transforms.api import Pipeline
import prediction_pipelines

prediction_pipeline = Pipeline()
prediction_pipeline.discover_transforms(prediction_pipelines)
