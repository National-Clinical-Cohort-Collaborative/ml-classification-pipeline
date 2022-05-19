from models.pysparkml import classification_model
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.tuning import ParamGridBuilder


class MultiLayerPerceptronModel(classification_model.ClassificationModel):
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns,
                 index_column_name, label_column_name, n_input_features):
        model_name = "Random Forest"

        mlp_model = MultilayerPerceptronClassifier(featuresCol='features', labelCol='label', layers=[n_input_features, 64, 32, 1])
        param_grid = ParamGridBuilder().addGrid(mlp_model.blockSize, [16, 32, 64, 128, 256]).addGrid(
            mlp_model.stepSize, [0.001, 0.01, 0.1]).build()

        classification_model.ClassificationModel.__init__(self, model_name, mlp_model, param_grid,
                                                          dataset, n_iterations, train_test_ratio,
                                                          preprocess_exclude_columns,
                                                          index_column_name, label_column_name)
