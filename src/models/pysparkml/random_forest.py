from models.pysparkml import classification_model
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder


class RandomForestModel(classification_model.ClassificationModel):
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns,
                 index_column_name, label_column_name):
        model_name = "Random Forest"

        rf_model = RandomForestClassifier(featuresCol='features', labelCol='label')
        param_grid = ParamGridBuilder().addGrid(rf_model.numTrees, [10, 50, 100, 500, 1000]).addGrid(
            rf_model.maxDepth, [None, 3, 5, 7, 9]).build()

        classification_model.ClassificationModel.__init__(self, model_name, rf_model, param_grid,
                                                          dataset, n_iterations, train_test_ratio,
                                                          preprocess_exclude_columns,
                                                          index_column_name, label_column_name)
