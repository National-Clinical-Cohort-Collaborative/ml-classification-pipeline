from models.pysparkml import classification_model
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder


class LogisticRegressionModel(classification_model.ClassificationModel):
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns,
                 index_column_name, label_column_name):
        model_name = "Logistic Regression"

        lr_model = LogisticRegression(maxIter=500)
        param_grid = ParamGridBuilder().addGrid(lr_model.regParam, [0.01, 0.03, 0.1, 0.3, 1, 3]).build()

        classification_model.ClassificationModel.__init__(self, model_name, lr_model, param_grid,
                                                          dataset, n_iterations, train_test_ratio,
                                                          preprocess_exclude_columns,
                                                          index_column_name, label_column_name)
