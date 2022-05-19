from models.pysparkml import classification_model
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder


class SupportVectorMachineModel(classification_model.ClassificationModel):
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns,
                 index_column_name, label_column_name):
        model_name = "Support Vector Machine"

        svm_model = LinearSVC(featuresCol='features', labelCol='label')
        # todo: convert to non-linear SVC (add kernel)
        param_grid = ParamGridBuilder().addGrid(svm_model.regParam, [0.1, 1, 10, 100, 1000]).build()

        classification_model.ClassificationModel.__init__(self, model_name, svm_model, param_grid,
                                                          dataset, n_iterations, train_test_ratio,
                                                          preprocess_exclude_columns,
                                                          index_column_name, label_column_name)
