from sklearn.linear_model import LogisticRegression
from models.sklearn import classification_model


class LogisticRegressionModel(classification_model.ClassificationModel):
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns):
        model_name = "Logistic Regression"
        lr_model = LogisticRegression(penalty="l1", solver="saga", max_iter=500)
        param_grid = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
        super().__init__(self, model_name, lr_model, param_grid, dataset, n_iterations,
                         train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns)
