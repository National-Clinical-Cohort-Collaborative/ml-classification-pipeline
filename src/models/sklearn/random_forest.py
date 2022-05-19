from sklearn.ensemble import RandomForestClassifier
from models.sklearn import classification_model


class RandomForestModel(classification_model.ClassificationModel):

    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns):
        model_name = "Random Forest"
        rf_model = RandomForestClassifier(class_weight="balanced")
        param_grid = {"n_estimators": [10, 50, 100, 500, 1000], "max_depth": [None, 3, 5, 7, 9]}
        super().__init__(self, model_name, rf_model, param_grid, dataset, n_iterations,
                         train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns)
