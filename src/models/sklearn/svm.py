from sklearn.svm import SVC
from models.sklearn import classification_model


class SupportVectorMachineModel(classification_model.ClassificationModel):
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns):
        model_name = "Support Vector Machine"
        svm_model = SVC(kernel="linear", probability=True, max_iter=500)
        param_grid = {'C': [0.1, 1, 10, 100, 1000], "gamma": [0.001, 0.0001]}
        super().__init__(self, model_name, svm_model, param_grid, dataset, n_iterations,
                         train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns)
