import data_preprocessing
from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd  # noqa


class LogisticRegressionModel:
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns):
        self.model_name = "Logistic Regression"
        self.dataset = dataset
        self.n_iterations = n_iterations
        self.dataset_preprocessing = data_preprocessing.DatasetPreprocessing(
            self.dataset, preprocess_exclude_columns, standardize_exclude_columns, train_test_ratio)
        self.model = None  # best model from cross validation, used for testing
        self.spark_obj = SparkSession.builder.appName("pandas_to_spark").getOrCreate()

    def run_iteration(self, X_train, y_train, X_test, y_test, itr):
        X_train = X_train.toPandas()
        y_train = y_train.toPandas()
        X_test = X_test.toPandas()
        y_test = y_test.toPandas()

        # K-FOLD CROSS VALIDATION #
        print(f"Itr {itr}: X_train size = {X_train.shape}")
        print(f"Itr {itr}: y_train size = {y_train.shape}")
        lr_model = LogisticRegression(penalty="l1", solver="saga", max_iter=500)

        # hyperparameter tuning using K-Fold Cross Validation with K = 5;
        # shuffle the data with given random seed before splitting into batches
        tuning_parameters = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
        # evaluation_params = ["average_precision"]
        evaluation_params = ["accuracy"]
        kfold_cv_model = KFold(n_splits=5, shuffle=True, random_state=12345)

        for evaluation_param in evaluation_params:
            print(f"Itr {itr}: Tuning hyper-parameters based on {evaluation_param}")
            cv_model = GridSearchCV(estimator=lr_model, param_grid=tuning_parameters,
                                    scoring=evaluation_param, cv=kfold_cv_model, verbose=2, return_train_score=True)
            cv_model.fit(X_train, y_train)

            # The best values chosen by KFold-crossvalidation
            print(f"Itr {itr}: Best parameters in trained model = {cv_model.best_params_}")
            print(f"Itr {itr}: Best score in trained model = {cv_model.best_score_}")
        self.model = cv_model.best_estimator_

        # TRAINING #
        print(f"Itr {itr}: Training best model from k-fold cross validation over full training set")
        self.model.fit(X_train, y_train)

        # TESTING #
        y_test = y_test[:, "label"].values
        print(f"Itr {itr}: X_test size = {X_test.shape}")
        print(f"Itr {itr}: y_test size = {y_test.shape}")

        y_pred = self.model.predict_proba(X_test)
        y_pred = [y[1] for y in y_pred]
        print(f"Itr {itr}: Size of y_test = {len(y_test)}")
        print(f"Itr {itr}: Size of y_pred = {len(y_pred)}")
        result_itr = pd.DataFrame({"test_label": y_test, "test_prediction": y_pred, "run": itr})
        return result_itr

    def run(self):
        output_df = None
        for itr in range(self.n_iterations):
            print(f"Iteration : {itr}")

            # get training and testing datasets
            X_train, y_train, X_test, y_test = self.dataset_preprocessing.run()
            result_itr = self.run_iteration(X_train, y_train, X_test, y_test, itr)
            print(f"Itr {itr}: result_itr shape = {result_itr.shape}")
            if output_df is None:
                output_df = result_itr
            else:
                output_df = pd.concat([output_df, result_itr])

        return self.spark_obj.createDataFrame(output_df)
