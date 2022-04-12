from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd  # noqa


class LogisticRegressionModel:
    def __init__(self, train_dataset, test_dataset, n_iterations):
        self.model_name = "Logistic Regression"
        self.train_df = train_dataset.toPandas()
        self.test_df = test_dataset.toPandas()
        self.n_iterations = n_iterations
        self.model = None  # best model from cross validation, used for testing
        self.spark_obj = SparkSession.builder.appName("pandas_to_spark").getOrCreate()

    def setup(self):
        # drop the person_id (identifier) and split (dataset split identifier) columns
        self.train_df = self.train_df.drop(columns=["person_id", "split"])
        self.test_df = self.test_df.drop(columns=["person_id", "split"])

    def run(self):
        output_df = None
        for itr in range(self.n_iterations):
            print(f"Iteration : {itr}")

            # K-FOLD CROSS VALIDATION #
            # retrieve the training dataset for this iteration
            train_df_itr = self.train_df[self.train_df["run"] == itr]
            # drop the run column
            train_df_itr = train_df_itr.drop(columns="run")
            # extract output dataset, drop the label column from features dataset
            X_train = train_df_itr.drop(columns="label")
            y_train = train_df_itr["label"]
            print(f"Itr {itr}: X_train size = {X_train.shape}")
            print(f"Itr {itr}: y_train size = {y_train.shape}")
            lr_model = LogisticRegression(penalty="l1", solver="saga", max_iter=500)

            # hyperparameter tuning using K-Fold Cross Validation with K = 5; shuffle the data with given random seed before splitting into batches
            tuning_parameters = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
            # evaluation_params = ["average_precision"]
            evaluation_params = ["accuracy"]
            kfold_cv_model = KFold(n_splits=5, shuffle=True, random_state=12345)

            for evaluation_param in evaluation_params:
                print(f"Itr {itr}: Tuning hyper-parameters based on {evaluation_param}")
                cv_model = GridSearchCV(estimator=lr_model, param_grid=tuning_parameters, scoring=evaluation_param, cv=kfold_cv_model, verbose=2, return_train_score=True)
                cv_model.fit(X_train, y_train)

                # The best values chosen by KFold-crossvalidation
                print(f"Itr {itr}: Best parameters in trained model = {cv_model.best_params_}")
                print(f"Itr {itr}: Best score in trained model = {cv_model.best_score_}")
            self.model = cv_model.best_estimator_

            # TRAINING #
            print(f"Itr {itr}: Training best model from k-fold cross validation over full training set")
            self.model.fit(X_train, y_train)

            # TESTING #
            # retrieve the testing dataset for this iteration
            test_df_itr = self.test_df[self.test_df["run"] == itr]
            # drop the run column
            test_df_itr = test_df_itr.drop(columns="run")
            # extract output dataset, drop the label column from features dataset
            X_test = test_df_itr.drop(columns="label")
            y_test = test_df_itr.loc[:, "label"].values
            print(f"Itr {itr}: X_test size = {X_test.shape}")
            print(f"Itr {itr}: y_test size = {y_test.shape}")

            y_pred = self.model.predict_proba(X_test)
            y_pred = [y[1] for y in y_pred]
            print(f"Itr {itr}: Size of y_test = {len(y_test)}")
            print(f"Itr {itr}: Size of y_pred = {len(y_pred)}")
            output_df_itr = pd.DataFrame({"test_label": y_test, "test_prediction": y_pred, "run": itr})
            if output_df is None:
                output_df = output_df_itr
            else:
                output_df = pd.concat([output_df, output_df_itr])

        return self.spark_obj.createDataFrame(output_df)
