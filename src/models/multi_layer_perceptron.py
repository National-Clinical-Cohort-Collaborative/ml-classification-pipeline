from pyspark.sql import SparkSession
from tensorflow import keras
import tensorflow as tf
import pandas as pd  # noqa
from models import dataset_preprocessing


class MultiLayerPerceptronModel:
    def __init__(self, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns, standardize_exclude_columns):
        self.model_name = "Multi Layer Perceptron"
        self.dataset = dataset
        self.n_iterations = n_iterations
        self.dataset_preprocessing = dataset_preprocessing.DatasetPreprocessing(
            self.dataset, preprocess_exclude_columns, standardize_exclude_columns, train_test_ratio)
        self.model = None  # best model from cross validation, used for testing
        self.spark_obj = SparkSession.builder.appName("pandas_to_spark").getOrCreate()

    def run_iteration(self, X_train, y_train, X_test, y_test, itr):
        X_train = X_train.toPandas()
        y_train = y_train.toPandas()
        y_train = y_train.iloc[:, 0].values
        X_test = X_test.toPandas()
        y_test = y_test.toPandas()
        y_test = y_test.iloc[:, 0].values

        print(f"Itr {itr}: X_train size = {X_train.shape}")
        print(f"Itr {itr}: y_train size = {y_train.shape}")

        # TRAINING #
        self.model = self.create_mlp_2_model()
        self.model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2)

        # TESTING #
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

    def create_mlp_2_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(keras.layers.Dense(32, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model
