from pyspark.sql import SparkSession
from tensorflow import keras
import tensorflow as tf
import pandas as pd  # noqa


class MultiLayerPerceptronModel:
    def __init__(self, train_dataset, test_dataset, n_iterations):
        self.model_name = "Multi Layer Perceptron"
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

            # TRAINING #
            # retrieve the training dataset for this iteration
            train_df_itr = self.train_df[self.train_df["run"] == itr]
            # drop the run column
            train_df_itr = train_df_itr.drop(columns="run")
            # extract output dataset, drop the label column from features dataset
            X_train = train_df_itr.drop(columns="label")
            y_train = train_df_itr["label"]
            print(f"Itr {itr}: X_train size = {X_train.shape}")
            print(f"Itr {itr}: y_train size = {y_train.shape}")
            self.model = self.create_mlp_2_model()
            self.model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2)

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
            y_pred = [y[0] for y in y_pred]
            print(f"Itr {itr}: Size of y_test = {len(y_test)}")
            print(f"Itr {itr}: Size of y_pred = {len(y_pred)}")
            output_df_itr = pd.DataFrame({"test_label": y_test, "test_prediction": y_pred, "run": itr})
            if output_df is None:
                output_df = output_df_itr
            else:
                output_df = pd.concat([output_df, output_df_itr])

        return self.spark_obj.createDataFrame(output_df)

    def create_mlp_2_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(keras.layers.Dense(32, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        return model
