import random
from pyspark.sql import SparkSession
import pandas as pd  # noqa
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


class DatasetPreprocessor:
    """
    Preparation of datasets before passing to the ML classification models
    1. Drop column(s), if required
    2. Split dataset into train and test datasets in desired ratio
    3. Create 'n' such splits for repetitive testing
    3. Sample train and test datasets to have approximately 1:1 ratio of positive and negative samples
    4. Standardize the dataset
    """

    def __init__(self, df, preprocess_exclude_columns, standardize_exclude_columns, train_test_ratio):
        self.df = df
        self.preprocess_exclude_columns = preprocess_exclude_columns
        self.standardize_exclude_columns = standardize_exclude_columns
        self.train_test_ratio = train_test_ratio
        self.spark_obj = SparkSession.builder.appName("pandas_to_spark").getOrCreate()

    def drop_columns(self, column_names):
        """
        column_names: array string
        e.g. column_names = ("col_1", "col_2", "col_3")
        """
        self.df = self.df.drop(*column_names)

    def split_dataset(self, train_test_ratio):
        """
        train_test_ratio = [train_proportion, test_proportion]
        e.g. train_test_ratio = [0.8, 0.2]
        return: train_df, test_df
        """
        # randomly selected seed value for creating the split
        seed = int(random.uniform(1000, 100000))
        print(f"seed={seed}")
        split_df = self.df.randomSplit(train_test_ratio, seed)
        return split_df[0], split_df[1]

    def standardize_dataset(self, df, ignore_columns):
        """
        dfs: Dataframe containing columns
            "person_id", <one or more feature columns>, "label":
            "split": "train" or "test"
                    specifying which split the record belongs to for the particular run (value of "run" column)
        ignore_columns = [] list of columns to be ignored while standardizing the columns
                            e.g. person_id, label
        returns: standardized Pandas Dataset using MinMaxScaler
        """
        df = df.toPandas()  # convert to Pandas
        column_names = list(df.columns)
        column_names = [col for col in column_names if col not in ignore_columns]  # retain only unignored columns
        standard_scaler = MinMaxScaler()

        std_df = None
        column_transformer = ColumnTransformer([("pass_through_id", "passthrough", ignore_columns)],
                                               remainder=standard_scaler)
        std_df = column_transformer.fit_transform(df)
        std_df = pd.DataFrame(std_df, columns=ignore_columns + column_names)
        return self.spark_obj.createDataFrame(std_df)

    def sample_balanced_dataset(self, df):
        """
        df: Dataframe containing: person_id, <one or more feature columns>, label
        return: sampled dataset with approx 1:1 ratio of positive and negative samples
        """
        df_sampled = None
        df_positives = df.filter(df.label == 1)  # postive records
        df_negatives = df.filter(df.label == 0)  # negative records
        # compute number of positives and negatives
        n_positives = df_positives.count()
        n_negatives = df_negatives.count()
        # choose the lower of the two
        n_ref = min(n_positives, n_negatives)
        print(f"# df records = {df.count()}")
        print(f"n_positives = {n_positives}")
        print(f"n_negatives = {n_negatives}")
        print(f"n_ref = {n_ref}")
        # sample from postives dataset
        df_positive_samples = df_positives.sample(fraction=n_ref/n_positives)
        # sample from negatives dataset
        df_negative_samples = df_negatives.sample(fraction=n_ref/n_negatives)
        print("number of sampled df_positive_samples = ", df_positive_samples.count())
        print("number of sampled df_negative_samples = ", df_negative_samples.count())

        df_sampled = df_positive_samples.union(df_negative_samples)   # noqa
        return df_sampled

    def run(self):
        # dataset within the preprocessing object is updated.
        self.drop_columns(self.preprocess_exclude_columns)
        train_df, test_df = self.split_dataset(self.train_test_ratio)
        train_df_std = self.standardize_dataset(train_df, self.standardize_exclude_columns)
        test_df_std = self.standardize_dataset(test_df, self.standardize_exclude_columns)
        balanced_train_df = self.sample_balanced_dataset(train_df_std)
        balanced_test_df = self.sample_balanced_dataset(test_df_std)

        # drop the person_id (identifier) and label columns from training and testing feature datasets.
        features_drop_cols = ("person_id", "label")
        X_train = balanced_train_df.drop(*features_drop_cols)
        y_train = balanced_train_df.select("label")
        X_test = balanced_test_df.drop(*features_drop_cols)
        y_test = balanced_test_df.select("label")
        print(f"X_train size = ({X_train.count()},{len(X_train.columns)})")
        print(f"y_train size = ({y_train.count()},{len(y_train.columns)})")
        print(f"X_test size = ({X_test.count()},{len(X_test.columns)})")
        print(f"y_test size = ({y_test.count()},{len(y_test.columns)})")

        return X_train, y_train, X_test, y_test
