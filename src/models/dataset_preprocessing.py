import random
from pyspark.sql import SparkSession
import pandas as pd  # noqa
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


class DatasetPreprocessing:
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

    # def split_dataset(self, train_test_ratio, n_iterations):
    #     """
    #     train_test_ratio = [train_proportion, test_proportion]
    #     e.g. train_test_ratio = [0.8, 0.2]
    #     n_iteration: number of train_test splits to be created for repeated experiments
    #     """
    #     split_dfs = None
    #     for i in range(n_iterations):
    #         print(f"Iteration: {i}")

    #         # randomly selected seed value for creating the split
    #         seed = int(random.uniform(1000, 100000))
    #         print(f"seed={seed}")
    #         split_df = self.df.randomSplit(train_test_ratio, seed)
    #         itr_train_df = split_df[0]
    #         itr_test_df = split_df[1]

    #         itr_train_df = itr_train_df.withColumn("split", F.lit("train"))
    #         itr_test_df = itr_test_df.withColumn("split", F.lit("test"))
    #         itr_df = itr_train_df.union(itr_test_df)  # noqa
    #         itr_df = itr_df.withColumn("run", F.lit(i))

    #         print(f"itr_train_df size = {itr_train_df.count()}")
    #         print(f"itr_test_df size = {itr_test_df.count()}")
    #         print(f"itr_df size = {itr_df.count()}")

    #         if split_dfs is None:
    #             split_dfs = itr_df
    #         else:
    #             split_dfs = split_dfs.union(itr_df)  # noqa
    #         print(f"split_dfs size = {split_dfs.count()}")
    #     return split_dfs

    # def sample_balanced_datasets(self, dfs, n_iterations, do_sample_test_dataset):
    #     """
    #     dfs: Dataframe containing below two columns along with person_id, features, label:
    #         "run": specifying the run_id or run number corresponding to the number of times the experiment is repeated
    #         "split": "train" or "test"
    #                 specifying which split the record belongs to for the particular run (value of "run" column)
    #     n_iterations: number of train_test splits to be created for repeated experiments
    #     do_sample_dataset: Boolean: whether the test dataset should
    #                     also be sampled to get equal number of positives and negatives
    #     """
    #     # keep the train and test samples separate: to be joined together at the end
    #     dfs_train = dfs.filter(dfs.split == "train")
    #     dfs_test = dfs.filter(dfs.split == "test")

    #     dfs_test_sampled = dfs_test  # When test dataset is NOT to be sampled
    #     dfs_train_sampled = self.sample_balanced_dataset(dfs_train, n_iterations, "train")
    #     if do_sample_test_dataset:
    #         dfs_test_sampled = self.sample_balanced_dataset(dfs_test, n_iterations, "test")

    #     return dfs_train_sampled.union(dfs_test_sampled)  # noqa

    # def sample_balanced_dataset(self, dfs, n_iterations, dataset_name: str):
    #     """
    #     dfs: Dataframe containing below two columns along with person_id, features, label:
    #         "run": specifying the run_id or run number corresponding to the number of times the experiment is repeated
    #         "split": "train" or "test"
    #                 specifying which split the record belongs to for the particular run (value of "run" column)
    #     n_iterations: number of train_test splits to be created for repeated experiments
    #     dataset_name: "train" or "test": name of dataset being sampled.
    #     """
    #     df_sampled = None

    #     for i in range(n_iterations):
    #         print(f"Iteration: {i}")
    #         print(f"Itr {i}: {dataset_name} Dataset Sampling")
    #         df_itr = dfs.filter(dfs.run == i)
    #         df_positives = df_itr.filter(df_itr.label == 1)  # postive records from training dataset
    #         df_negatives = df_itr.filter(df_itr.label == 0)  # negative records from training dataset

    #         # compute number of positives and negatives
    #         n_positives = df_positives.count()
    #         n_negatives = df_negatives.count()
    #         # choose the lower of the two
    #         n_ref = min(n_positives, n_negatives)
    #         print(f"n_{dataset_name} = {df_itr.count()}")
    #         print("n_positives = {n_positives}")
    #         print("n_negatives = {n_negatives}")
    #         print("n_ref = {n_ref}")

    #         # sample from postives dataset
    #         df_positive_samples = df_positives.sample(fraction=n_ref/n_positives)
    #         # sample from negatives dataset
    #         df_negative_samples = df_negatives.sample(fraction=n_ref/n_negatives)
    #         print(f"Itr {i}: number of sampled df_{dataset_name}_positive_samples = ", df_positive_samples.count())
    #         print(f"Itr {i}: number of sampled df_{dataset_name}_negative_samples = ", df_negative_samples.count())

    #         df_sampled_itr = df_positive_samples.union(df_negative_samples)   # noqa

    #         if df_sampled is None:
    #             df_sampled = df_sampled_itr
    #         else:
    #             df_sampled = df_sampled.union(df_sampled_itr)  # noqa
    #     return df_sampled

    # def standardize_datasets(self, dfs, dataset_name, n_iterations, ignore_columns):
    #     """
    #     dfs: Dataframe containing below two columns along with person_id, features, label:
    #         "run": specifying the run_id or run number corresponding to the number of times the experiment is repeated
    #         "split": "train" or "test"
    #                 specifying which split the record belongs to for the particular run (value of "run" column)
    #     dataset_name: "train" or "test": name of dataset being sampled.
    #     n_iterations: number of train_test splits to be created for repeated experiments
    #     ignore_columns = [] list of columns to be ignored while standardizing the columns
    #                         e.g. person_id, label, split, run
    #     returns: Pandas Dataset
    #     """
    #     dfs = dfs.toPandas()  # convert to Pandas
    #     dfs = dfs[dfs["split"] == dataset_name]
    #     column_names = list(dfs.columns)
    #     column_names = [col for col in column_names if col not in ignore_columns]

    #     std_dfs = None
    #     for i in range(n_iterations):
    #         print(f"Iteration: {i}", i)
    #         df_itr = dfs[dfs["run"] == i]
    #         print(f"Itr {i}: df_itr shape = {df_itr.shape}")

    #         standard_scaler = MinMaxScaler()
    #         column_transformer = ColumnTransformer([("pass_through_id", "passthrough", ignore_columns)],
    #                                                remainder=standard_scaler)
    #         std_df_itr = column_transformer.fit_transform(df_itr)
    #         std_df_itr = pd.DataFrame(std_df_itr, columns=ignore_columns + column_names)
    #         if std_dfs is None:
    #             std_dfs = std_df_itr
    #         else:
    #             std_dfs = pd.concat([std_dfs, std_df_itr])
    #     return self.spark_obj.createDataFrame(std_dfs)
