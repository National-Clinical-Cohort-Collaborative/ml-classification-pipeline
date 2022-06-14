import random
from pyspark.ml.feature import VectorAssembler, MinMaxScaler


class DatasetProcessor:
    """
    Preparation of datasets before passing to the ML classification models:

    1. Drop column(s), if required.
    2. Split dataset into train and test datasets in desired ratio.
    3. Sample train and test datasets to have approximately 1:1 ratio of positive and negative samples.
    4. Assembler: Format the datasets to meet the requirements of pyspark ml input:
       - one features column with a dense vector of all features.
       - one output column with the labels (classification)/ output (regression) values.
    5. Scaler: Standardize datasets using MinMaxScaler.

    TODO: add support for more scaling options.
    """

    def __init__(self, df, preprocess_exclude_columns, index_column_name, label_column_name, train_test_ratio):
        self.df = df
        self.preprocess_exclude_columns = preprocess_exclude_columns
        self.index_column_name = index_column_name
        self.label_column_name = label_column_name
        self.train_test_ratio = train_test_ratio

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

    def get_feature_column_names(self):
        """
        Features column names: all columns in dataset minus

        - index column (e.g.person_id) and
        - label column (e.g. label)
        """
        feature_names = self.df.columns
        feature_names.remove(self.index_column_name)
        feature_names.remove(self.label_column_name)
        return feature_names

    def get_dataset_assembler(self, output_col_name="features"):
        """
        Assembler: create a dataset in the libsvm format supported by pysparkml

        libsvm format:
         - features: DenseVector of all features
         - label: column to be predicted
        create a column called 'features' consisiting of DenseVectors of all features
        """
        assembler = VectorAssembler(
            inputCols=self.get_feature_column_names(),
            outputCol=output_col_name
        )
        return assembler

    def get_min_max_scaler(self, input_col_name, output_col_name):
        """
        Scaler: instantiate and return instance of MinMaxScaler

        """
        return MinMaxScaler(inputCol=input_col_name, outputCol=output_col_name)

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
        """
        1. Drop column(s), if required.
        2. Split dataset into train and test datasets in desired ratio.
        3. Sample train and test datasets to have approximately 1:1 ratio of positive and negative samples.
        """
        # dataset within the preprocessing object is updated.
        self.drop_columns(self.preprocess_exclude_columns)
        train_df, test_df = self.split_dataset(self.train_test_ratio)
        # train_df_std = self.standardize_dataset(train_df, self.standardize_exclude_columns)
        # test_df_std = self.standardize_dataset(test_df, self.standardize_exclude_columns)
        balanced_train_df = self.sample_balanced_dataset(train_df)
        balanced_test_df = self.sample_balanced_dataset(test_df)

        return balanced_train_df, balanced_test_df
