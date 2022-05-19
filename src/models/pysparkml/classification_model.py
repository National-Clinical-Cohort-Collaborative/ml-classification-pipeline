from models.pysparkml import dataset_processor

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql import functions as F


class ClassificationModel:
    def __init__(self, model_name, classification_model, param_grid, dataset, n_iterations, train_test_ratio, preprocess_exclude_columns,
                 index_column_name, label_column_name):
        self.model_name = model_name
        self.classification_model = classification_model
        self.param_grid = param_grid
        self.dataset = dataset
        self.n_iterations = n_iterations
        self.dataset_processor = dataset_processor.DatasetProcessor(
            self.dataset, preprocess_exclude_columns, index_column_name, label_column_name, train_test_ratio)
        self.model = None  # best model from cross validation, used for testing

    def run_iteration(self, train_df, test_df, itr):

        # K-FOLD CROSS VALIDATION #
        print(f"Itr {itr}: train_df size = ({train_df.count()},{len(train_df.columns)})")

        # hyperparameter tuning using K-Fold Cross Validation with K = 5;
        # TODO: shuffle the data with given random seed before splitting into batches
        evaluator = BinaryClassificationEvaluator()
        kfold_cv_model = CrossValidator(estimator=self.classification_model, estimatorParamMaps=self.param_grid,
                                        evaluator=evaluator, parallelism=2, numFolds=5,
                                        collectSubModels=False)
        assembler = self.dataset_processor.get_dataset_assembler()
        kfold_cv_pipeline = Pipeline(stages=[assembler, kfold_cv_model])
        kfold_cv_pipeline_model = kfold_cv_pipeline.fit(train_df)

        # CrossValidatorModel is a wrapper needed to select the best model
        kfold_cv_model = CrossValidatorModel(kfold_cv_pipeline_model)
        self.model = kfold_cv_model.bestModel
        # TODO: print(f"Itr {itr}: Best score in trained model = {cv_model.best_score_}")
        # The best values chosen by KFold-crossvalidation
        print(f"Itr {itr}: Best parameters in trained model = {kfold_cv_model.getEstimatorParamMaps()}")

        # TRAINING #
        print(f"Itr {itr}: Training best model from k-fold cross validation over full training set")
        self.model.fit(train_df)

        # TESTING #
        print(f"Itr {itr}: test_df size = ({test_df.count()},{len(test_df.columns)})")
        prediction_df = self.model.transform(test_df).select("probability", "prediction")

        split_prediction_udf = udf(lambda x: x[0].item(), FloatType())
        prediction_parsed_df = prediction_df.select(split_prediction_udf("probability").alias(
            "test_prediction"), prediction_df.prediction.alias("test_label"))
        prediction_parsed_df = prediction_parsed_df.withColumn("run", F.lit(itr))

        return prediction_parsed_df

    def run(self):
        print(f"Executing pipeline using {self.model_name}")
        output_df = None
        for itr in range(self.n_iterations):
            print(f"Iteration : {itr}")
            # get training and testing datasets
            train_df, test_df = self.dataset_processor.run()

            result_itr = self.run_iteration(train_df, test_df, itr)
            print(f"Itr {itr}: result_itr size = ({result_itr.count()},{len(result_itr.columns)})")
            if output_df is None:
                output_df = result_itr
            else:
                output_df = output_df.union(result_itr)  # noqa

        return output_df
