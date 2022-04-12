# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from dataset_prep.data_preprocessing import DatasetPrepocessing


@transform_df(
    Output("/UNITE/[RP-2AE058] [N3C Operational] Machine-learning resources for N3C/Blessy Antony/Cardiovascular Sequelae/src/repositories/datasets/congestive_heart_failure_prediction_dataset"),
    inpatients_df=Input(
        "/UNITE/[RP-2AE058] [N3C Operational] Machine-learning resources for N3C/Blessy Antony/Cardiovascular Sequelae/src/workbook-output/Dataset Definition/cp_inpatients"),
    outpatients_df=Input(
        "/UNITE/[RP-2AE058] [N3C Operational] Machine-learning resources for N3C/Blessy Antony/Cardiovascular Sequelae/src/workbook-output/Dataset Definition/cp_outpatients"),
    patients_df=Input(
        "/UNITE/[RP-2AE058] [N3C Operational] Machine-learning resources for N3C/Blessy Antony/Cardiovascular Sequelae/src/workbook-output/Dataset Definition/cp_patients"),
)
def compute(inpatients_df, outpatients_df, patients_df):
    inpatients_dataset_preprocessing = DatasetPrepocessing(inpatients_df)
    columns_to_drop = ("COVID_patient_death_indicator", "COVID_associated_hospitalization_indicator", "covid_index_date")
    n_iterations = 10
    inpatients_dataset_preprocessing.drop_columns(columns_to_drop)
    split_dfs = inpatients_dataset_preprocessing.split_dataset([0.8, 0.2], n_iterations)
    sampled_dfs = inpatients_dataset_preprocessing.sample_balanced_datasets(
        dfs=split_dfs, n_iterations=n_iterations, do_sample_test_dataset=True)

    std_ignore_cols = ["person_id", "label", "split", "run"]
    std_dfs = inpatients_dataset_preprocessing.standardize_datasets(
        dfs=sampled_dfs, dataset_name="train", n_iterations=n_iterations, ignore_columns=std_ignore_cols)

    return std_dfs
