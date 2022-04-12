from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd  # noqa


def get_auprc_distribution(self, model_ouput, n_iterations, model_name):
    """
    model_ouput: Dataframe containing the output of testing the model for 'n_iterations'
                    Dataframe shouls have the following columns: test_label, test_prediction, run
    returns: Dataframe with columns auprc, run, model
    """
    df = model_ouput.toPandas()
    auprc_list = []
    for itr in range(n_iterations):
        df_itr = df[df["run"] == itr]
        y_test = df_itr.loc[:, "test_label"].values
        y_pred = df_itr.loc[:, "test_prediction"].values

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        auprc = auc(recall, precision)
        auprc_list.append({"auprc": auprc, "run": itr, "model": self.model_name})
    return self.spark_obj.createDataFrame(pd.DataFrame(auprc_list))


def get_auroc_distribution(self, model_ouput, n_iterations, model_name):
    """
    model_ouput: Dataframe containing the output of testing the model for 'n_iterations'
                    Dataframe shouls have the following columns: test_label, test_prediction, run
    returns: Dataframe with columns auprc, run, model
    """
    df = model_ouput.toPandas()
    auroc_list = []
    for itr in range(n_iterations):
        df_itr = df[df["run"] == itr]
        y_test = df_itr.loc[:, "test_label"].values
        y_pred = df_itr.loc[:, "test_prediction"].values

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_roc = auc(fpr, tpr)
        auroc_list.append({"auc_roc": auc_roc, "run": itr, "model": self.model_name})
    return self.spark_obj.createDataFrame(pd.DataFrame(auroc_list))
