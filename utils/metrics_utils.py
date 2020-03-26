import pickle
import numpy as np

from datetime import datetime

from sklearn.metrics import (classification_report,
                             accuracy_score,
                             roc_auc_score,
                             confusion_matrix,
                             precision_recall_fscore_support as prfs)


def log_metrics(y_true,
                y_pred,
                metrics_write_path,
                average=""):
    """
    Log the precision/recall/f1-score/support

    :param y_true: (ndarray)
        True labels
    :param y_pred: (ndarray)
        Classifier predicted labels
    :param metrics_write_path: (str)
        Place to write metrics
    :param average: (str)
        Specifies micro/macro averaging

    :return:
        None
    """
    if not average:
        # Default write_file.write both micro and macro averages
        with open(metrics_write_path, "a") as write_file:
            write_file.write("\nModel Report\n")
            timestamp = str(datetime.fromtimestamp((datetime.timestamp(datetime.now()))))
            write_file.write("{}\n".format(timestamp))
            write_file.write("----------------------------------------\n")

        # Micro Avg
        log_metrics(y_true, y_pred, metrics_write_path, average="micro")

        # Macro Avg
        log_metrics(y_true, y_pred, metrics_write_path, average="macro")
    else:
        metrics = prfs(y_true, y_pred, average=average)
        with open(metrics_write_path, "a") as write_file:
            write_file.write("\n")
            write_file.write("\n- {} averaging -\n".format(average))
            write_file.write(classification_report(y_true, y_pred))

            write_file.write("\nPrecision: {}\n".format(metrics[0]))
            write_file.write("Recall: {}\n".format(metrics[1]))
            write_file.write("F1-Score: {}\n".format(metrics[2]))
            write_file.write("Support: {}\n".format(metrics[3]))
            write_file.write("Accuracy: {}\n".format(accuracy_score(y_true, y_pred)))
            write_file.write("AUC Score ({}): {}\n".format(average, roc_auc_score(y_true, y_pred,
                                                                                  average=average)))
            write_file.write(str(confusion_matrix(y_true, y_pred)) + "\n")


def write_feature_importances(model, feature_names, write_path):
    """
    Assumes the model has attribute `feature_importances_`
    Maps feature names to feature importances; then sorts by descending order.
    Finally, writes to file.

    :param model:
        The Estimator to use
    :param feature_names:
        Order list of all feature names; ordering should correspond to ordering of input into model
    :param write_path:
        Metrics filepath to save output

    :return:
        None
    """
    with open(write_path, "a") as feat_write_path:
        names_to_importances = sorted(dict(zip(feature_names,
                                               model.feature_importances_)).items(),
                                      key=lambda kv: kv[1],
                                      reverse=True)
        importance_tuples = [tup for tup in names_to_importances if tup[1] > 0][:50]
        feat_write_path.write("\nFeature Importances:\n{}\n".format(importance_tuples))


def write_model_params(model, feature_names, write_path, model_name="Unspecified Model Name"):
    """
    :param model: (Sklearn API model)
        A dictionary of the hyperparameters used by a model
    :param write_path: (str)
        Specifies the file to write to. We usually write to metrics file, for reproducibility.

    :return:
        None
    """
    # Tree-based Estimators have this attribute
    if hasattr(model, "feature_importances_"):

        with open(write_path, "a") as write_file:
            write_file.write("\n{}\n".format(model_name))
            write_file.write("\nEstimator params:\n")
            write_file.write(str(model.get_params()) + "\n")

        write_feature_importances(model,
                                  feature_names,
                                  write_path)

    # Cross Validators have this attribute
    elif hasattr(model, "best_estimator_"):

        # Saves CV Parameters
        with open(write_path, "a") as write_file:
            # Write info about the CV Tuner
            write_file.write("\nCV params:\n")
            write_file.write(str(model.get_params()) + "\n")

            write_model_params(model.best_estimator_, feature_names, write_path)

    # Put other feature-weights of estimators chosen by CV here
    else:
        pass


def save_performance_metrics(featurized_data, y_true,
                             tuned_and_fitted_model,
                             metrics_write_path,
                             model_name="",
                             feat_names_pkl_path=None):
    """
    Get and save metrics for this particular model

    :param featurized_data: (Dataframe)
    :param y_true: (Dataframe)
    :param tuned_and_fitted_model:
    :param metrics_write_path: (str)

    :return:
        None
    """
    if isinstance(featurized_data, np.ndarray) and feat_names_pkl_path:
        with open(feat_names_pkl_path, "rb") as feats_readfile:
            feature_names = pickle.load(feats_readfile)
        feature_vectors = featurized_data
    else:
        feature_names = featurized_data.columns.values
        feature_vectors = featurized_data.values

    y_pred = []
    for feat_vec in feature_vectors:
        if feat_vec.size > 0 and tuned_and_fitted_model.predict_proba([feat_vec])[0][1] >= 0.4:
            y_pred.append(1)
        else:
            y_pred.append(0)

    log_metrics(y_true, y_pred, metrics_write_path)
    write_model_params(tuned_and_fitted_model,
                       feature_names,
                       metrics_write_path,
                       model_name=model_name)
