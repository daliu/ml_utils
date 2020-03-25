import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_float

from app.models.prioritizer import (prioritizer_constants as pri_constants,
                                    train_constants,
                                    standardize)
from app.utils import metrics_utils
from app.log import logger


# Save model
def save_model(model, model_write_path):
    """
    :param model: (Sklearn Model)
    :param model_write_path: (str)

    :return: (None)
    """
    pickle.dump(model, open(model_write_path, "wb"))


def tune_model(X_features, Y_labels, model_name,
               param_grid={},
               feat_names_save_path=pri_constants.FEAT_NAMES_PATH,
               cv_folds=5,
               save=False):
    """
    :param X_features: (DataFrame)
        Feature Columns
    :param Y_labels: (Series)
        Labels
    :param model: (Sklearn Interface Model)
        The model to fit/predict with
    :param param_grid: (Dict)
        Dictionary object with hyperparameter names and ranges associated with them

    :return: (Sklearn Model)
        A hyperparamter tuned model fitted on X_features and Y_labels
    """

    # Important to Set Class Imbalance Params, and use max cores
    pos_weight_scalar = np.sum(Y_labels == 0) / float(np.sum(Y_labels == 1))
    if isinstance(pos_weight_scalar, pd.core.series.Series):
        pos_weight_scalar = pos_weight_scalar.values[0]

    upper_limit = min(len(X_features.columns), len(X_features.index))
    log_upper_limit = int(np.log(upper_limit))

    if model_name == "xgboost":
        # XGBoost with Hyperband Hyperparameter Optimization
        clf = xgb.XGBClassifier()
        clf.set_params(**{"scale_pos_weight": pos_weight_scalar,
                          "n_jobs": 4})

        # Hyperparameter search boundaries
        param_grid = {
                      # Parameters for Tree Booster
                      'eta': sp_float(0, 1),
                      'gamma': sp_randint(0, 100),
                      'max_depth': sp_randint(1, 10),
                      'learning_rate': sp_float(0, 1),
                      'n_estimators': sp_randint(100, 5000),

                      'min_child_weight': sp_randint(0, 50),
                      'max_delta_step': sp_randint(0, np.log(upper_limit)),
                      'subsample': sp_float(0, 1),

                      # Family of parameters for subsampling of columns
                      'colsample_bytree': sp_float(0, 1),
                      'colsample_bylevel': sp_float(0, 1),
                      'colsample_bynode': sp_float(0, 1),

                      # Regularization Params
                      'lambda': sp_randint(1, 100),
                      'alpha': sp_randint(0, 100),
                      }

    elif model_name == "random_forest_classifier":
        # Random Forest with max cores, balanced class weights
        clf = RandomForestClassifier()
        clf.set_params(n_jobs=-1,
                       class_weight={0: 1,
                                     1: pos_weight_scalar})
        # Hyperparameter search boundaries
        param_grid = {'max_depth': sp_randint(1, log_upper_limit),
                      'max_features': sp_randint(10, 500),
                      'min_samples_split': sp_randint(2, 50),
                      'min_samples_leaf': sp_randint(1, 50),
                      'n_estimators': sp_randint(100, 5000),
                      'bootstrap': np.array([True]),
                      'criterion': np.array(['gini', 'entropy'])
                      }

    else:
        # Default to Random Forest with specified params, n_jobs still -1
        clf = RandomForestClassifier(n_jobs=-1)
        clf.set_params(**param_grid)

    logger.debug("Now fitting model with params {}".format(param_grid))

    # # # # Comment this block out for production, because of Hyperband library
    # if param_grid:
    #     # Only tune if we actually have params to tune.
    #     from civismlext.hyperband import HyperbandSearchCV
    #     tuned_model = HyperbandSearchCV(clf,
    #                                     param_distributions=param_grid,
    #                                     cost_parameter_max={'n_estimators':
    #                                                         upper_limit},
    #                                     cost_parameter_min={'n_estimators': 10},
    #                                     scoring='roc_auc',
    #                                     n_jobs=4,
    #                                     cv=cv_folds)
    # else:
    #     # Skip tuning
    #     tuned_model = clf

    # Comment this code in for running local tests
    tuned_model = clf

    # Save the exact format of the columns, for standardization consistency
    # Also allows us to reload/peek feature names, to get a subset or analyze results
    if save:
        standardize.save_feature_names(X_features.columns.values, feat_names_save_path)

    # Need to use .values to convert to numpy matrices
    tuned_model.fit(X_features.values, Y_labels.values)

    logger.debug("Model fitted.")
    return tuned_model


def split_labeled_df(labeled_df,
                     label_col_name,
                     id_col_name,
                     write_path="tests/",
                     test_size=.15,
                     random_state=None):
    """
    Splits data by test_size and return

    :param labeled_df: (DataFrame)
        Assumed to be a dataframe with a labels column
    :param label_col_name: (str)
        Name of the labels column
    :param test_size: (int, 0, 1)
        The percentage of the split to use for testing (implies 1-test_size used for training)
    :param random_state: (None or int)
        Specifies random_state for reproducibility

    :return: (nd.array, nd.array, nd.array, nd.array)
        x_train, x_test, y_train, y_test
    """
    # sklearn train_test_split requires class to have at least two samples
    labeled_df = labeled_df.groupby(label_col_name).filter(lambda x: len(x) >= 2)

    if id_col_name in labeled_df.columns:
        labeled_df = labeled_df.set_index(id_col_name)

    train_data = labeled_df.drop(label_col_name, axis=1)
    labels = labeled_df[label_col_name]

    # Test size of zero implies use all of the train data to fit the model.
    if test_size == 0:
        return train_data, train_data, labels, labels

    logger.debug("Performing split with random_state {}".format(random_state))

    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                        labels,
                                                        test_size=test_size,
                                                        stratify=labels,
                                                        random_state=random_state)

    # Write x,y (train/test)
    x_train.to_pickle(write_path + train_constants.X_TRAIN_FILE_PATH)
    x_test.to_pickle(write_path + train_constants.X_TEST_FILE_PATH)

    # Pandas Series before v24 defaulted to False
    y_train.to_pickle(write_path + train_constants.Y_TRAIN_FILE_PATH)
    y_test.to_pickle(write_path + train_constants.Y_TEST_FILE_PATH)

    return x_train, x_test, y_train, y_test


# Try running and see if results produced
def get_fitted_model(overwrite_existing_files=False,
                     person_messages_filepath=pri_constants.MSGS_READ_PATH,
                     pkl_person_msg_filepath=pri_constants.PKL_MSGS_READ_PATH,
                     feat_names_save_path=pri_constants.FEAT_NAMES_PATH,
                     unlabeled_train_df_fpath=pri_constants.UNLABELED_TRAIN_DF_PATH,
                     labels_filepath=pri_constants.LABELS_FILEPATH,
                     labeled_df_filepath=pri_constants.LABELED_TRAIN_DF_PATH,
                     id_col_name=pri_constants.ID_COL_NAME,
                     label_name=pri_constants.LABEL_COL_NAME,
                     default_feature_val=pri_constants.DEFAULT_FEATURE_VAL,
                     unique_feats_threshold=pri_constants.UNIQUE_VALS,
                     split_write_path="tests/",
                     test_split_size=pri_constants.TEST_SPLIT_SIZE,
                     random_state=pri_constants.RANDOM_STATE,
                     param_grid=pri_constants.RF_PARAM_GRID,
                     model_name=pri_constants.MODEL_NAME,
                     model_filepath=pri_constants.MODEL_FILEPATH,
                     train_metrics_savepath=pri_constants.TRAIN_METRICS_FILEPATH,
                     validate_metrics_savepath=pri_constants.VALIDATE_METRICS_FILEPATH,
                     is_training_data=True,
                     cv_folds=pri_constants.CV_FOLDS):
    """
    Either get a pre-trained model from serialized file, or read-in, label, and train from scratch.

    :param label_name: (str)
        the name of the label column
    :param label_filepath: (str)
        filepath for where to get labels-column csv
    :param model_filepath: (str)
        filepath for where to save the model

    :return: (Classifier/Regressor)
        model to use
    """
    train_df = standardize.get_train_df(overwrite_existing_files=overwrite_existing_files,
                                        person_messages_filepath=person_messages_filepath,
                                        pkl_person_msg_filepath=pkl_person_msg_filepath,
                                        feat_names_save_path=feat_names_save_path,
                                        unlabeled_train_df_fpath=unlabeled_train_df_fpath,
                                        labels_filepath=labels_filepath,
                                        labeled_df_filepath=labeled_df_filepath,
                                        id_col_name=id_col_name,
                                        label_name=label_name,
                                        default_feature_val=default_feature_val,
                                        unique_feats_threshold=unique_feats_threshold,
                                        is_training_data=is_training_data)
    # Split data
    x_train, x_test, y_train, y_test = split_labeled_df(train_df,
                                                        label_name,
                                                        id_col_name,
                                                        write_path=split_write_path,
                                                        test_size=test_split_size,
                                                        random_state=random_state)
    # Get, train, and tune model
    tuned_and_fitted_model = tune_model(x_train, y_train, model_name, param_grid,
                                        feat_names_save_path,
                                        cv_folds=cv_folds,
                                        save=overwrite_existing_files)
    # Save Train Results
    metrics_utils.save_performance_metrics(x_train, y_train,
                                           tuned_and_fitted_model,
                                           train_metrics_savepath,
                                           model_name=model_name)
    #  Save Validation Results
    metrics_utils.save_performance_metrics(x_test, y_test,
                                           tuned_and_fitted_model,
                                           validate_metrics_savepath,
                                           model_name=model_name)

    if overwrite_existing_files:
        if hasattr(tuned_and_fitted_model, "best_estimator_"):
            save_model(tuned_and_fitted_model.best_estimator_, model_filepath)
        else:
            save_model(tuned_and_fitted_model, model_filepath)

    return tuned_and_fitted_model
