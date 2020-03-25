import os
import datetime  # noqa
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer

from app.models.general import (general_constants as pri_constants, tag_helpers,
                                education_helpers, employment_helpers,
                                achievement_helpers)
from app.log import logger


def keymap_replace_substrings(target_str,
                              mappings={"<": "left_angle_bracket",
                                        ">": "right_angle_bracket",
                                        "[": "left_square_bracket",
                                        "]": "right_square_bracket",
                                        "{": "left_curly_bracket",
                                        "}": "right_curly_bracket",
                                        "\n": "newline"}):
    """Replace parts of a string based on a dictionary.
    This function takes a dictionary of
    replacement mappings. For example, if I supplied
    the string "Hello world.", and the mappings
    {"H": "J", ".": "!"}, it would return "Jello world!".
    Warning: replacements are made iteratively,
    meaning multiple replacements could occur.
    :param target_str: (str)
        The string to replace characters in.
    :param mappings: (str)
        A dictionary of replacement mappings.
    :return: (str)
        String with values replaced
    """
    for substring, replacement in mappings.items():
        target_str = target_str.replace(substring, replacement)
    return target_str


def featurize(dicts, msg_type='person', is_training_data=False):
    """
    Takes in a list of dicts with type msg_type, and attempts to extract useful features
    :param obj_dicts: (list)
        contains one or more dictionaries, each of which is a person-dict payload
    :param msg_type: (str)
        specifies dict type
    :return (dict)
        keys are Person IDs(PIDs) mapped to a list containing every feature value for that person
    """
    pids_to_features = {}

    for msg_info in dicts:

        # Check to make sure each ID is valid, or else skip
        try:
            pid = msg_info.get(msg_type).get('id', {})
            pids_to_features[pid] = {}
        except AttributeError:
            logger.warning('general - missing pid for payload dict: {}'.format(msg_info))
            continue

        """
        Create features for this dict (from degrees, employments, tags, and achievements)
        Note: .update is almost 6x slower than [] = s, but I'm prioritizing legibility here.
        If there's a need for faster run-time, we can time these functions and check scale
        The syntax is so legible, that it's worth it.
        We should try to be efficient, but not go overboard.
        """
        feats_dict = pids_to_features.get(pid)

        feats_dict.update(tag_helpers.featurize_tags(msg_info, msg_type))
        feats_dict.update(education_helpers.featurize_degrees(msg_info))
        feats_dict.update(employment_helpers.featurize_employments(msg_info, is_training_data))
        feats_dict.update(achievement_helpers.featurize_achievements(msg_info))

        # This line is redundant, but including for clarity
        # feats_dict updates the same obj in pids_to_features[pid], but more explicit is safer here
        pids_to_features[pid] = feats_dict

    return pids_to_features


def save_feature_names(feat_names_lst, feat_names_wpath):
    """
    Saves the set of feature names into a file; one feature name per line
    This allows us to examine the features being saved, if we want to do an in-depth analysis.
    :param feat_names_lst: (set)
        unique feature names as strings
    :param feat_names_wpath: (str)
        write path for feature names
    :return: (None)
    """
    with open(feat_names_wpath, "w") as wfile:
        for feature_name in sorted(feat_names_lst):
            wfile.write(feature_name + "\n")

    with open(feat_names_wpath.replace(".csv", ".pkl"), "wb") as pkl_file:
        pickle.dump(feat_names_lst, pkl_file)


def load_feature_names(feat_names_readpath):
    """
    Loads the set of feature names from a csv; one feature name per line
    :param feat_names_readpath: (str)
        Read path for feature names
    :return: (set)
        set of features
    """
    try:
        with open(feat_names_readpath, "r") as rfile:
            return [feat_name.replace("\n", '') for feat_name in rfile]
    except FileNotFoundError:
        logger.error("Feature-names file not found.")
        return []


def build_unlabeled_features_df(pids_to_feats,
                                feat_names_save_path=pri_constants.FEAT_NAMES_PATH,
                                unlabeled_train_df_fpath=pri_constants.UNLABELED_TRAIN_DF_PATH,
                                default_feature_val=pri_constants.DEFAULT_FEATURE_VAL,
                                unique_feats_threshold=pri_constants.UNIQUE_VALS,
                                save=True):
    """
    1) Load obj_dicts and featurize them (dicts)
    2) Remove spaces from feature names
    3) Save all unique feature names
    4) Set default values for features that DNE
    5) Create/save (maybe)/return train_df
    :param pids_to_feats: (dict)
        pids mapped to feature vectors
    :param feat_names_save_path: (str)
        the filepath to store feature names
    :param unlabeled_train_df_fpath: (str)
        Training dataframe without a label column (still useful for unsupervised learning)
    :param default_feature_val: (int, or anything)
        default placeholder value when person doesn't have a specific feature
    :return: (DataFrame)
        the training dataframe (without labels)
    """
    if not pids_to_feats:
        return pids_to_feats

    # Establish train_df based on feature_vectors with placeholders
    feature_objects = []
    for pid in pids_to_feats:
        pids_to_feats[pid]["obj_id"] = pid
        feature_objects.append(pids_to_feats[pid])

    # Can't use Spare Matrix format for passing into Pandas Dataframe
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(feature_objects)
    column_labels = vectorizer.get_feature_names()
    train_df = pd.DataFrame(matrix, columns=column_labels).set_index("obj_id")
    logger.info("Managed to vectorize the data with rows: {}, columns: {}"
                .format(len(train_df), len(train_df.columns)))

    # Replace '[', ']', '<' chars (For XGBoost)
    train_df.rename(columns=keymap_replace_substrings, inplace=True)
    if save:
        # Drop columns with only one value in the entire column (speeds up computation)
        train_df = train_df.loc[:, train_df.apply(pd.Series.nunique) != 1]
        # Drop binary columns with less than ten occurances in each class
        # Otherwise, check that there are at least 10 unique values
        col_filter = train_df.apply(lambda s: (s.value_counts() > unique_feats_threshold).all()
                                    if s.nunique() < 3 else s.nunique() > unique_feats_threshold)
        train_df = train_df.loc[:, col_filter]
        logger.info("Dropped useless features. New rows: {}, columns: {}"
                    .format(len(train_df), len(train_df.columns)))

        # Store the unlabeled training dataframe as CSV
        # Helpful for when no labels (predict route or unsupervised algo)
        train_df.to_pickle(unlabeled_train_df_fpath)
    else:
        # If we're not saving, then we're loading features to predict.
        # In that case, make sure our features are only the ones used by the model.
        valid_cols = load_feature_names(feat_names_save_path)
        train_df = train_df.reindex(columns=valid_cols, fill_value=default_feature_val)

    return train_df


def add_labels_to_dataframe(unlabeled_train_df,
                            labels_filepath,
                            labeled_df_save_path,
                            id_col_name=pri_constants.ID_COL_NAME,
                            label_name=pri_constants.LABEL_COL_NAME,
                            save=True):
    """
    Add labels column to unlabeled_train_df
    :param unlabeled_train_df: (DataFrame)
        the training dataframe that needs labels
    :param id_col_name: (str)
        name of the ID column to use
    :param label_name: (str)
        the column name of the label to use (score/progress/is_X/etc)
    :param labels_filepath: (str)
        filepath with IDs and associated labels
    :param default_feature_val: (int, or anything)
        The default label to give when a obj_id has no associated label
    :return: (DataFrame)
        updated dataframe with labels
    """
    labels = pd.read_csv(labels_filepath).set_index(id_col_name)[label_name].to_frame()
    labeled_df = pd.merge(unlabeled_train_df,
                          labels,
                          how='inner',
                          left_index=True,
                          right_index=True)

    # Labeled df shouldn't have NaNs and should be rounded to 5 sig figs.
    labeled_df = labeled_df.round(5).reindex(sorted(labeled_df.columns), axis=1)

    if save:
        labeled_df.to_pickle(labeled_df_save_path)

    return labeled_df


def get_train_df(overwrite_existing_files=False,
                 obj_dicts_filepath=pri_constants.MSGS_READ_PATH,
                 pkl_obj_msg_filepath=pri_constants.PKL_MSGS_READ_PATH,
                 feat_names_save_path=pri_constants.FEAT_NAMES_PATH,
                 unlabeled_train_df_fpath=pri_constants.UNLABELED_TRAIN_DF_PATH,
                 labels_filepath=pri_constants.LABELS_FILEPATH,
                 labeled_df_filepath=pri_constants.LABELED_TRAIN_DF_PATH,
                 id_col_name=pri_constants.ID_COL_NAME,
                 label_name=pri_constants.LABEL_COL_NAME,
                 default_feature_val=pri_constants.DEFAULT_FEATURE_VAL,
                 unique_feats_threshold=pri_constants.UNIQUE_VALS,
                 is_training_data=False):
    """
    Get the train_df, either from CSV file, or direct from raw obj_dicts.

    :param overwrite_existing_files: (bool)
        if True, overwrite the labeled_df_filepath with a new CSV trained on obj_msgs,
        otherwise default to the stored CSV at labeled_df_filepath
        (WARNING: possibly overwrite existing files)
    :param obj_dicts_filepath: (str)
        filepath for obj_dicts
    :param feat_names_save_path: (str)
        filepath for feature names
    :param unlabeled_train_df_fpath: (str)
        filepath for unlabeled_train_df
    :param labels_filepath: (str)
        filepath to the label names
    :param labeled_df_filepath: (str)
        the path to the labeled training dataframe CSV
    :param id_col_name: (str)
        column name of IDs (e.g. "company_id", "id", "obj_id", etc)
    :param label_name: (str)
        the name of the labels column (e.g. "max_progress")
    :param default_feature_val: (int, float)
        default placeholder value when a feature value is not found/specified
    :param is_training_data: (bool)
        toggles tasks for only train/predict; e.g. building vectorized tf-idf model
    :return: (DataFrame)
        a training dataframe with labels for supervised learning
    """

    # Load decompressed raw dicts and featurizes the dictionaries
    logger.debug("Loading Person-dicts")
    if not os.path.isfile(pkl_obj_msg_filepath):  # or overwrite_existing_files

        # Read person dicts and featurize from raw string; then pickle to file
        with open(obj_dicts_filepath, "r") as rfile:
            pids_to_feats = featurize([eval(msg) for msg in rfile],
                                      is_training_data=is_training_data)

        with open(pkl_obj_msg_filepath, "wb") as pkl_msgs:
            pickle.dump(pids_to_feats, pkl_msgs)
    else:
        with open(pkl_obj_msg_filepath, "rb") as pkl_msgs:
            pids_to_feats = pickle.load(pkl_msgs)

    # Build (or load if already exists) unlabeled DataFrame
    logger.debug("building unlabeled df")
    if not os.path.isfile(unlabeled_train_df_fpath):  # or overwrite_existing_files
        train_df = build_unlabeled_features_df(pids_to_feats,
                                               feat_names_save_path=feat_names_save_path,
                                               unlabeled_train_df_fpath=unlabeled_train_df_fpath,
                                               default_feature_val=default_feature_val,
                                               unique_feats_threshold=unique_feats_threshold,
                                               save=overwrite_existing_files)
    else:
        train_df = pd.read_pickle(unlabeled_train_df_fpath)

    # Label (or load if already exists) the features DataFrame
    logger.debug("Now adding labels to df; Rows: {}, Columns: {}"
                 .format(len(train_df.index), len(train_df.columns)))
    if not os.path.isfile(labeled_df_filepath):  # or overwrite_existing_files
        train_df = add_labels_to_dataframe(train_df,
                                           labels_filepath,
                                           labeled_df_filepath,
                                           id_col_name=id_col_name,
                                           label_name=label_name,
                                           save=overwrite_existing_files)
    else:
        train_df = pd.read_pickle(labeled_df_filepath)

    return train_df
