import re
import string
import pandas as pd
import numpy as np

from datetime import datetime

from app.log import logger


def get_today():
    """
    Gets today's date in a `datetime` object.
    This is a helpful import when we don't want to
    explicitly `import datetime`, just to use it once in a file.
    :return: (datetime)
        today's date in datetime obj format
    """
    return datetime.today()


def add_vec_text_feats(feature_dict, target_text, corpus_name):
    """
    Parse through the text, removing stopwords. Then use word count as features.

    :param feature_dict: (dict)
        Feature dictionary to update
    :param target_text: (str)
        Target text to evaluate
    :param corpus_name: (str)
        The name of the category that target_text belongs to (patent/job/comp descript, title, etc.)
    :return: (dict)
        updated features dictionary
    """
    # Add Punctuation text feats
    for char in target_text:
        if char in string.punctuation:
            corpus_word = corpus_name + "_char_counter_" + char
            feature_dict[corpus_word] = feature_dict.get(corpus_word, 0) + 1

    return feature_dict


def get_timespan(end_date, start_date, time_granularity):
    """
    Calculates the timespan between start and end date.
    We don't include day in months/years timeframe because usually
    it isn't included in start/end dates and not helpful.

    :param end_date: (datetime or Falsey value (None, '', etc.))
        end of the time span
    :param start_date: (datetime or Falsey value)
        start of the time span
    :param time_granularity: (string)
        'days', 'months', 'years' supported
        The granularity of the timespan to count
    :return: (int)
        number of `time_granularity` between end and start dates
    """
    # Default start_date
    if not start_date:
        start_date = datetime.today()

    # Convert start/end dates to datetime objects
    if not isinstance(end_date, datetime) or not isinstance(start_date, datetime):
        try:
            start_date = pd.Timestamp(start_date).to_pydatetime()
            end_date = pd.Timestamp(end_date).to_pydatetime()
        except ValueError:
            print('Prioritizer - cannot convert event date for : {}'.format(end_date))
            return -1

    # Identify granularity of time-difference
    # Days is more of an estimate. Can by off by at most 10.
    if time_granularity == 'days':
        return (end_date.year - start_date.year)*365 + \
               abs(end_date.month - start_date.month)*30 + \
               abs(end_date.day - start_date.day)
    elif time_granularity == 'months':
        return (end_date.year - start_date.year)*12 + \
               abs(end_date.month - start_date.month)
    elif time_granularity == 'years':
        return (end_date.year - start_date.year) + \
               abs(end_date.month - start_date.month)/12
    else:
        print("Invalid granularity {} when calculating timespan.".format(time_granularity))
        return -1


def num_pattern_matches(text, patterns_lst):
    """
    Checks lowercase text and lowercase punctuation-removed text for patterns

    :param text: (str)
        the body of text to evaluate
    :param patterns_lst: (lst)
        list of regex string patterns to try and match
    :return:
        number of pattern-matches from list detected, NOT the number of total matches
        We do this to "classify" the text using the pattern-detectors.
    """
    if not isinstance(text, str) or not isinstance(patterns_lst, list):
        return 0

    # Replace forward slash, parentheses, and dashes with spaces
    low_replaced_txt = re.sub(r'/|\(|\)|-', ' ', text.lower())

    # Remove all punctuation
    no_punct_lower_text = low_replaced_txt.translate(str.maketrans('', '', string.punctuation))

    matches = 0
    for pattern in patterns_lst:
        pattern_found = re.search(pattern, low_replaced_txt) or \
                        re.search(pattern, no_punct_lower_text)
        if pattern_found:
            matches += 1

    return matches


def get_lst_avg(lst):
    """
    :param lst: (list)
        the list to average
    :return: (float)
        average value of the list
    """
    try:
        return sum(lst) / float(len(lst))
    except TypeError:
        print("get_lst_avg: bad lst or list contains non-numeric values (None, str, etc.)")
    except ZeroDivisionError:
        pass

    return -1


def add_min_max_avg_feats(feats_dict, lst, lst_name):
    """
    :param feats_dict: (dict)
        the feature dictionary to update
    :param lst: (lst)
        the list to aggregate
    :param lst_name: (str)
        a name or short description of the list
    :return: (dict)
        updated feature dictionary
    """
    avg_feat_name = "avg_" + lst_name
    max_feat_name = "max_" + lst_name
    min_feat_name = "min_" + lst_name

    try:
        feats_dict[avg_feat_name] = get_lst_avg(lst)
        feats_dict[max_feat_name] = max(lst)
        feats_dict[min_feat_name] = min(lst)
    except Exception as e:
        print("Empty list or invalid data "
                     "type when trying to get min/max/avg: {} for lst {}".format(e, lst_name))
        feats_dict[avg_feat_name] = -1
        feats_dict[max_feat_name] = -1
        feats_dict[min_feat_name] = -1

    return feats_dict


def get_aggregate_features(feats_dict, iterable_obj, feature_name,
                           preprocessing_fn=lambda x: x,
                           aggregation_fn=add_min_max_avg_feats,
                           allow_nulls=False):
    """
    Utility function that preprocesses elements of an iterable and then applies
    some aggregation function to the list of processed elements.
    Should return an updated feats_dict with the results of the aggregation function

    :param feats_dict: (dict)
        keys are feature-name strings, values are feature-values
    :param iterable_obj: (iterable)
        the iterable object to extract values from
    :param feature_name: ()
        the feature-name to append at the result of our aggregation function
    :param preprocessing_fn: (function)
        preprocessing function that might scale, filter, group values,
        or get values in sub-iterables
    :param aggregation_fn: (function)
        aggregates the processed values to produce feature(s) and store in feats_dict
        ASSUMPTIONS:
            - Takes in three parameters: feats_dict, values_lst, and feature_name
            - return an updated feats_dict, persisting any existing values in the dict
    :param allow_nulls:
        If True, allows None and np.nan in the aggregation list. Otherwise, filters them out.
        Use at your own risk. No assumptions about the model are made.
    :return: (dict)
        updated feature dictionary
    Warning: This function has to come after add_min_max_avg_feats() has been defined
    """
    values_lst = []
    for item in iterable_obj:
        value = preprocessing_fn(item)

        # Filter out None and np.na, unless we explicitly allow them.
        if not pd.isnull(value) or allow_nulls:
            values_lst.append(value)

    return aggregation_fn(feats_dict, values_lst, feature_name)


def get_decayed_rate(years_ago, slowdown_speed=0.02):
    """
    Calculates a decay-adjustment weight for some int; ideally years since some event

    :param years_ago: (int)
        Number of years_ago (int or float) between an event and today
    :param slowdown_speed: (float)
        exponential decay rate
    :return:
        numeric decay value according to recency of exit date
    """
    return np.exp(slowdown_speed * years_ago)
