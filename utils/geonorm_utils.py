import copy
from geopy.distance import great_circle
import numpy as np
import matplotlib.pyplot as plt
from openlocationcode import openlocationcode as olc
import pandas as pd
import math


def geonames2coords_dict(path):
    """
    Given the path to allCountries.txt file from geonames, generate a dictionary
    to map geonames ids to coordinates
    """
    # lee el archivo y selecciona solo las columnas especificadas
    geonames_df = pd.read_csv(path, usecols=[0, 1, 4, 5], sep='\t', low_memory=True,
                              header=None, names=['geoname_id', 'name', 'lat', 'long'])

    geonames_df = geonames_df.set_index('geoname_id')
    diccionario = geonames_df[['lat', 'long']].apply(tuple, axis=1).to_dict()
    return diccionario


def checkPluscode(pluscode):
    """
    Check if pluscode is correctly formated
    """

    return olc.isValid(pluscode)


def pluscode2coord(pluscode):
    """
    Transform pluscode list to latitude/longitude coordinates
    """
    try:
        coords = olc.decode(pluscode)
    except:

        return (None, None)

    return (coords.latitudeCenter, coords.longitudeCenter)


def calculate_scores(predictions, gold_standard, inspect=False, task="", write=True):
    """
    CODE Adaptaded from: https://github.com/milangritta/WhatsMissingInGeoparsing/blob/master/methods.py#L173
    Given the predictions and the gold annotations, calculate precision, recall, F Score and accuracy.
    :param inspect: If True, the differences between gold and predicted files will be printed
    :param predicted_list: List of lists. Each list represent the predictions made by the participants
     for each document. Each inner list should have the structure["doc_id","coords","span_ini",span_end"]
    :param gold: List of lists. Each list represent the gold standard labels for each document. Each
     inner list should have the structure ["doc_id","coords","span_ini",span_end"]
    :return: a list of errors per toponym i.e how far away is each correctly identified toponym from
    the gold location. This is used to measure the accuracy of the geocoding part
    """
    tp, fp, fn = 0.0, 0.0, 0.0
    accuracy = {}
    toponym_index = -1
    # Iterate over mentiones of each document
    # Dictonary is made to each document per key, making sure that if document is not in prediction, it will not affect other documents.
    dict1 = {}
    dict2 = {}
    for document in gold_standard:
        document_id = document[0][3]
        dict1[document_id] = document
    for document in predictions:
        document_id = document[0][3]
        dict2[document_id] = document

    for document_id in dict1.keys():
        gold_doc = dict1[document_id]
        if document_id not in dict2.keys():
            # this means that this document is not in the prediction file, therefore its empty
            predicted_doc = []
        else:
            predicted_doc = dict2[document_id]
        for gold_top in gold_doc[:]:
            # sumamos un índice para el AUC
            toponym_index += 1
            for predicted_top in predicted_doc[:]:
                # print(gold_top, predicted_top)
                if task == "PC":
                    # Remove everything after the '+' sign for a more lenient evaluation
                    gold_top[4] = gold_top[4].split("+")[0]
                    predicted_top[4] = predicted_top[4].split("+")[0]
                    # Spans are the same
                    if (gold_top[1] == predicted_top[1]) & (gold_top[2] == predicted_top[2]):
                        # Gold normalization is NOCODE -> ignore completely
                        if (gold_top[0] == (None, None)):
                            predicted_doc.remove(predicted_top)
                            gold_doc.remove(gold_top)
                        # Codes are the same in GS and prediction
                        elif ((gold_top[4] == predicted_top[4])):
                            # Add a true positive
                            tp += 1
                            # Remove elements from list to calculate later false positives and false negatives
                            predicted_doc.remove(predicted_top)
                            gold_doc.remove(gold_top)
                            # Get coordinates
                            predicted_coord = predicted_top[0]
                            gold_coord = gold_top[0]
                            if write:
                                # Calculate accuracy as paper said
                                # The logarithm ensure that the difference between small errors is more significant than the same different
                                # between large error.
                                accuracy[toponym_index] = np.log(
                                    1 + great_circle(predicted_coord, gold_coord).kilometers)
                        # Codes are not the same
                        else:
                            predicted_coord = predicted_top[0]
                            gold_coord = gold_top[0]
                            # If the predicted code is not valid, default to the furthest point from GS (i.e. antipodes)
                            if predicted_coord == (None, None):
                                lat = -gold_coord[0]
                                long = gold_coord[1] - \
                                       180 if gold_coord[1] > 0 else gold_coord[1] + 180
                                predicted_coord = (lat, long)
                            if write:
                                accuracy[toponym_index] = np.log(
                                    1 + great_circle(predicted_coord, gold_coord).kilometers)

                        break
                # Task isn't PC and spans are the same
                elif (gold_top[1] == predicted_top[1]) & (gold_top[2] == predicted_top[2]):
                    # If we are not calculating distance metrics, we can skip some steps
                    if (write == False):
                        # Codes match
                        if (gold_top[0] == predicted_top[0]):
                            # Add true positive
                            tp += 1
                            # Removve elements from list to calculate later false positives and false negatives
                            predicted_doc.remove(predicted_top)
                            gold_doc.remove(gold_top)

                        # task 4
                    # If we want to calculate distance metrics and codes match.
                    elif (gold_top[0] == predicted_top[0]):
                        # Add a true positive
                        tp += 1
                        # Get coordinates
                        predicted_coord = predicted_top[0]
                        gold_coord = gold_top[0]
                        # Remove elements from list to calculate later false positives and false negatives
                        predicted_doc.remove(predicted_top)
                        gold_doc.remove(gold_top)
                        # Calculate accuracy as paper said
                        # The logarithm ensures that the difference between small errors is more significant than the same difference
                        # between large errors.
                        if write:
                            accuracy[toponym_index] = np.log(
                                1 + great_circle(predicted_coord, gold_coord).kilometers)
                    # Ignore completely NOCODE
                    elif ((type(gold_top[0]) is tuple) == False):
                        gold_doc.remove(gold_top)
                        predicted_doc.remove(predicted_top)
                    # if there is an invalid code, then the opposite distance is taken from the goldstandard.
                    elif ((type(predicted_top[0]) is tuple) == False):
                        # the prediction was wrong
                        predicted_coord = predicted_top[0]
                        gold_coord = gold_top[0]
                        lat = -gold_coord[0]
                        long = gold_coord[1] - \
                            180 if gold_coord[1] > 0 else gold_coord[1] + 180
                        predicted_coord = (lat, long)

                        if write:
                            accuracy[toponym_index] = np.log(
                                1 + great_circle(predicted_coord, gold_coord).kilometers)

                    # annotation found but the code is not correct then only the distance is calculated
                    else:
                        predicted_coord = predicted_top[0]
                        gold_coord = gold_top[0]
                        if write:
                            accuracy[toponym_index] = np.log(
                                1 + great_circle(predicted_coord, gold_coord).kilometers)

                    break
        fp += len(predicted_doc)
        fn += len(gold_doc)
    f_score = (tp, fp, fn)
    print(f_score)
    output = {"f_score": f_score, "accuracy": accuracy}
    return output


def print_stats(accuracy, scores=None, plot=False, write=True):
    """
    CODE taken from: https://github.com/milangritta/WhatsMissingInGeoparsing/blob/master/methods.py#L173
    Take the list of errors and calculate the accuracy of the geocoding step, optionally plot as well.
    :param scores: A tuple (true_positive, false_positive, false_negative) to calculate the F Score
    :param accuracy: A list of geocoding errors per toponym i.e. how far off in km from true coordinates
    :param plot: whether to plot the accuracy line by toponym
    :return: N/A
    """

    MAX_ERROR = 20039  # Furthest distance between two points on Earth, i.e the circumference / 2
    if scores is not None:
        precision = scores[0] / (scores[0] + scores[1])
        recall = scores[0] / (scores[0] + scores[2])
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
    if write:
        median = np.median(sorted(np.exp(accuracy)))
        mean = np.mean(np.exp(accuracy))
        # This is the k in accuracy@k metric (see my Survey Paper for details)
        k = np.log(161)
        # accuracy_at_161 = sum([1.0 for dist in accuracy if dist < k]) / len(accuracy)
        if len(accuracy) != 0:
            accuracy_at_161 = sum(
                [1.0 for dist in accuracy if dist < k]) / (len(accuracy))
        else:
            accuracy_at_161 = 0

        auc_geo = np.trapz(accuracy) / (np.log(MAX_ERROR)
                                        * (len(accuracy) - 1))
    else:
        median = 0
        mean = 0
        k = 0
        accuracy_at_161 = 0
        auc_geo = 0

    return {"accuracy_at_161": accuracy_at_161, "auc": auc_geo, "mean": mean, "median": median, "f_score": f_score, "precision": precision, "recall": recall}


def calculat_fscore(gold_standard, predictions):
    tp = 0
    fp = 0
    fn = 0
    # Dictonary is made to each each document per key, making sure that if document is not in prediction, it will not affect other documents.
    dict1 = {}
    dict2 = {}
    for document in gold_standard:
        document_id = document[0][3]
        dict1[document_id] = document
    for document in predictions:
        document_id = document[0][3]
        dict2[document_id] = document

    for document_id in dict1.keys():
        gold_doc = dict1[document_id]
        if document_id not in dict2.keys():
            # this means that this document is not in the prediction file, therefore its empty
            predicted_doc = []
        else:
            predicted_doc = dict2[document_id]
        for gold_top in gold_doc[:]:
            # sumamos un índice para el AUC

            for predicted_top in predicted_doc[:]:
                # if nocode in goldstandard then, it is ignored
                if (gold_top[0] == predicted_top[0]) & (gold_top[1] == predicted_top[1]) & (gold_top[-1] == "NOCODE"):
                    predicted_doc.remove(predicted_top)
                    gold_doc.remove(gold_top)
                    break
                elif set(gold_top) == set(predicted_top):
                    # Add a true positive
                    tp += 1
                    # Removve elements from list to calculate later false positives and false negatives
                    predicted_doc.remove(predicted_top)
                    gold_doc.remove(gold_top)
                    break
        fp += len(predicted_doc)

        fn += len(gold_doc)

    scores = (tp, fp, fn)
    if scores is not None:
        precision = scores[0] / (scores[0] + scores[1])
        recall = scores[0] / (scores[0] + scores[2])
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)

    return {"recall": recall, "precision": precision, "f_score": f_score}


def calculat_fscore_per_entity(gold_standard, predictions, strict=True):

    # store the different labels in the test dataset.
    labels = []
    for predicted_doc_2, gold_doc_2 in zip(predictions, gold_standard):
        for gold_top in predicted_doc_2:
            if gold_top[-1] not in labels:
                labels.append(gold_top[-1])
    # Dictonary is made to each document per key, making sure that if document is not in prediction, it will not affect other documents.
    dict1 = {}
    dict2 = {}
    for document in gold_standard:
        document_id = document[0][0]
        dict1[document_id] = document
    for document in predictions:
        document_id = document[0][0]
        dict2[document_id] = document
    scores_final = {}
    TP = 0
    FP = 0
    FN = 0
    o_TP = 0
    o_FP = 0
    o_FN = 0
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        overlapping_tp = 0
        overlapping_fp = 0
        overlapping_fn = 0
        for document_id in dict1.keys():
            gold_doc_ = dict1[document_id]
            if document_id not in dict2.keys():
                # this means that this document is not in the prediction file, therefore its empty
                predicted_doc_ = []
            else:
                predicted_doc_ = dict2[document_id]
            predicted_doc = copy.deepcopy(predicted_doc_)
            predicted_doc = list(
                filter(lambda x: x[-1] == label, predicted_doc))
            gold_doc = copy.deepcopy(gold_doc_)
            gold_doc = list(filter(lambda x: x[-1] == label, gold_doc))

            for gold_top in gold_doc[:]:
                for predicted_top in predicted_doc[:]:

                    if set(gold_top) == set(predicted_top):
                        # Add a true positive
                        tp += 1
                        # Remove elements from list to calculate later false positives and false negatives
                        predicted_doc.remove(predicted_top)
                        gold_doc.remove(gold_top)
                        break
            if strict == False:
                overlapping_predicted_doc = copy.deepcopy(predicted_doc_)
                overlapping_predicted_doc = list(
                    filter(lambda x: x[-1] == label, overlapping_predicted_doc))
                overlapping_gold_doc = copy.deepcopy(gold_doc_)
                overlapping_gold_doc = list(
                    filter(lambda x: x[-1] == label, overlapping_gold_doc))
                for gold_top in overlapping_gold_doc[:]:
                    for predicted_top in overlapping_predicted_doc[:]:
                        if is_overlap_match(gold_top, predicted_top):
                            overlapping_tp += 1
                            overlapping_predicted_doc.remove(predicted_top)
                            overlapping_gold_doc.remove(gold_top)
                            break

            fp += len(predicted_doc)
            fn += len(gold_doc)

            if strict == False:
                overlapping_fp += len(overlapping_predicted_doc)
                overlapping_fn += len(overlapping_gold_doc)
        o_TP += overlapping_tp
        o_FP += overlapping_fp
        o_FN += overlapping_fn
        TP += tp
        FP += fp
        FN += fn
        scores = (tp, fp, fn)
        overlapping_scores = (overlapping_tp, overlapping_fp, overlapping_fn)
        if scores is not None:

            precision_key = label+"_precision"
            recall_key = label+"_recall"
            f_score_key = label+"_f_score"
            precision = scores[0] / (scores[0] + scores[1])
            recall = scores[0] / (scores[0] + scores[2])
            if precision == 0 or recall == 0:
                f_score = 0
            else:
                f_score = 2 * precision * recall / (precision + recall)

            scores_final[precision_key] = precision
            scores_final[recall_key] = recall
            scores_final[f_score_key] = f_score
        if strict == False:
            if overlapping_scores is not None:

                precision_key = "overlapping_"+label+"_precision"
                recall_key = "overlapping_"+label+"_recall"
                f_score_key = "overlapping_"+label+"_f_score"
                precision = overlapping_scores[0] / \
                    (overlapping_scores[0] + overlapping_scores[1])
                recall = overlapping_scores[0] / \
                    (overlapping_scores[0] + overlapping_scores[2])
                if precision == 0 or recall == 0:
                    f_score = 0
                else:
                    f_score = 2 * precision * recall / (precision + recall)
                scores_final[precision_key] = precision
                scores_final[recall_key] = recall
                scores_final[f_score_key] = f_score

    micro_recall = TP / (TP + FN)
    micro_precision = TP / (TP + FP)
    micro_F1 = TP / (TP + (0.5*(FP+FN)))

    if strict == False:
        overlapping_micro_recall = o_TP / (o_TP + o_FN)
        overlapping_micro_precision = o_TP / (o_TP + o_FP)
        overlapping_micro_F1 = o_TP / (o_TP + (0.5*(o_FP+o_FN)))
        scores_final["overlapping_f_score"] = overlapping_micro_F1
        scores_final["overlapping_recall"] = overlapping_micro_recall
        scores_final["overlapping_precision"] = overlapping_micro_precision
    scores_final["f_score"] = micro_F1
    scores_final["recall"] = micro_recall
    scores_final["precision"] = micro_precision
    return scores_final


def is_overlap(a, b):
    return b[1] <= a[1] <= b[2] or a[1] <= b[1] <= a[2]


def is_overlap_match(a, b):
    # return is_overlap(a, b) and a.ptid == b.ptid
    # Spanish task
    return is_overlap(a, b) and a[0] == b[0] and a[4] == b[4]
