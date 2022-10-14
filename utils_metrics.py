

import torch


def update_confusion_matrix(lab, pred, confusion_dict):
    if lab == 0 and pred == 0:
        confusion_dict['e_tp'] += 1
    elif lab == 0 and pred != 0:
        confusion_dict['e_fn'] += 1
    elif lab != 0 and pred == 0:
        confusion_dict['e_fp'] += 1
    elif lab != 0 and pred != 0:
        confusion_dict['e_tn'] += 1


    if lab == 1 and pred == 1:
        confusion_dict['n_tp'] += 1
    elif lab == 1 and pred != 1:
        confusion_dict['n_fn'] += 1
    elif lab != 1 and pred == 1:
        confusion_dict['n_fp'] += 1
    elif lab != 1 and pred != 1:
        confusion_dict['n_tn'] += 1


    if lab == 2 and pred == 2:
        confusion_dict['c_tp'] += 1
    elif lab == 2 and pred != 2:
        confusion_dict['c_fn'] += 1
    elif lab != 2 and pred == 2:
        confusion_dict['c_fp'] += 1
    elif lab != 2 and pred != 2:
        confusion_dict['c_tn'] += 1

    return confusion_dict


def calculate_f1_and_print(confusion_dict, model_log) -> None:

    if  confusion_dict['e_tp'] > 0:
        e_recall = confusion_dict['e_tp'] / (confusion_dict['e_tp'] + confusion_dict['e_fn'])
        e_precision = confusion_dict['e_tp'] / (confusion_dict['e_tp'] + confusion_dict['e_fp'])
        e_f1_score = 2*e_recall*e_precision/(e_precision + e_recall)

    if confusion_dict['n_tp'] > 0:
        n_recall = confusion_dict['n_tp'] / (confusion_dict['n_tp'] + confusion_dict['n_fn'])
        n_precision = confusion_dict['n_tp'] / (confusion_dict['n_tp'] + confusion_dict['n_fp'])
        n_f1_score = 2*n_recall*n_precision/(n_precision + n_recall)

    if confusion_dict['c_tp'] > 0:
        c_recall = confusion_dict['c_tp'] / (confusion_dict['c_tp'] + confusion_dict['c_fn'])
        c_precision = confusion_dict['c_tp'] / (confusion_dict['c_tp'] + confusion_dict['c_fp'])
        c_f1_score = 2*c_recall*c_precision/(c_precision + c_recall)

    all_tp = confusion_dict['e_tp'] + confusion_dict['n_tp'] + confusion_dict['c_tp']
    all_fp = confusion_dict['e_fp'] + confusion_dict['n_fp'] + confusion_dict['c_fp']
    all_fn = confusion_dict['e_fn'] + confusion_dict['n_fn'] + confusion_dict['c_fn']
    all_tn = confusion_dict['e_tn'] + confusion_dict['n_tn'] + confusion_dict['c_tn']

    if all_tp > 0:
        all_recall = all_tp / (all_tp + all_fn)
        all_precision = all_tp / (all_tp + all_fp)

    if confusion_dict['e_tp'] > 0 and confusion_dict['n_tp'] > 0 and confusion_dict['c_tp'] > 0:
        macro_f1 = (e_f1_score + n_f1_score + c_f1_score)/3
        model_log.msg(["F1 entailment: " + str(e_f1_score)])
        model_log.msg(["F1 neutral: " + str(n_f1_score)])
        model_log.msg(["F1 contradiction: " + str(c_f1_score)])
        model_log.msg(["macro F1: " + str(macro_f1)])
    else:
        model_log.msg(["Not all classes have at least one correct prediction"])

    if all_tp > 0:
        micro_f1 = 2*all_recall*all_precision/(all_recall+all_precision)
        model_log.msg(["micro F1: " + str(micro_f1)])
    else:
        model_log.msg(["No correct predictions"])



