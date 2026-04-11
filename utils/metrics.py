'''
Evaluation metrics for the three core GNN task types:
  - Node / graph classification  -> accuracy, f1_score
  - Link prediction              -> auc_roc
'''

import numpy as np


def accuracy(logits, y_true):
    '''
    Classification accuracy.
    logits : (N, C) raw scores or probabilities
    y_true : (N,) integer class labels
    '''
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y_true)


def auc_roc(y_scores, y_true):
    '''
    Area Under the ROC Curve — implemented from scratch via trapezoid rule.
    y_scores : (N,) predicted positive-class probabilities
    y_true   : (N,) binary labels {0, 1}
    '''
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5                         # degenerate case

    tps, fps = 0, 0
    tpr_list, fpr_list = [0.0], [0.0]
    for label in y_true_sorted:
        if label == 1:
            tps += 1
        else:
            fps += 1
        tpr_list.append(tps / n_pos)
        fpr_list.append(fps / n_neg)

    # Trapezoid rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2.0
    return auc


def f1_score(y_pred, y_true, average='macro'):
    '''
    F1 score — macro or binary.
    y_pred : (N,) predicted integer labels
    y_true : (N,) true integer labels
    '''
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1s.append(2 * precision * recall / denom if denom > 0 else 0.0)

    if average == 'macro':
        return float(np.mean(f1s))
    # binary: return score for positive class (class 1)
    pos_idx = list(classes).index(1) if 1 in classes else 0
    return f1s[pos_idx]
