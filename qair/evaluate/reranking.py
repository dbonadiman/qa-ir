import numpy as np
from sklearn import metrics


def zip_ex(exs):
    lbls = []
    pred = []
    for ex in exs:
        pred.append(ex.prediction)
        lbls.append(ex.label)
    return lbls, pred

def filter_dataset(dictionary, mode):

    def sort(cand):
        return list(sorted(cand, reverse=True, key=lambda x: x.prediction))

    filtered = {}
    for qid, candidates in dictionary.items():
        if not candidates:
            if mode in ['all_same', 'all_neg', 'empty']:
                continue
        else:
            label = [ex.label for ex in candidates]
            ln = len(label)
            sm = sum(label)
            if sm == 0 and mode in ['all_same', 'all_neg']:
                continue
            elif sm == ln and mode in ['all_same']:
                continue  
        filtered[qid] = sort(candidates)
    return filtered

def MAP(dataset, fltr='all_same'):
    dictionary = filter_dataset(dataset.q2e, mode=fltr)
    average_precisions = []
    for _, candidates in dictionary.items():
        if len(candidates) == 0:
            average_precisions.append(0.)
        else:
            label, scores_ = zip_ex(candidates)
            if sum(label) == 0:
                average_precisions.append(0.)
            else:
                average_precisions.append(metrics.average_precision_score(label, scores_))
    return np.mean(average_precisions)


def MRR(dataset, fltr='all_same'):
    dictionary = filter_dataset(dataset.q2e, mode=fltr)
    reciprocal_ranks = []
    for _, candidates in dictionary.items():
        ok = False
        label, _ = zip_ex(candidates)
        for i, lbl in enumerate(label, 1):
            if lbl > 0:
                reciprocal_ranks.append(1. / i)
                ok = True
                break
        if not ok:
            reciprocal_ranks.append(0.)
    return np.mean(reciprocal_ranks)

def P_at_n(dataset, n=1, fltr='all_same'):
    dictionary = filter_dataset(dataset.q2e, mode=fltr)
    hits_at_n = []
    for _, candidates in dictionary.items():
        if len(candidates) > 0:
            labels, _ = zip_ex(candidates[:n])
            hits_at_n.append(1. if 1 in set(labels) else 0.)
    return np.mean(hits_at_n)

def predictions_labels(dataset, mode='empty'):
    dictionary = filter_dataset(dataset.q2e, mode=mode)
    pred, lbl = [],  []
    for qid, lst in dictionary.items():
        for ex in lst:
            pred.append(float(ex.prediction))
            lbl.append(int(ex.label))
    return pred, lbl

def treshold(predictions, th=0.5):
    return [1 if p > th else 0 for p in predictions]

def roc(dataset, mode='empty'):
    predictions, labels = predictions_labels(dataset, mode)
    return metrics.roc_auc_score(labels, predictions)

def precision(dataset, th=0.5, mode='empty'):
    predictions, labels = predictions_labels(dataset, mode)
    return metrics.precision_score(labels, treshold(predictions, th))

def recall(dataset, th=0.5, mode='empty'):
    predictions, labels = predictions_labels(dataset, mode)
    return metrics.recall_score(labels, treshold(predictions, th))

def f1(dataset, th=0.5, mode='empty'):
    predictions, labels = predictions_labels(dataset, mode)
    return metrics.f1_score(labels, treshold(predictions, th))


def accuracy(dataset, th=0.5, mode='empty'):
    predictions, labels = predictions_labels(dataset, mode)
    return metrics.accuracy_score(labels, treshold(predictions, th))


def at_stats(dataset, th=0.5, mode='empty'):
    dictionary = filter_dataset(dataset.q2e, mode=mode)
    tp, tn, fp, fn = 0, 0, 0, 0
    for _, candidates in dictionary.items():
        if len(candidates) == 0:
            tn += 1
        else:
            label, scores = zip_ex(candidates)
            if sum(label) == 0:
                if scores[0] > th:
                    fp += 1
                else:
                    tn += 1
            else:
                if scores[0] > th and label[0] == 1:
                    tp += 1
                elif scores[0] > th:
                    fp += 1
                else:
                    fn += 1
    return tp, tn, fp, fn




def at_precision(dataset, th=0.5, mode='empty'):
    tp, tn, fp, fn = at_stats(dataset, th)
    return tp/(tp+fp+1e-13)

def at_recall(dataset, th=0.5, mode='empty'):
    tp, tn, fp, fn = at_stats(dataset, th)
    return tp/(tp+fn+1e-13)

def at_f1(dataset, th=0.5, mode='empty'):
    p = at_precision(dataset, th)
    r = at_recall(dataset, th)
    return 2*p*r/(p+r+1e-13)


def evaluate(dataset, th=0.5, mode='all_same'):
    metrics = {
        "P@1": P_at_n(dataset, 1, mode),
        "MAP": MAP(dataset, mode),
        "MRR": MRR(dataset, mode),
        "Prec": precision(dataset, th, mode),
        "Rec": recall(dataset, th, mode),
        "F1":f1(dataset, th, mode),
        "roc_auc": roc(dataset, mode),
        "accuracy":accuracy(dataset, th, mode),
        "answer triggering precision": at_precision(dataset, th),
        "answer triggering recall": at_recall(dataset, th),
        "answer triggering f1":at_f1(dataset, th),
    }
    return metrics
