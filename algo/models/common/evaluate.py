# Created by Hansi at 12/27/2021
from sklearn.metrics import recall_score, precision_score, f1_score


def get_eval_results(actuals, predictions):
    results = dict()
    r = recall_score(actuals, predictions, average='macro')
    results['recall'] = r
    p = precision_score(actuals, predictions, average='macro')
    results['precision'] = p
    f1 = f1_score(actuals, predictions, average='macro')
    results['f1'] = f1
    return results


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

