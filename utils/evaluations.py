import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support


def do_prc(scores, true_labels, file_name='test', directory='results', plot=True):
    """ Does the PRC curve

    Args :
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the PRC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the PRC curve or not
    Returns:
            prc_auc (float): area under the under the PRC curve
    """
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    prc_auc = auc(recall, precision)

    idx = (2 * precision * recall / (precision + recall)).argmax()
    thresh = thresholds[idx]

    y_pred = scores.copy()
    y_pred = np.array(y_pred)

    inds = (y_pred < thresh)
    inds_comp = (y_pred >= thresh)

    y_pred[inds] = 0
    y_pred[inds_comp] = 1
    mPrecision, mRecall, mF1, _ = precision_recall_fscore_support(true_labels, y_pred, average='binary')

    print("Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f " % (mPrecision, mRecall, mF1))


    if plot:
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AUC=%0.4f' 
                            %(prc_auc))
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('results/' + file_name + '_prc.jpg')
        plt.close()

    return prc_auc
