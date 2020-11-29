from typing import Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import logging
import sys
import itertools

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.data_cleaning import limit_features, split_data, fill_missing_data
from bac.util.config import parse_config

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


FIGURE_PARAMS = {
    'figure.figsize':(2.75,2.75),
    'figure.titlesize':11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'axes.labelsize': 11,
    'lines.linewidth': 1.5,
    'legend.fontsize': 8.5
}

def evaluate_model(y_true, y_pred, y_proba, out):
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba, average='micro', max_fpr=None)#sample_weight=scale_pos_weight, 
    print ('Accuracy: {:02.2f}%'.format(accuracy*100.))
    print ('ROC AUC: {:02.2f}%'.format(roc_auc*100.))
    # Log 
    print ('Accuracy: {:02.2f}%'.format(accuracy*100.), file=out) 
    print ('ROC AUC: {:02.2f}%'.format(roc_auc*100.), file=out)
    return accuracy, roc_auc


# #### Graphing Functions


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix\n")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(4,4))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    plt.ylim([-0.5, 1.5])
    plt.gca().invert_yaxis()
        
    # Annotate    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=13,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    

    plt.tight_layout()
    plt.show()
    # Save
    plt.savefig('/'.join([FIGURE_FOLDER,'/cm/Normalized_Confusion_Matrix_BAC_Classification_{}.png'.format(TODAY)]), bbox_inches='tight')
    
    
def extra_statistics(cm, outfile):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    print('sensitivity = recall = {0:4f}'.format(tp/(fn+tp)))
    print('sensitivity = recall = {0:4f}'.format(tp/(fn+tp)), file=outfile)
    print('precision = {:4f}'.format(tp/(tp+fp)))
    print('precision = {:4f}'.format(tp/(tp+fp)), file=outfile)
    print('specificity = {:4f}'.format(1 - (fp/(tn+fp))))
    print('specificity = {:4f}'.format(1 - (fp/(tn+fp))), file=outfile)
    print('1-specificity = {:4f}\n'.format(fp/(tn+fp)))
    print('1-specificity = {:4f}\n'.format(fp/(tn+fp)), file=outfile)
    
    
def full_report(figure_params, y_test, y_pred, target_names, outfile):
    
    # # Plot Confusion Matrix for Classification - http://scikit-learn.org/stable/modules/model_evaluation.html
    plt.rcParams.update(figure_params)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, ['low BAC','high BAC'], normalize=True)
    #fig.savefig('/'.join([FIGURE_FOLDER,'/cm/Normalized_Confusion_Matrix_BAC_Classification_{}.png'.format(TODAY)]), bbox_inches='tight')

    # Classification Report
    #print ('\nAccuracy: {:02.2f}%\n'.format(accuracy*100.))
    print (classification_report(y_test, y_pred, target_names=target_names), file=outfile)
    extra_statistics([[tn, fp], [fn, tp]], outfile)
    
    
def plot_ROC(y_test, y_pred_proba):
    # y_test can also be y_dev, depending on context
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate x and y for ROC-curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    #print (fpr.shape, tpr.shape, roc_auc)# DEBUG

    #Plot of a ROC curve for a specific class
    plt.rcParams.update(FIGURE_PARAMS)
    plt.figure()
    plt.plot(fpr, tpr, color='#336699',
             label='AUC: {:0.2f})'.format(roc_auc))#lw=2, 
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')#lw=2, 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    #plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('/'.join([FIGURE_FOLDER,'roc/roc_{}.png'.format(TODAY)]), bbox_inches='tight', dpi=300)
 

def plot_ROC_comparison(y_true, y_prob, labels):
    #y_test and y_prob will be lists of however many groups are in the comparison
    plt.rcParams.update(FIGURE_PARAMS)
    plt.figure()
    #https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    colors = ['indigo','darkgreen','goldenrod']
    c = []# ensure that c is the list of colors that is the same length as y_test, i.e., as the number of models to be graphed
    assert len(y_true)==len(y_prob)==len(labels)
    for i in range(len(y_true)):
        if i==0:
           c.append(colors[0])
        else:
           c.append(colors[i%len(colors)])

    for y_t, y_p, color, label in zip(y_true, y_prob, c, labels):
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label='{} AUC: {:0.2f})'.format(label, roc_auc))
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('/'.join([FIGURE_FOLDER, 'roc/roc_comparison_{}.png'.format(TODAY)]), bbox_inches='tight', dpi=300)
