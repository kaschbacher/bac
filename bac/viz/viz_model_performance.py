from typing import Sequence
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import datetime as dt
from pathlib import Path
import logging
import sys

from sklearn.metrics import roc_curve, auc

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
plt.rcParams.update(FIGURE_PARAMS)


def get_output_filepath(output_folder: str, pre: str, fname: str='roc') -> Path:
    """Creates a subfolder according to pre.
    Names file according to pre & date.
    Outputs a filepath to save a figure.

    Args:
        output_folder (str): the figures folder
        pre (str): a subfolder for this graph-type
        fname (str): filename without format

    Returns:
        output_fpath: the full filepath to save
    """
    output_folder = Path(output_folder) / pre
    if not Path.exists(output_folder):
        Path.mkdir(output_folder)
        
    #output_fname = Path(f"{fname}_{dt.date.today()}.pdf")
    output_fname = Path(f"{fname}.pdf")
    output_fpath = output_folder / output_fname
    return output_fpath
    
    
def plot_ROC(
    y_test: pd.Series, 
    y_prob: pd.Series, 
    model_name: str,
    output_folder: str='/mnt/data/figures',
    save_plot: bool=True):
    """Plot one ROC curve"""
    # Instantiate
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate x and y for ROC-curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    #print (fpr.shape, tpr.shape, roc_auc)# DEBUG

    #Plot of a ROC curve for a specific class
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
    
    if len(output_folder)<1:
        raise ValueError(f"Invalid output folder given to save ROC plot: {output_folder}.")
    if save_plot:
        output_fpath = str(get_output_filepath(output_folder, 'roc', 'roc_{model_name}'))
        plt.savefig(output_fpath, bbox_inches='tight', dpi=300)
        logging.info(f"Saving ROC-AUC Model Comparison Plot to:\n{output_fpath}")
    plt.show()
 

def plot_ROC_comparison(
        y_true: Sequence[pd.Series], 
        y_prob: Sequence[pd.Series], 
        labels: Sequence[str],
        output_folder: str='/mnt/data/figures',
        save_plot: bool=True):
    """Plot ROC Curves to compare multiple models.

    Args:
        y_true: List, observed y, len = number of models to compare
        y_prob: List, predicted y probabilities, len = n-models
        labels (Sequence[str]): List, plot-labels for each model
        output_folder: str, output path to server to save the plot
        save_plot: Optional boolean to save the plot
    """
    n_models = len(y_true)
    assert len(y_true)==len(y_prob)==len(labels)
    
    #https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    c = ['indigo', 'darkgreen', 'goldenrod', 'darkblue', 'teal', 'purple']
    # reformat so colors has the same length as y_test or n-models
    colors = [c[i % len(c)] for i in range(n_models)]
    
    # Sort by AUC
    auc_df = pd.DataFrame([y_true, y_prob, labels])

    plt.figure()
    
    for y_t, y_p, color, label in zip(y_true, y_prob, colors, labels):
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label='{} AUC: {:0.2f})'.format(label, roc_auc))
    
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    #plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.1, .5), loc='center left', ncol=1)
    
    if len(output_folder)<1:
        raise ValueError(f"Invalid output folder given to save ROC plot: {output_folder}.")
    if save_plot:
        output_fpath = str(get_output_filepath(output_folder, 'roc', 'roc_comparison'))
        plt.savefig(output_fpath, bbox_inches='tight', dpi=300)
        logging.info(f"Saving ROC-AUC Model Comparison Plot to:\n{output_fpath}")
    plt.show()
    
    
# TODO: Revise format to be consistent with above
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
    
    # Save
    plt.savefig('/'.join([FIGURE_FOLDER,'/cm/Normalized_Confusion_Matrix_BAC_Classification_{}.png'.format(TODAY)]), bbox_inches='tight')
    plt.show()
   