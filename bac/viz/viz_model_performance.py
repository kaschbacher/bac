import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import datetime as dt
import os

from sklearn.metrics import roc_auc_score, roc_curve, auc, \
    accuracy_score, confusion_matrix, classification_report


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


def get_output_filepath(output_folder: str, pre: str, fname: str='roc') -> output_fpath:
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
    output_folder = os.join.path(output_folder, pre)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    output_fname = f"{fname}_{dt.date.today()}.pdf"
    output_fpath = os.join(output_folder, output_fname)
    return output_fpath
    
    
def plot_ROC(y_test: pd.Series, y_proba: pd.Series, output_folder: str):
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate x and y for ROC-curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
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
    
    output_fpath = get_output_filepath(output_folder, 'roc')
    plt.savefig(output_fpath, bbox_inches='tight', dpi=300)
    plt.show()
 

def plot_ROC_comparison(y_true, y_prob, labels):
    #y_test and y_prob will be lists of however many groups are in the comparison
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
    plt.savefig('/'.join([FIGURE_FOLDER, 'roc/roc_comparison_{}.png'.format(TODAY)]), bbox_inches='tight', dpi=300)
    plt.show()
    
    
    
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
   