from typing import Sequence
import pandas as pd
import numpy as np

import logging
import sys
import click
from joblib import load

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.data_cleaning import split_data, fill_missing_data
from bac.util.config import parse_config
from bac.models.model_schemas import ModelSchemas

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# def evaluate_model(y_true, y_pred, y_proba, out):
#     accuracy = accuracy_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_proba, average='micro', max_fpr=None)#sample_weight=scale_pos_weight, 
#     print ('Accuracy: {:02.2f}%'.format(accuracy*100.))
#     print ('ROC AUC: {:02.2f}%'.format(roc_auc*100.))
#     # Log 
#     print ('Accuracy: {:02.2f}%'.format(accuracy*100.), file=out) 
#     print ('ROC AUC: {:02.2f}%'.format(roc_auc*100.), file=out)
#     return accuracy, roc_auc

 
# def extra_statistics(cm, outfile):
#     tn = cm[0][0]
#     fp = cm[0][1]
#     fn = cm[1][0]
#     tp = cm[1][1]
#     print('sensitivity = recall = {0:4f}'.format(tp/(fn+tp)))
#     print('sensitivity = recall = {0:4f}'.format(tp/(fn+tp)), file=outfile)
#     print('precision = {:4f}'.format(tp/(tp+fp)))
#     print('precision = {:4f}'.format(tp/(tp+fp)), file=outfile)
#     print('specificity = {:4f}'.format(1 - (fp/(tn+fp))))
#     print('specificity = {:4f}'.format(1 - (fp/(tn+fp))), file=outfile)
#     print('1-specificity = {:4f}\n'.format(fp/(tn+fp)))
#     print('1-specificity = {:4f}\n'.format(fp/(tn+fp)), file=outfile)
    
    
# def full_report(figure_params, y_test, y_pred, target_names, outfile):
    
#     # # Plot Confusion Matrix for Classification - http://scikit-learn.org/stable/modules/model_evaluation.html
#     plt.rcParams.update(figure_params)

#     cnf_matrix = confusion_matrix(y_test, y_pred)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

#     fig = plt.figure()
#     plot_confusion_matrix(cnf_matrix, ['low BAC','high BAC'], normalize=True)
#     #fig.savefig('/'.join([FIGURE_FOLDER,'/cm/Normalized_Confusion_Matrix_BAC_Classification_{}.png'.format(TODAY)]), bbox_inches='tight')

#     # Classification Report
#     #print ('\nAccuracy: {:02.2f}%\n'.format(accuracy*100.))
#     print (classification_report(y_test, y_pred, target_names=target_names), file=outfile)
#     extra_statistics([[tn, fp], [fn, tp]], outfile)

def load_serialized_models(ms: ModelSchemas, models_fpath: str) -> Sequence:
    """Load serialized model objects

    Args:
        ms (ModelSchemas): contains names for each model
        models_fpath (str): base filepath, needs names inserted

    Returns:
        List: a list of serialized models
    """
    saved_models = []
    temp_path = ''
    for model_schema in ms.schemas:
        model_name = model_schema["name"]
        temp_path = models_fpath.format(model_name)
        logging.info(f"Loading model schema from: {temp_path}")
        model = load(temp_path)
        saved_models.append(model)
    return saved_models

@click.command()
@click.argument(
    "io_config", type=click.Path(exists=True), default="/mnt/configs/io.yml",
)
@click.argument(
    "features_config", type=click.Path(exists=True), default="/mnt/configs/features.yml"
)
@click.argument(
    "eval_config", type=click.Path(exists=True), default='/mnt/configs/evaluate.yml'
)
def main(
    io_config = "/mnt/configs/io.yml", 
    features_config = "/mnt/configs/features.yml",
    eval_config = '/mnt/configs/evaluate.yml'
    ):
    """Load serialized models and evaluate

    Args:
        io_config: filepaths to load and save data
        features_config: Used to define feature sets for each model
        eval_config: filepath base for saved serialized models
    """
    
    # Load Configs:  Paths, Features, Saved Model-filepaths
    # TODO: eval_config can be part of io_config, if it doesn't need more args
    logger.info(f"Loading configs...")
    io_cfg = parse_config(io_config)
    features_cfg = parse_config(features_config)
    eval_cfg = parse_config(eval_config)
    models_fpath = eval_cfg["models_fpath"]
    
    # Load Data, subset features
    columns = features_cfg["features_to_keep"]
    test_set = io_cfg["partitions_filenames"][2]
    dfs_map = load_data_partitions(io_cfg["partitions_folder"], [test_set], columns)
    X_test, y_test = split_data(dfs_map['test']) 


	# Loop over models

	# -- predict -> proba
	# -- evaluate -> get model_eval object with associated stats per model


	# Build ModelSchemas -> use to Evaluate Multiple Models
    ms = ModelSchemas(X_test.columns, features_cfg)
    #logging.info(f"{ms.schemas}")
    
    saved_models = load_serialized_models(ms, models_fpath)
    print(len(saved_models))

    # for model_schema in ms.schemas:
    # 	# Set X and y
    #     model_name = model_schema["name"]
    #     logging.info(f"Loading model schema: {model_name}")
    #     subset_columns = model_schema['features']
    #     target = model_schema["target"]
    #     #X = X_test[subset_columns]
    #     #y = y_test
    

        
    
    

if __name__=="__main__":
    main()