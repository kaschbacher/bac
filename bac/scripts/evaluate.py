from typing import Sequence
import pandas as pd
import numpy as np

import logging
import sys
import click
import joblib
from pathlib import Path

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, \
    confusion_matrix, classification_report

from bac.util.config import parse_config
from bac.util.io import load_data_partitions, load_feature_labels, load_model
from bac.util.data_cleaning import split_data, fill_missing_data
from bac.models.model_schemas import ModelSchemas
from bac.features.feature_importances import compute_feature_importances

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_model_features_target(model_schema: ModelSchemas, 
        X_test: pd.DataFrame, y_test: pd.Series) -> (pd.DataFrame, pd.Series):
    """Given a ModelSchema, define X and y for evaluation.
    e.g., subset the features or perform another modification.

    Args:
        model_schema (ModelSchemas): A schema obj defining a model, features, target
        X_test (pd.DataFrame): The complete feature set for evaluation
        y_test (pd.Series): The original target for evaluation

    Returns:
        A new X and y, with a subset of features or other modification
    """
    model_name = model_schema["name"]
    logging.info(f"Loading model schema: {model_name}")
    subset_columns = model_schema['features']
    target = model_schema["target"]
    X = X_test[subset_columns]
    y = y_test
    return X, y


def evaluate_model(y_true: pd.Series, y_prob: pd.Series) -> Sequence[float]:
    y_pred = [round(prob) for prob in y_prob]
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob, average='micro', max_fpr=None) 
    avg_precision = average_precision_score(y_true, y_prob)
    logging.info(f'Accuracy: {accuracy}')
    logging.info(f'ROC AUC: {roc_auc}')
    logging.info(f'Average Precision: {avg_precision}')
    return [accuracy, roc_auc, avg_precision, y_pred]

 
def extra_statistics(cm: confusion_matrix) -> Sequence[float]:
    """ Compute additional statistics from the confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(fn+tp)# same as recall
    precision = tp/(tp+fp)
    specificity = 1 - (fp/(tn+fp))
    
    logging.info(f'sensitivity/recall = {sensitivity}')
    logging.info(f'precision = {precision}')
    logging.info(f'specificity = {specificity}')
    logging.info(f'1-specificity = {1 - specificity}\n')
    return sensitivity, precision, specificity


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
    models_base_fpath = eval_cfg["models_fpath"]
    
    # Load Data, subset features
    columns = features_cfg["features_to_keep"]
    test_set = io_cfg["partitions_filenames"][2]
    dfs_map = load_data_partitions(io_cfg["partitions_folder"], [test_set], columns)
    X_test, y_test = split_data(dfs_map['test']) 
    
    # Fill missing data
    [X_test] = fill_missing_data([X_test], missing_value = -999)

	# Loop over models
	# -- Build ModelSchemas -> use to Evaluate multiple trained models
    ms = ModelSchemas(X_test.columns, features_cfg)
    model_stats = {}

    for model_schema in ms.schemas:
        X, y = get_model_features_target(model_schema, X_test, y_test)
    
        # Load the model object
        model_name = model_schema["name"]
        model_fpath = models_base_fpath.format(model_name=model_name)
        model = load_model(model_fpath)
        
        # Get predicted probabilities in the test set
        y_prob = model.do_predict(X)

        # Evaluate model performance
        accuracy, roc_auc, avg_precision, y_pred = evaluate_model(y, y_prob)
        model_stats[model_name] = [accuracy, roc_auc, avg_precision, y_pred]
        
        cnf_matrix = confusion_matrix(y, y_pred)
        logging.info(f"\nConfusion Matrix:\n{cnf_matrix}\n")
        extra_statistics(cnf_matrix)
        
        # Classification Report
        target_names = ['low BAC','high BAC']
        logging.info(classification_report(y, y_pred, target_names=target_names))
        
        #plot_confusion_matrix(cnf_matrix, target_names, normalize=True)
        #fig.savefig('/'.join([FIGURE_FOLDER,'/cm/Normalized_Confusion_Matrix_BAC_Classification_{}.png'.format(TODAY)]), bbox_inches='tight')

        # Plot Shap Values for Best Model
        if model_name == "all":
            base_folder = Path(io_cfg["figures_fpath"])
            output_folder = base_folder / 'shap/'
            compute_feature_importances(model.model, X, output_folder)

if __name__=="__main__":
    main()