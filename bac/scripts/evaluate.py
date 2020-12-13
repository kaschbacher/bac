from typing import Sequence
import pandas as pd
import numpy as np

import logging
import sys
import click
import joblib
from pathlib import Path
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, \
    confusion_matrix, classification_report

from bac.util.config import parse_config
from bac.util.io import load_data_partitions, load_feature_labels, load_model
from bac.util.data_cleaning import split_data, fill_missing_data
from bac.models.model_schemas import ModelSchemas
from bac.viz.viz_model_performance import plot_ROC_comparison
from bac.features.feature_importances import compute_feature_importances

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# TODO: Would have to configure a logger elsewhere for all files to use
# handler = logging.FileHandler(filename='/mnt/data/logs/evaluate.log', mode='w')
# logger.addHandler(handler)


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


class ModelEvaluator():
    
    def __init__(self, y_true: pd.Series, y_prob: pd.Series):
        """Evaluate a classification model's performance, 
        given the vectors of true observatiosn & predicted probabilities

        Args:
            y_true (pd.Series): Obs y-values
            y_prob (pd.Series): Predicted probabilities
        """
        self.y_true = y_true
        self.y_prob = y_prob
        
        # Assumes a discrimination threshold of .5
        # TODO: May want to implement ability to use other thresholds
        self.y_pred = [round(prob) for prob in y_prob]
        
        # Compute model performance metrics for an sklearn classifier
        self.metrics = OrderedDict()
        self.cm = None
        
        self.get_classifier_metrics()
        self.get_confusion_matrix_metrics()
        self.log_metrics()


    def log_metrics(self):
        for score_type, value in self.metrics.items():
            logging.info(f'Evaluator -- {score_type}: {value}.')
        logging.info('\n')


    def get_classifier_metrics(self):
        """Evaluate a classification model's performance with standard metrics.
        Assigns a dictionary obj for self.metrics:
        """
        assert self.y_true.size>0
        assert self.y_prob.size == self.y_true.size
        
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_prob, average='micro', max_fpr=None) 
        self.metrics['f1_score'] = f1_score(self.y_true, self.y_pred, average='micro')


    def get_confusion_matrix(self):
        """ Compute the confusion matrix for a classifier model."""
        self.cm = confusion_matrix(self.y_true, self.y_pred)
        logging.info(f"\nConfusion Matrix:\n{self.cm}\n")

 
    def get_confusion_matrix_metrics(self) -> Sequence[float]:
        """ Compute additional statistics from the confusion matrix."""
        if self.cm is None:
            self.get_confusion_matrix()
        
        tn, fp, fn, tp = self.cm.ravel()
        self.metrics["sensitivity"] = tp/(fn+tp)# same as recall
        self.metrics["specificity"] = 1 - (fp/(tn+fp))
        self.metrics["precision"] = tp/(tp+fp)
        
        
    def log_classification_report(self, target_names: Sequence[str]):
        if len(target_names) != len(self.y_true.unique()):
            raise ValueError(f"N-target names doesn't match the number of unique targets.")
        report = classification_report(self.y_true, self.y_pred, target_names=target_names)
        logging.info(report)


def build_table_1(model_metrics: OrderedDict, table_folder: str="/mnt/data/tables") -> pd.DataFrame:
    """Builds and Saves Table 1: Summary of Eval Metrics for all Models

    Args:
        model_metrics (OrderedDict): key is model_name, value is metrics 
        table_folder (str): local filepath outside docker to write csv

    Returns:
        pd.DataFrame: Organized as a DataFrame to write to csv
    """
    # Build Table 1
    model_metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index').round(decimals=4)
    model_metrics_df = model_metrics_df.reset_index().rename(columns={'index':'model_name'})
    model_metrics_df = model_metrics_df.sort_values(by='roc_auc', ascending=False)
    logging.info(f"Table 1:  Model Performance Metrics")
    logging.info(model_metrics_df.head(10))
    
    # Save as csv
    table_folder = Path(table_folder)
    if not table_folder.exists:
        Path.mkdir(table_folder)
    table_fpath = str(table_folder / 'table1_model_performance.csv')
    model_metrics_df.to_csv(table_fpath, encoding='utf-8', index=False)
    return model_metrics_df


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
    figures_fpath = io_cfg["figures_fpath"]
    
    # Load Data, subset features
    columns = features_cfg["features_to_keep"]
    test_set = io_cfg["partitions_filenames"][2]
    dfs_map = load_data_partitions(io_cfg["partitions_folder"], [test_set], columns)
    X_test, y_test = split_data(dfs_map['test']) 
    
    # Fill missing data
    [X_test] = fill_missing_data([X_test], missing_value = -999)

	# Loop over models
    # -- Eval multiple trained models
    # -- Build Table 1 summary metrics
    # -- Plot ROC Comparison Figures
    ms = ModelSchemas(X_test.columns, features_cfg)# Multiple models
    model_metrics = {}# Table 1
    ys = []; probs=[]; plot_names=[]# ROC Figure

    for model_schema in ms.schemas:
        X, y = get_model_features_target(model_schema, X_test, y_test)
    
        # Load the model object
        model_name = model_schema["name"]
        model_fpath = models_base_fpath.format(model_name=model_name)
        model = load_model(model_fpath)
        
        # Evaluate model performance
        y_prob = model.do_predict(X)
        evaluator = ModelEvaluator(y, y_prob)
        model_metrics[model_name] = evaluator.metrics
        
        # Log Classification Report
        target_names = ['low BAC','high BAC']
        evaluator.log_classification_report(target_names)
        
        # TODO: Add to Eval Class, Reformat plotting function, and Test
        #plot_confusion_matrix(cnf_matrix, target_names, normalize=True)
        #fig.savefig('/'.join([FIGURE_FOLDER,'/cm/Normalized_Confusion_Matrix_BAC_Classification_{}.png'.format(TODAY)]), bbox_inches='tight')
        
        # Store kwargs for roc-comparison plot
        if model_name != "majority_class":
            ys.append(y)
            probs.append(y_prob)
            plot_names.append(model_schema["plot_name"])
            
        # # Plot Shap Values for Best Model
        # if model_name == "all":
        #     base_folder = Path(io_cfg["figures_fpath"])
        #     output_folder = base_folder / 'shap/'
        #     compute_feature_importances(model.model, X, output_folder)
    
    
    # Build & Save Table 1
    model_metrics_df = build_table_1(model_metrics)
    
    # TODO: Ordered dict for ys, probs, plot_names, model_names
    # TODO: provide the dict as a single arg to plot + plot_order.
    
    # # Plot final roc-comparison plot (w-o majority class)
    plot_order = model_metrics_df["model_name"].tolist()
    logging.info(plot_order)
    # plot_ROC_comparison(ys, probs, plot_names, figures_fpath, save_plot=True)
    # logging.info(f"\nEvaluate.py Complete.")

if __name__=="__main__":
    main()