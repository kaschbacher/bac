import logging
import sys
import pandas as pd
import datetime as dt
from typing import Sequence
import shap
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def compute_shap_values(
    sklearn_model: BaseEstimator, 
    X_explain: pd.DataFrame, 
    feature_labels: dict,
    output_folder: str
    ) -> None:
    """Given a tree model, compute, plot, save SHAP values.

    Args:
        sklearn_model (BaseEstimator): e.g., lightgbm
        X_explain (pd.DataFrame): feature set
        feature_labels (dict): maps feature_names onto feature_labels for plotting
        output_folder (str): prefix for storing plots & parquets
    """
    shap_arr = shap.TreeExplainer(sklearn_model).shap_values(X_explain)
    if isinstance(shap_arr, list):
        # Some output a list for each class
        shap_arr = shap_arr[1]
    
    # Save SHAP Values
    shap_df = pd.DataFrame(shap_arr, columns=X_explain.columns, index=X_explain.index)
    shap_df.to_parquet(f"{output_folder}shap_values.parquet")
    
    # Plot SHAP Values
    labels = [feature_labels[feat]["label"] for feat in X_explain.columns]
    shap.summary_plot(shap_arr, X_explain, labels, show=False)
    plt.gcf().set_size_inches(15, 8)
    #plt.yticks(labels, fontsize=14)
    #plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/shap_importances.pdf", bbox_inches="tight")
    plt.close()
    logging.info(f"SHAP value plots saved to {output_folder}")
    
    return shap_df
        
        
def compute_permutation_importances(
    sklearn_model: BaseEstimator, 
    X_explain: pd.DataFrame, 
    feature_labels: dict,
    output_folder: str,
    n_feat_display: int=10
    ) -> None:
    """[summary]

    Args:
        sklearn_model (BaseEstimator): [description]
        X_explain (pd.DataFrame): [description]
        feature_labels (Sequence): List of str labels to plot
        output_folder (str): [description]
        n_feat_display (int, optional): [description]. Defaults to 10.
    """
    # Build & Save
    permutation = pd.DataFrame(
        sklearn_model.feature_importances_,
        index=X_explain.columns,
        columns = ["permutation_imp"]
        )
    permutation = permutation.sort_values(by="permutation_imp", ascending=False)
    permutation.to_parquet(f"{output_folder}/permutation_importances.parquet")
    
    # Plot Permutation Importances
    p_display = permutation.iloc[:n_feat_display].copy().reset_index()
    p_display["label"] = p_display["index"].apply(lambda x: feature_labels[x]["label"])    
    p_display.plot.barh(x="label", y="permutation_imp")
    
    plt.gcf().set_size_inches(15, 8)
    plt.tight_layout()
    
    plt.savefig(f"{output_folder}/permutation_importances.pdf", bbox_inches="tight")
    plt.close()
    logging.info(f"Feature permutation plots saved to {output_folder}\n")
        
        
def compute_feature_importances(
    sklearn_model: BaseEstimator, X_explain: pd.DataFrame, 
    feature_labels: dict, figures_folder: str
    ) -> None:
    """Generate SHAP and permutation feature importances to assist interpretation

    Args:
        sklearn_model: a tree model that is sklearn-compatible
        X_explain: a user-day indexed dataframe of features
        feature_labels: a dict, in which ["label"] maps to a plotable label
        figures_folder: local path to save plots and parquets
        -- If directory given, include "/" at end.
    """
    figures_folder = Path(figures_folder)
    if not Path.exists(figures_folder):
        logging.info(f"Creating figures folder: {figures_folder}")
        Path.mkdir(figures_folder)
    
    # try:
    #     compute_shap_values(sklearn_model, X_explain, feature_labels, figures_folder)
    # except Exception as E:
    #     logger.warn(f"Could not compute SHAP")
    #     logger.warn(f"{E}")
        
    try:
        compute_permutation_importances(sklearn_model, X_explain, feature_labels, figures_folder)
    except AttributeError as E:
        logger.warn(f"Could not compute Permutation Importances")
        logger.warn(f"{E}")
        
            