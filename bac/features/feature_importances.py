import logging
import sys
import pandas as pd
import numpy as np
import datetime as dt
from typing import Sequence, Union
import shap
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm, colorbar
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Used by bac/scripts/shap_plots.py


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
    #shap_df.to_parquet(f"{output_folder}shap_values.parquet")
    
    # Plot SHAP Values
    formatter_params = {'xtick.labelsize': 18, 'ytick.labelsize': 18, 'axes.labelsize': 20}
    plt.rcParams.update(formatter_params)
    labels = [feature_labels[feat]["label"] for feat in X_explain.columns]
    
    shap.summary_plot(shap_arr, X_explain, labels, show=False)
    plt.gcf().set_size_inches(12, 8)
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
    # Colormap
    cmap = cm.get_cmap("Blues")
    colors = cmap(np.linspace(.25, 1, n_feat_display))# rgba
    
    # Build & Save
    permutation = pd.DataFrame(
        sklearn_model.feature_importances_,
        index=X_explain.columns,
        columns = ["permutation_imp"]
        )
    permutation = permutation.sort_values(by="permutation_imp", ascending=False)
    permutation.to_parquet(f"{output_folder}/permutation_importances.parquet")
    
    # Plot Permutation Importances
    formatter_params = {'xtick.labelsize': 13, 'ytick.labelsize': 15, 'axes.labelsize': 18}
    plt.rcParams.update(formatter_params)
    p_display = permutation.iloc[:n_feat_display].copy().reset_index()
    p_display["Feature Permutation Importance"] = p_display["index"].apply(lambda x: feature_labels[x]["label"])    
    p_display = p_display.sort_index(ascending=False)
    p_display.plot.barh(x="Feature Permutation Importance", y="permutation_imp", color=colors)
    
    plt.gcf().set_size_inches(12, 8)
    plt.legend("")
    plt.tight_layout()
    
    plt.savefig(f"{output_folder}/permutation_importances.pdf", bbox_inches="tight")
    plt.close()
    logging.info(f"Feature permutation plots saved to {output_folder}\n")
        
        
def shap_scatterplot(
    sklearn_model: BaseEstimator, 
    X_explain: pd.DataFrame, 
    feature_labels: dict,
    feature: str = "bac_guess",
    moderator: Sequence[str] = "episode",
    output_folder: str = "/mnt/data/figures/shap"
    ) -> None:
    """Partial Dependence Plot for SHAP

    Args:
        sklearn_model (BaseEstimator): e.g., lightgbm
        X_explain (pd.DataFrame): feature set
        feature_labels (dict): maps feature_names onto feature_labels for plotting
        feature (str): The main feature to scatterplot
        output_folder (str): prefix for storing plots & parquets
    """
    # Exclude missing data, which distorts visualization
    mask = X_explain[feature] > -999
    X_explain = X_explain.loc[mask]
    
    # Compute SHAP values
    shap_values = shap.TreeExplainer(sklearn_model).shap_values(X_explain)
    
    if isinstance(shap_values, list):
        # Some output a list for each class
        shap_values = shap_values[1]
        
    columns = X_explain.columns.tolist()
    if feature not in columns:
        raise ValueError(f"{feature} is not a column in the given feature df.")

    formatter_params = {'xtick.labelsize': 8, 'ytick.labelsize': 8}
    plt.rcParams.update(formatter_params)
    
    for mod in moderator:
        ax = shap.dependence_plot(feature, shap_values, X_explain, 
            interaction_index=mod, dot_size=2)
        if (feature == "bac_guess") or (feature == "bac_cumulative_avg"):
            plt.axvspan(.06, .10, alpha=.10, color='grey')
            plt.axvspan(.04, .12, alpha=.10, color='grey')
        plt.gcf().set_size_inches(6, 3)
        
        flabel = feature_labels[feature]["label"]
        plt.xlabel(flabel, fontsize=10)
        plt.ylabel(f"SHAP Value for {flabel}", fontsize=10)
        
        # cbarlabel = feature_labels[mod]["label"]
        # cbar = plt.colorbar()
        # cbar.ax.tick_params(labelsize=7) 
        #plt.colorbar().set_label(label=cbarlabel, fontsize=10)
        
        # # Hack to change fontsize on the legend/colorbar
        # cax = plt.gcf().axes[-1]
        # cax.tick_params(labelsize=8)
        # # Hack to change fontsize of the legend label
        # plt.gcf().figure.axes[-1].yaxis.label.set_size(10)# size of legend label
        
        plt.tight_layout()
        plt.savefig(f"{output_folder}/shap_scatterplot_{feature}_by_{mod}.pdf", bbox_inches="tight")
        plt.close()
        
        
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
        
    # try:
    #     compute_permutation_importances(sklearn_model, X_explain, feature_labels, figures_folder)
    # except AttributeError as E:
    #     logger.warn(f"Could not compute Permutation Importances")
    #     logger.warn(f"{E}")
        
    try:
        #moderators = ['episode', 'bac_cumulative_avg', 'gmt_diff_min']
        # moderators = ['distance_km', 'pct_zip_urban', 'bac_level_verified_sum', \
        #     'deep_engagement', 'n_days_engaged', 'dow', "episode", 'gmt_diff_min', \
        #     'hour_local']
        moderators = ['bac_guess']
        shap_scatterplot(sklearn_model, X_explain, feature_labels, 
                'bac_cumulative_avg', moderators, figures_folder)
        moderators=['bac_cumulative_avg']
        shap_scatterplot(sklearn_model, X_explain, feature_labels, 
                'bac_guess', moderators, figures_folder)
    except AttributeError as E:
        logger.warn(f"Could not compute SHAP scatterplot")
        logger.warn(f"{E}")
        
  