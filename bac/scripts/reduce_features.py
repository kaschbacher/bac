import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Sequence, List

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.config import parse_config

FEATURES_TO_DROP = [
            'guess_number',
            'monitor2','monitor3','monitor_yesno','monitor_measure','monitor_transport','monitor_number','monitor_content',
            'n_prior_zips', 'note_length', 'has_note', 'drinks_count','photo_count',
            'bac_last2','bac_last3','bac_last4','bac_last5', 'bac_cumulative_range','bac_avg_lag3','bac_avg_lag4','bac_avg_lag5',
            'dist_cumulative_range', 'dist_cumulative_min',
            'mv_death_rate_2012', 'pct_zip_rural', 'pop_zip_rural','pop_zip_all','pop_zip_urban',
            'official_number_margin', 'official_number_estimate', 'official_percent_margin', 'culture_id',
            'spm_number_estimate', 'spm_number_margin', 'spm_percent_estimate', 'spm_percent_margin', 'difference_number', 'statistically_significant_difference',
            'avg_heavy_drinking_black_non_hispanic', 'avg_heavy_drinking_hispanic', 'avg_heavy_drinking_multiracial_non_hispanic', 'avg_heavy_drinking_other_non_hispanic', 'avg_heavy_drinking_white_non_hispanic', 
            'sqrt_n_users_per_state','sq_n_users_per_state', 'sq_n_users_by_bac', 'n_adj_75q_bac','state_75q_bac_avg',
            'loser_state_code_le','winner_state_code_le','is_game','is_holiday',
            'sales_tax_per_gal_2014', 'gas_tax_per_gal_2014', 'cig_tax_per_20pack_2014', 'taxes'
        ]

def make_heat_map(data: np.ndarray, names: List[str], feature_labels: List[str]):
    """Heat map of feature correlations -> Reduce dimensionality

    Args:
        data (np.ndarray): [description]
        names (List): column names
        feature_labels (List): human readable labels

    Returns:
        [type]: [description]
    """
    df = pd.DataFrame(data, columns=names)
    corr = df.corr(method='pearson', min_periods=30)
    
    plt.figure(figsize=(24,24))
    sns.heatmap(data=corr, cmap='twilight', xticklabels=names, yticklabels=names, vmin=-1.0, vmax=1.0)
    sns.set(font_scale=1.4)
    
    fig = plt.gcf()
    return fig

def main(io_config = "/mnt/configs/io_config.yml"):

    # Set Parameters
    config = parse_config(io_config)
    fi_labels = load_feature_labels(config["feature_labels_fpath"])
    idx_uid = 1# column index of user id in bac feature file

    # Load Data
    dfs_map = load_data_partitions(config["partitions_folder"], config["partitions_filenames"])

    # Feature correlation heat map: All Features
    train = dfs_map["train"]
    names = train.columns[idx_uid+1:]
    fig = make_heat_map(train, names, fi_labels)

    fig_folder = config["figures_fpath"]
    figpath = os.path.join(fig_folder, "corr/feature_correlations_all.pdf")
    plt.savefig(figpath, format="pdf", dpi=300, bbox_inches='tight')
    print(f"Saved heatmap of all features to: {figpath}")

    # Reduce the Feature Set
    keep_names = [name for name in names if name not in FEATURES_TO_DROP]
    #fnames = [fi_labels[name]['label'] for name in keep_names]

    # Replot Feature Correlations in Reduced Feature Set
    fig = make_heat_map(train, keep_names, fi_labels)
    figpath = os.path.join(fig_folder, "corr/feature_correlations_reduced.pdf")
    plt.savefig(figpath, format="pdf", dpi=300, bbox_inches='tight')
    print(f"Saved heatmap of reduced feature subset to: {figpath}")

    # TODO: replace with a new LightGBC model
    # Fit Model with Reduced Feature Set; Evaluate the AUC_ROC
    # partitions, boosting_params = bac.randomize(by='users', data=data[:, keeps], names=keep_names)
    # user_roc = bac.run_balanced(boosting_params, partitions)
    # print ('With Refined set of {} features, randomization by Users AUC: {:.02f}\n'.format(len(keep_names)-2, user_roc))

    # ### Save output as a json

    # # TODO: Not sure what this is doing
    # # Build the dict
    # features_heatmap_out = {}
    # features_heatmap_out['names']=keep_names
    # features_heatmap_out['indices']= [int(keep) for keep in keeps]# json does not recognize numpy, convert to python int
    # print (features_heatmap_out)

    # # TODO: refactor to write yaml instead of json
    # feature_subset_fpath = config["feature_subset_fpath"]
    # with open(feature_subset_fpath, 'w') as json_file:
    #     json.dump(features_heatmap_out, json_file)
        

if __name__ == "__main__":
    main()

