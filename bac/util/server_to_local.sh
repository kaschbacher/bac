# Bash script to copy the figure files from the server to my local
# Must be run from the local computer's terminal window
# You will be prompted to enter the server password



# ROC-Comparison Plot
MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/figures/roc/roc_comparison.pdf"
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/figures/roc"
scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL


# Table 1 Model Performance Metrics
MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/tables/table1_model_performance.csv"
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/tables"
scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL


# Shap Feature Importances
MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/figures/shap/shap_importances.pdf"
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/figures/shap"
scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL

open figures/shap/shap_importances.pdf

# Permutation Importances
MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/figures/shap/permutation_importances.pdf"
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/figures/shap"
scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL

open figures/shap/permutation_importances.pdf


# SHAP partial dependence scatterplots - one file
#PLOT="shap_scatterplot_bac_cumulative_avg_by_bac_guess.pdf"
PLOT="shap_scatterplot_bac_guess_by_bac_cumulative_avg.pdf"
MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/figures/shap/"$PLOT
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/figures/shap"
scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL


# cp entire subfolder
FOLDER="corr"
MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/figures/"$FOLDER"/*"
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/figures/"$FOLDER
scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL

