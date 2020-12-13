# Must be run from my local computer

# Shap Feature Importances
MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/figures/shap/shap_importances.pdf"
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/figures/shap"

scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL


# ROC-Comparison Plot
SERVER=$MNT"/data/figures/roc/roc_comparison.pdf"
LOCAL="/Users/KAschbacher/desktop/eheart/bac_review/figures/roc"
scp kaschbacher@tisoncluster.ucsf.edu:$SERVER $LOCAL