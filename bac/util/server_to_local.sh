MNT="/media/TisonRaid01/Data_Backup/general_nas/kirstin/bac_repo/bac"
SERVER=$MNT"/data/figures/shap/shap_importances.pdf"
#LOCAL="~/desktop/eheart/bac_review/figures/shap"
#LOCAL="~/desktop"
LOCAL="/Users/KAschbacher/Desktop/eheart/bac_review/figures/shap/shap_importances.pdf"

scp $SERVER kaschbacher@tisoncluster.ucsf.edu:$LOCAL