import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

df_hyp = pd.read_csv('/COVID_8TB/vlp_results/csv_files/processed/processed_no_FIANL_seed777_pred.csv')
df_ref = pd.read_csv('/COVID_8TB/vlp_results/csv_files/processed/processed_no_FIANL_seed777_label.csv')
df_hyp_pos1 = (df_hyp == 1).astype(int)
del df_hyp_pos1["Reports"]
df_hyp_pos1 = np.array(df_hyp_pos1)

df_ref_pos1 = (df_ref == 1).astype(int)
del df_ref_pos1["Reports"]
df_ref_pos1 = np.array(df_ref_pos1)
df_hyp_0 = (df_hyp == 0).astype(int)
del df_hyp_0["Reports"]
df_hyp_0 = np.array(df_hyp_0)
df_ref_0 = (df_ref == 0).astype(int)
del df_ref_0["Reports"]
df_ref_0 = np.array(df_ref_0)
df_hyp_neg1 = (df_hyp == -1).astype(int)
del df_hyp_neg1["Reports"]
df_hyp_neg1 = np.array(df_hyp_neg1)
df_ref_neg1 = (df_ref == -1).astype(int)
del df_ref_neg1["Reports"]
df_ref_neg1 = np.array(df_ref_neg1)
df_hyp_all = df_hyp_pos1 + df_hyp_0 + df_hyp_neg1
df_ref_all = df_ref_pos1 + df_ref_0 + df_ref_neg1

# Accuarcy
accuracy_pos1 = (df_ref_pos1 == df_hyp_pos1).sum() / df_ref_pos1.size
accuracy_0 = (df_ref_0 == df_hyp_0).sum() / df_ref_0.size
accuracy_neg1 = (df_ref_neg1 == df_hyp_neg1).sum() / df_ref_neg1.size
accuracy_all = (df_ref_all == df_hyp_all).sum() / df_ref_all.size

# Precision
precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
precision_0 = precision_score(df_ref_0, df_hyp_0, average="micro")
precision_neg1 = precision_score(df_ref_neg1, df_hyp_neg1, average="micro")
precision_all = precision_score(df_ref_all, df_hyp_all, average="micro")

# Recall
recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
recall_0 = recall_score(df_ref_0, df_hyp_0, average="micro")
recall_neg1 = recall_score(df_ref_neg1, df_hyp_neg1, average="micro")
recall_all = recall_score(df_ref_all, df_hyp_all, average="micro")

# F1
f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro")
f1_0 = f1_score(df_ref_0, df_hyp_0, average="micro")
f1_neg1 = f1_score(df_ref_neg1, df_hyp_neg1, average="micro")
f1_all = f1_score(df_ref_all, df_hyp_all, average="micro")

print(accuracy_all)
print(precision_all)
print(recall_all)
print('evaluation finished.')