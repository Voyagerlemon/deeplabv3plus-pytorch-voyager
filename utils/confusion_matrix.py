import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

original_font = plt.rcParams['font.sans-serif']

file_path = "miou_out_deeplabv3-plus"
df_csv = pd.read_csv(os.path.join(file_path, "confusion_matrix.csv"), index_col=0)

df = df_csv.astype('float') / df_csv.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 9), dpi=300)

heatmap = sns.heatmap(df, annot=True, fmt=".2f", robust=True, linewidths=0.3, linecolor="silver", annot_kws={"size": 10}, 
                      cmap="GnBu", square=True)

heatmap.xaxis.tick_top()
# plt.xticks(ha="center")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='left', fontsize=12)

plt.ylabel('Ground Truth Label', fontsize=14, labelpad=15)
plt.xlabel('Prediction Label', fontsize=14, labelpad=15)

plt.rcParams['font.sans-serif'] = original_font
plt.savefig(os.path.join(file_path, "confusion_matrix.png"))