'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-11 16:42:17
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-06-11 17:06:57
FilePath: \deeplabv3_plus-voyager\confusion_matrix.py
Description: 混淆矩阵
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", kind="scatter", data=tips)
plt.show()