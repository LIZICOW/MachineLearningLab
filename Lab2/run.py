from Xgb import XGB
import pandas as pd

df = pd.read_csv('./data/StressLevelDataset.csv')
print(df)
X = df[[x for x in df.columns if x!='stress_level']]
Y = df['y']
xgb = XGB(n_estimators=2, max_depth=2, reg_lambda=1, min_child_weight=1, objective='linear')
xgb.fit(X,Y)
