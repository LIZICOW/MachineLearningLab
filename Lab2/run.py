from Xgb import XGB
import pandas as pd
from model import Model

df = pd.read_csv('./data/StressLevelDataset.csv')
# print(df)
X = df[[x for x in df.columns if x != 'stress_level']]
Y = df['stress_level']

# xgb = XGB(n_estimators=2, max_depth=2, reg_lambda=1, min_child_weight=1, objective='linear')
# xgb.fit(X,Y)
input_dim = len(X.columns)
print(input_dim)

model = Model(input_dim=input_dim, num_classes=3, lr=0.01)
model.load_data(X, Y)
model.train(epochs=1000)
model.test()
