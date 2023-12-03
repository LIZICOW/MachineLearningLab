import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import os, copy, random

class K_means():
    def __init__(self, data, K):
        self.data = np.array(data)
        print(self.data.shape)
        self.X = np.zeros((len(self.data), 1))
        print(self.X.shape)
        self.X = np.concatenate((self.X, self.data[:,:]), axis=1)
        self.K = K
        self.check = 1
        raw = len(self.X)//K
        col = K
        print(raw, col)
        self.echos = min(100, raw)
        self.randCenterIndex = np.arange(raw*col)
        np.random.shuffle(self.randCenterIndex)
        self.randCenterIndex = self.randCenterIndex.reshape((raw,col))
    def computeCenter(self):
        dis = 0
        for i in range(len(self.Center)):
            self.Center[i] = np.mean(np.array([self.X[j][1:] for j in range(len(self.X)) if self.X[j][0] == i]), axis=0)
            if True in np.isnan(self.Center[i]):
                self.Center[i] = self.X[random.randint(0, len(self.X)), 1:]
                continue
            try:
                dis += np.sum(np.sqrt(np.sum(np.square([self.Center[i] - self.X[j][1:] for j in range(len(self.X)) if self.X[j][0] == i]),axis=1)))
            except:
                print(self.X)
                print(0, self.Center[i], np.isnan(self.Center[i]))
                print(1, self.Center)
                tmp = [self.Center[i] - self.X[j][1:] for j in range(len(self.X)) if self.X[j][0] == i]
                print(2, tmp)
                print(3, np.sum(np.square(tmp), axis=1))
                print()
                self.check = 0
        return dis
    def classify(self):
        for i in range(len(self.X)):
            dis = 0x7fffffff
            for j in range(len(self.Center)):
                if dis > np.sqrt(np.sum(np.square(self.X[i][1:]-self.Center[j][:]))):
                    dis = np.sqrt(np.sum(np.square(self.X[i][1:]-self.Center[j][:])))
                    self.X[i][0] = j
    def fit(self):
        minMSE = 0x7fffffff
        Center = []
        X = []
        for j in range(self.echos):
            MSE = 0
            old_MSE = MSE
            if j % 10 == 0:
                print("=" * 50)
                print("Step", j)
                print("=" * 50)
            self.Center = np.array([self.X[i, 1:] for i in range(len(self.X)) if i in self.randCenterIndex[j]])
            while True:
                self.classify()
                MSE = self.computeCenter()
                if np.fabs(MSE - old_MSE) < 0.01:
                    if minMSE > MSE:
                        minMSE = MSE
                        Center = copy.deepcopy(self.Center)
                        X = copy.deepcopy(self.X)
                    break
                old_MSE = MSE
        return [minMSE, Center, X]

    def labels_(self):
        return self.X[:, 0]

print(os.getcwd())

file_name = "./data/OnlineRetail.csv"
df = pd.read_csv(file_name, sep=",", encoding="ISO-8859-1", header=0)
print(df.head(10))
df = df.dropna()
df['CustomerID'] = df['CustomerID'].astype(str)
# data_preparation

# 计算购买金额
df['Amount'] = df['Quantity'] * df['UnitPrice']
df_Monetary = df.groupby('CustomerID')['Amount'].sum()
df_Monetary = df_Monetary.reset_index()

# 计算购买频率
df_Frequency = df.groupby('CustomerID')['InvoiceNo'].count()
df_Frequency = df_Frequency.reset_index()
df_Frequency.columns = ['CustomerID', 'Frequency']

# 计算最后一次购买时间
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
df['No_buy'] = max(df['InvoiceDate']) - df['InvoiceDate']
df['No_buy'] = df['No_buy'].dt.days
df_Recency = df.groupby('CustomerID')['No_buy'].min()
df_Recency = df_Recency.reset_index()

# 合并
pd_MF = pd.merge(df_Monetary, df_Frequency, on='CustomerID', how='inner')
pd_MFR = pd.merge(pd_MF, df_Recency, on='CustomerID', how='inner')
pd_MFR.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

# Removing (statistical) outliers for Amount
Q1 = pd_MFR.Amount.quantile(0.05)
Q3 = pd_MFR.Amount.quantile(0.95)
IQR = Q3 - Q1
pd_MFR = pd_MFR[(pd_MFR.Amount >= Q1 - 1.5 * IQR) & (pd_MFR.Amount <= Q3 + 1.5 * IQR)]

# Removing (statistical) outliers for Recency
Q1 = pd_MFR.Recency.quantile(0.05)
Q3 = pd_MFR.Recency.quantile(0.95)
IQR = Q3 - Q1
pd_MFR = pd_MFR[(pd_MFR.Recency >= Q1 - 1.5 * IQR) & (pd_MFR.Recency <= Q3 + 1.5 * IQR)]

# Removing (statistical) outliers for Frequency
Q1 = pd_MFR.Frequency.quantile(0.05)
Q3 = pd_MFR.Frequency.quantile(0.95)
IQR = Q3 - Q1
pd_MFR = pd_MFR[(pd_MFR.Frequency >= Q1 - 1.5 * IQR) & (pd_MFR.Frequency <= Q3 + 1.5 * IQR)]

df = pd_MFR[['Amount', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = ['Amount', 'Frequency', 'Recency']
X = df_scaled

# 可以自行调整K的数值
km = K_means(X, 3)
km.fit()

labels = km.labels_()
df_scaled["Clus_Db"] = labels
print(labels)
df_scaled.sort_values("Clus_Db")
score = metrics.silhouette_score(X, df_scaled["Clus_Db"])
realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

print(set(labels))
print("realClusterNum:", realClusterNum)
print("clusterNum", clusterNum)
print(df_scaled.head(10))
print(score)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 根据 'Clus_Db' 的不同值来设置不同颜色
colors = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'pink', 5: 'black', 6: 'orange', 7: 'purple', 8: 'brown', 9: 'cyan'}
selected_colors = {i: colors[i] for i in range(realClusterNum)}
for cluster, color in selected_colors.items():
    cluster_points = df_scaled[df_scaled['Clus_Db'] == cluster]
    ax.scatter(cluster_points['Amount'], cluster_points['Frequency'], cluster_points['Recency'], c=color, label=cluster)

# 设置坐标轴标签
ax.set_xlabel('Amount')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

# 添加图例
ax.legend()

# 显示图形
plt.show()
