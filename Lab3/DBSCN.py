import numpy as np
import pandas as pd  
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
file_name = "OnlineRetail.csv"
df = pd.read_csv(file_name, sep=",", encoding="ISO-8859-1", header=0)
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

#可以自行调整
db = DBSCAN(eps=0.3, min_samples=15).fit(df_scaled)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
df_scaled["Clus_Db"] = labels
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
