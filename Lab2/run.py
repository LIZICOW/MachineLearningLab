import pandas as pd
from model import Model
from Visualizer import DataVisualizer

df = pd.read_csv('./data/StressLevelDataset.csv')

#可视化展现数据集特征
DataVisualizer(df).plot_heatmap(df)
DataVisualizer(df).plot_violin(df)
DataVisualizer(df).plot_boxplot(df)

X = df[[x for x in df.columns if x != 'stress_level']]
Y = df['stress_level']

input_dim = len(X.columns)
print(input_dim)

model = Model(input_dim=input_dim, num_classes=3, lr=0.01)
model.load_data(X, Y)
train_losses= model.train(epochs=1000)
y_pred, y_true, y_scores = model.test()
# 绘制损失函数的变化图
DataVisualizer(model).plot_loss(train_losses)
# 绘制混淆矩阵
DataVisualizer(model).plot_confusion_matrix(y_true, y_pred)


