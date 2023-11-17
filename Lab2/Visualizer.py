import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
class DataVisualizer:
    def __init__(self, model):
        self.model = model
    
    def plot_heatmap(self, df):
        corr = df.corr()
        plt.figure(figsize=(20,20))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()
    def plot_violin(self, df):
        for column in df.columns:
            if column != 'stress_level':
                plt.figure(figsize=(10,6))
                sns.violinplot(x='stress_level', y=column, data=df)
                plt.title(f'Violin plot of {column} vs stress_level')
                plt.show()
    def plot_boxplot(self, df):
        for column in df.columns:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=df[column])
            plt.title(f'Box plot of {column}')
            plt.show()            
    def plot_loss(self, losses):
        plt.figure(figsize=(10,6))
        plt.plot(losses)
        plt.title('Train loss value over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,6))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title('Confusion matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        
        
