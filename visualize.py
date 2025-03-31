import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# 加载训练集和验证集的 loss 以及验证集的 accuracy 数据
data = np.load('./model/checkpoints/training_metrics.npz')

train_losses = data['train_losses']
val_losses = data['val_losses']
val_accuracies = data['val_accuracies']

# 绘制训练集和验证集的 loss 曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 绘制验证集的 accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('image/training_metrics_plot.png')
plt.show()


# 加载保存的权重
model_data = np.load('./model/checkpoints/best_model.npy', allow_pickle=True).item()
weights = model_data['weights']
biases = model_data['biases']

# 创建一个包含所有层权重热力图的单一大图
plt.figure(figsize=(20, 7))  # 宽一些以适应横向排列的三个子图

# 为每一层的权重创建热力图，横向排列
for i, w in enumerate(weights):
    plt.subplot(1, len(weights), i+1)  # 1行，len(weights)列，第i+1个位置
    sns.heatmap(w, cmap='viridis', center=0, annot=False)
    plt.title(f'Layer {i+1} Weights')
    plt.xlabel('Neurons in layer {}'.format(i+2))
    plt.ylabel('Neurons in layer {}'.format(i+1))

plt.tight_layout()  # 调整布局，防止重叠
plt.savefig('image/all_weights_heatmap.png', dpi=300)  # 保存高分辨率图像
