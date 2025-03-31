import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import time
from model.cnn import CNN
from data_loader import data_loader

def train_cnn():
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测GPU是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    train_batches, (test_data, test_labels), meta_data = data_loader()
    
    # 准备训练数据和验证数据（从训练集中分出一部分作为验证集）
    val_ratio = 0.1  # 验证集占比
    all_train_data = []
    all_train_labels = []
    
    for batch_data, batch_labels in train_batches:
        all_train_data.append(batch_data)
        all_train_labels.extend(batch_labels)
    
    all_train_data = np.vstack(all_train_data)
    all_train_labels = np.array(all_train_labels)
    
    # 随机打乱数据
    indices = np.random.permutation(len(all_train_data))
    val_size = int(len(all_train_data) * val_ratio)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_data = all_train_data[train_indices]
    train_labels = all_train_labels[train_indices]
    val_data = all_train_data[val_indices]
    val_labels = all_train_labels[val_indices]
    
    # 转换为PyTorch张量
    train_data_tensor = torch.from_numpy(train_data)
    train_labels_tensor = torch.from_numpy(np.array(train_labels)).long()
    val_data_tensor = torch.from_numpy(val_data)
    val_labels_tensor = torch.from_numpy(np.array(val_labels)).long()
    test_data_tensor = torch.from_numpy(test_data)
    test_labels_tensor = torch.from_numpy(np.array(test_labels)).long()
    
    # 创建数据加载器
    batch_size = 128
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化CNN模型
    num_classes = 10  # CIFAR-10有10个类别
    model = CNN(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练设置
    num_epochs = 20
    
    # 用于记录训练和验证的损失与准确率
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # 开始训练
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练模式
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 计算训练损失和准确率
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 计算验证损失和准确率
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 打印每个epoch的结果
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s")
        print(f"  Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # 测试模式
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 计算测试损失和准确率
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    # 计算平均测试损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct_test / total_test
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # 绘制训练和验证的损失与准确率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.tight_layout()
    plt.savefig('image/training_metric_plot_cnn_2.png')
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), 'cnn_cifar10.pth')
    print("Model saved as 'cnn_cifar10.pth'")

if __name__ == "__main__":
    train_cnn()
