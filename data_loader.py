import os
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def data_loader():
    data_folder = 'cifar-10-batches-py'

    meta_path = os.path.join(data_folder, 'batches.meta')
    meta_data = unpickle(meta_path)

    train_batches = []
    for i in range(1, 6):
        batch_file = f'data_batch_{i}'
        batch_path = os.path.join(data_folder, batch_file)
        batch_data = unpickle(batch_path)

        # 读取数据
        data = batch_data[b'data']   # 10000x3072
        labels = batch_data[b'labels']

        # 将 data 转换为正确的形状 (样本数, 通道数, 高度, 宽度)
        num_samples = data.shape[0]  # 10000
        
        # 将每个样本分为R,G,B三个通道
        # 每个样本大小为3072，前1024为R通道，中间1024为G通道，最后1024为B通道
        data_reshaped = np.zeros((num_samples, 3, 32, 32), dtype=np.float32)
        
        # 按样本处理
        for j in range(num_samples):
            # 提取当前样本
            sample = data[j]
            
            # 分离三个通道并重塑
            data_reshaped[j, 0] = sample[0:1024].reshape(32, 32)    # R通道
            data_reshaped[j, 1] = sample[1024:2048].reshape(32, 32) # G通道
            data_reshaped[j, 2] = sample[2048:3072].reshape(32, 32) # B通道
        
        # 归一化
        data_reshaped = data_reshaped.astype(np.float32)  # 10000x3x32x32
        data_reshaped /= 255.0

        train_batches.append((data_reshaped, labels))

    test_batch_file = 'test_batch'
    test_batch_path = os.path.join(data_folder, test_batch_file)
    test_batch_data = unpickle(test_batch_path)

    # 读取测试数据
    test_data = test_batch_data[b'data']
    test_labels = test_batch_data[b'labels']

    # 将测试数据转换为正确的形状
    num_samples = test_data.shape[0]
    test_data_reshaped = np.zeros((num_samples, 3, 32, 32), dtype=np.float32)
    
    for j in range(num_samples):
        sample = test_data[j]
        test_data_reshaped[j, 0] = sample[0:1024].reshape(32, 32)    # R通道
        test_data_reshaped[j, 1] = sample[1024:2048].reshape(32, 32) # G通道
        test_data_reshaped[j, 2] = sample[2048:3072].reshape(32, 32) # B通道
    
    # 进行数据类型转换和归一化等操作
    test_data_reshaped = test_data_reshaped.astype(np.float32)
    test_data_reshaped /= 255.0

    return train_batches, (test_data_reshaped, test_labels), meta_data

