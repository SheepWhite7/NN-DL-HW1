import numpy as np
import logging
from data_loader import data_loader
from model.nn import SimpleNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yaml

# 配置日志记录
logging.basicConfig(filename='para_search.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def train_model(learning_rate, hidden_sizes, lambda_reg, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate_decay):
    # 创建SimpleNN类的实例
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    model = SimpleNN(input_size, output_size, hidden_sizes=hidden_sizes,
                     lambda_reg=lambda_reg, learning_rate=learning_rate)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            batch_X = X_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]

            # Forward pass
            predictions = model.forward(batch_X)

            # Compute loss
            loss = model.compute_loss(predictions, batch_y)
            epoch_loss += loss

            # Backward pass
            gradients_w, gradients_b = model.backward(batch_X, batch_y)

            # Update weights using SGD
            model.sgd_update(gradients_w, gradients_b, learning_rate)

        # 计算训练集 loss
        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)

        # 计算验证集 loss 和 accuracy
        val_predictions = model.forward(X_val)
        val_loss = model.compute_loss(val_predictions, y_val)
        val_losses.append(val_loss)

        val_pred_classes = model.predict(X_val)
        val_accuracy = np.mean(val_pred_classes == y_val) * 100
        val_accuracies.append(val_accuracy)

        # Learning rate decay
        learning_rate *= learning_rate_decay

        log_message = f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss}, Validation Accuracy: {val_accuracy}"
        print(log_message)
        logging.info(log_message)


    return train_losses, val_losses, val_accuracies


if __name__ == "__main__":
    # 定义超参数搜索空间
    learning_rates = [0.01]
    # hidden_sizes_list = [[64, 32], [64, 64], [128, 64], [128, 128], [256, 128], [256, 256], [512, 256], [512, 512]]
    hidden_sizes_list = [[512, 256]]
    lambda_regs = [0.01]
    epochs = 20
    batch_size = [8, 16, 32, 64, 128]
    learning_rate_decay = 0.99

    train_batches, (test_data, test_labels), meta_data = data_loader()

    all_train_data = np.concatenate([batch[0] for batch in train_batches], axis=0)
    all_train_labels = np.concatenate([batch[1] for batch in train_batches], axis=0)

    # Flatten the data for input to the NN
    X_all = all_train_data.reshape(-1, 3 * 32 * 32)
    y_all = np.array(all_train_labels)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    hyperparameter_labels = []

    with open('./model/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logging.info(config["activation_functions"])

    # 超参数搜索
    for lr in learning_rates:
        for hidden_sizes in hidden_sizes_list:
            for lambda_reg in lambda_regs:
                for batch_size_i in batch_size:
                    logging.info(f"lr:{lr}, hidden_sizes:{hidden_sizes}, lambda_reg:{lambda_reg}, batch_size:{batch_size_i}")
                    train_losses, val_losses, val_accuracies = train_model(lr, hidden_sizes, lambda_reg, X_train, y_train, X_val, y_val, epochs, batch_size_i,
                                                                        learning_rate_decay)
                    all_train_losses.append(train_losses)
                    all_val_losses.append(val_losses)
                    all_val_accuracies.append(val_accuracies)
                    label = f"LR: {lr}, HS: {hidden_sizes}, Lambda: {lambda_reg}, BS: {batch_size_i}"
                    hyperparameter_labels.append(label)

                    logging.info('------------------------------------------------------------------')

    # 绘制训练集和验证集的 loss 曲线
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for i in range(len(all_train_losses)):
        plt.plot(all_train_losses[i], label=f'Train - {hyperparameter_labels[i]}')
        plt.plot(all_val_losses[i], label=f'Val - {hyperparameter_labels[i]}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    # 调整图例
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)

    # 绘制验证集的 accuracy 曲线
    plt.subplot(1, 2, 2)
    for i in range(len(all_val_accuracies)):
        plt.plot(all_val_accuracies[i], label=f'Val Acc - {hyperparameter_labels[i]}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    # 调整图例
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)

    plt.tight_layout()
    plt.savefig('image/para_search_hidden_plot_batch.png')
    plt.show()