import numpy as np
from data_loader import data_loader
from model.nn import SimpleNN
from sklearn.model_selection import train_test_split
import yaml

def train_neural_network():
    train_batches, (test_data, test_labels), meta_data = data_loader()

    all_train_data = np.concatenate([batch[0] for batch in train_batches], axis=0)
    all_train_labels = np.concatenate([batch[1] for batch in train_batches], axis=0)

    # Flatten the data for input to the NN
    X_all = all_train_data.reshape(-1, 3 * 32 * 32)
    y_all = np.array(all_train_labels)

    # Split data into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=328)

    # Normalize using training data statistics
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Initialize and train the neural network
    input_size = X_train.shape[1]
    output_size = len(meta_data[b'label_names'])

    model = SimpleNN(input_size, output_size)

    epochs = 20
    batch_size = 32
    with open('./model/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    learning_rate = config["learning_rate"]

    # 用于记录训练集和验证集的 loss 以及验证集的 accuracy
    train_losses = []
    val_losses = []
    val_accuracies = []

    # For tracking the best model
    best_val_accuracy = 0
    best_model_path = './model/checkpoints/best_model.npy'

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

        # Evaluate training accuracy
        train_predictions = model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train) * 100

        # Evaluate validation accuracy
        val_predictions = model.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val) * 100

        # 计算验证集的 loss
        val_predictions_for_loss = model.forward(X_val)
        val_loss = model.compute_loss(val_predictions_for_loss, y_val)

        # Learning rate decay
        learning_rate_decay = 0.99
        learning_rate *= learning_rate_decay

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches}, Learning Rate: {learning_rate}")
        print(f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

        # 记录训练集和验证集的 loss 以及验证集的 accuracy
        train_losses.append(epoch_loss / num_batches)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_weights(best_model_path)
            print(f"New best validation accuracy: {best_val_accuracy:.2f}%. Model saved to {best_model_path}")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")

    # 保存训练集和验证集的 loss 以及验证集的 accuracy 数据
    np.savez('./model/checkpoints/training_metrics.npz', 
             train_losses=train_losses, 
             val_losses=val_losses, 
             val_accuracies=val_accuracies)

if __name__ == "__main__":
    train_neural_network()