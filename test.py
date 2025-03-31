import numpy as np
from data_loader import data_loader
from model.nn import SimpleNN
from sklearn.model_selection import train_test_split


def test():
    best_model_path = './model/checkpoints/best_model.npy'

    train_batches, (test_data, test_labels), meta_data = data_loader()

    all_train_data = np.concatenate([batch[0] for batch in train_batches], axis=0)
    all_train_labels = np.concatenate([batch[1] for batch in train_batches], axis=0)

    # Flatten the data for input to the NN
    X_all = all_train_data.reshape(-1, 3 * 32 * 32)
    y_all = np.array(all_train_labels)
    
    # Split data into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Flatten and normalize test data
    X_test = test_data.reshape(-1, 3 * 32 * 32)
    X_test = (X_test - mean) / std

    input_size = X_test.shape[1]
    output_size = len(meta_data[b'label_names'])
    
    model = SimpleNN(input_size, output_size)

    model.load_weights(best_model_path)

    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == test_labels) * 100
    print(f"Test Accuracy with the best model: {test_accuracy:.2f}%")

if __name__ == '__main__':
    test()