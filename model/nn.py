import numpy as np
import yaml
import os

# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x):
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    return dx

activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative)
}

class SimpleNN:
    def __init__(self, input_size, output_size, hidden_sizes=None, lambda_reg=None, learning_rate=None):
        with open('./model/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.input_size = input_size
        self.output_size = output_size

        # 如果没有传入超参数，则从配置文件中获取
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else config["hidden_sizes"]
        self.lambda_reg = lambda_reg if lambda_reg is not None else config["lambda_reg"]
        self.learning_rate = learning_rate if learning_rate is not None else config["learning_rate"]

        self.activation_functions = [activation_functions[af][0] for af in config["activation_functions"]]
        self.activation_derivatives = [activation_functions[af][1] for af in config["activation_functions"]]

        print(self.hidden_sizes)
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, sizes[i + 1])))

    def forward(self, X):
        # 前向传播
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                activation = self.activation_functions[i](z)
            else:
                activation = z  # 输出层不使用激活函数
            self.activations.append(activation)
        return self.activations[-1]

    def compute_loss(self, predictions, labels):
        # 应用softmax计算概率
        exp_scores = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 正确的对数概率
        correct_logprobs = -np.log(probs[range(len(labels)), labels])
        loss = np.mean(correct_logprobs)

        # L2正则化损失
        l2_loss = self.lambda_reg * sum([np.sum(w ** 2) for w in self.weights])
        loss += l2_loss

        return loss

    def backward(self, X, y):
        # 反向传播
        exp_scores = np.exp(self.activations[-1] - np.max(self.activations[-1], axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # softmax损失关于分数的梯度
        dZ = probs
        dZ[range(len(y)), y] -= 1
        dZ /= len(y)

        gradients_w = []
        gradients_b = []
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, dZ) + 2 * self.lambda_reg * self.weights[i]  # L2正则化
            db = np.sum(dZ, axis=0, keepdims=True)
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.activation_derivatives[i - 1](self.z_values[i - 1])

        return gradients_w, gradients_b

    def sgd_update(self, gradients_w, gradients_b, learning_rate):
        # 使用SGD更新权重
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

    def save_weights(self, filepath):
        """Save model weights to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'weights': self.weights,
            'biases': self.biases
        }
        np.save(filepath, model_data)
        
    def load_weights(self, filepath):
        """Load model weights from a file."""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.weights = model_data['weights']
        self.biases = model_data['biases']
    