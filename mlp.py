from nn_utils import *


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.sizes = (input_size, hidden_size, output_size)
        self.W1 = np.random.normal(0.0, 0.1, (hidden_size, input_size))
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.normal(0.0, 0.1, (output_size, hidden_size))
        self.b2 = np.zeros((output_size, 1))
        self.f1 = relu
        self.f1grad = relu_derivative
        self.f2 = sigmoid
        self.f2grad = sigmoid_derivative

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        self.W2grad = None
        self.b2grad = None
        self.W1grad = None
        self.b1grad = None

    def feedforward(self, X):
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = self.f1(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.f2(self.Z2)
        return self.A2

    def error(self, Y_train, Y_pred):
        return np.sum((Y_train - Y_pred) ** 2)

    def classify(self, X):
        Y_hat = self.feedforward(X)
        return self.predict(Y_hat)

    def predict(self, Y_hat):
        return np.argmax(Y_hat, axis=0)

    def accuracy(self, Y_target, Y):
        Y = Y[:, None]
        return np.sum(Y_target == Y) / len(Y_target)

    def train(self, X_batches, Y_batches, learning_rate, epochs, X_val, Y_val):
        for i in range(epochs):
            for (X_batch, Y_batch) in zip(X_batches, Y_batches):
                self.feedforward(X_batch)
                self.backpropagate(X_batch, Y_batch, learning_rate)
            Y_val_hat = self.classify(X_val)
            acc = self.accuracy(Y_val, Y_val_hat)
            print(acc)

    def backpropagate(self, X_batch, Y_batch, learning_rate):
        grad2 = self.f2grad(self.Z2)
        delta2 = (Y_batch - self.A2) * grad2
        delta2mean = np.mean(delta2, axis=1, keepdims=True)
        a1mean = np.mean(self.A1, axis=1, keepdims=True)
        self.W2grad = delta2mean @ a1mean.T
        self.b2grad = delta2mean
        grad1 = self.f1grad(self.Z1)
        delta1 = (self.W2.T @ delta2) * grad1
        delta1mean = np.mean(delta1, axis=1, keepdims=True)
        xmean = np.mean(X_batch, axis=1, keepdims=True)
        self.W1grad = delta1mean @ xmean.T
        self.b1grad = delta1mean
        self.update_weights(learning_rate)

    def update_weights(self, learning_rate):
        self.W2 = self.W2 + learning_rate * self.W2grad
        self.b2 = self.b2 + learning_rate * self.b2grad
        self.W1 = self.W1 + learning_rate * self.W1grad
        self.b1 = self.b1 + learning_rate * self.b1grad