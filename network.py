import numpy as np
import pickle as pkl
from typing import List
from nn_utils import *
from mlp import MLP
import csv
import matplotlib.pyplot as plt

class NormalGenerator:
    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance

    def generateWeights(self, layerSizes: List[int]):
        return [np.random.normal(self._mean, self._variance, (layerSizes[i + 1], layerSizes[i])) for i in range(len(layerSizes) - 1)]

class UniformGenerator:
    def __init__(self, lowerBound, upperBound):
        self._lowerBound = lowerBound
        self._upperBound = upperBound

    def generateWeights(self, layerSizes: List[int]):
        return [np.random.uniform(self._lowerBound, self._upperBound, (layerSizes[i + 1], layerSizes[i])) for i in range(len(layerSizes) - 1)]


class DeepNetwork:
    def __init__(self, sizes: List[int], hidden_activation: ActivationFunction, last_activation: ActivationFunction, weightGenerator):
        self.sizes = sizes
        self.activations = [hidden_activation for i in range(len(sizes) - 2)]
        self.activations.append(last_activation)
        self.biases = [np.zeros((size, 1)) for size in sizes[1:]]
        self.weights = weightGenerator.generateWeights(sizes)
        self._As = []
        self._Zs = []
        self._W_grads = [np.zeros_like(w) for w in self.weights]
        self._b_grads = [np.zeros_like(b) for b in self.biases]


    def classify(self, X):
        raw_output = self._feedforward(X)
        Y_classified = self._predict(raw_output)
        return Y_classified, raw_output

    def _feedforward(self, X):
        A = X
        for (W, b, f) in zip(self.weights, self.biases, self.activations):
            Z = W @ A + b
            A = f.activation(Z)
        return A

    def loss(self, Y_train, Y_pred):
        n = np.shape(Y_train)[1]
        return np.sum((Y_train - Y_pred) ** 2) / n

    def _predict(self, Y_hat):
        return np.argmax(Y_hat, axis=0)

    def accuracy(self, Y_target, Y):
        Y = Y[:, None]
        return np.sum(Y_target == Y) / len(Y_target)

    def error(self, Y_target, Y):
        return 1 - self.accuracy(Y_target, Y)

    def _feedforward_train(self, X):
        self._reset_cache()
        A = X
        self._As.append(A)
        for (W, b, f) in zip(self.weights, self.biases, self.activations):
            Z = W @ A + b
            self._Zs.append(Z)
            A = f.activation(Z)
            self._As.append(A)
        return A

    def train(self, X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, learning_rate, epochs, batch_size):
        X_batches, Y_batches = split_into_batches(batch_size, X_train, Y_train_one)
        train_losses = []
        val_losses = []
        val_errors = []
        for i in range(epochs):
            for (X_batch, Y_batch) in zip(X_batches, Y_batches):
                self._feedforward_train(X_batch)
                self._backpropagate(Y_batch, learning_rate)
            Y_val_hat, raw_val_output = self.classify(X_val)
            Y_train_hat, raw_train_output = self.classify(X_train)
            train_loss = self.loss(Y_train_one, raw_train_output)
            val_loss = self.loss(Y_val_one, raw_val_output)
            error = self.error(Y_val, Y_val_hat)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_errors.append(error)
            print(f"Train loss: {train_loss}, Val loss: {val_loss}, Val error: {error}")
        return train_losses, val_losses, val_errors

    def _cost_derivative(self, Y_target, Y):
        n = np.shape(Y_target)[1]
        return (Y_target - Y) / n

    def _backpropagate(self, Y_batch, learning_rate):
        n_examples = Y_batch.shape[1]
        grad = self.activations[-1].derivative(self._Zs[-1])
        delta = self._cost_derivative(Y_batch, self._As[-1]) * grad
        self._W_grads[-1] = delta @ self._As[-2].T / n_examples
        self._b_grads[-1] = np.mean(delta, axis=1, keepdims=True)

        for l in range(2, len(self.sizes)):
            grad = self.activations[-l].derivative(self._Zs[-l])
            delta = (self.weights[-l + 1].T @ delta) * grad
            self._W_grads[-l] = delta @ self._As[-l-1].T / n_examples
            self._b_grads[-l] = np.mean(delta, axis=1, keepdims=True)
        self.update_weights(learning_rate)

    def update_weights(self, learning_rate):
        for i in range(len(self._W_grads)):
            self.weights[i] += learning_rate * self._W_grads[i]
            self.biases[i] += learning_rate * self._b_grads[i]

    def reset_grads(self):
        for w in self._W_grads:
            w.fill(0)
        for b in self._b_grads:
            b.fill(0)

    def _reset_cache(self):
        self._As = []
        self._Zs = []


def read(name):
    with open(name, 'rb') as data:
        datasets = pkl.load(data, encoding='latin1')
        [X_train, Y_train], [X_val, Y_val], [X_test, Y_test] = datasets
        X_train = X_train.T
        Y_train = np.reshape(Y_train, (np.shape(Y_train)[0], 1))
        X_val = X_val.T
        Y_val = np.reshape(Y_val, (np.shape(Y_val)[0], 1))
        X_test = X_test.T
        Y_test = np.reshape(Y_test, (np.shape(Y_test)[0], 1))
        return X_train, Y_train, X_val, Y_val, X_test,  Y_test


def run_normal_test():
    variances = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
    results = []
    test_errors = []
    K_CLASSES = 10

    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()
    for variance in variances:
        weightGenerator = NormalGenerator(0, variance)
        net = DeepNetwork([np.shape(X_train)[0], 50, K_CLASSES], ActivationFunction(relu, relu_derivative),
                          ActivationFunction(sigmoid, sigmoid_derivative), weightGenerator)
        train_loss, val_loss, val_error = net.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, 0.5, 25,
                                                    50)
        y_test_pred, _ = net.classify(X_test)
        test_err = net.error(Y_test, y_test_pred)
        test_errors.append(test_err)
        results.append(train_loss)

    fig = plt.figure()
    epochs = np.arange(0, len(results[0]))
    for i, row in enumerate(results):
        plt.plot(epochs, row, label=str(variances[i]))
    plt.legend()
    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt w zależności od wariancji początkowych wag')
    plt.show()
    fig.savefig('normal_1.png', dpi=fig.dpi)

    fig = plt.figure()
    epochs = np.arange(0, len(results[0]))
    for i, row in list(enumerate(results))[-3:-1]:
        plt.plot(epochs, row, label=str(variances[i]))
    plt.legend()
    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt w zależności od wariancji początkowych wag')
    plt.show()
    fig.savefig('normal_2.png', dpi=fig.dpi)

    save_list_csv('normal.csv', test_errors, variances)


def run_uniform_test():
    intervals = [0.8, 0.5, 0.2, 0.1, 0.05]
    train_errors = []
    train_losses = []
    K_CLASSES = 10
    test_errors = []

    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()
    for interval in intervals:
        weightGenerator = UniformGenerator(-interval, interval)
        net = DeepNetwork([np.shape(X_train)[0], 50, K_CLASSES], ActivationFunction(relu, relu_derivative),
                          ActivationFunction(sigmoid, sigmoid_derivative), weightGenerator)
        train_loss, val_loss, val_error = net.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, 0.5, 25,
                                                    50)
        train_losses.append(train_loss)
        y_test_pred, _ = net.classify(X_test)
        test_err = net.error(Y_test, y_test_pred)
        test_errors.append(test_err)

    fig = plt.figure()
    epochs = np.arange(0, len(train_losses[0]))
    for i, row in enumerate(train_losses):
        plt.plot(epochs, row, label=str(f"[{-intervals[i]}; {intervals[i]}]"))
    plt.legend()
    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt w zależności od wielkości zakresu wag')
    plt.show()
    fig.savefig('unform_1.png', dpi=fig.dpi)

    fig = plt.figure()
    epochs = np.arange(0, len(train_losses[0]))
    for i, row in list(enumerate(train_losses))[-3:]:
        plt.plot(epochs, row, label=str(f"[{-intervals[i]}; {intervals[i]}]"))
    plt.legend()
    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt w zależności od wielkości zakresu wag')
    plt.show()
    fig.savefig('unform_2.png', dpi=fig.dpi)

    save_list_csv('uniform.csv', test_errors, test_errors)


def run_batch_size_tests():
    errors = []
    losses = []
    test_errors = []
    K_CLASSES = 10
    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()


    batch_sizes = [1, 5, 10, 25, 50, 100, 500, 1000, np.shape(X_train)[1]]
    for batch_size in batch_sizes:
        weightGenerator = NormalGenerator(0, 0.05)
        net = DeepNetwork([np.shape(X_train)[0], 50, K_CLASSES], ActivationFunction(relu, relu_derivative),
                          ActivationFunction(sigmoid, sigmoid_derivative), weightGenerator)
        train_loss, val_loss, val_error = net.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, 0.1, 25, batch_size)
        errors.append(val_error)
        losses.append(val_loss)
        y_test_pred, _ = net.classify(X_test)
        test_err = net.error(Y_test, y_test_pred)
        test_errors.append(test_err)

    fig = plt.figure()
    epochs = np.arange(0, len(losses[0]))
    for i, (loss, error) in enumerate(zip(losses, errors)):
        plt.plot(epochs, loss, label=str(batch_sizes[i]))

    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt w zależności od wielkości paczki danych')
    plt.legend()
    plt.show()
    fig.savefig('batch_sizes_1.png', dpi=fig.dpi)

    fig = plt.figure()
    epochs = np.arange(0, len(losses[0]))
    for i, (loss, error) in list(enumerate(zip(losses, errors)))[:5]:
        plt.plot(epochs, loss, label=str(batch_sizes[i]))

    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt w zależności od wielkości paczki danych')
    plt.legend()
    plt.show()
    fig.savefig('batch_sizes_2.png', dpi=fig.dpi)

    save_list_csv('batch_sizes.csv', test_errors, batch_sizes)


def run_neuron_number_tests():
    errors = []
    train_losses = []
    val_losses = []
    K_CLASSES = 10
    test_errors = []
    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()

    hidden_sizes = [1, 15, 50, 100, 500, 1000]
    for hidden_size in hidden_sizes:
        weightGenerator = NormalGenerator(0, 0.05)
        net = DeepNetwork([np.shape(X_train)[0], hidden_size, K_CLASSES], ActivationFunction(relu, relu_derivative),
                          ActivationFunction(sigmoid, sigmoid_derivative), weightGenerator)
        train_loss, val_loss, val_error = net.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, 0.5, 50, 10)
        errors.append(val_error)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        y_test_pred, _ = net.classify(X_test)
        test_err = net.error(Y_test, y_test_pred)
        test_errors.append(test_err)

    epochs = np.arange(0, len(train_losses[0]))
    for i, hidden_size in enumerate(hidden_sizes):
        fig = plt.figure()
        plt.plot(epochs, train_losses[i], label='Ciąg treningowy')
        plt.plot(epochs, val_losses[i], label='Ciąg walidacyjny')
        plt.xlabel('Numer epoki')
        plt.ylabel('Wartość funkcji kosztu')
        plt.title('')
        plt.legend()
        if hidden_size == 1:
            fig.suptitle(f'Koszt - {hidden_size} neuron w warstwie ukrytej')
        else:
            fig.suptitle(f'Koszt - {hidden_size} neuronów w warstwie ukrytej')
        fig.savefig(f'neuron_number_{hidden_size}.png', dpi=fig.dpi)

    save_list_csv('neuron_number.csv', test_errors, hidden_sizes)


def run_activation_function_tests():
    activation_functions = [
        ([ActivationFunction(sigmoid, sigmoid_derivative), ActivationFunction(sigmoid, sigmoid_derivative)], 'Sigmoid'),
        ([ActivationFunction(relu, relu_derivative), ActivationFunction(sigmoid, sigmoid_derivative)], 'ReLU')
    ]
    errors = []
    train_losses = []
    val_losses = []
    test_errors = []
    K_CLASSES = 10
    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()

    for funcs in activation_functions:
        ([a1, a2], _) = funcs
        weightGenerator = NormalGenerator(0, 0.05)
        net = DeepNetwork([np.shape(X_train)[0], 50, K_CLASSES], a1, a2, weightGenerator)
        train_loss, val_loss, val_error = net.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, 0.5, 25, 10)
        errors.append(val_error)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        y_test_pred, _ = net.classify(X_test)
        test_err = net.error(Y_test, y_test_pred)
        test_errors.append(test_err)
    fig = plt.figure()
    epochs = np.arange(0, len(val_losses[0]))
    for funcs, loss in zip(activation_functions, train_losses):
        plt.plot(epochs, loss, label=funcs[1])

    plt.xlabel('Numer epoki')
    plt.ylabel('Wartość funkcji kosztu na ciągu treningowym')
    plt.title('Koszt w zależności od funkcji aktywacji')
    plt.legend()
    plt.show()
    fig.savefig('sigmoid_vs_relu3.png', dpi=fig.dpi)
    save_list_csv('activation.csv', test_errors, ['sigmoid sigmoid', 'relu sigmoid', 'relu relu'])


def load_dataset():
    K_CLASSES = 10
    X_train, Y_train, X_val, Y_val, X_test, Y_test = read('mnist.pkl')
    Y_train_one, Y_val_one, Y_test_one = one_hot(Y_train, K_CLASSES), one_hot(Y_val, K_CLASSES), one_hot(Y_test,
                                                                                                         K_CLASSES)
    return X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one


def save_list_csv(filename, lst, header=None):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerow(lst)

if __name__ == '__main__':
    K_CLASSES = 10
    X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, X_test, Y_test, Y_test_one = load_dataset()
    weightGenerator = NormalGenerator(0, 0.05)
    net = DeepNetwork([np.shape(X_train)[0], 50, K_CLASSES], ActivationFunction(relu, relu_derivative),
                      ActivationFunction(sigmoid, sigmoid_derivative), weightGenerator)
    train_loss, val_loss, val_error = net.train(X_train, Y_train, Y_train_one, X_val, Y_val, Y_val_one, 0.5, 50,
                                                50)