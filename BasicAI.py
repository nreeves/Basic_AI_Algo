import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        # initialize weights and biases randomly
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i - 1]))
            self.biases.append(np.random.randn(layers[i], 1))

    def sigmoid(self, x):
        # sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs):
        # feed the inputs through the network and return the output
        a = inputs
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return a

    def train(self, inputs, labels, epochs, learning_rate):
        # train the network using backpropagation
        for epoch in range(epochs):
            # feed inputs forward through the network
            a = inputs
            activations = [a]
            zs = []
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, a) + b
                zs.append(z)
                a = self.sigmoid(z)
                activations.append(a)

            # backpropagation
            delta = self.cost_derivative(activations[-1], labels) * self.sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())

            for l in range(2, len(layers)):
                z = zs[-l]
                sp = self.sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

            # update weights and biases
            for i in range(len(layers) - 1):
                self.weights[i] -= learning_rate * nabla_w[i]
                self.biases[i] -= learning_rate * nabla_b[i]

    def cost_derivative(self, output_activations, labels):
        # return the derivative of the cost function
        return output_activations - labels

    def sigmoid_prime(self, z):
        # return the derivative of the sigmoid function
        return self.sigmoid(z) * (1 - self.sigmoid(z))