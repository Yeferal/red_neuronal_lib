import math
import random
import numpy as np


class Neuronal:
    def __init__(self, *, inputs: int, outputs: int, hidden_layers: int, neurons_hidden_layers: list,
                 hidden_layers_activation_function="SIGMOID", output_layers_activation_function="STEP"):
        """
        :param inputs (int): Number of inputs
        :param outputs (int): Number of outputs
        :param hidden_layers (int): Number of Hidden Layers
        :param neurons_hidden_layer (int[]): Number of Neurons per Hidden Layer
        :param hidden_layers_activation_function (str, optional): Activation function for hidden layers between (SIGMOID or HYPERBOLIC_TANGENT), default SIGMOID
        :param output_layers_activation_function (str, optional): Activation function for hidden layers between (IDENTITY or STEP), default STEP
        :return:
        """
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.neurons_hidden_layers = neurons_hidden_layers
        self.hidden_layers_activation_function = hidden_layers_activation_function
        self.output_layers_activation_function = output_layers_activation_function
        self.weights = []
        self.biases = []
        self.percentage_error = 100
        self.learning_rate = 0.5
        self.output_result = [None]
        self.generate_values()


    @staticmethod
    def sigmoid(x):
        z = (1 / (1 + np.exp(-x)))
        return z

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        # z = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        z = np.tanh(x)
        return z

    @staticmethod
    def tanh_derivative(x):
        # z = 1 - (math.pow(x, 2))
        z = 1.0 - np.tanh(x) ** 2
        return z

    @staticmethod
    def step_function(x, threshold=0):
        # print("STEP:", x, "threshold:", threshold)
        # print(np.where(x >= threshold, 1, 0))
        # return 1 if x >= threshold else 0
        return np.where(x >= threshold, 1, 0)

    @staticmethod
    def identity_function(x):
        return x

    @staticmethod
    def identity_derivated(x):
        return np.ones_like(x)
    def train(self, input_list: list, output_list: list, epoch):
        # Comprobas las capas que sea el mismo numero y todo lo demas antes en enviar al entrenamientos de capas
        self.__train_layers__(input_list, output_list, epoch)

    def __train_layers__(self, input_list: list, output_list: list, epoch):
        for e in range(epoch):
            for i in range(len(input_list)):
                new_output = self.forward_pass(input_list[i])
                loss = np.mean(np.square(input_list[i] - new_output))
                print("Iteration " + str(e) + ": perdida -> " + str(loss), ", res: ", str(new_output))
                self.backward_pass(input_list[i], output_list[i])

    def activation_function_hidden_layers(self, x):
        if self.hidden_layers_activation_function == "SIGMOID":
            return self.sigmoid(x)
        elif self.hidden_layers_activation_function == "HYPERBOLIC_TANGENT":
            return self.tanh(x)
        return 0

    def derivate_function_hidden_layers(self, x):
        if self.hidden_layers_activation_function == "SIGMOID":
            return self.sigmoid_derivative(x)
        elif self.hidden_layers_activation_function == "HYPERBOLIC_TANGENT":
            return self.tanh_derivative(x)
        return 0

    def activation_function_outputs(self, x, threshold=0):
        if self.output_layers_activation_function == "IDENTITY":
            return self.identity_function(x)
        elif self.output_layers_activation_function == "STEP":
            return self.step_function(x, threshold)
        return 0

    def forward_pass(self, input_list):
        self.output_result = []
        self.output_result.append(self.sigmoid(np.dot(input_list, self.weights[0]) + self.biases[0]))

        for i in range(1, self.hidden_layers):
            z = np.dot(self.output_result[i-1], self.weights[i])
            self.output_result.append(self.activation_function_hidden_layers(z + self.biases[i]))

        z = np.dot(self.output_result[self.hidden_layers-1], self.weights[-1])
        self.output_result.append(self.activation_function_outputs(z + self.biases[-1]))

        return self.output_result[-1]

    def backward_pass(self, input_list: list, output_list):
        # errors = y - output_res
        error = output_list - self.output_result[-1]
        # delta error
        if self.output_layers_activation_function == "STEP":
            delta_errors = [error]
        else:
            delta_errors = [self.identity_derivated(error * self.output_result[-1])]

        for i in range(self.hidden_layers - 1, -1, -1):
            error = np.dot(delta_errors[-1], self.weights[i + 1].T)
            delta_errors.append(error * self.derivate_function_hidden_layers(self.output_result[i]))
        delta_errors.reverse()

        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] += np.dot(np.array([input_list]).T, delta_errors[i]) * self.learning_rate
            else:
                self.weights[i] += np.dot(self.output_result[i-1].T, delta_errors[i]) * self.learning_rate

            self.biases[i] += np.sum(delta_errors[i], axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        return self.forward_pass(X)

    def generate_values(self):
        self.weights = []
        self.biases = []

        # YA QUE LAS CAPAS OCULTAS PUEDENTE VARIAR EL EN NUMERO DE NEURONAS
        # ES NECESARIO NIVELAR LOS PESOS CON LAS ENTRADAS
        for i in range(self.hidden_layers):
            layer_weights = []
            if i == 0:
                self.weights.append(np.random.randn(self.inputs, self.neurons_hidden_layers[i]))
                self.biases.append(np.zeros((1, self.neurons_hidden_layers[i])))
            else:
                self.weights.append(np.random.randn(self.neurons_hidden_layers[i-1], self.neurons_hidden_layers[i]))
                self.biases.append(np.zeros((1, self.neurons_hidden_layers[i])))

        self.weights.append(np.random.randn(self.neurons_hidden_layers[-1], self.outputs))
        self.biases.append(np.zeros((1, self.outputs)))


