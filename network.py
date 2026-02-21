from typing import List, Tuple, TextIO
from abc import ABC, abstractmethod
import math
import json
import random

def multiply_matrix_by_vector(matrix: List[List[float]], vector: List[float]) -> List[float]:
    result = []
    if len(matrix[0]) != len(vector):
        raise ValueError('The number of columns in the matrix should be equal to the number of elements in the vector')
    length = len(vector)
    for row in matrix:
        val = 0
        if len(row) != length:
            raise ValueError('The two-dimensional list should be a matrix')
        for i in range(length):
            val += row[i] * vector[i]
        result.append(val)
        
    return result
    
def fill_vector_to_needed_length(vector: List[float], length: int) -> List[float]:
    return vector + [0]*(length - len(vector))

def add_vectors(vector1: List[float], vector2: List[float]) -> List[float]:
    lv1 = len(vector1)
    lv2 = len(vector2)
    if lv1 != lv2:
        raise ValueError('The vectors should have equal number of elements')
        
    result = [vector1[i] + vector2[i] for i in range(lv1)]
    return result

def multiply_vector_by_number(vector: List[float], number: float) -> List[float]:
    return [el * number for el in vector]

class Wrapper(ABC):
    @abstractmethod
    def function(self, x: float) -> float:
        pass

    @abstractmethod
    def derivative(self, x: float) -> float:
        pass

class Sigmoid(Wrapper):
    def function(self, x: float) -> float:
        return 1/(1+math.exp(-x))

    def derivative(self, x: float) -> float:
        return math.exp(-x)/(1+math.exp(-x))**2

class Linear(Wrapper):
    def function(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1

class Network:
    def __init__(self, layers: List[int], wrapper: Wrapper):
        self._layers = layers
        self._layer_num = len(self._layers)
        self._wrapper = wrapper
        self._weights = []
        self._biases = []

        for ind in range(1, len(layers)):
            self._weights.append([])
            self._biases.append([])
            for i in range(layers[ind]):
                self._biases[ind-1].append(random.uniform(-5, 5))
                self._weights[ind-1].append([])
                for j in range(layers[ind-1]):
                    self._weights[ind-1][i].append(random.uniform(-5, 5))

    def set_weights(self, weights: List[List[List[float]]]):
        self._weights = weights

    def set_biases(self, biases: List[List[float]]):
        self._biases = biases

    @classmethod
    def load_model(cls, file: TextIO, wrapper: Wrapper):
        options = json.loads(file.read())
        instance = cls(options['layers'], wrapper)
        instance.set_weights(options['weights'])
        instance.set_biases(options['biases'])
        return instance

    def save_model(self, name: str):
        model_info = {
            'layers': self._layers,
            'weights': self._weights,
            'biases': self._biases,
        }
        with open(f'{name}.json', 'w') as file:
            file.write(json.dumps(model_info))
                        
    def _calculate_weighted_sums_and_activations(self, activations: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        if self._layers[0] != len(activations):
            raise ValueError('The input length should be equal to the given length of the first layer')
        
        layer_sums = [activations]
        layer_activations = [activations]

        for i in range(self._layer_num - 1):
            weighted_sums = add_vectors(multiply_matrix_by_vector(self._weights[i], activations), self._biases[i])
            layer_sums.append(weighted_sums)
            activations = []
            for el in weighted_sums:
                activations.append(self._wrapper.function(el))

            layer_activations.append(activations)

        return layer_sums, layer_activations

    def get_answer(self, input_data: List[float]) -> List[float]:
        return self._calculate_weighted_sums_and_activations(input_data)[1][-1]
        
    def _calculate_partial_gradient(self, input_data: List[float], output_data: List[float]) -> List[float]:
        model_sums, model_activations = self._calculate_weighted_sums_and_activations(input_data)
        partial_gradient = []
        current_layer = self._layer_num - 1
        cost_derivatives_to_use = []
        while current_layer > 0:
            new_cost_derivatives = [0] * self._layers[current_layer - 1]
            for current in range(self._layers[current_layer]):
                if current_layer == self._layer_num - 1:
                    cost_derivative = 2*(model_activations[current_layer][current] - output_data[current])
                else:
                    cost_derivative = cost_derivatives_to_use[current]
                activation_derivative = self._wrapper.derivative(model_sums[current_layer][current])

                cost_derivative_over_bias = cost_derivative * activation_derivative
                for previous in range(self._layers[current_layer - 1]):
                    weighted_sum_derivative = model_activations[current_layer - 1][previous]
                    cost_derivative_over_weight = weighted_sum_derivative * cost_derivative * activation_derivative

                    partial_gradient.append(cost_derivative_over_weight)

                    weight = self._weights[current_layer-1][current][previous]
                    new_cost_derivatives[previous] += cost_derivative * activation_derivative * weight

                partial_gradient.append(cost_derivative_over_bias)

            cost_derivatives_to_use = new_cost_derivatives

            current_layer -= 1
        return partial_gradient

    def _calculate_gradient(self, dataset: List[Tuple[List[float], List[float]]], test_examples_num: int) -> List[float]:
        gradient_sum = []
        for test in dataset:
            new_gradient = self._calculate_partial_gradient(test[0], test[1])
            if gradient_sum == []:
                gradient_sum = new_gradient
            else:
                gradient_sum = add_vectors(gradient_sum, new_gradient)

        return multiply_vector_by_number(gradient_sum, 1/test_examples_num)

    def _descent(self, gradient: List[float], learning_rate: float):
        gradient_index = 0
        for layer in range(self._layer_num - 1, 0, -1):
            for current in range(self._layers[layer]):
                for previous in range(self._layers[layer-1]):
                    self._weights[layer - 1][current][previous] -= learning_rate * gradient[gradient_index]
                    gradient_index += 1
                self._biases[layer - 1][current] -= learning_rate * gradient[gradient_index]
                gradient_index += 1

    def _cost(self, dataset: List[Tuple[List[float], List[float]]]):
        value = 0
        for test in dataset:
            activations = self.get_answer(test[0])
            for ind in range(self._layers[-1]):
                value += (activations[ind] - test[1][ind])**2

        return value

    def _epoch(self, dataset: List[Tuple[List[float], List[float]]],
                     learning_rate: float, batch_size: int, train_examples_num: int):
        if batch_size != 0:
            random.shuffle(dataset)
            for lbound in range(0, train_examples_num, batch_size):
                rbound = lbound + batch_size
                batch = dataset[lbound:rbound]
                gradient = self._calculate_gradient(batch, batch_size)
                self._descent(gradient = gradient, learning_rate = learning_rate)
        else:
            gradient = self._calculate_gradient(dataset, train_examples_num)
            self._descent(gradient = gradient, learning_rate = learning_rate)

    def train(self, dataset: List[Tuple[List[float], List[float]]],
              learning_rate: float, epochs: int,
              validation: List[Tuple[List[float], List[float]]] = [], patience: int = 0,
              batch_size: int = 0):
        train_examples_num = len(dataset)
        last_cost = 0
        being_patient = 0
        for it in range(epochs):
            self._epoch(dataset=dataset, learning_rate=learning_rate, batch_size=batch_size,
                              train_examples_num=train_examples_num)
            if validation and patience:
                cost = self._cost(dataset=validation)
                if cost >= last_cost:
                    being_patient += 1
                else:
                    being_patient = 0
                last_cost = cost
                if being_patient == patience//2:
                    print('Dividing learning rate')
                    learning_rate /= 2
                if being_patient == patience:
                    print('Reached patience value')
                    break