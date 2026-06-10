import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Prevent overflow
    return exp_x / exp_x.sum()

def leaky_reLU(x):
    return np.maximum(0.01 * x, x)

def deriv_leaky_reLU(x):
    x = np.array(x)  # Ensure NumPy array
    return np.where(x > 0, 1, 0.01)

def lerp(a, b, t):
    return a + t * (b - a)

def generate_random_list(size):
    return [random.uniform(-1, 1) for _ in range(size)]  # More efficient random initialization

class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, learning_rate=0.1):
        self.lr = learning_rate
        self.input_count = inputs
        self.hidden_count = hidden
        self.output_count = outputs
        
        self.hidden_data = []
        self.pre_hidden = []
        self.pre_outputs = []
        
        # Initialize weights and biases
        self.weights_ih = np.random.uniform(-1, 1, (hidden, inputs))  # Use NumPy for efficiency
        self.bias_h = np.random.uniform(-1, 1, hidden)
        
        self.weights_ho = np.random.uniform(-1, 1, (outputs, hidden))
        self.bias_o = np.random.uniform(-1, 1, outputs)
    
    def feed_forward(self, inputs):
        self.hidden_data = []
        self.pre_hidden = []
        self.pre_outputs = []
        
        # Convert inputs to NumPy array
        inputs = np.array(inputs)
        
        # Input to hidden layer
        self.pre_hidden = np.dot(self.weights_ih, inputs) + self.bias_h
        self.hidden_data = leaky_reLU(self.pre_hidden)
        
        # Hidden to output layer
        self.pre_outputs = np.dot(self.weights_ho, self.hidden_data) + self.bias_o
        outputs = softmax(self.pre_outputs)  # Use softmax for classification
        
        return outputs
    
    def cost(self, inputs, results):
        predictions = self.feed_forward(inputs)
        mse = np.mean((np.array(results) - predictions) ** 2)
        return mse
    
    def train(self, inputs, results):
        inputs = np.array(inputs)
        results = np.array(results)
        
        predictions = self.feed_forward(inputs)
        cost_deriv = predictions - results  # Softmax + Cross-Entropy simplification

        # Calculate output layer derivatives
        deriv_output_weights = np.outer(cost_deriv, self.hidden_data)  # Faster weight updates
        deriv_bias_o = cost_deriv  # Gradient of bias is just the error
        
        # Calculate hidden layer errors
        deriv_hidden_errors = np.dot(self.weights_ho.T, cost_deriv)
        
        # Calculate hidden layer derivatives
        deriv_hidden_weights = np.outer(deriv_hidden_errors * deriv_leaky_reLU(self.pre_hidden), inputs)
        deriv_bias_h = deriv_hidden_errors * deriv_leaky_reLU(self.pre_hidden)
        
        # Update weights and biases
        self.weights_ho -= self.lr * deriv_output_weights
        self.bias_o -= self.lr * deriv_bias_o
        
        self.weights_ih -= self.lr * deriv_hidden_weights
        self.bias_h -= self.lr * deriv_bias_h

    def print_parameters(self):
        print("Input to Hidden Weights:\n", self.weights_ih)
        print("\nHidden Biases:\n", self.bias_h)
        print("\nHidden to Output Weights:\n", self.weights_ho)
        print("\nOutput Biases:\n", self.bias_o)
