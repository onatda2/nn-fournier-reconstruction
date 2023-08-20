"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.t = 1
        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        self.m = {}
        self.v = {}
        for i in range(1, self.num_layers + 1):
          self.m["m" + str(i)] = np.zeros_like(self.params['W' + str(i)])
          self.v["v" + str(i)] = np.zeros_like(self.params['W' + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        return ((X@W) + b)

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        return np.where(X > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        def f1(a):
            return 1/(1+np.exp(-a))
        def f2(a):
            return np.exp(a)/(1+np.exp(a))
        y = np.where(x >= 0, f1(x), f2(x))
        return y

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (np.mean(np.mean(np.square(y-p), axis=0)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        # Store the output of each layer in
        # self.outputs as it will be used during back-propagation. Use
        # the same keys as self.params. Use functions like
        # self.linear, self.relu, and self.mse in here.
        self.outputs = {}
        self.outputs["nonlinear"+str(0)] = X
        X_cpy = X
        for i in range(1, self.num_layers):
            Y = self.linear(self.params["W" + str(i)], X_cpy, self.params["b" + str(i)])
            self.outputs["linear"+str(i)] = Y
            Y = self.relu(Y)
            self.outputs["nonlinear"+str(i)] = Y
            X_cpy = Y
        Y = self.linear(self.params["W" + str(self.num_layers)], X_cpy, self.params["b" + str(self.num_layers)])
        self.outputs["linear" + str(self.num_layers)] = Y
        Y = self.sigmoid(Y)
        self.outputs["nonlinear" + str(self.num_layers)] = Y
        return Y

    
    def sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
        return (self.sigmoid(x) * (1-self.sigmoid(x)))
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (-2*(y-p))/(p.shape[0] * p.shape[1])
    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        
        total_loss = self.mse(y, self.outputs["nonlinear" + str(self.num_layers)])
        
        
        self.gradients["nonlinear" + str(self.num_layers)] = self.mse_grad(y, self.outputs["nonlinear" + str(self.num_layers)]) * self.sigmoid_grad(self.outputs["linear" + str(self.num_layers)])
        
        
        self.gradients["W" + str(self.num_layers)] = self.outputs["nonlinear" + str(self.num_layers-1)].T @ self.gradients["nonlinear" + str(self.num_layers)]
        
        self.gradients["b" + str(self.num_layers)] = np.sum(self.gradients["nonlinear" + str(self.num_layers)], axis=0)
        
        self.gradients["nonlinear" + str(self.num_layers)] = self.gradients["nonlinear" + str(self.num_layers)] @ self.params["W"+str(self.num_layers)].T
        
        i = self.num_layers - 1
        
        while (i > 0):
            self.gradients["nonlinear" + str(i)] = self.gradients["nonlinear" + str(i+1)] * self.relu_grad(self.outputs["linear"+str(i)])
            
            self.gradients["W" + str(i)] = self.outputs["nonlinear" + str(i-1)].T @ self.gradients["nonlinear" + str(i)]
            
            self.gradients["b" + str(i)] = np.sum(self.gradients["nonlinear" + str(i)], axis=0)
            
            self.gradients["nonlinear" + str(i)] = self.gradients["nonlinear" + str(i)] @ self.params["W"+str(i)].T
            
            i = i-1
            
        return total_loss


    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # Handles updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
          for i in range(1, self.num_layers + 1):
                self.params['W' + str(i)] -= lr * self.gradients['W' + str(i)]
                self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)]
        elif opt == "Adam":
          self.t += 1
          for i in range(1, self.num_layers + 1):
            g_t = self.gradients['W' + str(i)]
            self.m['m' + str(i)] = b1*self.m['m' + str(i)]+(1.0-b1)*g_t
            self.v['v' + str(i)] = b2*self.v['v' + str(i)]+(1.0-b2)*(g_t**2)
            m_hat = self.m['m' + str(i)]/(1-(b1**self.t))
            v_hat = self.v['v' + str(i)]/(1-(b2**self.t))

            self.params['W' + str(i)] -= lr*m_hat/(np.sqrt(v_hat)+eps)
            self.params['b' + str(i)] -= lr*self.gradients['b' + str(i)]
