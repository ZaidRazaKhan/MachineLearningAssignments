import numpy as np
import pandas as pd

class LinearPredictor():

    # Homogenize
    def homogenize(self, X):
        shape = X.shape
        X_homo = np.ones((shape[0], shape[1]+1))
        X_homo[:,:-1] = X
        return X_homo

    # Loss Function
    def loss(self, Y_true, Y_pred, lmbda):
        return (np.square(Y_true-Y_pred).mean() + lmbda * np.square(self.W).mean()) / 2

    # Funciton to find the Linear Hypothesis
    def fit(self, data, alpha, iterations, lmbda, verbose=False):

        Y_true = data['Sample Measurement'].values
        Y_true = np.reshape(Y_true, (Y_true.shape[0], 1))

        X = data.iloc[:, 1:-1].values
        m = X.shape[0]

        # Make X homogeneous
        X = self.homogenize(X)

        # Initialize Weights    
        self.W = np.zeros((X.shape[1], 1))

        loss_data = []

        for i in range(iterations):
            Y_pred = np.dot(X, self.W)
            error = Y_pred - Y_true
            gradient = np.dot(X.T, error)
            self.W -= np.dot(alpha/m, gradient + lmbda*self.W)
            loss_data.append(self.loss(Y_true, Y_pred, lmbda))
            if verbose:
                print("Iteration " + str(i) + ": Loss = ", loss_data[-1])

        return loss_data

    # Test Model
    def test(self, data):

        Y_true = data['Sample Measurement'].values
        Y_true = np.reshape(Y_true, (Y_true.shape[0], 1))

        X = data.iloc[:, 1:-1].values
        
        Y_pred = np.dot(self.homogenize(X), self.W)
        return (np.square(Y_true-Y_pred).mean())/2
