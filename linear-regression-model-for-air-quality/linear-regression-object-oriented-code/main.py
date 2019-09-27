import time
import argparse
from DataLoader import DataLoader
from DataManipulator import DataManipulator
from LinearPredictor import LinearPredictor
import matplotlib.pyplot as plt
import numpy as np

def main():
    
	# Arguments Parser Structure
    parser = argparse.ArgumentParser(description='Linear Regression.')
    parser.add_argument('-v', '--verbose', action='store_true', help='print onging learning status')
    parser.add_argument('-a', '--alpha', action='store', help='learning rate (default=0.1)', default=0.1, type=float)
    parser.add_argument('-l', '--lmbda', action='store', help='regularization constant (default=0.1)', default=0.1, type=float)
    parser.add_argument('-i', '--iterations', action='store', help='number of iterations for learning (default=10000)', default=5000, type=int)

	# Parse Arguments
    parsed = parser.parse_args()

    verbose = parsed.verbose
    alpha = parsed.alpha
    iterations = parsed.iterations
    lmbda = parsed.lmbda

    # Initialize Data Loader
    print('Initializing Data Loader ......... ', end='', flush=True)
    start = time.time()   
    loader = DataLoader()
    print('done -- ' + str(round(time.time()-start, 3)) + 's')
    
    # Load Input Files
    print('Loading Input Files .............. ', end='', flush=True)
    start = time.time()
    SO2_train_data = loader.loadCSVFile('SO2_train.csv')
    O3_train_data = loader.loadCSVFile("O3_train.csv")
    PM2_5_train_data = loader.loadCSVFile("PM2.5_train.csv")
    SO2_test_data = loader.loadCSVFile('SO2_test.csv')
    O3_test_data = loader.loadCSVFile("O3_test.csv")
    PM2_5_test_data = loader.loadCSVFile("PM2.5_test.csv")
    print('done -- ' + str(round(time.time()-start, 3)) + 's')

    # Initialize Linear Models
    print('Initializing Models .............. ', end='', flush=True)
    start = time.time()
    SO2_model = LinearPredictor() 
    O3_model = LinearPredictor()
    PM2_5_model = LinearPredictor()
    print('done -- ' + str(round(time.time()-start, 3)) + 's')

    # Learn Hypothesis
    print('Training SO2 Model ............... ', end='', flush=True)
    start = time.time()
    history = SO2_model.fit(SO2_train_data, alpha, iterations, lmbda, verbose)
    plt.plot(range(0, len(history)), history, label='SO2')
    print('done -- ' + str(round(time.time()-start, 3)) + 's' + ' Training MSE: ' + str(history[-1]))

    print('Training O3 Model ................ ', end='', flush=True)
    start = time.time()
    history = O3_model.fit(O3_train_data, alpha, iterations, lmbda, verbose)
    plt.plot(range(0, len(history)), history, label='O3')
    print('done -- ' + str(round(time.time()-start, 3)) + 's' + ' Training MSE: ' + str(history[-1]))

    print('Training PM2.5 Model ............. ', end='', flush=True)
    start = time.time()
    history = PM2_5_model.fit(PM2_5_train_data, alpha, iterations, lmbda, verbose)
    plt.plot(range(0, len(history)), history, label='PM2.5')
    print('done -- ' + str(round(time.time()-start, 3)) + 's' + ' Training MSE: ' + str(history[-1]))

    # Test Hypothesis
    print('Testing SO2 Model ................ ', end='', flush=True)
    start = time.time()
    error = SO2_model.test(SO2_test_data)
    print('done -- ' + str(round(time.time()-start, 3)) + 's' + ' MSE: ' + str(error))

    print('Testing O3 Model ................. ', end='', flush=True)
    start = time.time()
    error = O3_model.test(O3_test_data)
    print('done -- ' + str(round(time.time()-start, 3)) + 's' + ' MSE: ' + str(error))

    print('Testing PM2.5 Model .............. ', end='', flush=True)
    start = time.time()
    error = PM2_5_model.test(PM2_5_test_data)
    print('done -- ' + str(round(time.time()-start, 3)) + 's' + ' MSE: ' + str(error))

    # Plot    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()