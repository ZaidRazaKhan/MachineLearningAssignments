import pandas as pd
import numpy as np
import math


df_inner = pd.read_csv('joined_data.csv')


df_inner = df_inner.drop(columns = 'Date_Time')
df_inner = df_inner.drop(columns = 'Date')

def remove_null_columns(data_frame):
    for col in data_frame.columns:
        data_frame = data_frame[pd.notnull(data_frame[col])]
    return data_frame


df_inner = remove_null_columns(df_inner)
# for col in df_inner.columns:
#     df_inner = df_inner[pd.notnull(df_inner[col])]
# df_inner.shape

def normalize(data_frame):
    normalized_data_frame = ((data_frame-data_frame.min())/(data_frame.max()-data_frame.min()))
    return normalized_data_frame

# normalization

df_inner = normalize(df_inner)
# df_inner=((df_inner-df_inner.min())/(df_inner.max()-df_inner.min()))

def data_frame_to_numpy(df_inner, output_column):
    output = df_inner[output_column]
    df_inner = df_inner.drop(columns = output_column)
    return df_inner.to_numpy(), (output).to_numpy()

    
B = np.zeros((1,54)).T
alpha = 0.1
X,Y = data_frame_to_numpy(df_inner, 'PPM')
m = len(Y)
ones = np.ones((m,1)) # for affine to homogeneous conversion
X = np.hstack((X,ones))
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, Y, B)
print(inital_cost)

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

number_of_iterations = 100
# Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, number_of_iterations)

# New Values of B
print(newB)

# Final Cost of new B
print(cost_history[-1])