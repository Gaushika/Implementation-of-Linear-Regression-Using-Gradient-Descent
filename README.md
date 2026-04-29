# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the necessary libraries. 
2.Load the dataset from a CSV file and initialize the independent and dependent variables. 
3.Scale the features using a standard scaler to normalize the data. 
4.Initialize parameters. 
5.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function. 
6.Plot the data.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Gaushika RR
RegisterNumber: 212225040091 
*/

import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (X = input, y = output)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Initialize parameters
m = 0  # slope
b = 0  # intercept

# Hyperparameters
learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Compute gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Final parameters
print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions
y_pred = m * X + b

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

## Output:
<img width="725" height="541" alt="image" src="https://github.com/user-attachments/assets/71de36ea-3971-4341-901f-c120267aba95" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
