import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(gradient, start, learn_rate, n_iter=5000,
                     tolerance=1e-06):
    vector = start
    path = [start]
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector = vector + diff
        path.append(vector)
    return vector, np.array(path)

# Define the function y = 2x^2
def func(x):
    return 2 * x ** 2

# Define the gradient of the function
def gradient(x):
    return 4 * x

# Perform gradient descent
value, path = gradient_descent(gradient, start=200, learn_rate=0.1)

# Generate x values for plotting
x_values = np.linspace(-50, 50, 100)
# Compute y values using the function
y_values = func(x_values)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Function: y = 2x^2', color='blue')

# Plot the path taken by gradient descent
plt.scatter(path, func(path), color='red', label='Gradient Descent Path')

# Mark the starting point and the final value
plt.scatter(path[0], func(path[0]), color='green', label='Start')
plt.scatter(path[-1], func(path[-1]), color='black', label='Final Value')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Visualization')
plt.legend()
plt.grid(True)
plt.show()
