import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(gradient, start, learn_rate, n_iter=5000,
                     tolerance=1e-06):
    vector = start
    trajectory = [vector]  # Store the trajectory of optimization
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.linalg.norm(diff) <= tolerance:
            # Convergence condition
            break
        vector = vector + diff
        trajectory.append(vector)  # Append the new vector to the trajectory
    return np.array(trajectory)

# Define the loss function and its gradient
def loss_function(v):
    return v[0]**2 + v[1]**4

def gradient(v):
    return np.array([2*v[0], 4*v[1]**3])

# Call the gradient descent function
trajectory = gradient_descent(gradient, 
                              start=np.array([1, 1]), 
                              learn_rate=0.1)

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100), 
             [[loss_function(np.array([x, y])) for x in np.linspace(-2, 2, 100)] 
              for y in np.linspace(-2, 2, 100)], levels=50, cmap='viridis')
plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', marker='o')
plt.xlabel('v1')
plt.ylabel('v2')
plt.title('Gradient Descent Optimization Path')
plt.colorbar(label='Loss')
plt.grid(True)
plt.show()
