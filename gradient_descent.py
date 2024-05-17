import numpy as np
def gradient_descent(gradient, start, learn_rate, n_iter=5000,
                     tolerance = 1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff)<=tolerance):
        #it can also happen near a local 
        #minimum or a saddle point.
            break
        vector = vector + diff
    return vector

#calling the gradient descent function for a convex function
#y = 2x^2

#value = gradient_descent(lambda v: 4*v,start=200,learn_rate=0.01)
#print(value)

# For example, you can find the minimum of the function 𝑣₁² + 𝑣₂⁴ #that has the gradient vector (2𝑣₁, 4𝑣₂³):

optimized_loss_function = gradient_descent(
                        lambda v: np.array([2*v[0],4*v[1]**3]),
                        start = np.array([1,1,]),
                        learn_rate=0.1,
)

print(optimized_loss_function)