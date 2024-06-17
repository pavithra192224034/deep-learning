import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    # Calculating the mean squared error
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost

def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    # Initializing weights and biases
    current_weight = 0.1
    current_bias = 0.01
    n = float(len(x))
    
    costs = []
    weights = []
    previous_cost = None
    
    # Gradient descent iterations
    for i in range(iterations):
        y_predicted = (current_weight * x) + current_bias
        
        # Calculating the current cost
        current_cost = mean_squared_error(y, y_predicted)
        
        # Stopping criteria
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break
        
        previous_cost = current_cost
        
        # Updating weights and biases using gradients
        weight_derivative = -(2/n) * np.sum(x * (y - y_predicted))
        bias_derivative = -(2/n) * np.sum(y - y_predicted)
        
        current_weight -= learning_rate * weight_derivative
        current_bias -= learning_rate * bias_derivative
        
        # Storing weights and costs for visualization
        costs.append(current_cost)
        weights.append(current_weight)
        
        # Print every 100th iteration for visibility
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: Cost {current_cost:.4f}, Weight {current_weight:.4f}, Bias {current_bias:.4f}")
    
    # Plotting cost vs weights
    plt.figure(figsize=(8, 6))
    plt.plot(weights, costs, color='blue')
    plt.scatter(weights, costs, marker='o', color='red', label='Cost vs Weights')
    plt.title('Cost vs Weights')
    plt.xlabel('Weight')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return current_weight, current_bias

def main():
    # Input data
    X = np.array([52.50234527, 63.42680403, 81.53035803, 47.47563963, 89.81320787,
                  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
                  45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
                  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    
    Y = np.array([41.70700585, 78.77759598, 82.5623823 , 91.54663223, 77.23092513,
                  78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
                  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
                  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
    
    # Run gradient descent to estimate weights and bias
    estimated_weight, estimated_bias = gradient_descent(X, Y, iterations=2000)
    print(f"Estimated Weight: {estimated_weight:.4f}\nEstimated Bias: {estimated_bias:.4f}")
    
    # Make predictions using the estimated parameters
    Y_pred = estimated_weight * X + estimated_bias
    
    # Plot the regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='orange', label='Data Points')
    plt.plot(X, Y_pred, color='blue', linestyle='dashed', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
