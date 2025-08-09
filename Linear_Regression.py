### SAU DAY LA MAN CODE LINEAR REGRESSION CUA TOI ###

# IMPORT LIBRARY #

import numpy as np
import matplotlib.pyplot as plt


########## CLASS LINEAR REGRESSION ########

class LinearRegression():
    def __init__(self, learning_rate=1e-3, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = None
        self.b = None
    

    def fit(self, X_train, y_train):
        n_samples = len(X_train)
        
        self.w = 0.0
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = self.w * X_train + self.b

            dw = (-2/n_samples) * np.sum((y_train - y_pred)*X_train)
            db = (-2/n_samples) * np.sum((y_train - y_pred))

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 100 == 0:
                loss = np.mean((y_train - y_pred) **2)
                print(f"Epoch: {epoch}, Loss: {loss:.4f}.")
        
    def predict(self, X_test):
        return self.w * X_test + self.b
    
if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate=1e-3, epochs=1000)
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"Learned parameters: w = {model.w}, b = {model.b}.")
    plt.scatter(X, y, color='green', label='Gound Truth')
    plt.plot(X, y_pred, color='red', label='Predicted')
    plt.title('NgocSon Linear Regression')
    plt.legend()
    plt.show()