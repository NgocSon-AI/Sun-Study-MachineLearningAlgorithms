import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=1e-3, epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs

        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples = len(X)

        self.w = 0.0
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = self.w * X + self.b

            dw = (-2/n_samples) * np.sum((y-y_pred) * X)
            db = (-2/n_samples) * np.sum((y-y_pred))

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 100 == 0:
                loss = np.mean((y-y_pred)**2)
                print(f"Epoch: {epoch}, Loss: {loss}.")
        
    def predict(self, X):
        return self.w * X + self.b


if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 7, 9, 11])


    model = LinearRegression(learning_rate=1e-3, epochs=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"Learned parameters: w = {model.w}, b = {model.b}.")
    plt.scatter(X, y, color='green', label='GroundTruth')
    plt.plot(X, y_pred, color='red', label='Predict')
    plt.title("Ngoc Son Linear Regression.")
    plt.legend()
    plt.grid(True)
    plt.show()
