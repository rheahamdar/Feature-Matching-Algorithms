from copy import copy 
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
rng = default_rng()

class RANSACNearestPoint:
    def __init__(self, n=4, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n #minimum number of data points to estimate
        self.k = k #maximum number of iterations allowed in the algorithm
        self.t = t #threshold value to determine data points that are fit well by the model(inlier)
        self.d = d #number of close data points(inliers) required to assert that the model fits well to the data
        self.model = model #linear regression object implementing 'fit' and 'predict'
        self.loss = loss #function of 'y_true' and 'y_pred' that returns a vector
        self.metric = metric #function of 'y_true' and 'y_pred' that returns a float
        self.best_fit = None #model parameters which best fit the data
        self.best_error = np.inf #to sharpen the model parameters to the best data fitting as iterations go on

    def fit(self,X,y):
        for _ in range(self.k):
            indices = rng.permutation(X.shape[0])
            maybe_inliers = indices[:self.n]
            maybe_model = copy(self.model).fit(X[maybe_inliers],y[maybe_inliers])

            thresholded = (self.loss(y[indices][self.n:], maybe_model.predict(X[indices][self.n:]))< self.t)

            inlier_indices = indices[self.n:][np.flatnonzero(thresholded).flatten()]

            if inlier_indices.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_indices])
                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

                this_error = self.metric(y[inlier_points], better_model.predict(X[inlier_points]))

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = better_model
                    self.best_inliers = inlier_indices

        return self

    def nearest_point(self, target):
        return self.best_fit.predict(target.reshape(1, -1))


def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params


if __name__ == "__main__":
    # Example usage
    X = np.random.rand(100, 2)  # Random 2D data with 100 data points and 2 features
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, size=(100,))  # Linear relationship with noise

    target = np.array([0.5, 0.5])  # Example target point

    ransac_nearest_point = RANSACNearestPoint(n=4, k=100, t=0.1, d=10, model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)
    ransac_nearest_point.fit(X, y)

    nearest_point = ransac_nearest_point.nearest_point(target)

    
    # Plotting
    plt.scatter(X[:, 0], X[:, 1], label='Data Points')
    plt.scatter(target[0], target[1], marker='x', color='red', label='Target Point')
    plt.scatter(nearest_point, nearest_point, marker='o', color='green', label='Nearest Point')

    plt.legend()
    plt.title('RANSAC Nearest Point')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    