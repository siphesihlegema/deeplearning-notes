import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=32, random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0, 0.01, size=(n_features,))
        self.bias = 0.0

    def predict(self, X):
        return X @ self.weights + self.bias

    def _mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def _compute_gradients(self, X_batch, y_batch):
        n = len(y_batch)
        y_pred = X_batch @ self.weights + self.bias
        error = y_pred - y_batch
        grad_w = (2 / n) * (X_batch.T @ error)
        grad_b = (2 / n) * np.sum(error)
        return grad_w, grad_b

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_params(n_features)
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.n_epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad_w, grad_b = self._compute_gradients(X_batch, y_batch)
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            epoch_loss = self._mse_loss(self.predict(X), y)
            self.loss_history.append(epoch_loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs} — MSE Loss: {epoch_loss:.4f}")

        return self

    def score(self, X, y):
        """R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
