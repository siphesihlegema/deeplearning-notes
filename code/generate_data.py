import numpy as np


def generate_linear_data(
    n_samples=500,
    weights=(3.5, -2.1),
    bias=5.0,
    noise_std=1.5,
    feature_range=(-5, 5),
    random_state=42,
):
    """
    Generate synthetic regression data with two features.

    y = w1*x1 + w2*x2 + bias + noise

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    true_params : dict with the ground-truth weights and bias
    """
    rng = np.random.default_rng(random_state)

    low, high = feature_range
    X = rng.uniform(low, high, size=(n_samples, 2))

    noise = rng.normal(0, noise_std, size=n_samples)
    y = X @ np.array(weights) + bias + noise

    true_params = {"weights": np.array(weights), "bias": bias}
    return X, y, true_params


def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == "__main__":
    X, y, params = generate_linear_data()
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"True weights : {params['weights']}")
    print(f"True bias    : {params['bias']}")
    print(f"y range      : [{y.min():.2f}, {y.max():.2f}]")
