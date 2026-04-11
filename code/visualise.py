import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from generate_data import generate_linear_data, train_test_split
from linear_regression import LinearRegression


# ── data ─────────────────────────────────────────────────────────────────────
X, y, true_params = generate_linear_data(n_samples=500, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ── model ─────────────────────────────────────────────────────────────────────
model = LinearRegression(learning_rate=0.05, n_epochs=500, batch_size=32, random_state=42)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
r2 = model.score(X_test, y_test)

print(f"\nLearned weights : {model.weights}")
print(f"Learned bias    : {model.bias:.4f}")
print(f"True weights    : {true_params['weights']}")
print(f"True bias       : {true_params['bias']}")
print(f"Test R²         : {r2:.4f}")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Linear Regression with SGD — Two-Feature Synthetic Data", fontsize=14, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. 3-D scatter: data + fitted plane
ax3d = fig.add_subplot(gs[0, :2], projection="3d")
ax3d.scatter(X_test[:, 0], X_test[:, 1], y_test, alpha=0.5, s=18, label="Test data", color="steelblue")

x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
zz = model.weights[0] * xx1 + model.weights[1] * xx2 + model.bias
ax3d.plot_surface(xx1, xx2, zz, alpha=0.3, color="tomato", label="Fitted plane")
ax3d.set_xlabel("Feature 1")
ax3d.set_ylabel("Feature 2")
ax3d.set_zlabel("y")
ax3d.set_title("Fitted plane vs. test data")
ax3d.legend(loc="upper left", fontsize=8)

# 2. Training loss curve
ax_loss = fig.add_subplot(gs[0, 2])
ax_loss.plot(model.loss_history, color="darkorange", linewidth=1.5)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("MSE Loss")
ax_loss.set_title("SGD Training Loss")
ax_loss.grid(True, alpha=0.3)

# 3. Predicted vs actual
ax_pva = fig.add_subplot(gs[1, 0])
ax_pva.scatter(y_test, y_pred_test, alpha=0.5, s=18, color="steelblue")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
ax_pva.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
ax_pva.set_xlabel("Actual y")
ax_pva.set_ylabel("Predicted y")
ax_pva.set_title(f"Predicted vs Actual  (R²={r2:.3f})")
ax_pva.legend(fontsize=8)
ax_pva.grid(True, alpha=0.3)

# 4. Residuals vs predicted
residuals = y_test - y_pred_test
ax_res = fig.add_subplot(gs[1, 1])
ax_res.scatter(y_pred_test, residuals, alpha=0.5, s=18, color="mediumpurple")
ax_res.axhline(0, color="red", linestyle="--", linewidth=1)
ax_res.set_xlabel("Predicted y")
ax_res.set_ylabel("Residual")
ax_res.set_title("Residuals vs Predicted")
ax_res.grid(True, alpha=0.3)

# 5. Partial regression plots (y vs each feature with the other held at mean)
ax_feat = fig.add_subplot(gs[1, 2])
colors = ["steelblue", "darkorange"]
feature_names = ["Feature 1", "Feature 2"]
for i in range(2):
    x_range = np.linspace(X[:, i].min(), X[:, i].max(), 100)
    X_line = np.full((100, 2), X.mean(axis=0))
    X_line[:, i] = x_range
    y_line = model.predict(X_line)
    ax_feat.plot(x_range, y_line, color=colors[i], linewidth=2, label=feature_names[i])

ax_feat.set_xlabel("Feature value")
ax_feat.set_ylabel("Predicted y")
ax_feat.set_title("Partial effect of each feature\n(other feature held at mean)")
ax_feat.legend(fontsize=8)
ax_feat.grid(True, alpha=0.3)

plt.savefig("linear_regression_results.png", dpi=150, bbox_inches="tight")
print("\nFigure saved to linear_regression_results.png")
plt.show()
