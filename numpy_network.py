import argparse

import matplotlib.pyplot as plt
import numpy as np


def build_dataset(start=-2.0, stop=2.0, step=0.1, seed=42):
    rng = np.random.default_rng(seed)
    x = np.arange(start, stop + step, step, dtype=float).reshape(1, -1)
    noise = rng.random(x.shape) - 0.5
    y = x ** 2 + noise
    return x, y


def activation_forward(values):
    return np.arctan(values)


def activation_derivative(values):
    return 1.0 / (1.0 + values ** 2)


def initialize_parameters(rng, hidden_units):
    limit_w1 = np.sqrt(6.0 / (1 + hidden_units))
    limit_w2 = np.sqrt(6.0 / (hidden_units + 1))
    w1 = rng.uniform(-limit_w1, limit_w1, size=(hidden_units, 1))
    b1 = np.zeros((hidden_units, 1))
    w2 = rng.uniform(-limit_w2, limit_w2, size=(1, hidden_units))
    b2 = np.zeros((1, 1))
    return w1, b1, w2, b2


def train_network(
    x,
    y,
    hidden_units=5,
    learning_rate=0.15,
    epochs=8000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    w1, b1, w2, b2 = initialize_parameters(rng, hidden_units)

    history = []
    sample_count = x.shape[1]
    report_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        z1 = w1 @ x + b1
        a1 = activation_forward(z1)
        y_pred = w2 @ a1 + b2

        error_output = y - y_pred
        error_hidden = w2.T @ error_output
        hidden_grad = activation_derivative(z1) * error_hidden

        w2 += learning_rate * (error_output @ a1.T) / sample_count
        b2 += learning_rate * np.mean(error_output, axis=1, keepdims=True)
        w1 += learning_rate * (hidden_grad @ x.T) / sample_count
        b1 += learning_rate * np.mean(hidden_grad, axis=1, keepdims=True)

        mse = np.mean(error_output ** 2)
        history.append(mse)

        if (epoch + 1) % report_interval == 0 or epoch == 0:
            print(f"Epoka {epoch + 1:4d}/{epochs} | MSE: {mse:.6f}")

    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "history": np.array(history),
        "prediction": y_pred,
    }


def plot_results(x, y, prediction, history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(x.ravel(), y.ravel(), color="crimson", label="Dane")
    axes[0].plot(x.ravel(), prediction.ravel(), color="navy", label="Predykcja")
    axes[0].set_title("Aproksymacja funkcji, aktywacja: arctan")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history, color="darkgreen")
    axes[1].set_title("Przebieg błędu MSE")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("wynik_sieci_arctan.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Port sieci z Matlaba/Scilaba do Pythona z użyciem numpy."
    )
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--hidden-units", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, y = build_dataset(seed=args.seed)
    result = train_network(
        x,
        y,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )

    final_mse = result["history"][-1]
    print(f"\nFinalne MSE: {final_mse:.6f}")

    plot_results(x, y, result["prediction"], result["history"])


if __name__ == "__main__":
    main()
