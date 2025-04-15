# src/utils.py
# Author: @ledondodo

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from transformers import AutoTokenizer


def get_smollm():
    """Get the SmolLM model"""
    # local
    checkpoint = "./checkpoints/smollm"

    # online
    if not os.path.exists(checkpoint):
        print("\nDownloading the model online... (SmolLM-135M-Instruct)")
        checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"

    return checkpoint


def get_tokenizer():
    """Get the tokenizer from the checkpoint"""
    checkpoint = get_smollm()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer


def print_vocab(tokenizer):
    """Print the entire vocab of the tokenizer"""
    for token, id in tokenizer.get_vocab().items():
        print(token, "\t", id, "\t", tokenizer.decode([id]))


def make_path(dir, name):
    """Create a path for a file, in a directory of the same name"""
    path = os.path.join(dir, name)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


def tokenizer_config(tokenizer, device, input):
    """Configure tokenizer for input
    and return tokenized input"""
    # Transform input
    messages = [{"role": "user", "content": input}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    return input_tokens


def metrics_predictions(path, size_pred=1000, model_type="quad"):
    """Fit a model to predict time"""
    assert model_type in ["quad", "exp"], "Model type must be 'quad' or 'exp'"
    if not os.path.exists(path):
        os.makedirs(path)

    sizes = np.array([10, 20, 50, 100, 200, 500])
    times = np.array([1.042, 1.319, 3.432, 15.536, 136.565, 1792.830])

    def model(x, a, b, c):
        if model_type == "quad":
            return a + b * x + c * x**2
        elif model_type == "exp":
            return a * b**x

    popt, pcov = curve_fit(model, sizes, times)

    time_pred = model(size_pred, *popt)

    sizes_fit = np.linspace(min(sizes), max(size_pred, max(sizes)), 100)
    times_fit = model(sizes_fit, *popt)

    # Metrics
    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        return 1 - (ss_res / ss_tot)

    preds = model(sizes, *popt)
    mse = np.mean((times - preds) ** 2)
    r2 = r_squared(times, preds)

    print(
        f"\nModel {model_type}:\nPredicted time for size {size_pred}: {time_pred/3600:.2f}h (MSE={mse:.2f}, R2={r2:.3f})\n"
    )

    # Plot the data and the fit
    plt.scatter(sizes, times, label="Data", color="red")
    if model_type == "quad":
        plt.plot(
            sizes_fit,
            times_fit,
            label=f"Fit: {popt[0]:.2f} + {popt[1]:.2f}x + {popt[2]:.2f}x^2",
            color="blue",
        )
    if model_type == "exp":
        plt.plot(
            sizes_fit,
            times_fit,
            label=f"Fit: {popt[0]:.2e} * {popt[1]:.2f}^x",
            color="blue",
        )
    plt.axvline(
        size_pred,
        color="green",
        linestyle="--",
        label=f"Prediction for size {size_pred} = {time_pred/3600:.2f} h",
    )
    plt.axhline(time_pred, color="green", linestyle="--")
    plt.xlabel("Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title("Time vs Size Prediction", fontsize=16, fontweight="bold", y=1.05)
    plt.figtext(
        0.5,
        0.9,
        f"Model: {model_type}, MSE={mse:.2f}, R2={r2:.3f}",
        ha="center",
        fontsize=10,
        style="italic",
        color="gray",
    )
    plt.grid()
    plt.savefig(path + f"/time_model_{model_type}.png")
    plt.close()
    print("Graph saved at", path + f"/time_model_{model_type}.png\n")

    return time_pred, mse, r2


class MatchNotFoundError(Exception):
    """Raised when a required match is not found."""

    def __init__(self, target):
        super().__init__(f"No match found for: {target}")
