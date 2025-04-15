# algorithms/algo_7.py
# Author: @ledondodo

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_data(data_path):
    """
    Load CSV data

    Args:
        data_path (str): path to the CSV file

    Returns:
        df (pd.DataFrame): data loaded
    """
    pickle_file = os.path.splitext(data_path)[0] + ".pkl"
    if os.path.exists(pickle_file):
        print("Loading data from pickle file...\n")
        df = pd.read_pickle(pickle_file)
    else:
        print("Loading data from CSV file...\n")
        df = pd.read_csv(data_path)
        df.to_pickle(pickle_file)
    return df


def dataset(data_path):
    """
    Process dataset for FSM

    Args:
        data_path (str): path to the CSV file

    Returns:
        words (list): list of unique words
    """
    df = load_data(data_path)
    print("Data raw: size", df.shape, "\n", df.head(), "\n")
    # Filter data
    dff = df[df.cefr.apply(lambda x: "A1" in x)]
    print(
        "Data A1: size",
        dff.shape,
        f"= {100*round(dff.shape[0]/df.shape[0],4):.2f}%",
        "\n",
        dff.head(),
        "\n",
    )
    # list of unique words
    words = dff.word.unique().tolist()
    print("Unique words:", len(words), "\n")
    return words


def expand_words(words, breaks, spaces):
    """
    Expand words with common variations

    Args:
        words (list): list of unique words
        breaks (list): list of common breaks
        spaces (list): list of common spaces

    Returns:
        data (dict): expanded words data"""
    data = {"words": [], "breaks": breaks, "spaces": spaces}
    for w in words:
        data["words"].append(w)
        # Case variations
        data["words"].append(w.lower())
        data["words"].append(w.title())
        data["words"].append(w.upper())
        # Special characters variations
        data["words"].append(w + " ")
        data["words"].append(" " + w)
    # Remove duplicates
    data["words"] = list(set(data["words"]))
    print("Expanded words:", len(data["words"]), "\n")
    return data


def metrics_predictions(path, size_pred=1000, model_type="quad"):
    """
    Fit a model to predict time

    Args:
        path (str): path to save the graph
        size_pred (int): size to predict time
        model_type (str): type of model to fit, "quad" or "exp"

    Returns:
        time_pred (float): predicted time
        mse (float): mean squared error
        r2 (float): R-squared
    """
    assert model_type in ["quad", "exp"], "Model type must be 'quad' or 'exp'"
    if not os.path.exists(path):
        os.makedirs(path)

    sizes = np.array([10, 20, 50, 100, 200, 500, 931])
    times = np.array([1.55, 1.79, 3.14, 11.33, 91.23, 1222.404, 5407.22])
    # times pynini
    sizes = np.array([10, 20, 50, 100, 200, 500])
    times = np.array([0.771, 0.908, 6.795, 48.180, 234.421, 2762.521])

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
    plt.xlabel("Data Subset Size", fontsize=14)
    plt.ylabel(
        "Time (s)",
        fontsize=14,
    )
    plt.legend(fontsize=14, loc="upper right", bbox_to_anchor=(0.83, 0.94))
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
    plt.tick_params(axis="both", labelsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + f"/time_model_{model_type}.png")
    plt.close()
    print("Graph saved at", path + f"/time_model_{model_type}.png\n")

    return time_pred, mse, r2
