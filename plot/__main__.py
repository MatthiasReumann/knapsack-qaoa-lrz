import numpy as np
import matplotlib.pyplot as plt


def plot_runtimes(filename: str):
    qubits = ("5", "10", "15", "20")

    # Data copied from output.
    runtimes = {
        "AER Simulator": (1.2, 7, 17, 36),
        "QExa20": (354, 393, 468, 724),
        "AQT20": (5722, 11832, 0, 0),
    }

    x = np.arange(len(qubits))  # the label locations
    width = 0.32  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in runtimes.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Max Runtime / Time To Solution [s]")
    ax.set_xlabel("# Qubits")
    ax.set_xticks(x + width, qubits)
    ax.set_yscale('log')
    ax.set_ylim(top=20000)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(filename)


def plot_objectives(filename: str):
    qubits = ("5", "10", "15", "20")

    # Data copied from output.
    runtimes = {
        "AER Simulator": (10, 29, 57, 78),
        "QExa20": (10, 29, 55, 74),
        "AQT20": (10, 29, 0, 0),
        # "Exact": (10, 29, 58, 80), 
    }

    x = np.arange(len(qubits))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    ax.axhline(y=10, color="gray", linewidth=0.5)
    ax.axhline(y=29, color="gray", linewidth=0.5)
    ax.axhline(y=58, color="gray", linewidth=0.5)
    ax.axhline(y=80, color="gray", linewidth=0.5)

    for attribute, measurement in runtimes.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=10)
        multiplier += 1

    ax.set_ylabel("Avg. Objective Value")
    ax.set_xlabel("# Qubits")
    ax.set_xticks(x + width, qubits)
    ax.set_ylim(top=90)
    
    # fig.set_size_inches(8, 8)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(filename)


if __name__ == "__main__":
    plot_runtimes(filename="plots/runtimes_p1.png")
    plot_objectives(filename="plots/objectives_p1.png")
