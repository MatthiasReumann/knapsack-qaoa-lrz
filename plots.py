import numpy as np
import matplotlib.pyplot as plt


def plot_runtimes(filename: str):
    qubits = ("5", "10", "15", "20")
    runtimes = {
        "AER Simulator": (1.19, 6.08, 12.93, 33.46),
        "QExa20": (334.30, 413.90, 607.27, 589.31),
    }

    # plt.bar(qubits, runtime, width=0.25, label="AER Simulator")

    x = np.arange(len(qubits))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in runtimes.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Runtime / Time To Solution [s]")
    ax.set_xlabel("# Qubits")
    ax.set_xticks(x + width, qubits)
    ax.set_ylim(top=700)

    plt.legend(loc="upper left")
    plt.savefig(filename)


if __name__ == "__main__":
    plot_runtimes(filename="plots/runtimes_p1.png")
