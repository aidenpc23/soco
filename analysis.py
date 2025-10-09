import matplotlib.pyplot as plt
import numpy as np


def plot_all(histories):
    plot_tracking_results(histories)
    plot_cost_distributions(histories)


def plot_tracking_results(histories):
    plt.figure(figsize=(10, 4))

    first_hist = next(iter(histories.values()))
    y = np.array(first_hist["y"])
    plt.plot(y, label="true target (y_t)", color='black', linestyle='--', linewidth=1.2)

    for model_name, history in histories.items():
        x = np.array(history["x"])
        plt.plot(x, label=f"{model_name} decision (x_t)", linewidth=2)

    plt.title("model tracking comparison")
    plt.xlabel("time step (t)")
    plt.ylabel("decision value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_cost_distributions(histories_dict):
    cost_dict = {name: np.array(hist["total"]) for name, hist in histories_dict.items()}

    plt.figure(figsize=(8, 4))
    plt.boxplot(list(cost_dict.values()), labels=list(cost_dict.keys()), patch_artist=True)
    plt.title("cost distribution per model")
    plt.ylabel("total cost per step")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    for name, costs in cost_dict.items():
        plt.hist(costs, bins=20, alpha=0.5, label=name)
    plt.title("cost histogram comparison")
    plt.xlabel("cost per step")
    plt.ylabel("frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\nsummary: ")
    for name, costs in cost_dict.items():
        total_cost = np.sum(costs)
        avg_cost = np.mean(costs)
        hit_cost = np.sum(histories_dict[name]["hitting"])
        move_cost = np.sum(histories_dict[name]["movement"])
        print(f"{name}:")
        print(f"  total cost: {total_cost:.4f}")
        print(f"  hitting: {hit_cost:.4f}")
        print(f"  move: {move_cost:.4f}")
        print(f"  avg cost per step: {avg_cost:.4f}")
    print("\n\n")
