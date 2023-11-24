import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import seaborn as sns


def output_value_iteration_performance_metrics_graph(
    df: pd.DataFrame,
    mdp: str,
    gamma: float,
    grid_style: str = "whitegrid",
) -> None:
    title = f"{mdp.replace('_', ' ').title()}: \nHistorical Delta Over Iterations; gamma = {gamma}"
    output_location = f"../outputs/value_iteration/{mdp}_historical_performance_metrics_gamma_{str(gamma).replace('.', '_')}.png"

    os.makedirs(os.path.dirname(output_location), exist_ok=True)

    sns.set_style(grid_style)

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["Iteration"],
        df["Historical Delta"],
        label="Historical Delta",
        color="blue",
    )
    plt.plot(
        df["Iteration"],
        df["Historical Value Function"],
        label="Historical Value Function",
        color="red",
    )
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_location)
    plt.close()


def output_policy_iteration_performance_metrics_graph(
    df: pd.DataFrame,
    mdp: str,
    gamma: float,
    grid_style: str = "whitegrid",
) -> None:
    title = f"{mdp.replace('_', ' ').title()}: \nHistorical Delta Over Iterations; gamma = {gamma}"
    output_location = f"../outputs/policy_iteration/{mdp}_historical_performance_metrics_gamma_{str(gamma).replace('.', '_')}.png"

    os.makedirs(os.path.dirname(output_location), exist_ok=True)

    sns.set_style(grid_style)

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["Iteration"],
        df["Historical Delta"],
        label="Historical Delta",
        color="blue",
    )
    plt.plot(
        df["Iteration"],
        df["Historical Value Function"],
        label="Historical Value Function",
        color="red",
    )
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_location)
    plt.close()

def plot_train_DQN_performance(
    mdp: str,
    df: pd.DataFrame,
    grid_style: str = "whitegrid",
) -> None:
    title = f"{mdp.replace('_', ' ').title()}: \nDQN: Performance Over Episodes"
    output_location = f"../outputs/DQN/{mdp}_performance_over_episodes.png"

    os.makedirs(os.path.dirname(output_location), exist_ok=True)

    sns.set_style(grid_style)

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["Episode"],
        df["Total Rewards"],
        label="Total Rewards",
        color="purple",
    )
    plt.plot(
        df["Episode"],
        df["Epsilon Value"],
        label="Epsilon Value",
        color="green",
    )
    plt.plot(
        df["Episode"],
        df["10-Episode Rolling Avg Rewards"],
        label="10-Episode Rolling Avg Rewards",
        color="blue",
    )
    plt.plot(
        df["Episode"],
        df["30-Episode Rolling Avg Rewards"],
        label="30-Episode Rolling Avg Rewards",
        color="red",
    )
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_location)
    plt.close()