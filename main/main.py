import numpy as np
import pandas as pd

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional for enhanced aesthetics

import itertools


def simple_weather_model_mdp() -> tuple[np.ndarray, np.ndarray]:
    # Number of states and actions
    n_states = 2
    n_actions = 2

    # Transition Probability Matrix [actions x states x states]
    P = np.zeros((n_actions, n_states, n_states))

    # Carrying an umbrella
    P[0, 0, 0] = 0.9  # Sunny today, sunny tomorrow
    P[0, 0, 1] = 0.1  # Sunny today, rainy tomorrow
    P[0, 1, 0] = 0.5  # Rainy today, sunny tomorrow
    P[0, 1, 1] = 0.5  # Rainy today, rainy tomorrow

    # Not carrying an umbrella
    P[1] = P[0]  # Same transition probabilities

    # Reward Matrix [actions x states]
    R = np.array(
        [
            # Rewards for carrying an umbrella
            [1, -1],
            # Rewards for not carrying an umbrella
            [0, -2],
        ]
    )

    return (P, R)


def vending_machine_mdp(
    max_count: int = 10, n_item_types: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an MDP for a vending machine.

    Parameters:
    max_count (int): Maximum count for each item type.
    n_item_types (int): Number of different item types in the vending machine.

    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing the transition probabilities and rewards.
    """
    # Define the state space and action space
    n_states = (max_count + 1) ** n_item_types
    n_actions = 2 * n_item_types  # Actions: Add or remove each item type

    # Check if the state space size exceeds the limit
    if n_states < 1000:
        raise ValueError(
            "The state space size is less than 1000. Increase max_count or n_item_types."
        )

    # Initialize transition probabilities and rewards
    P = np.zeros((n_actions, n_states, n_states))
    R = np.zeros((n_actions, n_states))

    # Generate all possible states
    states = list(itertools.product(range(max_count + 1), repeat=n_item_types))

    # Define transitions and rewards
    for i, state in enumerate(states):
        state_number = np.ravel_multi_index(state, (max_count + 1,) * n_item_types)

        # Define transitions for each action
        for action in range(n_actions):
            new_state = list(state)
            item_type = action // 2
            adding = action % 2 == 0

            # Add or remove an item
            if adding and new_state[item_type] < max_count:
                new_state[item_type] += 1
            elif not adding and new_state[item_type] > 0:
                new_state[item_type] -= 1

            new_state_number = np.ravel_multi_index(
                new_state, (max_count + 1,) * n_item_types
            )
            P[action, state_number, new_state_number] = 1  # Deterministic transition

            # Simple reward: positive for adding items, negative for removing
            R[action, state_number] = 1 if adding else -1

    return (P, R)


def value_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float = 1e-3, threshold: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_actions, n_states = R.shape  # 3 actions, 1024 states

    V = np.zeros(n_states)  # Initialize value function for 1024 states

    historical_delta = []
    historical_value_function = []
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]

            # Update value function
            V[s] = max(
                [
                    sum(
                        [
                            P[a, s, s_prime] * (R[a, s_prime] + gamma * V[s_prime])
                            for s_prime in range(n_states)
                        ]
                    )
                    for a in range(n_actions)
                ]
            )
            delta = max(delta, abs(v - V[s]))

            historical_value_function.append(v)
            historical_delta.append(delta)

        if delta < threshold:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        # Determine the best action for each state
        policy[s] = np.argmax(
            [
                sum(
                    [
                        P[a, s, s_prime] * (R[a, s_prime] + gamma * V[s_prime])
                        for s_prime in range(n_states)
                    ]
                )
                for a in range(n_actions)
            ]
        )

    historical_delta_np = np.array(historical_delta)
    historical_value_function_np = np.array(historical_value_function)
    performance_metrics_np = np.vstack(
        (historical_delta_np, historical_value_function_np)
    )

    performance_metrics_df = pd.DataFrame(
        performance_metrics_np.T,
        columns=["Historical Delta", "Historical Value Function"],
    )
    performance_metrics_df = performance_metrics_df.reset_index(
        inplace=False, drop=False
    ).rename(columns={"index": "Iteration"})
    return (policy, V, performance_metrics_df)


def policy_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float = 0.99, threshold: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_actions, n_states = R.shape

    policy = np.random.randint(n_actions, size=n_states)  # Random initial policy
    V = np.zeros(n_states)  # Initialize value function

    historical_delta = []
    historical_value_function = []

    is_policy_stable = False
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]
                V[s] = sum(
                    [
                        P[policy[s], s, s_prime]
                        * (R[policy[s], s_prime] + gamma * V[s_prime])
                        for s_prime in range(n_states)
                    ]
                )
                delta = max(delta, abs(v - V[s]))

                historical_value_function.append(v)
                historical_delta.append(delta)

            if delta < threshold:
                break

        # Policy Improvement
        is_policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            policy[s] = np.argmax(
                [
                    sum(
                        [
                            P[a, s, s_prime] * (R[a, s_prime] + gamma * V[s_prime])
                            for s_prime in range(n_states)
                        ]
                    )
                    for a in range(n_actions)
                ]
            )

            if old_action != policy[s]:
                is_policy_stable = False

    historical_delta_np = np.array(historical_delta)
    historical_value_function_np = np.array(historical_value_function)
    performance_metrics_np = np.vstack(
        (historical_delta_np, historical_value_function_np)
    )

    performance_metrics_df = pd.DataFrame(
        performance_metrics_np.T,
        columns=["Historical Delta", "Historical Value Function"],
    )
    performance_metrics_df = performance_metrics_df.reset_index(
        inplace=False, drop=False
    ).rename(columns={"index": "Iteration"})

    return (policy, V, performance_metrics_df)


def output_value_iteration_performance_metrics_graph(
    df: pd.DataFrame,
    mdp: str,
    gamma: float,
    value_function_color: str = "green",
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
    value_function_color: str = "green",
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


if __name__ == "__main__":
    RUN_VALUE_ITERATION = False
    RUN_POLICY_ITERATION = True
    if RUN_VALUE_ITERATION:
        ###########################################
        ## Value Iteration
        ###########################################
        # Simple Weather Model MDP
        mdp = "simple_weather_model"
        (P, R) = simple_weather_model_mdp()
        gamma = 1e-3
        (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
        output_value_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        gamma = 0.9
        (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
        output_value_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        gamma = 0.99
        (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
        output_value_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        # Vending Machine MDP
        mdp = "vending_machine"
        (P, R) = vending_machine_mdp()
        gamma = 1e-3
        (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
        output_value_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        gamma = 0.9
        (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
        output_value_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

    if RUN_POLICY_ITERATION:
        ###########################################
        ## Policy Iteration
        ###########################################
        # Simple Weather Model MDP
        mdp = "simple_weather_model"
        (P, R) = simple_weather_model_mdp()
        gamma = 1e-3
        (policy, V, performance_metrics_df) = policy_iteration(P, R, gamma=gamma)
        output_policy_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        gamma = 0.9
        (policy, V, performance_metrics_df) = policy_iteration(P, R, gamma=gamma)
        output_policy_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        gamma = 0.99
        (policy, V, performance_metrics_df) = policy_iteration(P, R, gamma=gamma)
        output_policy_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        # Vending Machine MDP
        mdp = "vending_machine"
        (P, R) = vending_machine_mdp()
        gamma = 1e-3
        (policy, V, performance_metrics_df) = policy_iteration(P, R, gamma=gamma)
        output_policy_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )

        gamma = 0.9
        (policy, V, performance_metrics_df) = policy_iteration(P, R, gamma=gamma)
        output_policy_iteration_performance_metrics_graph(
            df=performance_metrics_df, mdp=mdp, gamma=gamma
        )
