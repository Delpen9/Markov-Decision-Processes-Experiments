import numpy as np
import pandas as pd

import itertools

import matplotlib.pyplot as plt
import seaborn as sns

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

def one_hot_encode_policy_and_create_heatmap(
    policy : np.ndarray,
    n_actions : int,
    additional_details : str,
    model : str = "value_iteration",
    mdp : str = "simple_weather_model",
    output_filepath : str = "../outputs/reward_mappings/"
) -> None:
    output_filepath = fr"{output_filepath}{mdp}_{model}_policy_heatmap_{additional_details}.png"
    
    n_states = policy.shape[0]
    policy_matrix = np.zeros((n_states, n_actions))

    for s in range(n_states):
        action = policy[s]
        policy_matrix[s, action] = 1

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(policy_matrix.T, annot=True, cmap="viridis")
    plt.title("Policy Heatmap")
    plt.xlabel("State")
    plt.ylabel("Action")

    # Save the heatmap to a file
    plt.savefig(output_filepath)
    plt.close()