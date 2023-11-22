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


def library_book_management_mdp(
    n_books: int = 10, return_prob: float = 0.1, buy_influence: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    n_states = 2 ** n_books  # Each book can be either in or out
    n_actions = 3  # Buy, Sell, Do nothing

    # Initialize transition probabilities and rewards
    P = np.zeros((n_actions, n_states, n_states))
    R = np.zeros((n_actions, n_states))

    # Generate all possible states
    states = list(itertools.product([0, 1], repeat=n_books))

    for i, state in enumerate(states):
        # State representation as a binary number
        state_number = sum([bit * (2 ** idx) for idx, bit in enumerate(state)])

        # Action 0: Buy a new book
        for j, next_state in enumerate(states):
            next_state_number = sum(
                [bit * (2 ** idx) for idx, bit in enumerate(next_state)]
            )
            if next_state_number == state_number:
                P[0, state_number, next_state_number] = 1 - buy_influence
            else:
                # Assume buying a new book slightly increases the probability of a random change
                P[0, state_number, next_state_number] = buy_influence / (n_states - 1)
        R[0, state_number] = -2  # Cost of buying a new book, adjusted

        # Action 1: Sell a book
        # Model which book is sold and how it affects the state
        if sum(state) > 0:  # Can only sell if there's at least one book in the library
            for idx, book_state in enumerate(state):
                if book_state == 1:  # Book is in the library and can be sold
                    new_state = list(state)
                    new_state[idx] = 0  # Remove this book
                    new_state_number = sum(
                        [bit * (2 ** k) for k, bit in enumerate(new_state)]
                    )
                    P[1, state_number, new_state_number] = 1 / sum(
                        state
                    )  # Equal probability for each book that's in
                    R[1, state_number] = 5  # Gain from selling a book, adjusted
        else:
            P[
                1, state_number, state_number
            ] = 1  # Stay in the same state if no book to sell

        # Action 2: Do nothing
        for j, next_state in enumerate(states):
            # Calculate probability of each book being returned
            prob = 1
            for idx, (current, next) in enumerate(zip(state, next_state)):
                if current == 0 and next == 1:  # Book returned
                    prob *= return_prob
                elif current == 1 and next == 0:  # Book remains out
                    prob *= 1 - return_prob
                elif current == next:  # No change
                    prob *= 1
                else:  # Book cannot be taken out if it's already in
                    prob = 0
                    break

            next_state_number = sum(
                [bit * (2 ** idx) for idx, bit in enumerate(next_state)]
            )
            P[2, state_number, next_state_number] = prob

        # Complex reward structure based on the number of books in and additional factors
        R[2, state_number] = 10 * sum(state) - 5 * len(
            [1 for x in state if x == 0]
        )  # More reward for books in, penalty for books out

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


def output_value_iteration_performance_metrics_graph(
    df: pd.DataFrame,
    mdp: str,
    gamma: float,
    delta_color: str = "blue",
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
        color=delta_color,
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
    # Simple Weather Model MDP
    mdp = "simple_weather_model"
    (P, R) = simple_weather_model_mdp()
    # (P, R) = library_book_management_mdp()
    gamma = 1e-3
    (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
    output_value_iteration_performance_metrics_graph(df=performance_metrics_df, mdp=mdp, gamma=gamma)

    gamma = 0.9
    (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
    output_value_iteration_performance_metrics_graph(df=performance_metrics_df, mdp=mdp, gamma=gamma)

    gamma = 0.99
    (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
    output_value_iteration_performance_metrics_graph(df=performance_metrics_df, mdp=mdp, gamma=gamma)

    # Library Book Management MDP
    mdp = "library_book_management"
    (P, R) = library_book_management_mdp()
    gamma = 1e-3
    (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
    output_value_iteration_performance_metrics_graph(df=performance_metrics_df, mdp=mdp, gamma=gamma)

    gamma = 0.9
    (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
    output_value_iteration_performance_metrics_graph(df=performance_metrics_df, mdp=mdp, gamma=gamma)

    gamma = 0.99
    (policy, V, performance_metrics_df) = value_iteration(P, R, gamma=gamma)
    output_value_iteration_performance_metrics_graph(df=performance_metrics_df, mdp=mdp, gamma=gamma)