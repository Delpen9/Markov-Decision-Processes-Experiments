import numpy as np
import pandas as pd

import itertools
import random


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


class SimpleWeatherEnv:
    def __init__(self):
        # Define the states and actions
        self.n_states = 2
        self.n_actions = 2

        self.states = list([0, 1])
        self.state = self._one_hot_encode_state(0)  # Initial state as one-hot

        self.reset_to_initial_state = False

        # Transition Probability Matrix [actions x states x states]
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))
        # Carrying an umbrella
        self.P[0, 0, 0] = 0.9  # Sunny today, sunny tomorrow
        self.P[0, 0, 1] = 0.1  # Sunny today, rainy tomorrow
        self.P[0, 1, 0] = 0.5  # Rainy today, sunny tomorrow
        self.P[0, 1, 1] = 0.5  # Rainy today, rainy tomorrow
        # Not carrying an umbrella
        self.P[1] = self.P[0]  # Same transition probabilities

        # Reward Matrix [actions x states]
        self.R = np.array(
            [
                # Rewards for carrying an umbrella
                [1, -1],
                # Rewards for not carrying an umbrella
                [0, -2],
            ]
        )

    def _one_hot_encode_state(self, state_index):
        # Create a one-hot encoded state
        state = np.zeros(self.n_states)
        state[state_index] = 1
        return state

    def step(self, action):
        # Use the transition probabilities to determine the next state
        probabilities = self.P[action, np.argmax(self.state)]
        next_state = np.random.choice([0, 1], p=probabilities)
        reward = self.R[action, np.argmax(self.state)]
        done = False  # Example condition to end an episode

        self.state = self._one_hot_encode_state(next_state)
        return self.state, reward, done

    def reset(self):
        if self.reset_to_initial_state:
            self.state = self._one_hot_encode_state(0)  # Initial state of 0
            return self.state
        else:
            self.state = self._one_hot_encode_state(np.random.choice(range(self.n_states)))  # Random initial state
            return self.state


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


class VendingMachineEnv:
    def __init__(self, max_count=10, n_item_types=3):
        # Initialize parameters
        self.max_count = max_count
        self.n_item_types = n_item_types
        self.n_states = (max_count + 1) ** n_item_types

        self.reset_to_initial_state = False

        self.n_actions = 2 * n_item_types

        # Check if the state space size exceeds the limit
        if self.n_states < 1000:
            raise ValueError(
                "The state space size is less than 1000. Increase max_count or n_item_types."
            )

        # Generate all possible states
        self.states = list(itertools.product(range(max_count + 1), repeat=n_item_types))
        self.state = self._one_hot_encode_state(0)  # Initial state index

        # Initialize transition probabilities and rewards
        self.P, self.R = self._initialize_matrices()

    def _one_hot_encode_state(self, state_index):
        # Create a one-hot encoded state
        state = np.zeros(self.n_states)
        state[state_index] = 1
        return state

    def _initialize_matrices(self):
        P = np.zeros((self.n_actions, self.n_states, self.n_states))
        R = np.zeros((self.n_actions, self.n_states))

        # Define transitions and rewards
        for i, state in enumerate(self.states):
            state_number = np.ravel_multi_index(
                state, (self.max_count + 1,) * self.n_item_types
            )

            for action in range(self.n_actions):
                new_state = list(state)
                item_type = action // 2
                adding = action % 2 == 0

                if adding and new_state[item_type] < self.max_count:
                    new_state[item_type] += 1
                elif not adding and new_state[item_type] > 0:
                    new_state[item_type] -= 1

                new_state_number = np.ravel_multi_index(
                    new_state, (self.max_count + 1,) * self.n_item_types
                )
                P[
                    action, state_number, new_state_number
                ] = 1  # Deterministic transition
                R[action, state_number] = 1 if adding else -1

        return P, R

    def step(self, action):
        current_state_tuple = self.states[np.argmax(self.state)]
        probabilities = self.P[action, np.argmax(self.state)]
        next_state = np.random.choice(range(self.n_states), p=probabilities)
        reward = self.R[action, np.argmax(self.state)]
        done = False  # You can define your own condition to end an episode

        self.state = self._one_hot_encode_state(next_state)
        return self.state, reward, done

    def reset(self):
        if self.reset_to_initial_state:
            self.state = self._one_hot_encode_state(0)  # Initial state of 0
            return self.state
        else:
            self.state = self._one_hot_encode_state(np.random.choice(range(self.n_states)))  # Random initial state
            return self.state