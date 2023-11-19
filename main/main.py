import numpy as np
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


if __name__ == "__main__":
    (P, R) = library_book_management_mdp()
    print(P.shape)
    print(R.shape)
