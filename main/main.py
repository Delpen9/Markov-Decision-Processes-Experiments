import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import seaborn as sns

import itertools

from MDP.environments import (
    simple_weather_model_mdp,
    SimpleWeatherEnv,
    vending_machine_mdp,
    VendingMachineEnv,
)

from Models.models import (
    value_iteration,
    policy_iteration,
)

from Models.DQNAgent import (
    DQNAgent,
)


def train_DQNAgent(
    episodes: int = 1000,
    batch_size: int = 8,
    environment: callable = SimpleWeatherEnv,
    max_steps: int = 5,
) -> tuple[DQNAgent, list[float], list[float], list[float]]:
    env = environment()
    state_size = env.n_states
    action_size = env.n_actions
    agent = DQNAgent(state_size, action_size)

    # Initialize lists to store metrics
    total_rewards = []
    steps_per_episode = []
    epsilon_values = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        agent.update_target_model()

        # Append metrics after each episode
        total_rewards.append(total_reward)
        print(total_rewards)
        steps_per_episode.append(time + 1)
        epsilon_values.append(agent.epsilon)

        # Print the episode summary
        print(
            f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Steps: {time + 1}, Epsilon: {agent.epsilon}"
        )

    # Return the agent and the recorded metrics
    return (agent, total_rewards, steps_per_episode, epsilon_values)


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
    RUN_POLICY_ITERATION = False
    RUN_DQN_AGENT = True
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

    if RUN_DQN_AGENT:
        # Simple Weather Model MDP
        episodes = 1000
        batch_size = 8
        environment = SimpleWeatherEnv
        max_steps = 5
        (
            trained_agent,
            total_rewards,
            steps_per_episode,
            epsilon_values,
        ) = train_DQNAgent(
            episodes=episodes,
            batch_size=batch_size,
            environment=environment,
            max_steps=max_steps,
        )

        # Vending Machine MDP
        # episodes = 1000
        # batch_size = 8
        # environment = VendingMachineEnv
        # max_steps = 5
        # (
        #     trained_agent,
        #     total_rewards,
        #     steps_per_episode,
        #     epsilon_values,
        # ) = train_DQNAgent(
        #     episodes=episodes,
        #     batch_size=batch_size,
        #     environment=environment,
        #     max_steps=max_steps,
        # )
