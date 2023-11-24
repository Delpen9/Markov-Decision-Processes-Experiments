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

from Graphing.graphing import (
    output_value_iteration_performance_metrics_graph,
    output_policy_iteration_performance_metrics_graph,
    plot_train_DQN_performance,
)


def train_DQNAgent(
    episodes: int = 1000,
    batch_size: int = 8,
    environment: callable = SimpleWeatherEnv,
    max_steps: int = 5,
) -> tuple[DQNAgent, pd.DataFrame]:
    env = environment()
    state_size = env.n_states
    action_size = env.n_actions
    agent = DQNAgent(state_size, action_size)

    # Initialize lists to store metrics
    total_rewards = []
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
        epsilon_values.append(agent.epsilon)

        total_rewards_np = np.array(total_rewards)
        epsilon_values_np = np.array(epsilon_values)
        performance_np = np.vstack((total_rewards_np, epsilon_values_np)).T

        performance_df = pd.DataFrame(
            performance_np, columns=["Total Rewards", "Epsilon Value"]
        )
        performance_df = performance_df.reset_index(drop=False).rename(
            columns={
                "index": "Episode",
            }
        )
        performance_df["10-Episode Rolling Avg Rewards"] = (
            performance_df["Total Rewards"].rolling(window=10).mean()
        )
        performance_df["30-Episode Rolling Avg Rewards"] = (
            performance_df["Total Rewards"].rolling(window=30).mean()
        )

        # Print the episode summary
        print(
            f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Steps: {time + 1}, Epsilon: {agent.epsilon}"
        )

    # Return the agent and the recorded metrics
    return (agent, performance_df)


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
        ###########################################
        ## Random State Resets
        ###########################################
        # Simple Weather Model MDP
        episodes = 150
        batch_size = 8
        environment = SimpleWeatherEnv
        environment.reset_to_initial_state = False
        max_steps = 5
        (trained_agent, performance_df) = train_DQNAgent(
            episodes=episodes,
            batch_size=batch_size,
            environment=environment,
            max_steps=max_steps,
        )

        plot_train_DQN_performance(
            mdp="simple_weather_model",
            df=performance_df,
            random_resets=True,
        )

        # Vending Machine MDP
        episodes = 150
        batch_size = 8
        environment = VendingMachineEnv
        environment.reset_to_initial_state = False
        max_steps = 5
        (trained_agent, performance_df) = train_DQNAgent(
            episodes=episodes,
            batch_size=batch_size,
            environment=environment,
            max_steps=max_steps,
        )

        plot_train_DQN_performance(
            mdp="vending_machine",
            df=performance_df,
            random_resets=True,
        )

        ###########################################
        ## Initial State Resets
        ###########################################
        # Simple Weather Model MDP
        episodes = 150
        batch_size = 8
        environment = SimpleWeatherEnv
        environment.reset_to_initial_state = True
        max_steps = 5
        (trained_agent, performance_df) = train_DQNAgent(
            episodes=episodes,
            batch_size=batch_size,
            environment=environment,
            max_steps=max_steps,
        )

        plot_train_DQN_performance(
            mdp="simple_weather_model",
            df=performance_df,
            random_resets=False,
        )

        # Vending Machine MDP
        episodes = 150
        batch_size = 8
        environment = VendingMachineEnv
        environment.reset_to_initial_state = True
        max_steps = 5
        (trained_agent, performance_df) = train_DQNAgent(
            episodes=episodes,
            batch_size=batch_size,
            environment=environment,
            max_steps=max_steps,
        )

        plot_train_DQN_performance(
            mdp="vending_machine",
            df=performance_df,
            random_resets=False,
        )