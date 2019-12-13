import gym
import math
import numpy as np
from model import create_generation
from mutation import mutate


def experience_env(model, env, episodes, render):

    total_score = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            if render:
                env.render()

            obs_reshaped = obs.reshape((1, 4))
            action = model.predict(obs_reshaped)

            #print(action[0][0])
            action = int(round(action[0][0]))

            obs, reward, done, info = env.step(action)
            score += reward

        total_score += score

    if render:
        print(total_score)

    return total_score


def generation_experience_env(generation, env):
    output = []

    for agent in generation:
        score = experience_env(agent, env, 1, False)
        output.append((agent, score))

    output.sort(key=lambda tup: tup[1], reverse=True)

    return output


def environmental_pressure(experienced_gen, threshold=0.1):
    total_population = len(experienced_gen)
    n_survived = round(total_population * threshold)

    assert n_survived > 0, "No population survived"

    return list(map(lambda x: x[0], experienced_gen[:n_survived]))


def find_optimal_agent(epochs, n_pop):
    env = gym.make('CartPole-v1')
    generation = create_generation(n_pop)

    for _ in range(epochs):
        experienced_gen = generation_experience_env(generation, env)
        survived_population = environmental_pressure(experienced_gen)
        population_multiplier = int(round(len(generation)/len(survived_population)))

        new_generation = []

        for _ in range(1, population_multiplier):
            new_generation.extend(survived_population[:])

        severity = max(1, math.floor(epochs/10))

        for agent in new_generation:
            mutate(agent, severity)

        new_generation.extend(survived_population[:])

        generation = new_generation

        experience_env(survived_population[0], env, 1, True)


find_optimal_agent(30, 30)
