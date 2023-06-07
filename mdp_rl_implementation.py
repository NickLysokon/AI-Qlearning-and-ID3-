from copy import deepcopy
import random
from typing import List
import copy
from typing import List, Tuple, Dict

import numpy as np
import termcolor


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    U_new = copy.deepcopy(U_init)

    while True:

        U_init = copy.deepcopy(U_new)

        # Compute the new value estimates

        for i in range(mdp.num_row):
            for c in range(mdp.num_col):
                max_list = []

                s = tuple([i, c])
                if mdp.board[i][c] == 'WALL':
                    continue

                if s in mdp.terminal_states:
                    U_new[i][c] = float(mdp.board[i][c])
                    continue

                for a in mdp.actions:
                    sigma = float(mdp.board[i][c])
                    p_list = mdp.transition_function[a]

                    if list(mdp.step(s, 'UP')) != s:
                        sigma += mdp.gamma * float(p_list[0]) * float(
                            U_new[list(mdp.step(s, 'UP'))[0]][list(mdp.step(s, 'UP'))[1]])

                    if list(mdp.step(s, 'DOWN')) != s:
                        sigma += mdp.gamma * float(p_list[1]) * \
                                 float(U_new[list(mdp.step(s, 'DOWN'))[0]][list(mdp.step(s, 'DOWN'))[1]])

                    if list(mdp.step(s, 'RIGHT')) != s:
                        sigma += mdp.gamma * float(
                            p_list[2]) * \
                                 float(U_new[list(mdp.step(s, 'RIGHT'))[0]][list(mdp.step(s, 'RIGHT'))[1]])

                    if list(mdp.step(s, 'LEFT')) != s:
                        sigma += mdp.gamma * float(
                            p_list[3]) * \
                                 float(U_new[list(mdp.step(s, 'LEFT'))[0]][list(mdp.step(s, 'LEFT'))[1]])

                    max_list.append(sigma)

                U_new[i][c] = copy.deepcopy(max(max_list))

        for i in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[i][c] == 'WALL':
                    continue

                if abs(U_new[i][c] - U_init[i][c]) < epsilon * mdp.gamma * (1 - mdp.gamma) and tuple(
                        [i, c]) not in mdp.terminal_states:
                    return U_new


def get_policy(mdp, U):
    policy = copy.deepcopy(U)

    for i in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[i][c] == 'WALL':
                continue
            max_grade = []

            s = tuple([i, c])

            if (mdp.step(s, 'UP')) != s and mdp.board[mdp.step(s, 'UP')[0]][mdp.step(s, 'UP')[1]] != 'WALL':
                max_grade.append(float(U[list(mdp.step(s, 'UP'))[0]][list(mdp.step(s, 'UP'))[1]]))

            if (mdp.step(s, 'DOWN')) != s and mdp.board[mdp.step(s, 'DOWN')[0]][mdp.step(s, 'DOWN')[1]] != 'WALL':
                max_grade.append(float(U[list(mdp.step(s, 'DOWN'))[0]][list(mdp.step(s, 'DOWN'))[1]]))

            if (mdp.step(s, 'RIGHT')) != s and mdp.board[mdp.step(s, 'RIGHT')[0]][mdp.step(s, 'RIGHT')[1]] != 'WALL':
                max_grade.append(float(U[list(mdp.step(s, 'RIGHT'))[0]][list(mdp.step(s, 'RIGHT'))[1]]))

            if (mdp.step(s, 'LEFT')) != s and mdp.board[mdp.step(s, 'LEFT')[0]][mdp.step(s, 'LEFT')[1]] != 'WALL':
                max_grade.append(float(U[list(mdp.step(s, 'LEFT'))[0]][list(mdp.step(s, 'LEFT'))[1]]))

            max_val = max(max_grade)

            if float(U[list(mdp.step(s, 'UP'))[0]][list(mdp.step(s, 'UP'))[1]]) == max_val and (mdp.step(s, 'UP')) != s:
                policy[i][c] = 'UP'
                continue
            if float(U[list(mdp.step(s, 'DOWN'))[0]][list(mdp.step(s, 'DOWN'))[1]]) == max_val and (
                    mdp.step(s, 'DOWN')) != s:
                policy[i][c] = 'DOWN'
                continue
            if float(U[list(mdp.step(s, 'RIGHT'))[0]][list(mdp.step(s, 'RIGHT'))[1]]) == max_val and (
                    mdp.step(s, 'RIGHT')) != s:
                policy[i][c] = 'RIGHT'
                continue
            if float(U[list(mdp.step(s, 'LEFT'))[0]][list(mdp.step(s, 'LEFT'))[1]]) == max_val and (
                    mdp.step(s, 'LEFT')) != s:
                policy[i][c] = 'LEFT'
                continue

    return policy


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
               max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # Set the discount factor (gamma) and the learning rate (alpha)
    gamma = decay_rate
    alpha = learning_rate

    # Set the initial values for the Q-table
    Q = {}
    for i in range(mdp.num_row):
        for c in range(mdp.num_col):
            s = tuple([i, c])

            Q[s] = {}
            Q[s]['UP'] = 0
            Q[s]['DOWN'] = 0
            Q[s]['RIGHT'] = 0
            Q[s]['LEFT'] = 0

    # Set the number of iterations and the rewards
    num_iterations = max_steps

    curr_episode = 0

    while curr_episode <= total_episodes:
        state = init_state
        for i in range(num_iterations):

            if state in mdp.terminal_states:
                break

            # Choose an action based on an epsilon-greedy policy
            if np.random.uniform(0, 1) > epsilon:
                local_max = max(Q[state].values())
                if local_max == Q[state]['UP']:
                    action = 'UP'
                if local_max == Q[state]['DOWN']:
                    action = 'DOWN'
                if local_max == Q[state]['RIGHT']:
                    action = 'RIGHT'
                if local_max == Q[state]['LEFT']:
                    action = 'LEFT'





            else:

                action = list(random.choice(list(mdp.actions.items())))[0]

            next_state = mdp.step(state, action)
            if next_state in mdp.terminal_states:

                if float(mdp.board[next_state[0]][next_state[1]]) == 1:
                    Q[next_state]['UP'] = 100000
                    Q[next_state]['DOWN'] = 100000
                    Q[next_state]['RIGHT'] = 100000
                    Q[next_state]['LEFT'] = 100000
                if float(mdp.board[next_state[0]][next_state[1]]) == -1:
                    Q[next_state]['UP'] = 0
                    Q[next_state]['DOWN'] = 0
                    Q[next_state]['RIGHT'] = 0
                    Q[next_state]['LEFT'] = 0

            reward = float(mdp.board[state[0]][state[1]])

            local_max = max(Q[next_state].values())
            if local_max == Q[state]['UP']:
                best_action = 'UP'
            if local_max == Q[state]['DOWN']:
                best_action = 'DOWN'
            if local_max == Q[state]['RIGHT']:
                best_action = 'RIGHT'
            if local_max == Q[state]['LEFT']:
                best_action = 'LEFT'
            Q[state][action] = Q[state][action] + alpha * (
                    reward + gamma * Q[next_state][best_action] - Q[state][action])
            # Set the current state to the next state
            state = next_state

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-gamma * curr_episode)

        curr_episode += 1

    return Q


def q_table_policy_extraction(mdp, qtable):
    policy = [['ACT' for x in range(mdp.num_col)] for y in range(mdp.num_row)]
    for i in range(mdp.num_row):
        for c in range(mdp.num_col):

            state = (i, c)

            local_max = max(qtable[state].values())
            if local_max == qtable[state]['UP']:
                action = 'UP'
            if local_max == qtable[state]['DOWN']:
                action = 'DOWN'
            if local_max == qtable[state]['RIGHT']:
                action = 'RIGHT'
            if local_max == qtable[state]['LEFT']:
                action = 'LEFT'

            policy[i][c] = action

    return policy


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
