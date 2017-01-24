#!/usr/bin/env python

import numpy as np
import argparse

GAMMA = 0.9  # discount factor

STATES = [  # set of all states
     0,  1,  2,  3,  4,
     5,  6,  7,  8,  9,
    10, 11, 12, 13, 14,
    15, 16, 17, 18, 19,
    20, 21, 22, 23, 24
]
TERM_STATES = [6, 8, 12, 16, 18]  # terminal states

ACTIONS = [  # set of all actions
    0,  # 0 = move RIGHT
    1,  # 1 = move LEFT
    2,  # 2 = move UP
    3   # 3 = move DOWN
]

N_STATES = len(STATES)
N_ACTIONS = len(ACTIONS)

P = None  # transition probabilities: <state, action, state> -> 0/1
R = None  # reward function: <state, action, state> -> 0/1
POLICY = None  # policy: <state> -> <action>
V = None  # <state> -> #

def init_global_vars():
    """
    Initialize global variables:
    Transition probabilities (P : <state,action,state> -> 0/1)
    Reward function (R : <state,action,state> -> 0/-10/100)
    Deterministic policy (POLICY : <state> -> <action>)
    Value-function (V : <state> -> #)
    """
    # Deterministic environment: P(s, a, s') = 0/1
    global P
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # transition probability (default is 0 for all states and all actions)
    for s in STATES:
        if s in TERM_STATES:
            # P[s,:,s] = 1  # whatever the action, stay in the terminal state.
            P[s,:,:] = 0  # no transition
        else :
            for a in ACTIONS:
                if a == 0 and s not in [4,9,14,19,24]:
                    # if action=RIGHT and state is not on the right-most border: valid move!
                    P[s,a,s+1] = 1.0
                elif a == 1 and s not in [0,5,10,15,20]:
                    # if action=LEFT and state is not on the left-most border: valid move!
                    P[s,a,s-1] = 1.0
                elif a == 2 and s not in [0,1,2,3,4]:
                    # if action=UP and state is not on the up-most border: valid move!
                    P[s,a,s-5] = 1.0
                elif a == 3 and s not in [20,21,22,23,24]:
                    # if action=DOWN and state is not on the bottom-most border: valid move!
                    P[s,a,s+5] = 1.0

    # Rewards only at terminal states:
    global R
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # rewards (default is zero for all states)
    R[5,0,6] = R[7,1,6] = R[11,2,6] = R[1,3,6] = -10
    R[7,0,8] = R[9,1,8] = R[13,2,8] = R[3,3,8] = -10
    R[11,0,12] = R[13,1,12] = R[17,2,12] = R[7,3,12] = 100
    R[15,0,16] = R[17,1,16] = R[21,2,16] = R[11,3,16] = -10
    R[17,0,18] = R[19,1,18] = R[23,2,18] = R[13,3,18] = -10

    # initialize policy and value arbitrarily
    global POLICY
    POLICY = np.zeros(N_STATES, dtype=np.int)  # policy = move right
    global V
    V = np.zeros(N_STATES)


def iterative_policy_eval(epsilon=0.1):
    """
    Policy Evaluation step.
    Iterate through all states to update value function V(s) until the update becomes < epsilon.
    :param epsilon: small positive number to tell when to stop iteration.
    """
    delta = 0
    for s in STATES:
        v = V[s]  # old state-value
        V[s] = sum([P[s,POLICY[s],s1] * (R[s,POLICY[s],s1] + GAMMA*V[s1]) for s1 in STATES])
        delta = max(delta, abs(v-V[s]))
    if delta >= epsilon:
        iterative_policy_eval(epsilon)


def policy_improvement():
    """
    Policy Improvement step.
    Check if taking any other action yields in a better value.
    If so, change the policy, and return false.
    :return: true only if the policy wasn't changed for all sates.
    """
    policy_stable = True
    for s in STATES:
        current_v = sum([P[s,POLICY[s],s1] * (R[s,POLICY[s],s1] + GAMMA*V[s1]) for s1 in STATES])
        # Taking best action with respect to current value function V:
        for a in ACTIONS:
            temp = sum([P[s,a,s1] * (R[s,a,s1] + GAMMA*V[s1]) for s1 in STATES])
            if temp > current_v:
                POLICY[s] = a  # update policy
                current_v = temp
                policy_stable = False
    return policy_stable


def make_greedy_policy():
    """
    Just like policy_improvement, make policy greedy with respect to V.
    Only difference: this is the ONLY policy update we make since we
    assume that V ~ V*
    """
    policy_improvement()  # make policy greedy with respect to V~V*


def value_iteration(epsilon=0.1, i=1):
    """
    Just like iterative_policy_eval(), but take the max over all ACTIONS for V(s).
    :param epsilon: small positive number to tell when to stop iteration.
    :param i: iteration index
    """
    print "Iteration:", i
    print "V:", V
    delta = 0
    for s in STATES:
        v = V[s]  # old state-value
        # Taking best action with respect to current value function V:
        for a in ACTIONS:
            temp = sum([P[s,a,s1] * (R[s,a,s1] + GAMMA*V[s1]) for s1 in STATES])
            if temp > V[s]:
                V[s] = temp  # update value function
        delta = max(delta, abs(v-V[s]))
    if delta >= epsilon:
        value_iteration(epsilon, i+1)


def main():
    parser = argparse.ArgumentParser(description='MDP Dynamic Programming.')
    parser.add_argument(
        'method',
        choices=["policy_iteration", "value_iteration", "prioritize_sweeping"],
        help="The algorithm to solve a simple grid world MDP."
    )
    args = parser.parse_args()


    init_global_vars()
    # print "P:\n", P
    # print "R:\n", R
    # print "V:", V
    # print "POLICY:", POLICY

    if args.method == "policy_iteration":
        ###
        # POLICY ITERATION
        ###
        iteration = 0
        policy_stable = False
        while not policy_stable:
            iteration += 1
            iterative_policy_eval()
            policy_stable = policy_improvement()
            print "Iteration:", iteration
            print "V:", V
            print "policy:", POLICY

    elif args.method == "value_iteration":
        ###
        # VALUE ITERATION
        ###
        value_iteration()  # compute optimal value function V*
        make_greedy_policy()  # get optimal policy by being greedy with respect to V*
        print "policy:", POLICY

    elif args.method == "prioritize_sweeping":
        ###
        # prioritize_sweeping
        ###
        # TODO
        print "TODO"


if __name__ == '__main__':
    main()

