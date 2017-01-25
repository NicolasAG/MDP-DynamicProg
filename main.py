#!/usr/bin/env python

import numpy as np
import argparse
from datetime import datetime as dt

GAMMA = 0.  # discount factor: to be set

S_WIDTH = 0  # to be set
STATES = []  # 2D square of border S_WIDTH

TERM_STATES = []  # terminal states: 4 corners (<0) + center (>0)

ACTIONS = [  # set of all actions
    0,  # 0 = move RIGHT
    1,  # 1 = move LEFT
    2,  # 2 = move UP
    3   # 3 = move DOWN
]

P = None  # transition probabilities: <state, action, state> -> 0/1
R = None  # reward function: <state, action, state> -> 0/1
POLICY = None  # policy: <state> -> <action>
V = None  # <state> -> #

def init_global_vars(gamma=0.9, s_width=5):
    """
    Initialize global variables:
    :GAMMA: discount factor for iterative algorithms.
    :S_WIDTH: number of states per border (try with 5, 11, 21, 31, 41)
    :STATES: array representing a 2D square grid world (try with 25, 121, 441, 961, 1681).
    :TERM_STATES: list of 5 terminal states (4 corners are <0 and center is >0).
    :P: Transition probabilities <state,action,state> -> 0/1
    :R: Reward function <state,action,state> -> 0/-10/100
    :POLICY: Deterministic policy <state> -> <action>
    :V: Value-function <state> -> #
    """
    global GAMMA  # discount factor
    GAMMA = gamma
    global S_WIDTH  # size of the world
    S_WIDTH = s_width
    global STATES  # 2D square represented in a 1D array
    STATES = range(S_WIDTH**2)
    global TERM_STATES  # terminal states
    TERM_STATES = [
        S_WIDTH+1,  # top left corner
        2*(S_WIDTH-1),  # top right corner
        (S_WIDTH/2) * (S_WIDTH+1),  # center cell
        S_WIDTH * (S_WIDTH-2) + 1,  # bottom left corner
        S_WIDTH * (S_WIDTH-1) - 2  # bottom right cell
    ]

    r_border = range(S_WIDTH-1, S_WIDTH**2, S_WIDTH)  # states on the right border
    l_border = range(0, S_WIDTH**2, S_WIDTH)  # states on the left border
    t_border = range(S_WIDTH)  # states on the top border
    b_border = range(S_WIDTH*(S_WIDTH-1), S_WIDTH**2)  # states at the bottom border

    # Deterministic environment: P(s, a, s') = 0/1
    global P
    P = np.zeros((len(STATES), len(ACTIONS), len(STATES)))  # transition probability (default is 0 for all states and all actions)
    for s in STATES:
        if s in TERM_STATES:
            P[s,:,:] = 0  # no transitions allowed in terminal states.
        else :
            for a in ACTIONS:
                if a == 0 and s not in r_border:
                    # if action=RIGHT and state is not on the right-most border: valid move!
                    P[s,a,s+1] = 1.0
                elif a == 1 and s not in l_border:
                    # if action=LEFT and state is not on the left-most border: valid move!
                    P[s,a,s-1] = 1.0
                elif a == 2 and s not in t_border:
                    # if action=UP and state is not on the up-most border: valid move!
                    P[s,a,s-S_WIDTH] = 1.0
                elif a == 3 and s not in b_border:
                    # if action=DOWN and state is not on the bottom-most border: valid move!
                    P[s,a,s+S_WIDTH] = 1.0

    # Rewards only at terminal states:
    global R
    R = np.zeros((len(STATES), len(ACTIONS), len(STATES)))  # rewards (default is zero for all states)
    tl = TERM_STATES[0]  # top left corner
    tr = TERM_STATES[1]  # top right corner
    c = TERM_STATES[2]  # center
    bl = TERM_STATES[3]  # bottom left corner
    br = TERM_STATES[4]  # bottom right corner
    R[tl-1, 0, tl] = R[tl+1, 1, tl] = R[tl+S_WIDTH, 2, tl] = R[tl-S_WIDTH, 3, tl] = -10
    R[tr-1, 0, tr] = R[tr+1, 1, tr] = R[tr+S_WIDTH, 2, tr] = R[tr-S_WIDTH, 3, tr] = -10
    R[c-1, 0, c] = R[c+1, 1, c] = R[c+S_WIDTH, 2, c] = R[c-S_WIDTH, 3, c] = 100
    R[bl-1, 0, bl] = R[bl+1, 1, bl] = R[bl+S_WIDTH, 2, bl] = R[bl-S_WIDTH, 3, bl] = -10
    R[br-1, 0, br] = R[br+1, 1, br] = R[br+S_WIDTH, 2, br] = R[br-S_WIDTH, 3, br] = -10

    # initialize policy and value arbitrarily
    global POLICY
    POLICY = np.zeros(len(STATES), dtype=np.int)  # policy = move right
    global V
    V = np.zeros(len(STATES))


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
    def restricted_float(x):  # Custom type for argparse argument --gamma
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x

    parser = argparse.ArgumentParser(description='MDP Dynamic Programming.')
    parser.add_argument(
        'method',
        choices=["policy_iteration", "value_iteration", "prioritize_sweeping"],
        help="The algorithm to solve a simple grid world MDP."
    )
    parser.add_argument(
        '--width', type=int, default=5, choices=range(5,42,2),  # min 5x5 , max 41x41 square
        help="The width of the 2D square grid world."
    )
    parser.add_argument(
        '--gamma', type=restricted_float, default=0.9,
        help="Discount factor for iterative algorithms."
    )
    args = parser.parse_args()


    init_global_vars(args.gamma, args.width)

    start = dt.now()
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

    print "took:", dt.now() - start


if __name__ == '__main__':
    main()

