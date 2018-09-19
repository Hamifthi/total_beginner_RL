import numpy as np
import pprint
import sys
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta = 0.00001):
    """
    Evaluate a policy in an environment given a full description of the environment's dynamics

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """

    # 1.Start with a random (all 0) value Function
    V = np.zeros(env.ns)

    # 2. Run until convergence.
    while True:
        delta = 0

        # 3.For each state, perform a "full-backup" across all states
        for s in range(env.nS):
            # 3a. little v is the value of each individual state, set to 0 before
            # it's going to be calculated
            v = 0

            # 4. Look at possible next actions and their probabilities for each state

            # remember our policy here is a random_policy so thats:
            # random_policy = np.ones([env.nS, env.nA]) / env.nA
            # so policy[s] will be the policy for our individual state
            # and it will be the average of the number of actions over the number of spaces
            # so this action_prob will be 25%, or .25

            # in gridworld, there are 4 options, so a in enumerate(policy[s]) will be '4'
            # and action_prob in enumerate(policy[s]) will be '.25'
            for a, action_prob in enumerate(policy[s])

                # 5. For each action, look at the possible next states after that action
                # remember env.P[s][a] is a list of transition tuples, which we are searching all of

                #in gridworld, there will be 4 25% likely actions, each leading to 1 state
                # with a certain reward. So this next 'for loop' is utilizing the
                # Bellman Expectation Equation 4 times assigning values to our tuples
                # along the way
                for prob, next_state, reward, done in env.P[s][a]
                    # 6. Calculate the expected value
                    # using the Bellman Expectation Equation for Vπ, iteratively (using +=)
                    # for each individual state
                    v += action_prob * prob * (reward + discount_factor * V[next_state])

            # 6. How much our value function changed across that individual state
            # where v is newly calcualted function of all possible values and
            # V[s] is the original value of that particular state

            # this is important because the change is what is allowing for convergence
            # to occur. After the first evaluation of all states, the delta will be
            # large integer (1, or 1.25, or .78... w/e)
            # (need to figure this out exactly)
            # so obviously that's much higher than our threshold, so we go around again,
            # but as we see below, we will update our value of each state to be the
            # newly calculated value, so we will start the next go around with
            delta = max(delta, np.abs(v - V[s]))

            # 6a. Set our old V to our newly calcuated expected value
            # again, V[s] is keeping track of the last "go-round", or Vk value, so
            # that's how we calcualte delta, and now we'll update our V[s] to
            # our just-calculated "v" value. 
            V[s] = v

        # 7. Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break

    return np.array(V)
