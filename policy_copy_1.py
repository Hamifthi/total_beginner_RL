import numpy as np
import pprint
import sys
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

#policy evaluation function from Denny Britz.
def policy_eval(policy, env, discount_factor=1.0, theta=.00001):
    """
    Evaluate a policy in an environment given a full description of the environment's dynamics.

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

    #Start with a random (all 0) value function
    V = np.zeros(env.nS)
    print("This is the range of spaces: " + (str(range(env.nS))))
    print("V initial: " + (str(V)) + "\n")
    round = 0
    while True:
        print("Begin round " + str(round) + "\n")
        delta = 0

        # For each state, perform a "full backup"
        print("Now we will evaluate 'v' using the Bellman Expectation Equation")
        print("by looking in each direction and inputing the probability,")
        print("the value of the next state, and the reward in to our equation.\n")
        for s in range(env.nS):
            print("state: " + str(s) + ", round: " + str(round) + "\n")
            v = 0
            #Look at possible next actions
            for a, action_prob in enumerate(policy[s]):
                #for each action, look at the possible next states
                #the bellman equation
                for prob, next_state, reward, done in env.P[s][a]:
                    print("action: " + str(a + 1))
                    if a == 0:
                        print("direction up")
                        print("prob: " + str(prob))
                        print("action_prob: " + str(action_prob))
                        print("next_state: " + str(next_state))
                        print("reward: " + str(reward))
                    elif a == 1:
                        print("direction right")
                        print("prob: " + str(prob))
                        print("action_prob: " + str(action_prob))
                        print("next_state: " + str(next_state))
                        print("reward: " + str(reward))

                    elif a == 2:
                        print("direction down")
                        print("prob: " + str(prob))
                        print("action_prob: " + str(action_prob))
                        print("next_state: " + str(next_state))
                        print("reward: " + str(reward))
                    else:
                        print("direction left")
                        print("prob: " + str(prob))
                        print("action_prob: " + str(action_prob))
                        print("next_state: " + str(next_state))
                        print("reward: " + str(reward))
                    #calculate the expected value. Ref: Sutton book eq. 4.6.
                    print("v for this state in this round: " + str(v))
                    print("V[next_state value]: " + str(V[next_state]))
                    print("calculation: v += action_prob * prob * (reward + discount_factor * V[next_state])")
                    print("calculation: v += " + str(action_prob) + " * " + str(prob) + " * " + " (" + str(reward) + " + " + str(discount_factor) + " * " + str(V[next_state]) + ")")

                    # I don't think we need "prob" here, just to simplify
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
                    #v += action_prob * (reward + discount_factor * V[next_state])

                    print("v after calculation in this round: " + str(v) + "\n")
            #How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v


            print("delta: " + str(delta))
            print("V[s] where state = " + str(s) + ": " + str(V[s]) + "\n")

        #stop evaluating once our value functin change is below a threshold

        print("array of values V after round " + str(round) + " : " + str(V))
        print("\n")
        round += 1
        #if delta < theta:
        if round == 2:
            break

    return np.array(V)


if __name__ == "__main__":
    env = GridworldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    print("This is the random policy at work in GridWorld.")
    print("In other words, for each of the")
    print("possible 16 states there is a 25 percent chance the agent")
    print("will move in 1 of 4 directions; Either up (0), right (1), down (2)")
    print("or left (3). (If you go into the envs/gridworld.py file, you will see that")
    print("that the directions are associated with these index numbers.)")
    print("This is shown below as an array of possible actions for each state")
    print("and their probability of occuring: \n")
    print(str(random_policy))

    print("\n In theory we would run the policy evaluation algorithm")
    print("for as many rounds 'k' as possible until we get convergence.")
    print("However, since this generates a lot of data, we're only going")
    print("to execute 2 'rounds' of calculation, which we've done artificially")
    print("by telling the algorithm to stop after 2 rounds...\n")

    v = policy_eval(random_policy, env)

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    print("Insight: Notice that the way this algorithm works, as we go around")
    print("the second time, the algorithm takes the most up-to-date values into")
    print("account. For example, in state 10, when evaluating it's V(s), we take")
    print("state 6's V(s) from the current round 1, whereas we take state 11's")
    print("V(s) from round 0, since we have yet to calculate it. This helped")
    print("me gain more inisight into how the algorithm works and also made me")
    print("question how efficient it is. I guess as our values converge the differences")
    print("between previous state and this state become minimal.")

    expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    #np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
