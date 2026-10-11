"""
Finite-horizon dynamic programming + parameterized reward (goal knob) + Bayesian inference of the reward from behavior
"""

import math
import random

# ENVIRONMENT SETUP 
# hallway of 5 tiles: [0] [1] [2] [3] [4]
# 3 actions: LEFT, STAY or RIGHT

N_STATES = 5
states = list(range(N_STATES))

ACTIONS = [-1, 0, +1]  # 0 = STAY (new)
ACTION_NAMES = {-1: "LEFT", 0: "STAY", +1: "RIGHT"}

# movement is noisy (for STAY, the "intended" tile is the tile you are on, so the button can still nudge you 1 tile either way)
P_INTENDED = 0.9
P_NOISE_EACH = 0.05

def clamp(s):
    return max(0, min(N_STATES - 1, s))

def transition_probs(s, a):
    """
    environment: P(s' | s, a) as a dict {next_tile: probability}
    fixed "lookup table": it never depends on the policy or on the reward, it is just how the world responds to an action!
    """
    intended_next = clamp(s + a)
    neighbor_1 = clamp(s + a + 1)  # overshoot
    neighbor_2 = clamp(s + a - 1)  # undershoot

    probs = {}
    for s_prime in [intended_next, neighbor_1, neighbor_2]:
        probs[s_prime] = probs.get(s_prime, 0.0)
    # near the walls two outcomes can be the same tile, so we ADD the probabilities instead of overwriting!
    probs[intended_next] += P_INTENDED
    probs[neighbor_1] += P_NOISE_EACH
    probs[neighbor_2] += P_NOISE_EACH
    return probs

def sample_state(probs, rng):
    """Turn {tile: probability} into ONE random outcome (a weighted die roll)"""
    return rng.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]

# PARAMETERIZED REWARD:  r(s,a; goal)
# +10 if you are already ON the goal and your action keeps you there 
# NOTE: with STAY this is possible on ANY goal tile, before, only end tiles could earn it by pushing into the wall)
# +1   if the action moves you closer to the goal, -1 otherwise

BIG_REWARD = 10.0
def reward(s, a, goal):
    """immediate reward for taking action a in tile s, for a given goal tile"""
    s_next = clamp(s + a)  # where the action is *meant* to take you
    if s == goal and s_next == goal:
        return BIG_REWARD
    moves_closer = abs(s_next - goal) < abs(s - goal)
    return 1.0 if moves_closer else -1.0

def build_reward_table(goal):
    """R[s][a] for one value of the goal knob"""
    return {s: {a: reward(s, a, goal) for a in ACTIONS} for s in states}

# FINITE-HORIZON DYNAMIC PROGRAMMING!
# the game lasts T steps, label the decision points "stages" t = 0, 1, ...,
# V[t][s] = best total reward from tile s at stage t until the game ends
# Q[t][s][a] = ^^ if you commit to action a right now
# 
# start at the END, where nothing is left to earn: V[T][s] = 0
# then fill in one stage at a time, moving backward:
# Q[t][s][a] = R(s,a) + sum_{s'} P(s'|s,a) * V[t+1][s']
# V[t][s]    = max_a Q[t][s][a] (keep the best)    
#
# 1. goal gives you a reward table: +1 for stepping closer, +10 for staying on the goal
# 2. Bellman’s equation turns that reward table into Q values: 
#       for each tile, stage and action, how much total reward you expect from here if you play well (backward-fill table)
# 3. best action on each tile and stage is the one with the highest Q - that’s what the agent does
# 4. observer uses the same Q values in the softmax to ask “how likely was this move if the goal were g?”                  

HORIZON = 12  # T: number of steps in the game

def finite_horizon_dp(R, T):
    """Solve the T-step game for reward table R. Returns (V, Q)."""
    V = [None] * (T + 1)
    Q = [None] * T
    V[T] = {s: 0.0 for s in states}  # nothing left to earn after the last step

    for t in range(T - 1, -1, -1):  # T-1, T-2, ..., 0  (backward in time)
        Q[t] = {}
        V[t] = {}
        for s in states:
            Q[t][s] = {}
            for a in ACTIONS:
                probs = transition_probs(s, a)
                # weighted average of next-stage values over where the noisy move might actually land us
                expected_future = sum(p * V[t + 1][sp] for sp, p in probs.items())
                Q[t][s][a] = R[s][a] + expected_future
            V[t][s] = max(Q[t][s].values())
    return V, Q


def best_actions(q_for_state, tol=1e-9):
    """action(s) with the highest Q val"""
    top = max(q_for_state.values())
    return [a for a, q in q_for_state.items() if q >= top - tol]

# AI's addition for visualizing output!
def print_dp_tables(V, Q, T):
    print(f"Backward induction, T = {T} steps (read from the bottom up)")
    print("  stage  steps_left | " + "".join(f"  tile {s}  " for s in states))
    for t in range(T, -1, -1):
        row = "".join(f"{V[t][s]:8.2f} " for s in states)
        print(f"  {t:5d}  {T - t:10d} | {row}")
    print("\n  Best action at each stage (stage 0 = start of the game):")
    for t in range(T):
        names = []
        for s in states:
            acts = best_actions(Q[t][s])
            names.append("/".join(ACTION_NAMES[a] for a in acts))
        print(f"  stage {t:2d}: " + "  ".join(f"{n:>10}" for n in names))


# THE AGENT WE OBSERVE (it has a HIDDEN true goal)
# agent plays the optimal finite-horizon policy for its goal
# the observer later sees only (stage, tile, action, landed-on), never the goal itself

def simulate_agent(goal, T, start, rng, Q=None):
    """play one T-step game and return a list of (stage, tile, action, next_tile) - ties broken at random"""
    if Q is None:
        _, Q = finite_horizon_dp(build_reward_table(goal), T)
    s = start
    trajectory = []
    for t in range(T):
        a = rng.choice(best_actions(Q[t][s]))  # follow the optimal policy
        s_next = sample_state(transition_probs(s, a), rng)  # noisy world
        trajectory.append((t, s, a, s_next))
        s = s_next
    return trajectory

# HOW LIKELY IS AN ACTION, GIVEN A CANDIDATE GOAL?
# to use Bayes' rule we need P(action | state, stage, goal): "if this were the goal, how likely is the agent to do THAT?"
# softmax over Q-values: P(a | s, t, goal) = exp(Q/tau) / sum over actions of exp(Q/tau)

TAU = 0.3
MIN_LIKELIHOOD = 1e-12

def action_probability(q_for_state, a, tau=TAU):
    """softmax probability of action a"""
    top = max(q_for_state.values())
    weights = {b: math.exp((q - top) / tau) for b, q in q_for_state.items()}
    return max(weights[a] / sum(weights.values()), MIN_LIKELIHOOD)


# BAYESIAN INFERENCE OF GOAL (posterior-update half of BCR)
# candidates = every possible goal tile
# for each one we SOLVE the dynamic program, which answers: "if this were their reward, how would they behave?" 
# then every observed action either supports or hurts each candidate: 
# new belief(goal) [proportional to] old belief(goal) * P(action | goal)

# environment's transition probabilities are the same for every candidate, so they cancel (this is the "same for all m" step in Steines' eq. 3.17)

GOAL_CANDIDATES = list(states)

def solve_all_candidates(T):
    """solve the T-step game once for each candidate goal, returns {goal: Q}"""
    return {g: finite_horizon_dp(build_reward_table(g), T)[1] for g in GOAL_CANDIDATES}


def update_posterior(posterior, t, s, a, Q_by_goal, tau=TAU):
    """one Bayes step: fold in the observation 'at stage t, in tile s, did a'"""
    unnormalized = {
        g: posterior[g] * action_probability(Q_by_goal[g][t][s], a, tau)
        for g in posterior
    }
    total = sum(unnormalized.values())
    return {g: p / total for g, p in unnormalized.items()}


def infer_goal(trajectory, Q_by_goal, tau=TAU):
    """
    returns the list of posteriors: index 0 is the prior (no evidence yet), index k is the belief after k observed actions
    """
    posterior = {g: 1.0 / len(GOAL_CANDIDATES) for g in GOAL_CANDIDATES}  # uniform prior
    history = [posterior]
    for (t, s, a, _s_next) in trajectory:
        posterior = update_posterior(posterior, t, s, a, Q_by_goal, tau)
        history.append(posterior)
    return history


# DEMO
def bar(p, width=24):
    return "#" * int(round(p * width))

def demo(true_goal=4, start=0, T=HORIZON, seed=3):
    rng = random.Random(seed)
    Q_by_goal = solve_all_candidates(T)

    trajectory = simulate_agent(true_goal, T, start, rng, Q_by_goal[true_goal])
    print(f"Hidden truth: the agent wants tile {true_goal} (the observer is NOT told this).")
    print("What the observer sees:")
    for (t, s, a, s_next) in trajectory:
        print(f"  step {t:2d}: on tile {s}, chose {ACTION_NAMES[a]:>5}, landed on tile {s_next}")

    history = infer_goal(trajectory, Q_by_goal)
    print("\nObserver's belief about the goal tile (each bar = probability):")
    header = "  after step | " + "  ".join(f"goal={g}" for g in GOAL_CANDIDATES)
    print(header)
    for k, post in enumerate(history):
        label = "prior" if k == 0 else f"{k:5d}"
        print(f"  {label:>10} | " + "  ".join(f"{post[g]:6.2f}" for g in GOAL_CANDIDATES))

    final = history[-1]
    print("\nFinal belief:")
    for g in GOAL_CANDIDATES:
        print(f"  goal = {g}: {final[g]:5.3f} {bar(final[g])}")
    guess = max(final, key=final.get)
    print(f"\nObserver's best guess: tile {guess}   (truth: tile {true_goal})")


def evaluate(horizons=(12, 30, 60), games_per_goal=300, seed=0):
    """How often does the observer's best guess match the hidden truth, as a function of how many steps it gets to watch?"""
    rng = random.Random(seed)
    print(f"Observer's best guess is correct (%), {games_per_goal} random games per cell, tau = {TAU}")
    print("  steps watched | true goal: " + "".join(f"{g:6d} " for g in GOAL_CANDIDATES))
    for T in horizons:
        Q_by_goal = solve_all_candidates(T)
        cells = []
        for true_goal in GOAL_CANDIDATES:
            correct = 0
            for _ in range(games_per_goal):
                start = rng.choice(states)
                traj = simulate_agent(true_goal, T, start, rng, Q_by_goal[true_goal])
                final = infer_goal(traj, Q_by_goal)[-1]
                correct += max(final, key=final.get) == true_goal
            cells.append(100 * correct / games_per_goal)
        print(f"  {T:13d} |            " + "".join(f"{c:6.1f} " for c in cells))


def main():
    V, Q = finite_horizon_dp(build_reward_table(4), 5)
    print_dp_tables(V, Q, 5)
    print()

    demo()
    print()
    evaluate()


if __name__ == "__main__":
    main()