import importlib
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

MODULE = "Bellman_v2" 

TRUE_GOAL = 2   # the agent's hidden goal (the observer is not told)
START = 3       # tile the agent starts on
T = 12          # number of steps in the game
SEED = 3        # change this to get different glitchy-button luck
MS_PER_STEP = 800

m = importlib.import_module(MODULE)

# run the game: the agent plays its best moves for TRUE_GOAL
rng = random.Random(SEED)
Q_by_goal = m.solve_all_candidates(T)
trajectory = m.simulate_agent(TRUE_GOAL, T, START, rng, Q_by_goal[TRUE_GOAL])

# observer watches the moves and updates its belief after each one
history = m.infer_goal(trajectory, Q_by_goal)   # history[k] = belief after k moves
positions = [START] + [next_tile for (_t, _s, _a, next_tile) in trajectory]

# plots
fig, (ax_hall, ax_bel) = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [1, 2]})

for tile in m.states:
    ax_hall.add_patch(Rectangle((tile - 0.45, -0.45), 0.9, 0.9, fill=False, linewidth=1.5))
    ax_hall.text(tile, -0.32, str(tile), ha="center", va="center", fontsize=10)
(agent,) = ax_hall.plot([START], [0.05], "o", markersize=26, color="tab:blue")
ax_hall.set_xlim(-0.6, m.N_STATES - 0.4)
ax_hall.set_ylim(-0.6, 0.6)
ax_hall.set_aspect("equal")
ax_hall.axis("off")
hall_title = ax_hall.set_title("")

bars = ax_bel.bar(m.states, [1 / m.N_STATES] * m.N_STATES, color="tab:gray")
labels = [ax_bel.text(g, 0.0, "", ha="center", va="bottom") for g in m.states]
ax_bel.set_ylim(0, 1.1)
ax_bel.set_xticks(m.states)
ax_bel.set_xticklabels([f"goal {g}" for g in m.states])
ax_bel.set_ylabel("observer's belief")


def draw_frame(k):
    agent.set_data([positions[k]], [0.05])
    if k == 0:
        hall_title.set_text("Before any moves (hidden goal: ?)")
    else:
        _t, s, a, s_next = trajectory[k - 1]
        glitch = "  (glitch!)" if m.clamp(s + a) != s_next else ""
        hall_title.set_text(f"Step {k} of {T}: chose {m.ACTION_NAMES[a]} on tile {s}, landed on {s_next}{glitch}")
    belief = history[k]
    for g in m.states:
        bars[g].set_height(belief[g])
        bars[g].set_color("tab:gray")
        labels[g].set_position((g, belief[g] + 0.01))
        labels[g].set_text(f"{belief[g]:.2f}")
    if k == T:
        guess = max(belief, key=belief.get)
        bars[TRUE_GOAL].set_color("tab:green")
        hall_title.set_text(f"Done!!!! :D Hidden goal was tile {TRUE_GOAL}; observer's best guess: tile {guess}")
    return []


if __name__ == "__main__":
    anim = FuncAnimation(fig, draw_frame, frames=T + 1, interval=MS_PER_STEP, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()