# Reinforcement learning
### RL vs supervised learning
- Reinforcement learning is **teaching by experience**: the agent tries actions, observes outcomes, and learns from reward signals. No one tells the agent the correct action - it must discover which actions yield the most reward through trial and error.
- Supervised learning is **teaching by example**: the model is shown labeled examples of correct answers and learns to imitate them.

RL is fundamentally about making a **sequence of decisions**, not a single prediction. Each action changes the state of the environment, which affects what actions and rewards are available later. Rewards may be **delayed** - a chess move may only pay off many moves later - so the agent must learn which earlier decisions deserve **credit** for eventual outcomes (the credit assignment problem). This sequential, delayed-feedback structure is what distinguishes RL from supervised learning, where each prediction is independent and feedback is immediate.

### Real-world application
- Game Playing: Beating world champions in complex board games (Go, Chess) and real-time strategy video games (Dota 2, StarCraft II).
- Robotics: Training robotic arms to grasp objects, or teaching quadrupedal robots to walk over uneven terrain.
- Autonomous Driving: Optimizing trajectory planning, lane-changing behavior, and collision avoidance systems.
- Advertising: Real-time bid optimization in ad auctions, budget pacing across campaigns, and sequential ad selection that maximizes long-term conversions instead of immediate clicks.
- [Large Language Models (LLMs)](transformer.md#llms-large-language-models): Fine-tuning models using RLHF (Reinforcement Learning from Human Feedback) to ensure AI responses align with human preferences regarding safety and helpfulness.

### RL Terminology
An **agent** interacts with an **environment** and learns to take **action** by maximizing a cumulative **reward**.
- **Agent**: The AI system, decision-maker, or learner (e.g., a self-driving car software or a chess-playing bot).
- **Environment**: Everything outside the agent that it interacts with (e.g., the physical roads or the chessboard).
- **State** ($S$): The current situation or configuration of the environment at a specific time.
- **Action** ($A$): The choices available to the agent (e.g., turn left, move pawn to E4).
- **Reward** ($R$): The feedback signal sent from the environment to evaluate the agent's last action. It can be positive (a reward) or negative (a penalty). Examples include:
   - Games: win, maximize score
   - Finance: gains, gains minus risk
   - Drone delivery: positive delivery reward, penalty for collision.
   
<img width="1025" height="415" alt="345fadfa-549a-462a-b757-9ab258e747f3" src="https://github.com/user-attachments/assets/67239e31-45fd-4195-b4ab-d5242e7380a8" />

- **Observation (0)** The information the agent receives from the environment at each step. If the observation captures the complete state, the environment is fully observable (like a chessboard); if not, it is partially observable (like a poker hand or a car's camera view).

- **Terminal state** is one where the episode ends — no further actions or rewards follow. Examples include:
   - Chess or Go: checkmate, resignation, or a draw
   - Robotics: A drone landing at its target — or crashing
   - Finance and wagering: Bankruptcy — bankroll hits zero
   - Finance: An options position expiring
   - A treatment-planning MDP in healthcare: patient recovery or death
 
### Policy
- **Policy** ($\pi$): The decision-making rule the agent is learning — a mapping from states to actions. The policy defines the agent's behavior.
   - A **deterministic** policy returns a single action for each state: $A = \pi(S)$.
   - A **stochastic** policy returns a probability distribution over actions: $\pi(A|S)$.
- In small, tractable problems the policy can be derived from a **Q-table** — a lookup table storing an estimated value for every state-action pair, where the agent simply picks the action with the highest value. In complex problems the state space is too large to enumerate, so the mapping is approximated with a neural network (the "deep" in deep RL).

### The Bellman Equation
The **Bellman equation** expresses the core recursive idea of RL: the value of where you are now = the reward you get now + the value of where you end up next.
- Instead of evaluating a state by playing out an entire episode, the agent can break the problem into one step at a time: take an action, collect the immediate reward, and rely on its estimate of the next state's value to account for everything after that.
- Future rewards are typically discounted by a factor 𝛾 (gamma, between 0 and 1), meaning a reward now is worth slightly more than the same reward later. This keeps values finite and makes the agent prefer faster paths to reward.
- This recursive structure is what makes learning practical: the agent doesn't need to see the end of the game to update its estimates — it can bootstrap, improving its value estimate for the current state using its estimate of the next one. Q-learning and DQN are built directly on this idea.

### Categories of RL agents
RL algorithms differ in *what* the agent learns:
- **Value-based**: The agent learns a value function (like a Q-table or DQN) and derives its policy implicitly by picking the highest-value action. Examples: Q-learning, DQN.
- **Policy-based**: The agent learns the policy directly, optimizing the parameters of $\pi(A|S)$ to maximize expected reward without ever estimating state values. Examples: REINFORCE.
- **Actor-critic**: A hybrid — the **actor** learns the policy while the **critic** learns a value function that evaluates the actor's actions, reducing the variance of policy updates. Examples: A2C, PPO.

A separate axis is whether the agent models the environment:
- **Model-free**: The agent learns purely from experience, with no model of how the environment transitions between states. Most deep RL (DQN, PPO) is model-free.
- **Model-based**: The agent learns or is given a model of the environment's dynamics and can plan by simulating outcomes before acting. Example: AlphaZero, which uses tree search over a learned model.

### PPO
**PPO (Proximal Policy Optimization)** is an actor-critic policy gradient algorithm that improves training stability by clipping each update so the new policy can't move too far from the old one.

### References
- 2013 deep mind DQN paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

- 2015 Silver et al. DQN nature paper [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
 [code](https://github.com/google-deepmind/dqn)

- 2017 Open AI PPO paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### Class
[DeepMind RL class](https://www.youtube.com/watch?v=TCCjZe0y4Qc&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&index=1&)   
[Stanford 230 lecture](https://www.youtube.com/watch?v=4E27qlfYw0A&list=PLoROMvodv4rNRRGdS0rBbXOUGA0wjdh1X&index=5)
