# Reinforcement learning
### RL vs supervised learning
- Supervised learning is **teaching by example**: the model is shown labeled examples of correct answers and learns to imitate them.
- Reinforcement learning is **teaching by experience**: the agent tries actions, observes outcomes, and learns from reward signals. No one tells the agent the correct action - it must discover which actions yield the most reward through trial and error.

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
- **Reward** ($R$): The feedback signal sent from the environment to evaluate the agent's last action. It can be positive (a reward) or negative (a penalty).
<img width="1025" height="415" alt="345fadfa-549a-462a-b757-9ab258e747f3" src="https://github.com/user-attachments/assets/67239e31-45fd-4195-b4ab-d5242e7380a8" />

- **Observation (0)** The information the agent receives from the environment at each step. If the observation captures the complete state, the environment is fully observable (like a chessboard); if not, it is partially observable (like a poker hand) or a car's camera view).

- **Policy** is the decision-making rule the agent is learning - a mapping from states to actions. Policy defines the agent's behavior.  
Policy can be deterministic A = π(S) or stochastic A = π(A|S).

### References
- 2013 deep mind DQN paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

- 2015 Silver et al. DQN nature paper [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
 [code](https://github.com/google-deepmind/dqn)

- 2017 Open AI PPO paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### Class
[DeepMind RL class](https://www.youtube.com/watch?v=TCCjZe0y4Qc&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&index=1&)
