# Flappy bird reinforcement learning agent
This code provides an environment for the game flappy bird. Using this environment one can train an deep-q network (DQN) agent for playing the game using reinforcement learning. Additionally one can choose freely between the following agents:
- **User agent:** User played game (bump bird with space)
- **Random agent:** Agent which performs random actions
- **DQN agent:** Agent using a dueling double deep Q-Network for performing actions

The game and its agent is run by initializing with
```python
g = game.Game(agent_name = "dqn_agent", device = "cpu")
```
To run a game execute the main function
```python
g.main(draw = "True")
```
If using a DQN agent one can train with
```python
g.train_agent(draw = "False", episodes = 100, batches = 100, hyperparameters)
```

## Deep Q-learning setup
The game environment returns three features as input for the DQN agent:
- Horizontal distance to next pipe
- Vertical distance to lower next pipe
- Speed of bird

The returned reward from the environment after performing an action is:
## base line surving
- 0.1 for surviving 
## Excellent vertical alignment
- +0.5 if vertical distance to the pipe center < 0.2
## Good vertical alignment
- +0.2 if vertical distance < 0.4
## Pass pipe
- +5 when the bird successfully passes a pipe
## High velocity
- -0.3 if absolute bird velocity > 0.8
## Collision
- -10 for colliding
## Score > 50
- If total score > 50, the reward is multiplied by 1.2

The agent has two possible actions:
- bump the bird
- doing nothing

The whole states/rewards/actions are stored in an experience buffer of the agent used for training after each episode. During the training procedure the agent uses the $\epsilon$-greedy policy with decreasing $\epsilon$.<br />
The DQN agents architecture is a neural network with one hidden layer of size 128.

## Trained DQN agent example
Example of trained DQN agent using a training of 100 episodes each with 100 batches of size 128 and learning rate of $\tau = 1e^{-4}$ with hyperparameters $\gamma = 0.8$, $\epsilon_s = 0.9$, $\epsilon_e = 1e^{-2}$. <br /><br />
![](https://github.com/Srinivas-Kantheti-2005/Flappy_Bird_RL)

## Required packages
- numpy
- pytorch
- pygame
