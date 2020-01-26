# CartPoleGame
1 INTRODUCTION
In recent years Reinforcement Learning became one of the newest trends in Machine Learning and Artificial Intelligence. When our team was searching an appropriate project for Introduction to Artificial Intelligence course, we found a new trend topic which uses neural networks and agents together which is called Reinforcement Learning. The purpose of our project is making an agent which is capable of learning based on mistakes and trials and solving the cartpole problem.
Besides getting knowledge on neural networks, we also acquired knowledge on agents and learned how reinforcement learning is a powerful technic.
	2 CARTPOLE PROBLEM DETAIL
Before to understand cartpole game, we need to define OpenAI Gym. It is a toolkit for developing and comparing reinforcement learning algorithms. It provides us to use environments for implementing reinforcement learning algorithms. In our project, we decided to use “Cartpole” environment. This environment describes cart-pole problem. “A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center” [1].
	3 METHOD AND APPROACH
The environment we just defined is successful if we can balance the pole for 250 frames, and it will be unsuccessful if the pole has more than 15 degrees from vertical or the cart moves more than 2.4 units from the center. 
We researched reinforcement learning to make an intelligent agent that learns from its mistakes and improves its performance. Related to agent actions, reinforcement algorithm gives punishments and rewards. To understand how it works on Cartpole, we need to understand Markov Chain Model. Because, cartpole is built on that model which is shown in below. During our project we brainstormed on model-based learning and model-free learning, but we moved on model-based learning after examining similar works.
 (Markov Chain Model)


In this model, an agent takes current state which is St, then picks the best action At based on model prediction and execute it on an environment. After that environment returns a reward which is Rt+1 for a given action and returns a new state St+1 and an information if the new state is terminal. This process repeats until termination.
To find the best action for observation, there is a reinforcement learning algorithm which is called Deep Q-Learning algorithm according to related works. We learned how to use neural networks for Deep Q Learning and apply to play Cartpole game. We also needed to learn how to use Keras and Gym for implementation parts. 
We use three hidden layers in our neural network, and they consist of 100,100 and 64 neurons for each layer respectively. Relu activation function is also used for improving performance. Output layer includes 2 neurons which are used for left and right actions. Our neural network predicts the reward value from a certain state. It means that the model is capable of predicting the reward from unseen data.
Deep Q Learning algorithm has also some key methods which are “remember” and “replay”. We also saw some different methods for further works which is Double Deep Q Learning. It can also consider for solving this kind of problems with high accuracy. The point of using remember function is for training. Because we need to update memory list which includes previous experiences with new experiences to re-train our model. We need to call memory list and use this function to append state, action, reward, and next state to the list. Replay function provides training on our neural network with data in the memory. We defined “minibatch” for some experiences from the memory. We used Numpy library for selecting experiences randomly. We setted batch size as 50 for our minibatch. If memory size is less than 50, then we take everything is in our memory.
We also used some parameters for our reinforcement learning agent in our model. You can see them with detail in below.
•	Episodes: We used 250, it represents number of games that agent plays.
•	Gamma: We used 0.95, it represents discount rate to calculate the future discounted reward.
•	Epsilon: We used 1.0, this is a exploration rate in which an agent randomly decides its action rather than prediction.
•	Epsilon decay: We used 0.999, we need to decrease the number of explorations as it gets good at playing games.
•	Epsilon min: We used 0.001, this is the minimum amount for the agent to explore.
•	Learning rate: We used 0.001, it represents  how much neural network learns in each iteration.
•	Batch size: We used 50, it shows how much memory the model will use to learn.

4 RESULTS
In our model, we run 250 episodes of game for training. When the pole is balanced, we set +1 point to score for every frame. Our target was getting 400 points. We reached that score in 136 episodes during training.  we reached our target for the first time in third episode during testing through our trained model.
  

In this project, we have learned how reinforcement learning works, how to write neural network and how to work with agents.

REFERENCES
[1] [Barto83]	AG Barto, RS Sutton and CW Anderson, "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem", IEEE Transactions on Systems, Man, and Cybernetics, 1983.
