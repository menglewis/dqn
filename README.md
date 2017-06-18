# DQN

This is an implementation of Deep Q Learning using Keras and Tensorflow.

## Examples

There is an example of using this DQN implementation on the CartPole problem from OpenAI Gym.
To train the agent on CartPole, run

    $ python cartpole.py
    
This will train the agent and save the weights for the model as well as save the rewards for each episode during training.

## Use the trained agent
After training a model, you can load it into an agent and play it against an OpenAI Gym environment using play.py

    $ python play.py --env-name CartPole-v0 --model-path model_CartPole-v0_DQNAgent.h5 --episodes 1
    
