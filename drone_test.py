import numpy as np
from Simulator import SimulatorEnv
from drone_network import DroneNetwork
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from drone import Drone

# Get the environment and extract the number of actions.
env = Drone()
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

n = DroneNetwork(nb_actions=nb_actions, observation_shape=env.observation_space.shape)

# Next, we build a very simple model.
actor = n.create_actor()
critic = n.create_critic()
action_input = n.get_action_input()

actor.summary()
critic.summary()
print(action_input)

memory = SequentialMemory(limit=100000, window_length=1)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory)

agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.load_weights('ddpg_{}_weights.h5f'.format('drone'))
agent.test(env, nb_episodes=100000, visualize=True)
#agent.test(env, nb_episodes=20, visualize=True, nb_max_episode_steps=50)
env.close()
