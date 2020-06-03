import numpy as np
from Simulator import SimulatorEnv
from drone_network import DroneNetwork
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory

# Get the environment and extract the number of actions.
env = SimulatorEnv()
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

n = DroneNetwork(nb_actions=nb_actions, observation_shape=env.observation_space.shape)

# Next, we build a very simple model.
actor = n.create_actor()
critic = n.create_critic()
action_input = n.get_action_input()
critic.summary()
memory = SequentialMemory(limit=100000, window_length=1)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory)

agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=15000, visualize=True, verbose=2, nb_max_episode_steps=200)
agent.save_weights('ddpg_{}_weights.h5f'.format('drone'), overwrite=True)
print('testing')
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)
env.close()

