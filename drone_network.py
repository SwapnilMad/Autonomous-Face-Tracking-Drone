from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Dropout


class DroneNetwork:

    def __init__(self, observation_shape, nb_actions):
        self.observation_shape = observation_shape
        self.nb_actions = nb_actions
        self.action_input = Input(shape=(self.nb_actions,), name='action_input')

    def create_actor(self):
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + self.observation_shape))
        actor.add(Dense(600, activation='relu'))
        actor.add(Dropout(0.5))
        actor.add(Dense(600, activation='relu'))
        actor.add(Dropout(0.5))
        actor.add(Dense(self.nb_actions, activation='tanh'))
        return actor

    def create_critic(self):
        observation_input = Input(shape=(1,) + self.observation_shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(600, activation='relu')(flattened_observation)
        x = Dropout(0.5)(x)
        x = Concatenate()([x, self.action_input])
        x = Dense(600, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear')(x)
        return Model(inputs=[self.action_input, observation_input], outputs=x)

    def get_action_input(self):
        return self.action_input
