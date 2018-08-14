'''
##########################
Markov decision process
-----------------------
author        : T de Klijn
created       : 2018-08-13
last modified : 2018-08-14
##########################
Implementation of MDP using keras

Uses Game.py for interacting with a maze, Game.py extends global_params.py. 

adapted from: 
http://www.samyzaf.com/ML/rl/qmaze.html
https://keras.io/models/sequential/
'''

import numpy as np
import os, sys, time, datetime, json, random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU

from game.Game import Game
from game.global_param import *

# possible actions for game and the amount of actions
actions = ['u', 'd', 'l', 'r']
num_actions = len(actions)
# Eagerness to learn
epsilon = 0.9

class Experience(object):
    '''
    Save learned information from the training. Convert it into a formate that
    model (keras) accepts, and train the model.
    '''

    def __init__(self,model, max_memory=100, discount=0.95):

        self.model = model                        # keras sequential
        self.max_memory = max_memory              # lenght of memory list
        self.discount = discount                  # discount factor
        self.memory = list()                      # memory list
        self.num_actions = model.output_shape[-1] # number of movement actions

    def remember(self, episode):
        '''
        Append episode info, except when max is reached, then append and
        discard older episodes
        '''
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate = flattened board
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        '''
        Let the model predict the next move
        '''
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        '''
        Combine everything to tensorflow target
        '''
        # Some local constants
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        # Input and targets are fed to the model
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        # Choose a subset of episodes to train the model on
        for i,j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            # Q_value of best action
            Q_sa = np.max(self.predict(envstate_next))
            # Update targets
            if game_over:
                targets[i,action] = reward
            else:
                targets[i,action] = reward + self.discount * Q_sa
        return inputs, targets

def qtrain(model, game, **opt):
    '''
    Train a model with a neural net to solve the maze created in Game using
    reinforment learning (Markov Decision Process)
    '''
    # Global parameters
    global epsilon
    # get parameters from **opt, else take default
    n_epoch = opt.get('epochs', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', '')
    name = opt.get('name', 'model')
    # Set start time of learning exercise
    start_time = datetime.datetime.now()
    
    # Prep epsilon update - become less exploratory over time
    d_epsilon = n_epoch / 10
    update_epsilon_list = [i*d_epsilon for i in range(10)][1:]
 
    # Load weights from previous session
    if weights_file:
        print('Loading weights from file %s'.format(weights_file))
        model.load_weights(weights_file)

    # Controls all learned data/weights
    experience = Experience(model, max_memory=max_memory)
    # Save wins and losses
    win_history = []
    # amount of free space on the board
    n_free_cells = game.size - danger_number - 1 
    # If total reward < min reward, escape epoch
    min_reward = -0.5 * game.size
    # history window size.size
    hsize = game.size // 2
    # Container for win ratio
    win_rate = 0.0
    # Start learning
    for epoch in range(n_epoch):
        # initialize game + params
        loss = 0.0
        game_over = False
        # first state
        envstate = observe_board(game.board, game.position_x, game.position_y)
        n_episodes = 0
        tot_reward = 0.0
        # Sve trajectory to penalize backtracking
        trajectory = []
        # Play the game
        while not game_over:
            # Get possible actions
            valid_actions = actions
            # Set new state to old
            prev_envstate = envstate
            # Determin action epsilon determines how curious the player is
            if np.random.rand() > epsilon:
                action = random.choice([i for i in range(num_actions)])
            else:
                # get action from model
                action = np.argmax(experience.predict(prev_envstate))
            # Perform action
            game_status = game.update_board(actions[action])
            # Set reward, if step is state is already explored 0.25 else get_reward_from_state
            if [game.position_x,game.position_y] in trajectory:
                reward = -0.25
            else:
                reward = get_reward_from_state(game_status)
            # Append current position to trajectory
            trajectory.append([game.position_x, game.position_y])
            # Total reward updata
            tot_reward += reward
            # Update current environment 
            envstate = observe_board(game.board, game.position_x,
                    game.position_y)
            # Check if episode is over
            if game_status == 'F':
                win_history.append(1)
                game = Game()
                game_over = True
            elif game_status == 'D' or tot_reward < min_reward:
                win_history.append(0)
                game = Game()
                game_over = True
            else:
                game_over = False
            # Save episode
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1
            # Modify state/reward info to train the model
            inputs, targets = experience.get_data(data_size = data_size)
            # Trein the model
            h = model.fit(
                    inputs,
                    targets,
                    epochs=8,
                    batch_size=16,
                    verbose=0,
                    )
            # Evaluate model
            loss = model.evaluate(inputs, targets, verbose=0)
        # Escape if 100% win ratio
        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
        # Print episode info
        t = format_time((datetime.datetime.now() -
            start_time).total_seconds())
        template = 'Epoch: {:4d}/{:4d} | Loss: {:.4f} | Epsodes: {:3d} | Win count: {:4d} | Win rate: {:.3f} | time: {}'
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        # Update epsilon at intervals - make less curious over time
        if epoch in update_epsilon_list:
            epsilon *= 1.02
        # If win_rate is high enough stop training
        if sum(win_history[-hsize:]) == hsize: 
            print('Reached 100%% win rate at epoch {:d}'.format(epoch))
            break

    # output model - weights + nearal network info
    h5file = name + '.h5'
    json_file = name + '.json'
    # Save weights
    model.save_weights(h5file, overwrite = True)
    # Save model
    with open(json_file, 'w') as outfile:
        json.dump(model.to_json(), outfile)
    # nicer time string
    t = format_time((datetime.datetime.now() - start_time).total_seconds())
    # Print summary of learning session
    print('files: {} {}'.format(h5file, json_file))
    print('n_epoch: {:d}, max_mem: {:d}, data: {:d}, time: {}'.format(epoch,max_memory,data_size,t))

def get_reward_from_state(state : str) -> float:
    '''
    Return reward from state
    '''
    if state == 'F':
        return 1.0
    elif state == 'D':
        return -0.75
    elif state == 'A':
        return -0.04
    else:
        raise Excetpion('State not recognised')
    
def build_model(game, lr=0.001):
    '''
    Setup model (sequential) and the nearal network
    '''
    model = Sequential()
    model.add(Dense(game.size, input_shape=(game.size,)))
    model.add(PReLU())
    model.add(Dense(game.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

def format_time(seconds : int) -> str:
    '''
    Convert seconds to a usefull time format
    '''
    if seconds < 400:
        s = float(seconds)
        return '{:.1f} seconds'.format(s)
    elif seconds < 4000:
        m = seconds / 60.0
        return '{:.2f} minutes'.format(m)
    else:
        h = seconds / 3600.0
        return '{:.2f} hours'.format(h)

def observe_board(board,x,y):
    '''
    Convert (state of) the board to a 1D array
    '''
    board[np.where(board == 1)] = 1.0
    board[np.where(board == 5)] = 1.0
    board[np.where(board == 3)] = 1.0
    board[np.where(board == -10)] = 0.0
    sp = np.shape(board)
    if 0 <= x < sp[0] and 0 <= y < sp[1]:
        board[x,y] = 0.5
    return board.reshape((1,-1)).astype(float)

def run():
    # Initialize the game
    game = Game()
    # Build model
    model = build_model(game)
    # Train model
    qtrain(model, game, epochs=15000, max_memory=8*game.size, data_size=32,
            weights_file='')

if __name__ == '__main__':
    run()
