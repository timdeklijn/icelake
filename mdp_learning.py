'''
##########################
Markov decision process
-----------------------
author        : T de Klijn
created       : 2018-08-09
last modified : 2018-08-09
##########################
Implementation of MDP
'''

import numpy as np
import os, sys, time, datetime, json, random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU

from Game import Game

# possible actions for game and the amount of actions
actions = ['u', 'd', 'l', 'r']
num_actions = len(actions)

# Eagernes to learn
epsilon = 0.1

class Experience(object):

    def __init__(self,model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        '''
        Append episode info, except when max is reached, then append and
        discard old episode
        '''
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate = flattened board
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        '''
        Predict new state
        '''
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        '''
        Combine everything to tensorflow target
        '''
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i,j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i,action] = reward
            else:
                targets[i,action] = reward + self.discount * Q_sa
        return inputs, targets

def qtrain(model, game, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', '')
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    if weights_file:
        print('Loading weights from file %s'.format(weights_file))
        model.load_weights(weights_file)

    # Controls all learned data/weights
    experience = Experience(model, max_memory=max_memory)

    # Save wins and losses
    win_history = []
    # amount of free space on the board
    n_free_cells = game.size - game.danger_number - 1 
    min_reward = -0.5 * game.size
    # TODO look up what this is !!!
    hsize = game.size // 2  # history window size
    # Container for win ratio
    win_rate = 0.0
    # TODO find what this is
    imctr = 1

    print('Start Learning')
    for epoch in range(n_epoch):
        loss = 0.0
        start = [game.position_x, game.position_y]
        game_over = False
        
        envstate = observe_board(game.board)

        n_episodes = 0
        
        trajectory = []
        
        tot_reward = 0.0

        while not game_over:
            # Get possible actions
            valid_actions = actions
            
            # Set new state to old
            prev_envstate = envstate

            # Determin action epsilon determines how curious the player is
            if np.random.rand() < epsilon:
                action = random.choice([i for i in range(num_actions)])
            else:
                # get action from model
                action = np.argmax(experience.predict(prev_envstate))
            
            # Perform action
            game_status = game.update_board(actions[action])
            # Set reward, if step is taken -0.25 else get_reward_from_state
            if [game.position_x,game.position_y] in trajectory:
                reward = -0.25
            else:
                reward = get_reward_from_state(game_status)
            # Append current position to trajectory
            trajectory.append([game.position_x, game.position_y])

            tot_reward += reward

            # Update current environment 
            envstate = observe_board(game.board)

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

            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            inputs, targets = experience.get_data(data_size = data_size)
            h = model.fit(
                    inputs,
                    targets,
                    epochs=8,
                    batch_size=16,
                    verbose=0,
                    )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = 'Epoch: {:03d}/{:d} | Loss: {:.4f} | Epsodes: {:02d} | Win count: {:02d} | Win rate: {:.3f} | time: {}'
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
    
        # Make less curious if stuff is going good
        if win_rate > 0.9 : epsilon = 0.5

        # If win_rate is high enough stop training
        if sum(win_history[-hsize:]) == hsize: 
            print('Reached 100%% win rate at epoch {:d}'.format(epoch))

    # output model + neural network
    h5file = name + '.h5'
    json_file = name + '.json'
    model.save_weights(h5file, overwrite = True)
    with open(json_file, 'w') as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = end_time - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    # Print summary
    print('files: {} {}'.format(h5file, json_file))
    print('n_epoch: {:d}, max_mem: {:d}, data: {:d}, time: {}'.format(epoch,max_memory,data_size,t))
    return seconds

def get_reward_from_state(state):
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

def format_time(seconds):
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

def observe_board(board):
    '''
    Convert (state of) the board to a 1D array
    '''
    board[np.where(board == 1)] = 1.0
    board[np.where(board == 5)] = 1.0
    board[np.where(board == 3)] = 1.0
    board[np.where(board == -10)] = 0.0
    return board.reshape((1,-1)).astype(float)

def main():
    # Initialize the game
    game = Game()
    # Build model
    model = build_model(game)
    # Train model
    qtrain(model, game, epochs=1000, max_memory=4*game.size, data_size=32)

if __name__ == '__main__':
    main()
