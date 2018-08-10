'''
##########################
Q-learning using table 
----------------------
author        : T de Klijn
created       : 2018-08-07
last modified : 2018-08-09
##########################
Every action on a board is determined by values in the Q-table (with a certain
distortion), the Q-table is being updated each step, taking into acount the
current and previous steps. This way, the algorithm will have a tendency to
prefer future rewards instead of direct rewards.

TODO: 
    - Implement plotting / visualization
    - RIC (reset of initial conditions), set all Q-table values to the reward
      of the first state
    - Data collection (a lot)
'''

from global_param import Global
from Game import Game
import numpy as np
import gc, os
import csv

class Q_learning(Global):
    '''
    Simple Q-learning:

    * make q-table, shape = (number of tiles, number of options)
    * Loop over n-episodes, per episode walk board based on q-table
    * per action update q-table
    '''
    def __init__(self):

        # parameters:
        self.lr = 0.8            # learning rate
        self.y = 0.95            # discount factor
        self.sub_steps = 100     # max steps per episode
        self.n_episodes = 1000   # Number of episodes 

        # Reward lookup - based on status of the board
        self.r_table = {'A' : 0,
                        'D' : -1,
                        'F' : 1}

        # options - input options for the icelake game
        self.action_list = ['u','d','l','r']

        # Initialize q-table - (length = number of states (positions), width =
        # number of options
        self.q_table = np.zeros((self.width*self.height,4))

        self._init_data_collection()

    def _convert_pos(self, p):
        '''
        Convert position [x,y] to index (1d)
        '''
        return (p[0] * self.width) + p[1]

    def _init_data_collection(self):
        '''
        Initialize data containers used for further analysis
        '''

        # episode list, for plotting purposes
        self.episode_list = [i for i in range(self.n_episodes)]

        # Empty board, save empty board (contains start, finish and exit)
        self.empty_board = None

        # shortes_path should be changed when path is shorter then shortest
        # path, and the player plositions should be saved in
        # shortes_path_container
        self.shortest_path = 10e6
        self.shortest_path_container = []

        # append False if player did not reach the finish
        self.survival_list = []

    def write_data(self):
        '''
        Output data collected during q_learning to a file for later analysis
        '''
        # Check if data directory exists, if not create
        data_dir = 'data/'
        if not os.path.isdir(data_dir): os.makedirs(data_dir)

        # Write empty board to file
        np.save(data_dir + f'new_board_{Global.width}.npy', self.empty_board)

        # Write shortes path to file
        with open(data_dir + f'path_{Global.width}.csv','w') as path_file:
            path_writer = csv.writer(path_file, delimiter=',')
            path_writer.writerow(['index', 'x', 'y'])

            for i, xy in enumerate(self.shortest_path_container):
                path_writer.writerow([i,xy[0],xy[1]])

        # Write survival to file
        with open(data_dir + f'survival_{Global.width}.csv','w') as survival_file:
            survival_writer = csv.writer(survival_file, delimiter=',')
            survival_writer.writerow(['episode','survival'])

            for i, b in enumerate(self.survival_list):
                survival_writer.writerow([self.episode_list[i],b])

    def learn(self):
        '''
        Perform simple Q-learning algorithm based on a Q-table
        Iterate over the episodes, and for each episodes try steps based on the
        Q-values in the q_table, update the q-values during these steps. Escape
        an episode when status is 'D' (dead) or 'F' (finished)
        '''

        r_list = []
        episode_reward = 0
        for i in range(self.n_episodes):
            
            # If a new board is generated, do this only the first game
            if Global.new_board == True and i == 1:
                print('Locking board')
                Global.new_board = False

            # Initialize a new game for each episode
            game = Game()                    

            if i == 0: self.empty_board = game.board 

            j = 0
            path_container = []
            s = [game.start_x, game.start_y]

            # Append state (player coordinates) to shortest path
            path_container.append(s)

            while j < self.sub_steps:
                j += 1
                
                # state is 1d index in q-table
                s = self._convert_pos(s)
                
                # Determine action based on q_table + random noise
                # Choose the max value from the Q-table for this state
                # (position on board), this is being distorted by a random
                # value (scaled by episode number)
                a = np.argmax(
                        self.q_table[s,:] +
                        np.random.randn(1,len(self.action_list))
                        *(1.0/(i+1)))

                # take step on board
                status = game.update_board(self.action_list[a])

                # New gamestate (x and y position of the player
                s1 = [game.position_x, game.position_y]

                # append new position to path
                path_container.append(s1)

                # reward
                r = self.r_table[status]

                # Based on the rewards on the current position (s1) as well as
                # the previous position (s) the Q-value for this state is
                # updated. This is determined by the learining rate (lr), the
                # discount factor (y) and the reward in the current position
                # (r). If s is outside the Q-table, reward = -1.
                if self._convert_pos(s1) < len(self.q_table):
                    self.q_table[s,a] = self.q_table[s,a] + self.lr * ( r + self.y *
                            np.max(self.q_table[self._convert_pos(s1),:] - self.q_table[s,a]))
                else:
                    self.q_table[s,a] = self.q_table[s,a] + self.lr * ( r + self.y *
                            (-1) - self.q_table[s,a])

                # update reward per episode
                episode_reward += r

                # new state -> old state
                s = s1

               # break out when dead
                if status == 'D':
                    self.survival_list.append(False)
                    break
                if status == 'F':
                    self.survival_list.append(True)
                    if j < self.shortest_path: 
                        self.shortest_path = j
                        self.shortest_path_container = path_container
                    break

            # append episode score
            r_list.append(episode_reward)

            # Garbage collection
            gc.collect()

        # write data
        self.write_data()

def main():

    q_learn = Q_learning()
    q_learn.learn()

if __name__ == '__main__':
    main()
