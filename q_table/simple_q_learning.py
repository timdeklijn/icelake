'''
##########################
Q-learning using table 
----------------------
author        : T de Klijn
created       : 2018-08-07
last modified : 2018-08-14
##########################
Every action on a board is determined by values in the Q-table (with a certain
distortion), the Q-table is being updated each step, taking into acount the
current and previous steps. This way, the algorithm will have a tendency to
prefer future rewards instead of direct rewards.

TODO: 
    - RIC (reset of initial conditions), set all Q-table values to the reward
      of the first state
'''

from game.global_param import *
from game.Game import Game

import pandas as pd
import numpy as np
import gc, os
import csv

class Qlearn():
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
        self.q_table = np.zeros((width*height,4))

    def _convert_pos(self, p):
        '''
        Convert position [x,y] to index (1d)
        '''
        return (p[0] * width) + p[1]

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
            
            # Initialize a new game for each episode
            game = Game()                    

            if i == 0: self.empty_board = game.board 

            j = 0
            s = [game.start_x, game.start_y]

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

               # break out when dead or finished
                if status == 'D':
                    self.final_status = 'D'
                    break
                if status == 'F':
                    self.final_status = 'F'
                    break

            # append episode score
            r_list.append(episode_reward)

            # Garbage collection
            gc.collect()

def run():
    '''
    Run a q_learning algorithm on the game/maze/board
    '''
    learn = Qlearn()
    learn.learn()

if __name__ == '__main__':
    run()
