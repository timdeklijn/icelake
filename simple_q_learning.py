'''
##########################
Q-learning using table 
----------------------
author        : T de Klijn
created       : 2018-08-07
last modified : 2018-08-10
##########################
Every action on a board is determined by values in the Q-table (with a certain
distortion), the Q-table is being updated each step, taking into acount the
current and previous steps. This way, the algorithm will have a tendency to
prefer future rewards instead of direct rewards.


The MetaLearning class will run Qlearning over and over on different boards
with different sizes. It will collect data and write it to file.

TODO: 
    - RIC (reset of initial conditions), set all Q-table values to the reward
      of the first state
    - Data collection (a lot)
'''

from global_param import Global
from Game import Game

import pandas as pd
import numpy as np
import gc, os
import csv

class MetaLearning(Global):
    '''
    Meta learning class, will run Q_learning over and over to collect data,
    which will be outputted to file for later analysis. 

    One such meta run will run Q_learning n_runs times, per board size
    board = {'width' : n, 'height' : n, 'dangers' : d}.
    '''

    def __init__(self):

        # Run control
        Global.show_board = False

        # Number of succesful runs per board
        self.n_runs = 10

        # final data file
        self.data_file = 'data/meta_data.csv'

        # meta parameters:
        n = list(range(3,6))
        dangers = n 
        # Dict with desired board options:
        self.boards = []
        for i, wh in enumerate(n):
            self.boards.append({'width': wh, 'height' : wh, 'dangers' :
                dangers[i]-1})

    def meta_learn(self):
        '''
        Run Qlearn multiple times and collecting different parameters/variables
        to later do statistics on. 
        '''

        # Containers
        size, dangers, first_succes, n_fails, n_finish = [],[],[],[],[]

        # Start looping over boards, every board has to succeed n_runs times.
        for board in self.boards:

            # Set global board parameters
            Global.width = board['width']
            Global.height = board['height']
            Global.danger_number = board['dangers']

            # Counter for number of succesfull runs, ensures every board has
            # the same number of final succesfull runs
            succes_runs = 0

            # Perform q_learning runs untill the n_runs has been reached
            # succesfuly
            while succes_runs < self.n_runs:

                # Start Qlearning
                Global.new_board = True
                q_learn = Qlearning()
                q_learn.n_episodes = 100
                q_learn.learn()

                # Count when succesfull
                if q_learn.final_status == 'F':
                    succes_runs +=1

                    size.append(board['width'])
                    dangers.append(board['dangers'])
                    first_succes.append(q_learn.first_succes)
                    n_fails.append(q_learn.n_fail)
                    n_finish.append(q_learn.n_finish)
                    print(f'''{board['width']} - {board['dangers']} - {q_learn.first_succes} - {q_learn.n_fail} - {q_learn.n_finish}''') 
                gc.collect()

        # Data frame for saving
        self.df = pd.DataFrame({'size' : size, 'n_dangers' : dangers,
            'first_succes' : first_succes, 'n_fails' : n_fails, 'n_finish' :
            n_finish})

        # Final results
        print(self.df.head())

        # Output data to file
        self.df.to_csv(self.data_file, sep=',')
           

class Qlearning(Global):
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

        # will contain the final status of a q-learn run
        self.final_status = ''

        # Containers for meta learning
        self.n_fail, self.n_finish = 0 , 0
        self.first_succes = None

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

               # break out when dead or finished, adjust counters
                if status == 'D':
                    self.n_fail +=1
                    self.survival_list.append(False)
                    self.final_status = 'D'
                    break
                if status == 'F':
                    if self.first_succes == None:
                        self.first_succes = i 
                    self.n_finish += 1
                    self.survival_list.append(True)
                    self.final_status = 'F'
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

#    q_learn = Qlearning()
#    q_learn.learn()
    meta_learn = MetaLearning()
    meta_learn.meta_learn()

if __name__ == '__main__':
    main()
