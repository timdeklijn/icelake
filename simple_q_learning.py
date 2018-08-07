from global_param import Global
from Game import Game
import numpy as np

class Q_learning(Global):
    '''
    Simple Q-learning:

    * make q-table, shape = (number of tiles, number of options)
    '''
    def __init__(self):

        # parameters:
        self.lr = 0.8
        self.y = 0.95
        self.sub_steps = 100
        self.n_episodes = 1000

        # Reward lookup
        self.r_table = {'A' : 0,
                        'D' : -1,
                        'F' : 1}

        # options
        self.action_list = ['u','d','l','r']

        # Initialize q-table
        self.q_table = np.zeros((self.width*self.height,4))


    def convert_pos(self, p):
        '''
        Convert position [x,y] to index (1d)
        '''
        return (p[0] * self.width) + p[1]

    def learn(self):
        '''
        Perfome simple Q-learning algorithm
        '''
        print('start learning')
        r_list = []
        finish_counter = 0
        for i in range(self.n_episodes):
            game = Game()
            episode_reward = 0
            j = 0
            s = [0,0]
            while j < self.sub_steps:
                j += 1

                # state is 1d index in q-table
                s = self.convert_pos(s)
                
                # Determine action based on q_table + random noise
                a = np.argmax(
                        self.q_table[s,:] +
                        np.random.randn(1,len(self.action_list))
                        *(1.0/(i+1)))

                # take step on board
                tmp = game.update_board(self.action_list[a])

                # New gamestate
                s1 = tmp['position']

                # reward
                r = self.r_table[tmp['status']]

                # If next state is not in q_table, give -1 penalty
                if self.convert_pos(s1) < len(self.q_table):
                    # update q-table
                    self.q_table[s,a] = self.q_table[s,a] + self.lr * ( r + self.y *
                            np.max(self.q_table[self.convert_pos(s1),:] - self.q_table[s,a]))
                else:
                    self.q_table[s,a] = self.q_table[s,a] + self.lr * ( r + self.y *
                            (-1) - self.q_table[s,a])

                # update reward per episode
                episode_reward += r

                # new state -> old state
                s = s1

                # break out when dead
                if tmp['status'] == 'D':
                    break
                if tmp['status'] == 'F':
                    finish_counter += 1
                    break

            # append episode score
            r_list.append(episode_reward)

def main():

    q_learn = Q_learning()
    q_learn.learn()

if __name__ == '__main__':
    main()
