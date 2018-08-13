'''
##########################
Frozen lake game
----------------
author        : T de Klijn
created       : 2018-08-06
last modified : 2018-08-08
##########################
Classic frozen lake game. Genarate a 'board' with a start, finish and holes in
the ice. This is done random and based on width, hight and numeber of dangers.
The board can be displayed in the terminal. A move input for 'update_board' can
be left (l), right (r), up (u) or down (d). The after every board_update, the
class well return a status which is alive (A), dead (D) or finished (F). Other
parameters such as player position (x or y) can be extracted from the class as
well. 
'''

import numpy as np
import global_param
from global_param import Global

class Game(Global):
    '''
    !!!! Global parameters in global_param.py !!!!

    * Create a board with dangers (D) a start (S) and an exit (E)
        an existing board can also be loaded
    * let the player (P) begin at start and based on move inputs walk towards
        the finish (or not)
    * returns the status of the playes as a string: alive (A), dead (D) or finished (F)
    '''

    def __init__(self):
        '''
        Initialze board class
        '''
       
        # initialize finish based on gloabel width and height
        self.finish_x, self.finish_y = Global.width-1, Global.height-1 

        # A player will start the game being alive 'A'
        self.alive = True

        self.size = self.width * self.height

        # Create a new board with dangers placed
        if self.new_board:
            self._create_danger_list()
            self.create_board()
            self.place_start_finish()
            self.place_danger()
            self.save_board()

        # Load an existing board
        else:
            self.load_board()

        # Setup board
        self.place_player()
        if self.show_board:
            print('\n'+10*'<'+'  START  ' + 10*'>' + '\n')
            self.display_board()

    def update_board(self, move: str) -> str:
        '''
        * Update player position based on move input
        * check if player is alive
        * redraw board
        * Return status alive (A), dead (D) or finish (F)
        '''
        self.update_player_position(move)
        self.place_start_finish()
        self.place_danger()
        self.player_alive()
        if self.alive: 
            self.place_player()
        if self.show_board:
            self.display_board()
        if not self.alive:
            status = 'D'
        elif self.player_at_finish():
            status = 'F'
        else:
            status = 'A'       
        return status

    def create_board(self):
        '''
        returns a numpy zero array with width and height
        '''
        self.board = np.ones((self.width,self.height), dtype = int)

    def _mkstr(self,icon : str) -> str:
        '''
        Construct line element for terminal printing
        '''
        space = '  '
        return space + str(icon) +  space + '|' 
        
    def display_board(self):
        '''
        Print the board in its current state
        * row by row check the element value in board array and
            apply drawing rules
        '''
        brd = []
        for row in self.board:
            s = '|' 
            for elem in row:
                if elem == self.elem_val:
                    s += self._mkstr(self.elem_icon) 
                elif elem == self.danger_val:
                    s += self._mkstr(self.danger_icon)
                elif elem == self.start_val:
                    s += self._mkstr(self.start_icon)
                elif elem == self.finish_val:
                    s += self._mkstr(self.finish_icon)
                elif elem == self.player_val:
                    s += self._mkstr(self.player_icon)
                else:
                    raise Exception('Problem constructing board')
            brd.append(s)
            brd.append(len(s)*'-')
        brd = [(len(s)*'-')] + brd
        for l in brd: print(l)   
        
    def place_start_finish(self):
        '''
        Place start and finish icons on the board
        '''
        self.board[self.start_x, self.start_y]   = self.start_val
        self.board[self.finish_x, self.finish_y] = self.finish_val

    def _create_danger_list(self):
        '''
        Make a randeom list of coordinates of danger locations
        cannot contain duplicates, start or finish position`
        '''
        self.danger_list = []
        strt = [self.start_x, self.start_y]
        fnsh = [self.finish_x, self.finish_y]
        while len(self.danger_list) < self.danger_number:
                elem = [np.random.randint(self.width-1),
                        np.random.randint(self.height-1)]
                if elem not in self.danger_list and elem != strt and elem != fnsh:
                    self.danger_list.append(elem)
                else:
                    continue
        self.danger_list = np.array(self.danger_list)

    def place_danger(self):
        '''
        Create the danger locations on the board
        ''' 
        for i in self.danger_list: 
            self.board[i[0],i[1]] = self.danger_val

    def place_player(self):
        '''
        Place player its loaction
        '''
        self.board[self.position_x, self.position_y] = self.player_val

    def update_player_position(self, move):
        '''
        Based on move, update player location
        ''' 
        old_x, old_y = self.position_x, self.position_y
        if   move == 'l': self.position_y -=1
        elif move == 'r': self.position_y +=1
        elif move == 'u': self.position_x -=1
        elif move == 'd': self.position_x +=1
        else:
            raise Exception('Incorrect input for player movement')
        # reset old player position
        self.board[old_x,old_y] = self.elem_val
        
    def player_alive(self) -> bool:
        '''
        Check new player position and return False if it 
        is equal to a danger position or outside of board
        '''
        # If out of bounds
        if self.position_x < 0 or self.position_x > self.height-1 or self.position_y < 0 or self.position_y > self.width-1:
            self.alive = False
            return False
        # If on a danger field
        elif self.board[self.position_x, self.position_y] == self.danger_val:
            self.alive = False
            return False
        else:
            return True

    def player_at_finish(self) -> bool:
        '''
        Check if player is at finish, else return false
        '''
        if self.position_x == self.finish_x and self.position_y == self.finish_y:
            return True
        else:
            return False

    def load_board(self):
        '''
        Load board from numpy array and exctract danger_list
        '''
        self.board = np.load('board.npy')
        tmp =  np.where(self.board == self.danger_val)
        self.danger_list = np.array(list(map(lambda x,y: [x,y], tmp[0],
            tmp[1])))

    def save_board(self):
        '''
        write board as a numpy array
        '''
        np.save('board.npy', self.board)
