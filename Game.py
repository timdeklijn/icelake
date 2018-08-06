import numpy as np

class Board(object):
    '''
    Create a board, place player, update when input was given
    '''

    def __init__(self):
        '''
        Initialze board class
        '''
         # Display features
        self.player_icon = 'P'
        self.start_icon  = 'S'
        self.finish_icon = 'E'
        self.danger_icon = 'D'
        
        # Board params
        self.width = 10
        self.height = 10
        self.danger_number = 20
        self.position_x, self.position_y = 0,0
        self.start_x, self.start_y       = 0,0
        self.finish_x, self.finish_y     = self.width-1, self.height-1

        # Initialize board
        self.create_board()
        self.place_start_finish()
        self.place_danger()
        self.display_board()
        self.place_player()
        
    def create_board(self):
        '''
        returns a numpy zero array with width and height
        '''
        self.board = np.zeros((self.width,self.height), dtype = str)

    def display_board(self):
        '''
        Print the board in its current state
        '''
        print('\n\n')
        brd = []
        space = '  '
        for row in self.board:
            s = '|' 
            for elem in row:
                if elem == '': elem = '0'
                s += space + str(elem) +  space + '|' 
            brd.append(s)
            brd.append(len(s)*'-')
        brd = [(len(s)*'-')] + brd
        for l in brd: print(l)   
        
    def place_start_finish(self):
        '''
        Place start and finish icons on the board
        '''
        self.board[self.start_x, self.start_y] = self.start_icon
        self.board[self.finish_x, self.finish_y] = self.finish_icon

    def place_danger(self):
        '''
        Create X number of dangers on the board
        ''' 
        dangers = [ [ np.random.randint(self.width-1),
                    np.random.randint(self.height-1)] for i in
                    range(self.danger_number) ]
        for i in dangers: 
            self.board[i[0],i[1]] = self.danger_icon

    def place_player(self):
        '''
        Place player its loaction
        '''
        self.board[self.position_x, self.position_y] = self.player_icon

    def update_player_position(self, move):
        '''
        Based on move, update player location
        ''' 
        if move == 'l': self.position_x -=1
        if move == 'r': self.position_x +=1
        if move == 'u': self.position_y +=1
        if move == 'd': self.position_y -=1

    def player_alive(self) -> bool:
        '''
        Check new player position and return False if it is equal to a danger
        '''
        if self.board[self.position_x, self.position_y] == self.danger_icon:
            return False
        else:
            return True

    def update_board(self, move: str):
        '''
        Update player position based on move input
        check if player is alive
        redraw board
        '''
        self.update_player_position(move)
        if not self.player_alive():
            print('XXXXXXX DIED XXXXXXX')
        else:
            self.place_player()
            self.display_board()
