'''
##########################
Global params
--------------
author        : T de Klijn
created       : 2018-08-07
last modified : 2018-08-07
##########################
'''

class Global:
    '''
    * Store global variables for the Game class
    '''

    # Start with a new board?
    new_board = False

    # Global board parameters
    width                  = 5                 # Board with
    height                 = 5                 # Board height
    danger_number          = 5                 # number of danger tiles
    start_x, start_y       = 0,0               # Entrance of maze
    position_x, position_y = start_x, start_y  # Player starting position
    finish_x, finish_y     = width-1, height-1 # Exit of maze

    # Display features
    player_icon = 'P'
    player_val  =  5
    start_val   =  2
    start_icon  = 'S'
    finish_icon = 'E'
    finish_val  =  3
    danger_icon = 'D'
    danger_val  = -10
    elem_icon   = ' ' 
    elem_val    =  1

