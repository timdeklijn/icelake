'''
##########################
Global params
--------------
author        : T de Klijn
created       : 2018-08-07
last modified : 2018-08-08
##########################
Store global variables for the Game class
This can also be used by machine learning algorithms
'''

# Start with a new board?
new_board = False

# Global board parameters
width                  = 4                 # Board with
height                 = 4                 # Board height
danger_number          = 4                 # number of danger tiles
start_x, start_y       = 0,0               # Entrance of maze
position_x, position_y = start_x, start_y  # Player starting position

show_board = True

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

