'''
##########################
Show MDP
--------
author        : T de Klijn
created       : 2018-08-13
last modified : 2018-08-15
##########################
Load the nearal network model created in train, run through game once, save
everything to be displayed later
'''
import numpy as np

from game.Game import Game
from MDP.train import build_model, observe_board, Experience

def run():
    '''
    Run through the board until the finish is reached. Only listen to the
    model/policy
    '''
    # Needed for coupling to Experience class
    max_memory = 1000
    # possible actions for game and the amount of actions
    actions = ['u', 'd', 'l', 'r']
    num_actions = len(actions)
    # init game
    game = Game()
    # Build model
    model = build_model(game)
    # Load weights into model
    weights_file = 'model.h5'
    # Load weights from previous session
    print('Loading weights from file %s'.format(weights_file))
    model.load_weights(weights_file)
    # predict is part of the experience class
    experience = Experience(model, max_memory=max_memory)
    # Get current state
    envstate = observe_board(game.board, game.position_x, game.position_y)
    # Run until game over 
    game_over = False
    # Save coordinates 
    trajectory = []
    while game_over == False:
        # Get possible actions
        valid_actions = actions
        # Set new state to old
        prev_envstate = envstate
        # get action from model
        action = np.argmax(experience.predict(prev_envstate))
        # Perform action
        game_status = game.update_board(actions[action])
        # Append trajectory 
        trajectory.append([game.position_x, game.position_y])
        # new state
        envstate = observe_board(game.board, game.position_x,
                game.position_y)
        # Check if episode is over
        if game_status == 'F':
            game = Game()
            game_over = True
        elif game_status == 'D':
            game = Game()
            game_over = True
        else:
            game_over = False
    # Trajectory converted to string
    traj = '00' + coord_to_string(trajectory)
    # Danger coordinates converted to string
    tmp = np.where(game.board == -10)
    dng = coord_to_string(list(zip(list(tmp[0]),list(tmp[1]))))
    print(traj,'\n',dng)

def coord_to_string(crd) -> str:
    '''
    Convert a list of coordinates to a string with numbers
    '''
    s = ''
    # Iterate over list of coordinates
    for r in crd:
        for c in r:
            s+=str(c)
    return s 

if __name__ == '__main__':
    run()

