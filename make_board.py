import sys

from game.Game import Game
from game.global_param import *
from game import global_param

def new_board():
    '''
    Create a random new board based on global parameters and save it
    '''
    show_board = True
    new_board = True
    global_param.new_board = True
    game = Game()
    game._create_danger_list()
    game.create_board()
    game.place_start_finish()
    game.place_danger()
    game.save_board()
    game.display_board()

    print('New board created and saved')


def run(args):
    if not 'custom' in args:
      new_board()


if __name__ == '__main__':
    run(sys.argv[:])
