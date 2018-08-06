from Game import Board
from time import sleep

def main():
    game = Board()
    game.update_board('R')
    sleep(1)
    game.update_board('R')
    sleep(1)
    game.update_board('R')
    sleep(1)
    game.update_board('R')
    sleep(1)
    game.update_board('R')
    sleep(1)

if __name__ == '__main__':
    main()
