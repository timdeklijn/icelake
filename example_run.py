from Game import Game
from time import sleep

def main():
    game = Game()
    moves = ['d','d','d','d','r','r','r','r']
    status_list = []
    for move in moves:
        status = game.update_board(move)
        status_list.append(status)
    print(f'\n{moves}')
    print('\n' + str(status_list))

if __name__ == '__main__':
    main()
