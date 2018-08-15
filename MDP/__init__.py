import sys
from game.global_param import *
from MDP import train 

in_list = ['train', 'show']

print('Markov decision Proces module')

if 'train' in sys.argv[:]:
    print(width,height)
    train.run()
elif 'show' in sys.argv[:]:
    show.run()
else:
    print(f'Give input: {in_list}')
