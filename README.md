# Goal

Create a simple game with a playingboard with a beginning and an end. Other
tiles can be either a hole or not. A player will walk aling the board, en can
die by stepping of the board or on a hole. The game is finished when the player
is dead or when the player finished. Create AI to walk over the board and find
the optimal path.

The game can be found in Game.py, there is a class Game. The class can be
initiated by:

```
from Game import Game
game = Game ()
```

A player move can be inputted by handing eiter up ('u'), down ('d'), left ('l')
or right ('r') to update\_board

```
move = 'u'
status = game.update_board(move)
```

The game will move the playes, update the board and return a status which can
be alive ('A'), dead ('D') or finished ('F').


