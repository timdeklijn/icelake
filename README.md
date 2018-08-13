# Icelake

## Goal

Create a simple game with a playingboard with a beginning and an end. Other
tiles can be either a hole or not. A player will walk aling the board, en can
die by stepping of the board or on a hole. The game is finished when the player
is dead or when the player finished. Create AI to walk over the board and find
the optimal path.

## Game

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

The game will move the player, update the board and return a status which can
be alive ('A'), dead ('D') or finished ('F').

## Simple Q-learning


## Meta learning


## Markov Decision Process

### Markov process

a state (St) has the Markov property if:

    P[St+1 | St] = P[St+1 | S1,....,St]

So all relevant history is captured in the current state.

The probability to transition form Markov state (s) to a successor state (S')
is defined transition function:

    P_ss' = P[S_t+1 = s' | S_t = s]

A probability distribution over next possible successor states, given the
current state. So each possible 'action' (move to next state) has a
probability.

The transition function can be rewritten to a matrix formulation where each row
sums to 1:

             | P11 . . . P1n |
    P = from | ... . . .     |
             | Pn1 . . . Pnn |

A markov process is memory-LESS and random. It is a sequence of random states
S1, S2, ... with the Markov property. A Markov provcess or chain is a tuple
(S,P) on the state space S, and the transition function P. The dynamics of the
system are defined by these two components. Sampling from a MDP, we end up with
a sequence of states, or an episode.

### Markov reward process

To prevent all state transitions to be random a reward should be added, this
will lead to actual judgement during the learning phase. This is also called a
Markov Reward Process (MRP).

MRP depends on two extra parameters compared to the MDP and can be defined as
(S,P,R,y) (y=gamma). S is an infinite state space, P is the transition
probability function and R is the reward function:

	R_s = E[R_t +1 | St = S]

The function R determines the immediate reward to be expected to get from going
to a certain state.

A discount factor is added, to make the agent care more about the total reward
compare to immediate reward. The goal is to maximize the return.

	Gt = Sum\_k=0 y^k R\_t+k+1

y is between 0 and 1. y=0 leads to a short sighted agent, while y=1 ignores
immediate and only cares about future rewards. Moreover, adding a discount
factor will garantee a convergence in the Markov Process. y = 1 can be
simplified to:

	Gt = Sun\_k=0 y^k = 1 / (1-y)

When there is no information about the future, y will help guide the process
towards a more immediate reward. it effectively downscales the problem.

The value function informs on how optimal each state/action is. The value
function informs the agent on the reward of a paticular action or state.

The state-value function can be defined as:

	v(s) = E[G_t | S_t = s]

So for a particular episode (multiple states and decisions) it will return the
total reward all states and transitions.

### Bellman Equation

An agent will always attempt to get the maximum reward from every state.
Therefor the value-function should be optimized. This will lead to the maximum
sum of cumulative rewards.

The Bellman equation will decompose the value function into two parts; it
effectively splits it in an immediate reward part (R\_t+1), and the discounted
value of the successor state (yV(S\_t+1)):

	v(s) = E[G_t | S_t = s]

Expanding Gt results in:

	v(s) = E[R\_t+1 + yR\_t+2 + y^2R\_t+3 + ... | S_t = s]
	     = E[R\_t+1 + y(R\_t+2 + yR\_t+3 + ...) | S_t = s]

The replace the expanded part for states bigger then +1 by the discount funtion.

	     = E[R\_t+1 + yG\_t+1 | S\_t = s ]

Since the value-function (v(s)) is linear, the expected value for G\_t+1 =
S\_t+1.

	     = E[R\_t+1 + yv(s\_t+1) | S\_t = s]

This is the final Bellman Equation:


	v(s) = E[R\_t+1 + yv(s\_t+1) | S\_t = s]

Or rewritten as:

	v(s) = R\_s + y Sum_\s'->S P\_ss' v(s')

Simply said: the reward at a current state, is the current state reward plus
the reward of the next possible states multiplied by the probability of going
to the next states.

The Bellman equation is linear and can be solved directly:

	      v = R + yPv
	(l-yP)v = R
	      v = (1-yP)^-1 R

The computation is correlated to the number of states with n^3. So MRP's should
be relatively small.

### Markov Decision Process

The Markov Decision Process adds additional parameters to the MRP, and is
dependend on (S,A,P,R,y). S is still the space state, A is a set of actions, P
is the state transition probability function:

	P\_ss'^a = P[S\_t+1 = s' | S\_t = s, A\_t = a]

The reward function can be written as:

	R\_s^a = E[R\_t+1 | S\_t = s, A\_t = a]

Again, the discount value y is between 0 and 1.

A policy (pi) is a distribution over actions (given states) and defines the
behaviour of an agent (ONLY the policy defines the behaviour).

	pi(a|s) = P[A\_t = a | S\_t = s]

Policies ONLY depend on the current state, so at a particular state, a policy
will make the same decision everytime. The can be made stochastic to encourage
exploration of state space.

Both the Markov Process and the Markov Reward Process can be recoverd form a
policy. A Markov Process is just the state sequences determined by a policy.
The MRP is the state/reward sequence (S1,R1,S2,R2,...) given a certain policy:

	P\_s,s'^pi = Sum\_a->A pi(a|s)P\_ss'^a

Averaging over all possible outcoms / state transitions depending on a policy
pi results in a transition dynamics function.

We have a probalbility to take an action a under policy pi from state S. We
multiply the probability that we take the action with what would happen in
successor states.

Also the reward function is dependent on the policy pi:

	R\_s^pi = Sum\_a -> A pi(a|s)R\_s^a

The above results in a policy that will choose the optimal path through a
Markov process.

The value function determines how good it is to be in state S based on all
actions sampled by the policy.

	v\_pi(s) = E\_pi[G\_t | S\_t = s] = E\_pi [Sum\_k=0 y^k R\_t+k+1 |
	S\_t=s], for all s in S

The result will determine how good it is to take an action from a particular
state.

    q\_pi(s,a) = E\_pi[G\_t | S\_t = s, A\_t = a] = E\_pi [ Sum\_k=0 y^k
    R\_t+k+1 | S\_t = s, A\_t = a]


