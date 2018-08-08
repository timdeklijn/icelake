'''
##########################
Vizualize Trajectory
--------------------
author        : T de Klijn
created       : 2018-08-08
last modified : 2018-08-08
##########################
NOT FINISHED

Visualize a trajectory
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def trajectory_movie(data):
    '''
    Create animation from board arrays using matplotlib
    '''
    # Start figure
    fig = plt.figure()

    # will be filled with [imshow] objects
    image_list = []
     
    # Loop over arrays
    for image in data:
        image_list.append([plt.imshow(image, animated=True)])

    # Animate 
    ani = animation.ArtistAnimation(fig, image_list, interval=200, blit=True,
            repeat_delay=500)

    # Show
    plt.show()

data = []
for i in range(10):
    data.append(np.random.randn(8,8))

trajectory_movie(data)
