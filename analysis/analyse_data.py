'''
##########################
Analyse data
------------
author        : T de Klijn
created       : 2018-08-09
last modified : 2018-08-09
##########################
Analyse data 
'''

import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import brewer, Category20

def plot_dead_alive(files):

    # Find all datagiles
    data_dir = 'data/'
    files = [data_dir + f for f in files if 'survival' in f]

    # Plot width / height
    p_width = 1200
    p_height = 400
    
    # first file for getting some values
    df = pd.read_csv(files[0])

    # Generate ranges
    eps = df['episode'].size
    x_r = (0, int(eps/8))
    y_r = (-1,2)

    # Create plot
    p = figure(x_range=x_r, y_range=y_r, 
        plot_width=p_width, plot_height=p_height)

    # customize plot
    p.title.text = 'Survival per Episode'
    p.xaxis.axis_label = 'Episode'
    p.yaxis.axis_label = 'Survival'

    # Change y tick labels
    p.yaxis.ticker = [0,1]
    p.yaxis.major_label_overrides = {0 : 'A', 1: 'D'}
    p.yaxis.minor_tick_line_color = None

    # remove grd
    p.xgrid.visible = False
    p.ygrid.visible = False

    color = brewer['PuBu'][len(files)]

    for i,f in enumerate(files):

        df = pd.read_csv(f)

        # convert survival 
        df['survival'] = (df['survival']).astype(int)


        # draw line
        p.line(x=df['episode'],y=df['survival'],
                line_width = 4, color = color[i])

    return p

def conv_np_where_to_usefull(n):

    return [n[0][0] + 0.5, n[1][0] + 0.5]

def create_board_player(p,b):
    '''
    Plot image of board with player trajectory on it
    '''

    # Plot width/height
    p_width = 600
    p_height = 600

    # Read shortest path
    df_path = pd.read_csv(p)

    # Read and plot board
    empty_board = np.load(b)

    # Find components of the board
    start = np.where(empty_board == 2)
    finish = np.where(empty_board == 3)
    wall = np.where(empty_board == -10)

    # modify coordinates to + 0.5 
    wall_x = np.array(wall[0])+0.5
    wall_y = np.array(wall[1])+0.5
    start = conv_np_where_to_usefull(start)
    finish = conv_np_where_to_usefull(finish)

    # modify path to +0.5
    x = df_path['x']+0.5
    y = df_path['y']+0.5

    # Find board with and height
    board_shape = np.shape(empty_board)
    x_r = (0,board_shape[0])
    y_r = (0,board_shape[1])

    # Size of glyphs in scatter plot
    path_size = int((p_width / x_r[1])*0.6)
    square_size = int(p_width/ x_r[1]*0.9)

    # Create figure
    p = figure(x_range=x_r, y_range=y_r, 
            plot_width=p_width, plot_height=p_height)

    # style info - remove all axes
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None 
    p.xaxis.axis_line_color = None
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.xaxis.major_label_text_font_size = '0pt'
    p.yaxis.axis_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_font_size = '0pt'

    p.title.text = 'Shortest Path'

    # Plot wall
    p.scatter(x=wall_x,y=wall_y, size=square_size, color='black', marker='square')

    # Plot Start
    p.scatter(x=start[0],y=start[1],size=square_size, color='navy', alpha=0.5,
            marker='square')
    # Finish
    p.scatter(x=finish[0],y=finish[1],size=square_size, color='green', alpha=0.5,
            marker='square')

    # Plot Path
    p.scatter(x=x, y=y, size=path_size) 

    return p

