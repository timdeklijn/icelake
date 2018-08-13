from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Tabs
from bokeh.layouts import row, column
from bokeh.plotting import show, figure

from analysis import analyse_data
import os
import numpy as np

data_dir = 'data/'
files = os.listdir(data_dir)

board_list = np.sort([data_dir + f for f in files if 'board' in f])
path_list = np.sort([data_dir + f for f in files if 'path' in f]) 

boards = []
for i, board in enumerate(board_list):
    boards.append(analyse_data.create_board_player(path_list[i],board))

t2 = analyse_data.plot_dead_alive(files)

board_row = row(boards)
t2_row = row(t2)

layout = column(board_row, t2_row)

curdoc().add_root(layout)
curdoc().title = 'Q-Learning'
