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

from bokeh.plotting import figure, show, figure
from bokeh.palettes import brewer, Category20
from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import RadioGroup
from bokeh.models import ColumnDataSource

C_LIST = ['navy', 'green', 'orange']

data_file = 'data/meta_data.csv'
df = pd.read_csv(data_file)

print(df.head())

size_options = [i for i in df['size'].unique()]

source_hist = ColumnDataSource(data=dict(size=[],first_succes=[]))

def first_succes_histo():

    p = figure(title='Fist succes histograms')

    p.vbar(x='first_succes',top='size', source=source_hist,width=0.005)

    return p

def bs():

    for dat in histo_plot.data['hist']:
        print(dat)
        p.quad(top=dat[0], bottom=0, left=dat[1][:-1],
                right=dat[1][1:],fill_color=C_LIST[i])

#        p.quad(top=hist, bottom=0, left=edges[:-1],
#                right=edges[1:],fill_color=C_LIST[i], line_color='black',
#                legend=f'size: {s}')
#
    p.xaxis.axis_label = 'First Succesfull Attampt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label = None
    p.yaxis.major_label_text_font_size = '0pt'

    return p

def prep_histo():

    unique = np.unique(source_hist.data['size'])
    hist_l = []
    for i in unique:
        hist, edges = np.histogram(source_hist.data['first_succes'][np.where(source_hist.data['size'] == i)], density=True,
                bins=3)
        hist_l.append([hist,edges])
    d_tmp = {'hist' : hist_l}
    df_tmp = pd.DataFrame(d_tmp)
    histo_plot.data = histo_plot.from_df(df_tmp[['hist']])

def first_succes_radiogroup():

    return RadioGroup(labels = [str(i) for i in size_options], active =0)

def sliced_copy_df_size(df, selection):

    if not isinstance(selection, list):
        selection = [selection]
    for i in size_options:
        if i not in selection:
            df.drop(df[df['size'] == i].index, inplace=True)
    return df

def radiogroup_handler(new):
    update()

def update():
    '''
    Something changes, update plots
    '''
    # Update size selectoin based on checkboxgroup
    size = w_succes.active
    size = size_options[size]
    # Create data frame for histogram plotting
    df_c = df.copy()
    df_c = sliced_copy_df_size(df_c, size)
    # Place in source_column_data
    source_hist.data = source_hist.from_df(df_c[['size','first_succes']])

# Initialize page elements
w_succes = first_succes_radiogroup()
p1 = first_succes_histo()

# Set interactions
w_succes.on_click(radiogroup_handler)

# Init data
update()

# Initialize Layour
layout = row(p1,widgetbox(w_succes))

# Add elements and title
curdoc().add_root(layout)
curdoc().title = 'Meta Learning'
