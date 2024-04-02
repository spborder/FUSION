"""

Various utility functions used in FUSION

"""

import os
import sys
import numpy as np

import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import textwrap



def get_pattern_matching_value(input_val):
    """
    Used to extract usable values from components generated using pattern-matching syntax.
    """
    if type(input_val)==list:
        if len(input_val)>0:
            return_val = input_val[0]
        else:
            return_val = None
    elif input_val is None:
        return_val = input_val
    else:
        return_val = input_val

    return return_val


def extract_overlay_value(structure_list,overlay_prop):
    """
    Used to pull out raw value to add to list used to generate heatmaps/colorbars
    """
    raw_values_list = []
    if not overlay_prop['name'] is None:
        for st in structure_list:
            if overlay_prop['name'] in st:
                if not overlay_prop['value'] is None:
                    if overlay_prop['value'] in st[overlay_prop['name']]:
                        if not overlay_prop['sub_value'] is None:
                            if overlay_prop['sub_value'] in st[overlay_prop['name']][overlay_prop['value']]:
                                raw_values_list.append(float(st[overlay_prop['name']][overlay_prop['value']][overlay_prop['sub_value']]))
                        else:
                            raw_values_list.append(float(st[overlay_prop['name']][overlay_prop['value']]))
                    elif overlay_prop['value']=='max':
                        # Getting the max index of st[overlay_prop['name']][overlay_prop['value']]
                        check_values = list(st[overlay_prop['name']].values())
                        raw_values_list.append(np.argmax(check_values))
                else:
                    raw_values_list.append(float(st[overlay_prop['name']]))
    
    return raw_values_list

def gen_violin_plot(feature_data, label_col, label_name, feature_col, custom_col):
    """
    Generate a violin plot for a single feature across one/multiple labels including customdata    
    """
    figure = go.Figure(data = go.Violin(
                        x = feature_data[label_col],
                        y = feature_data[feature_col],
                        customdata = feature_data[custom_col],
                        points = 'all',
                        pointpos=0
                    ))
    figure.update_layout(
        legend = dict(
            orientation='h',
            y = 0,
            yanchor='top',
            xanchor='left'
        ),
        title = '<br>'.join(
            textwrap.wrap(
                f'{feature_col}',
                width=30
            )
        ),
        yaxis_title = dict(
            text = '<br>'.join(
                textwrap.wrap(
                    f'{feature_col}',
                    width=15
                )
            ),
            font = dict(size = 10)
        ),
        xaxis_title = dict(
            text = '<br>'.join(
                textwrap.wrap(
                    label_name,
                    width=15
                )
            ),
            font = dict(size = 10)
        ),
        margin = {'r':0,'b':25}
    )

    return figure










































