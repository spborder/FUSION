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

import dash_leaflet as dl

from skimage import draw
from scipy import ndimage

from sklearn.cluster import DBSCAN

from umap import UMAP

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
                        customdata = feature_data[custom_col] if not custom_col is None else None,
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

def gen_umap(feature_data, feature_cols, label_and_custom_cols):
    """
    Generating umap embeddings dataframe from input data
    """

    quant_data = feature_data.loc[:,[i for i in feature_cols if i in feature_data.columns]].values
    label_data = feature_data.loc[:,[i for i in label_and_custom_cols if i in feature_data.columns]]

    feature_data_means = np.nanmean(quant_data,axis=0)
    feature_data_stds = np.nanstd(quant_data,axis=0)

    scaled_data = (quant_data-feature_data_means)/feature_data_stds
    scaled_data[np.isnan(scaled_data)] = 0.0
    scaled_data[~np.isfinite(scaled_data)] = 0.0

    umap_reducer = UMAP()
    embeddings = umap_reducer.fit_transform(scaled_data)
    umap_df = pd.DataFrame(data = embeddings, columns = ['UMAP1','UMAP2'],index = label_data.index)

    umap_df = pd.concat((umap_df,label_data),axis=1)
    umap_df.columns = ['UMAP1','UMAP2'] + label_and_custom_cols

    return umap_df

def gen_clusters(feature_data, feature_cols, eps = 0.3, min_samples = 10):
    """
    Implementation of DBSCAN for generating cluster labels and noise labels
    """

    if feature_data.shape[0]<min_samples:
        return None

    quant_data = feature_data.loc[:,[i for i in feature_cols if i in feature_data.columns]].values
    #label_data = feature_data.loc[:,[i for i in label_and_custom_cols if i in feature_data.columns]]

    feature_data_means = np.nanmean(quant_data,axis=0)
    feature_data_stds = np.nanstd(quant_data,axis=0)

    scaled_data = (quant_data-feature_data_means)/feature_data_stds
    scaled_data[np.isnan(scaled_data)] = 0.0
    scaled_data[~np.isfinite(scaled_data)] = 0.0

    clusterer = DBSCAN(eps = eps, min_samples = min_samples).fit(scaled_data)
    cluster_labels = clusterer.labels_

    string_labels = [f'Cluster {i}' if not i==-1 else 'Noise' for i in cluster_labels]

    return string_labels

def process_filters(input_keys,input_values,input_styles,cell_names_key=None):
    """
    Taking keys, values, and styles and returning a list of filter dictionaries that can be used to remove unwanted FTUs
    """
    filter_list = []
    for prop,style,values in zip(input_keys,input_styles,input_values):
        if not style['display'] is None:
            
            if '-->' in prop:
                filter_parts = prop.split(' --> ')
                m_prop = filter_parts[0]
                val = filter_parts[1]
            else:
                m_prop = prop
                val = None

            if m_prop=='Max Cell Type':
                m_prop = 'Main_Cell_Types'
                val = 'max'

            if cell_names_key:
                if val in cell_names_key:
                    val = cell_names_key[val]

            filter_list.append({
                'name': m_prop,
                'value': val,
                'sub_value': None,
                'range': values
            })

    return filter_list

# Taken from plotly image annotation tutorial: https://dash.plotly.com/annotations#changing-the-style-of-annotations
def path_to_indices(path):
    """
    From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)

def path_to_mask(path, shape):
    """
    From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)

    # Clipping values for rows and columns to "shape" (annotations on the edge are counted as dimension+1)
    rr = np.clip(rr,a_min=0,a_max=int(shape[0]-1))
    cc = np.clip(cc,a_min=0,a_max=int(shape[1]-1))

    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

def make_marker_geojson(bbox_list,convert_coords = True, wsi = None):
    """
    Make GeoJSON-formatted markers where the coordinates are equal to the center point of the bounding boxes
    
    use convert_coords = True and provide WSI if giving bounding boxes in terms of slide coordinates as the
    GeoJSON has to be rendered in map coordinates

    bounding boxes should be in [minx, miny, maxx, maxy] format 
    """

    marker_geojson = {
        'type':'FeatureCollection',
        'features': []
    }
    marker_list = []

    for bbox in bbox_list:

        # Converting the original coordinates (if using map)
        #if convert_coords:
        #    bbox = wsi.convert_slide_coords([[bbox[0],bbox[1]],[bbox[2],bbox[3]]])
        #    bbox = [bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]

        # Finding average of extrema
        bbox_center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]

        marker_geojson['features'].append({
            'type': 'Feature',
            'properties':{'type':'marker'},
            'geometry':{'type':'Point','coordinates': bbox_center}
        })
        
        marker_list.append(
            dl.Marker(position = bbox_center[::-1])
        )

    return marker_geojson, marker_list



