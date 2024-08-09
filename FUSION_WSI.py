"""

External file to hold the WholeSlide class used in FUSION


"""
import os
import numpy as np

from shapely.geometry import shape, Polygon, box
import random
import json
import geojson
import shutil
import requests

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc
import dash_bootstrap_components as dbc
from dash_extensions.enrich import html

from typing_extensions import Union

import geopandas
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function

from tqdm import tqdm
from FUSION_Utils import (
    extract_overlay_value,
    gen_violin_plot,
    gen_umap)

#NOTE: All spatial indices will be in (x,y)/(row,column)/(horizontal,vertical)/(width,height) format for consistency

class SlideHandler:
    """
    More generalized handler for slide-related methods. Takes slide_info_store as input into most
    functions which will contain the following information at a minimum:
        - "slide_info": (/item/{id}), item info from Girder such as _id, folderId, metadata, etc.
        - "tiles_metadata": (/item/{id}/tiles), large-image metadata for image item such as sizeX, sizeY, frames, etc.
        - "slide_type": (/item/{id} ["meta"]["Spatial Omics Type"]), type of spatial --omics such as Visium, CODEX, Xenium, or Regular
        - "annotations": (/annotation?itemID={id}?limit=0), annotation ids available for this item
        - "overlay_prop": property to use to generate structure overlay heatmaps
        - "cell_vis_val": transparency value for overlaid property heatmaps
        - "filter_vals": list of properties and values to filter visible structures
        - "current_channels": list of overlaid channels to include and their colors (CODEX only)
    """
    def __init__(
            self,
            girder_handler
    ):
        self.girder_handler = girder_handler

        #TODO: cement the schema here for more generalizability
        # Validated properties for overlaid visualizations
        self.visualization_properties = [
        'Main_Cell_Types',
        'Gene Counts',
        'Morphometrics',
        'Cluster',
        'Cell_Subtypes',
        'Cell Label',
        'Cell Type',
        'Pathway Expression',
        'Transcript Counts'
        ]

    def __str__(self):
        return f'SlideHandler object for {self.girder_handler.apiUrl}'

    def get_slide_map_data(self, item_id:str, user_info:dict):
        # Getting all of the necessary materials for loading a new slide

        # Get resource tile metadata
        tile_metadata = self.girder_handler.get_tile_metadata(item_id)
        # Get item info
        item_info = self.girder_handler.get_item_info(item_id)

        # Calculate scale to apply to annotations and for setting nativeZoom
        zoom_levels = tile_metadata['levels']

        # Zoom levels increase spatial dimensions by 2**(level), base_dims therefore
        # determines the size of the smallest zoom level (0)
        base_dims = [
            tile_metadata['sizeX']/(2**(zoom_levels-1)),
            tile_metadata['sizeY']/(2**(zoom_levels-1))
        ]

        tile_dims = [
            tile_metadata['tileWidth'],
            tile_metadata['tileHeight']
        ]

        image_dims = [
            tile_metadata['sizeX'],
            tile_metadata['sizeY']
        ]
        
        x_scale = base_dims[0]/image_dims[0]
        y_scale = -(base_dims[1]/image_dims[1])

        #TODO: figure out how to actually set the map bounds (lat,lng)
        map_bounds = [-tile_dims[0], tile_dims[1]]
        
        # Grabbing available annotation ids for current item
        annotation_ids = self.girder_handler.get_available_annotation_ids(item_id)

        # Getting initial tile url(s)
        frame_names = []
        if 'Spatial Omics Type' in item_info['meta']:
            if not item_info['meta']['Spatial Omics Type']=='CODEX':
                if not 'frames' in tile_metadata:
                    tile_url = self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+user_info['token']
                else:
                    # This is for 3-channel RGB images (.ome.tif)
                    tile_url = self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+user_info['token']+'&style={"bands": [{"framedelta":0,"palette":"rgba(255,0,0,255)"},{"framedelta":1,"palette":"rgba(0,255,0,255)"},{"framedelta":2,"palette":"rgba(0,0,255,255)"}]}'
            else:
                # For CODEX items there are multiple tile_urls possible
                if 'frames' in tile_metadata:
                    tile_url = []
                    frame_names = [i['Channel'] for i in tile_metadata['frames']]
                    
                    # Checking for registered RGB/Histology image
                    if all([i in frame_names for i in ['red','green','blue']]):
                        # Setting the style of the red, green, and blue frames to go from black to red/green/blue with the right framedelta
                        rgb_style_dict = {
                            "bands": [
                                {
                                    "palette": ["rgba(0,0,0,0)",'rgba('+','.join(['255' if i==c_idx else '0' for i in range(3)]+['0'])+')'],
                                    "framedelta": frame_names.index(c)
                                }
                                for c_idx,c in enumerate(['red','green','blue'])
                            ]
                        }

                        rgb_url = self.girder_handler.gc.urlBase+f'item/{item_id}/tiles/zxy/'+'/{z}/{x}/{y}?token='+user_info["token"]+'&style='+json.dumps(rgb_style_dict)

                        frame_names = [i for i in frame_names if not i in ['red','green','blue']]
                        frame_names.append('Histology (H&E)')

                    tile_url = [
                        self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/'+'/{z}/{x}/{y}?token='+user_info["token"]+'&style={"bands": [{"palette":["rgba(0,0,0,0)","rgba(255,255,255,255)"],"framedelta":'+str(idx)+'}]}'
                        if not name=='Histology (H&E)' else rgb_url
                        for idx, name in enumerate(frame_names) 
                    ]

        else:
            tile_url = self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+user_info['token']

        # This should be all the needed information to generate TileLayers
        return_dict = {
            'tiles_metadata': tile_metadata,
            'scale': [x_scale,y_scale],
            'zoom_levels': zoom_levels,
            'tile_dims': tile_dims,
            'map_bounds': map_bounds,
            'tile_url': tile_url,
            'annotations': annotation_ids,
            'frame_names': frame_names,
            'manual_ROIs': [],
            'marked_FTUs': []
        }
        
        return return_dict
        
    def get_annotation_geojson(self, slide_info:dict, user_info: dict, idx:int):
        """
        Get annotation in GeoJSON format, apply scale factors and save locally
        """
        
        if idx>=len(slide_info['annotations']):
            print(f'Uh oh! Tried to get annotation index: {idx} when only {len(slide_info["annotations"])} are available!')
            raise IndexError

        this_annotation = slide_info['annotations'][idx]
        f_name = this_annotation['annotation']['name']

        save_path = f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}/{this_annotation["_id"]}.json'
        if not os.path.exists(save_path):
            if not os.path.exists(f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}'):
                os.makedirs(f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}')

            # Getting annotation in GeoJSON form from API
            try:
                annotation_geojson = self.girder_handler.gc.get(f'./annotation/{this_annotation["_id"]}/geojson?token={user_info["token"]}')
            except requests.exceptions.ChunkedEncodingError:
                # This means there was an error in the GeoJSON conversion (possibly on Girder's end?)
                annotation_json = self.girder_handler.gc.get(f'./annotation/{this_annotation["_id"]}?token={user_info["token"]}')

                # Converting to GeoJSON manually
                annotation_geojson = {
                    'type': 'FeatureCollection',
                    'features': [
                        {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Polygon',
                                'coordinates': [el['points']]
                            },
                            'properties': {
                                'user': el['user']
                            }
                        }
                        for el in annotation_json['annotation']['elements'] if el['type']=='polyline'
                    ]
                }

            x_scale = slide_info['scale'][0]
            y_scale = slide_info['scale'][1]
            scaled_annotation = geojson.utils.map_geometries(lambda g: geojson.utils.map_tuples(lambda c: (c[0]*x_scale,c[1]*y_scale, c[2]), g), annotation_geojson)

            if len(scaled_annotation["features"])>0:
                for f in scaled_annotation['features']:
                    f['properties']['name'] = f_name

                    if not 'user' in f['properties']:
                        f['properties']['user'] = {}

        else:
            try:
                with open(save_path,'r') as f:
                    scaled_annotation = geojson.load(f)
                    f.close()
            except json.decoder.JSONDecodeError:
                scaled_annotation = None
        
        if not scaled_annotation is None and len(scaled_annotation['features'])>0:
            # Saving geojson locally
            if not os.path.exists(save_path):
                with open(save_path,'w') as f:
                    geojson.dump(scaled_annotation,f)
                    f.close()

    def find_intersecting_ftu(self, query_poly: Union[list,Polygon], ftu: Union[str,list], slide_info:dict):
        """
        Find intersecting FTU's with a given query_poly (either boundary coordinates for a viewport or a manual ROI polygon)
        Searches for each FTU separately by given id      
        """
        if type(query_poly)==list:
            box_poly = query_poly
            query_poly = box(*box_poly)
        else:
            # Getting bounding box for manual annotation
            box_poly = list(query_poly.bounds)

        intersecting_ftus_polys = []
        intersecting_ftus_props = []

        ann_names = [i['annotation']['name'] for i in slide_info['annotations']]

        if isinstance(ftu,list):
            intersecting_ftus_polys = {}
            intersecting_ftus_props = {}
            
            for f in ftu:
                intersecting_ftus_polys[f] = []
                intersecting_ftus_props[f] = []
                ann_id = slide_info['annotations'][ann_names.index(f)]['_id']
                # Reading GeoJSON
                with open(f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}/{ann_id}.json') as g:
                    geojson_data = json.load(g)
                    g.close()
                
                geo_df = geopandas.GeoDataFrame.from_features(geojson_data['features'])
                intersect_ftus = geo_df[geo_df.intersects(query_poly)]

                intersecting_ftus_polys[f] = [
                    shape(i)
                    for i in intersect_ftus['geometry']
                ]
                intersecting_ftus_props[f] = [
                    i
                    for i in intersect_ftus['user']
                ]

        elif isinstance(ftu,str):
            
            ann_id = slide_info['annotations'][ann_names.index(ftu)]['_id']
            # Reading GeoJSON
            with open(f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}/{ann_id}.json') as g:
                geojson_data = json.load(g)

                g.close()
            
            geo_df = geopandas.GeoDataFrame.from_features(geojson_data['features'])
            intersect_ftus = geo_df[geo_df.intersects(query_poly)]

            intersecting_ftus_polys = [
                shape(i)
                for i in intersect_ftus['geometry']
            ]
            intersecting_ftus_props = [
                i
                for i in intersect_ftus['user']
            ]


        return intersecting_ftus_props, intersecting_ftus_polys

    def spatial_aggregation(self, agg_polygon: Polygon, slide_info: dict):
        """
        Generalized aggregation of underlying structure properties for a given polygon
        """

        # Getting shape properties for the polygon itself:
        x_scale = slide_info['scale'][0]
        y_scale = slide_info['scale'][1]
        polygon_properties = {
            'Area (pixels)': round(agg_polygon.area * 1/(x_scale * -(y_scale)))
        }

        ignore_columns = ['unique_index','name','structure','ftu_name','image_id','ftu_type',
                          'Min_x_coord','Max_x_coord','Min_y_coord','Max_y_coord',
                          'x_tsne','y_tsne','x_umap','y_umap']
        
        categorical_columns = ['Cluster']
        ftu_names = [i['annotation']['name'] for i in slide_info['annotations']]
        # Step 1: Find intersecting structures with polygon
        aggregated_properties = {}
        for ftu_idx, ftu in enumerate(ftu_names):
            overlap_properties, overlap_polys = self.find_intersecting_ftu(agg_polygon, ftu, slide_info)

            overlap_area = [(i.intersection(agg_polygon).area)/(agg_polygon.area) for i in overlap_polys]
            # overlap_properties is a list of properties for each intersecting polygon
            agg_prop_df = pd.DataFrame.from_records(overlap_properties)
            agg_prop_df = agg_prop_df.drop(columns = [i for i in ignore_columns if i in agg_prop_df.columns.tolist()])

            # string and dict types will be called "object" dtypes in pandas
            agg_numeric_props = agg_prop_df.select_dtypes(exclude = 'object')

            # Remove properties designated to be categorical
            agg_numeric_props = agg_numeric_props.drop(columns = [i for i in categorical_columns if i in agg_numeric_props.columns.tolist()])

            # Scaling numeric props by area
            #for row_idx, area in enumerate(overlap_area):
                #print(f'area: {area}')
            #    agg_numeric_props.iloc[row_idx,:] *= area

            agg_numeric_dict = agg_numeric_props.mean(axis=0).to_dict()
            
            aggregated_properties[ftu] = agg_numeric_dict
            aggregated_properties[ftu][f'{ftu} Count'] = len(overlap_area)

            agg_object_props = agg_prop_df.select_dtypes(include='object')
            # Adding categorical properties
            if any([i in categorical_columns for i in agg_prop_df.columns.tolist()]):
                combined_column_names = agg_object_props.columns.tolist()+[i for i in categorical_columns if i in agg_prop_df and not i in agg_object_props]
                agg_object_props = pd.concat([agg_object_props,agg_prop_df.iloc[:,[i for i in range(len(categorical_columns)) if categorical_columns[i] in agg_prop_df and categorical_columns[i] not in agg_object_props]]],axis=1,ignore_index=True)
                agg_object_props.columns = combined_column_names
            for col_idx, col_name in enumerate(agg_object_props.columns.tolist()):
                col_values = agg_object_props[col_name].tolist()
                col_vals_dict = {col_name: {}}
                
                if type(col_values[0])==dict:
                    
                    # Test for single nested dictionary
                    sub_values = list(col_values[0].keys())
                    if not type(col_values[0][sub_values[0]])==dict:
                        col_df = pd.DataFrame.from_records([i for i in col_values if isinstance(i,dict)]).astype(float)

                        # Scaling by intersection area
                        for row_idx, area in enumerate(overlap_area):
                            col_df.iloc[row_idx,:] *= area

                        col_df_norm = col_df.sum(axis=0).to_frame()
                        col_df_norm = (col_df_norm/col_df_norm.sum()).fillna(0.000).round(decimals=18)
                        col_df_norm[0] = col_df_norm[0].map('{:.19f}'.format)
                        col_vals_dict[col_name] = col_df_norm.astype(float).to_dict()[0]
                    
                    else:
                        for sub_val in sub_values:
                            if type(col_values[0][sub_val])==dict:
                                col_df = pd.DataFrame.from_records([i[sub_val] for i in col_values if isinstance(i[sub_val],dict)]).astype(float)
                                
                                # Scaling by intersection area
                                for row_idx, area in enumerate(overlap_area):
                                    col_df.iloc[row_idx,:] *= area

                                col_df_norm = col_df.sum(axis=0).to_frame()
                                col_df_norm = (col_df_norm/col_df_norm.sum()).fillna(0.000).round(decimals=18)
                                col_df_norm[0] = col_df_norm[0].map('{:.19f}'.format)
                                col_vals_dict[col_name][sub_val] = col_df_norm.astype(float).to_dict()[0]
                    
                elif type(col_values[0])==str:
                    # Just getting the count of each unique value here
                    col_vals_dict[col_name] = {i:col_values.count(i) for i in np.unique(col_values).tolist()}

                elif type(col_values[0])==list:
                    # Getting an average of all the lists
                    #TODO: This could include the channel names if it's for Channel Means with CODEX
                    mean_list_vals = np.mean(np.array(col_values),axis=0)
                    col_vals_dict[col_name] = {
                        f'Value {i}': j
                        for i,j in enumerate(mean_list_vals.tolist())
                    }


                aggregated_properties[ftu] = aggregated_properties[ftu] | col_vals_dict

        return aggregated_properties, polygon_properties

    def intersecting_frame_intensity(self, box_poly: Union[list,Polygon], frame_list: Union[list,str], slide_info:dict):
        """
        Finding intensity distribution of each frame within a specified region
        """
        if type(box_poly)==list:
            box_poly = box(*box_poly)

        if frame_list=='all':
            frame_list = list(range(len([i for i in slide_info['frame_names'] if not i=='Histology (H&E)'])))

        #TODO: Get rid of convert map coords thing
        box_coordinates = np.array(self.convert_map_coords(list(box_poly.exterior.coords), slide_info))
        min_x = int(np.min(box_coordinates[:,0]))
        min_y = int(np.min(box_coordinates[:,1]))
        max_x = int(np.max(box_coordinates[:,0]))
        max_y = int(np.max(box_coordinates[:,1]))
        
        # Box size then can be determined by (maxx-minx)*(maxy-miny)
        box_size = (max_x-min_x)*(max_y-min_y)
        # Or probably also by multiplying some scale factors by box_poly.area
        # Pulling out those regions of the image

        # Slide coordinates list should be [minx,miny,maxx,maxy]
        slide_coords_list = [min_x,min_y,max_x,max_y]

        frame_properties = {}
        for frame in frame_list:
            # Get the image region associated with that frame
            # Or just get the histogram for that channel? not sure if this can be for a specific image region
            image_histogram = self.girder_handler.gc.get(f'/item/{slide_info["slide_info"]["_id"]}/tiles/histogram',
                                                         parameters = {
                                                            'top': min_y,
                                                            'left': min_x,
                                                            'bottom': max_y,
                                                            'right': max_x,
                                                            'frame': frame,
                                                            'bins':100,
                                                            #'rangeMax':255,
                                                            'roundRange': True, 
                                                            'density': True
                                                            }
                                                        )

            # Fraction of total intensity (maximum intensity = every pixel is 255 for uint8)
            frame_properties[slide_info['frame_names'][frame]] = image_histogram[0]
        
        return frame_properties

    def pathway_expression_histogram(self, pathway_expression_df: pd.DataFrame, f_values:Union[None,list]):
        """
        Create a histogram from a dataframe containing different columns for each pathway represented
        """

        pathway_hist_df = pd.DataFrame()
        if type(f_values)==list:
            if all([f is None for f in f_values]):
                f_values = [pathway_expression_df.columns.tolist()[0]]
            else:
                f_values = [pathway_expression_df.columns.tolist()[i] for i in f_values]
        elif f_values is None:
            f_values = [pathway_expression_df.columns.tolist()[0]]

        for path in f_values:
            if not path is None:
                path_data = pathway_expression_df[path].values

                hist, bin_edges = np.histogram(path_data,bins = 50,density = False)

                path_data_list = [
                    {
                    'Frequency': i,
                    'Expression': j,
                    'Pathway': path
                    }
                    for i,j in zip([0]+hist,bin_edges)
                ]

                if pathway_hist_df.empty:
                    pathway_hist_df = pd.DataFrame.from_records(path_data_list)
                else:
                    pathway_hist_df = pd.concat([pathway_hist_df,pd.DataFrame.from_records(path_data_list)],axis=0,ignore_index=True)

        return pathway_hist_df

    def update_viewport_data(self, bounds_box: list, view_type: dict, slide_info: dict):
        """
        Grabbing data from the current viewport and returning visualization components and data

        Available view_types = cell_composition (Main_Cell_Types, Cell Subtypes, Cell Type), features, histograms
            Main_Cell_Types and Cell Subtypes are only available for Visium
            Nucleus Features and histogram are only available for CODEX
            Cell Type and Pathway Expression are only available for Xenium
        """
        viewport_data_components = []
        viewport_data = {}

        if slide_info['slide_type']=='Visium':
            available_views = [
                {'label': 'Main Cell Types','value': 'Main_Cell_Types'},
                {'label': 'Cell Subtypes','value': 'Cell_Subtypes'},
                {'label': 'Features','value': 'features','disabled': True}
            ]
        elif slide_info['slide_type']=='CODEX':
            available_views = [
                {'label': 'Channel Intensity Histograms','value': 'channel_hist'},
                {'label': 'Nucleus Features','value': 'Channel Means'},
                {'label': 'Cell Type','value': 'Cell Type'}
            ]
        elif slide_info['slide_type']=='Xenium':
            available_views = [
                {'label': 'Cell Type', 'value': 'Cell Type'},
                {'label': 'Pathway Expression','value': 'Pathway Expression'},
                {'label': 'Features','value': 'features','disabled': True}
            ]
        else:
            available_views = [
                {'label': 'Features','value': 'features','disabled': True}
            ]


        if view_type['name'] is None:
            # Returning default options:
            viewport_data_components = html.Div([
                dbc.Row([
                    dbc.Col(
                        children = [
                            dbc.Label('Select View Type: ')
                        ],
                        md = 4
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            options = available_views,
                            value = view_type['name'],
                            multi = False,
                            id = {'type': 'roi-view-data','index': 0}
                        )
                    )
                ])
            ],style = {'height': '30vh'})

            return viewport_data_components, viewport_data

        intersecting_ftus = {}
        intersecting_ftu_polys = {}

        pie_chart_views = [
            'Main_Cell_Types',
            'Cell_Subtypes',
            'Cell Type'
        ]

        bar_chart_views = [
            'channel_hist',
            'Pathway Expression'
        ]

        ftu_names = [i['annotation']['name'] for i in slide_info['annotations']]

        if 'values' in view_type:
            f_values = view_type['values']
        else:
            f_values = None

        if 'label' in view_type:
            f_label = view_type['label']
        else:
            f_label = None
        
        possible_values = []
        disable_label_drop = True

        if not view_type['name']=='channel_hist':

            for ftu in ftu_names:
                intersecting_ftus[ftu], intersecting_ftu_polys[ftu] = self.find_intersecting_ftu(bounds_box, ftu, slide_info)

            for m_idx, m_ftu in enumerate(slide_info['manual_ROIs']):
                for int_ftu in m_ftu['geojson']['features'][0]['properties']['user']:
                    intersecting_ftus[f'Manual ROI: {m_idx+1}, {int_ftu}'] = [m_ftu['geojson']['features'][0]['properties']['user'][int_ftu]]

            for n_idx, n_ftu in enumerate(slide_info['marked_FTUs']):
                intersecting_ftus[f'Marked FTUs: {n_idx+1}'] = [i['properties']['user'] for i in n_ftu['geojson']['features']]

        else:
            if all([i is None for i in view_type['values']]):
                # Returning default options:
                viewport_data_components = html.Div([
                    dbc.Row([
                        dbc.Col(
                            children = [
                                dbc.Label('Select View Type: ')
                            ],
                            md = 4
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                options = available_views,
                                value = view_type['name'],
                                multi = False,
                                id = {'type': 'roi-view-data','index': 0}
                            )
                        )
                    ]),
                    dbc.Row([
                        dbc.Tabs(
                            dbc.Tab([
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Label('Select additional values: ',html_for = {'type': 'viewport-values-drop','index': 0}),
                                        md = 3, align = 'center'
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            options = [{'label': i, 'value': idx} for idx,i in enumerate(slide_info['frame_names']) if not i in ['Histology (H&E)','red','green','blue']],
                                            value = None,
                                            multi = True,
                                            id = {'type': 'viewport-values-drop','index': 0},
                                            disabled = False
                                        ),
                                        md = 6, align = 'center'
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            'Plot it!',
                                            className = 'd-grid col-12 mx-auto',
                                            n_clicks = 0,
                                            style = {'width': '100%'},
                                            id = {'type': 'viewport-plot-butt','index': f_idx},
                                            disabled = False
                                        ),
                                        md = 3
                                    )
                                ])
                            ])
                        )
                    ])
                ],style = {'height': '30vh'})

                return viewport_data_components, viewport_data

            intersecting_ftus['Tissue'] = self.intersecting_frame_intensity(bounds_box,view_type['values'],slide_info)


        if len(list(intersecting_ftus.keys()))>0:
            tab_list = []

            for f_idx, f in enumerate(list(intersecting_ftus.keys())):
                viewport_data[f] = {}
                if len(intersecting_ftus[f])==0:
                    continue
                
                # Grabbing data for plotting
                if not view_type['name']=='channel_hist':
                    structure_data = []
                    for st_idx,st in enumerate(intersecting_ftus[f]):
                        structure_dict = {}
                        if view_type['name'] in st:
                            # For CODEX Channel names:
                            if type(st[view_type['name']])==list:

                                possible_values = [i['Channel'] for i in slide_info['tiles_metadata']['frames']]
                                for frame in f_values:
                                    if not frame is None:
                                        frame_name = slide_info['tiles_metadata']['frames'][frame]['Channel']
                                        structure_dict[frame_name] = st[view_type['name']][frame]
                                    
                                # Adding bounding box info
                                structure_dict['Hidden'] = {'Slide_Id': slide_info['slide_info']["_id"],'Bounding_Box':list(intersecting_ftu_polys[f][st_idx].bounds)}

                                if not f_label is None:
                                    if f_label<len(st[view_type['name']]):
                                        structure_dict['label'] = st[view_type['name']][f_label]
                                    else:
                                        structure_dict['label'] = "Unlabeled"
                                else:
                                    structure_dict['label'] = 'Unlabeled'

                            # For spatially aggregated values, Pathway Expression, Main_Cell_Types, and Cell_Subtypes
                            elif type(st[view_type['name']])==dict:
                                if view_type['name']=='Channel Means':
                                    for frame in f_values:
                                        if not frame is None:
                                            frame_name = slide_info['frame_names'][frame]
                                            structure_dict[frame_name] = st[view_type['name']][frame_name]

                                    if not f_label is None:
                                        frame_label = slide_info['frame_names'][f_label]
                                        if frame_label in st[view_type['name']]:
                                            structure_dict['label'] = st[view_type['name']][slide_info['tiles_metadata']['frames'][f_label]]
                                        else:
                                            structure_dict['label'] = 'Unlabeled'
                                    else:
                                        structure_dict['label'] = 'Unlabeled'
                                
                                elif view_type['name']=='Pathway Expression':
                                    possible_values = list(set(possible_values) | set(list(st[view_type['name']].keys())))
                                    for path in list(st[view_type['name']].keys()):
                                        structure_dict[path] = st[view_type['name']][path]
                                    
                                elif view_type['name']=='Cell Type':
                                    # this will be a count for each cell type that is present
                                    structure_dict[view_type['name']] = st[view_type['name']]

                                elif view_type['name'] in ['Main_Cell_Types','Cell_Subtypes']:
                                    
                                    if view_type['name']=='Main_Cell_Types':
                                        structure_dict[view_type['name']] = st[view_type['name']]

                                        if "Cell_States" in st:
                                            structure_dict['states'] = st['Cell_States']
                                        else:
                                            structure_dict['states'] = {}

                                    elif view_type['name']=='Cell_Subtypes':
                                        main_types = list(st[view_type['name']].keys())
                                        sub_type_dict = {}
                                        for m in main_types:
                                            for s_t in st[view_type['name']][m]:
                                                sub_type_dict[s_t] = st[view_type['name']][m][s_t] * st['Main_Cell_Types'][m]
                                        
                                        structure_dict[view_type['name']] = sub_type_dict

                            elif type(st[view_type['name']])==str:
                                structure_dict[view_type['name']] = {st[view_type['name']]: 1}

                            else:
                                structure_dict[view_type['name']] = st[view_type['name']]
                                structure_dict['Hidden'] = {'Slide_Id': slide_info['slide_info']['_id'],'Bounding_Box': list(intersecting_ftu_polys[f][st_idx].bounds)}

                                if not f_label is None:
                                    if f_label in st:
                                        structure_dict['label'] = st[f_label]
                                    else:
                                        structure_dict['label'] = 'Unlabeled'
                                else:
                                    structure_dict['label'] = 'Unlabeled'

                            if len(list(structure_dict.keys()))>0:
                                structure_data.append(structure_dict)

                    if len(structure_data)>0:
                        structure_data_df = pd.DataFrame.from_records(structure_data)
                        viewport_data[f]['data'] = structure_data_df.to_dict('records')
                    else:
                        viewport_data[f]['data'] = []

                    viewport_data[f]['count'] = len(structure_data)
                else:
                    possible_values = [i for i in slide_info['frame_names'] if not i in ['Histology (H&E)','red','green','blue']]
                    structure_data = []
                    for frame in f_values:
                        if not frame is None:
                            frame_name = slide_info['frame_names'][frame]
                            frame_data = intersecting_ftus[f][frame_name]
                            
                            frame_data_list = [
                                {
                                'Frequency': i,
                                'Intensity': j,
                                'Channel': frame_name
                                }
                                for i,j in zip([0]+frame_data['hist'],frame_data['bin_edges'])
                            ]
                            structure_data.extend(frame_data_list)

                    viewport_data[f]['data'] = structure_data
                    viewport_data[f]['count'] = 1

                if len(viewport_data[f]['data'])==0:
                    continue

                # Generating plot based on assembled data
                if view_type['name']=='channel_hist':
                    chart_label = 'Channel Intensity Histogram'
                    histogram_df = pd.DataFrame.from_records(viewport_data[f]['data'])
                    histogram_df = histogram_df[histogram_df['Frequency']>=0.01]
                    #bar_width = (1/100)*(histogram_df['Intensity'].max() - histogram_df['Intensity'].min())
                    f_tab_plot = px.bar(
                        data_frame = histogram_df,
                        x = 'Intensity',
                        y = 'Frequency',
                        color = 'Channel'
                    )
                    #f_tab_plot.update_traces(width = bar_width)

                elif view_type['name']=='Pathway Expression':
                    chart_label = 'Pathway Expression Histogram'
                    if len(viewport_data[f]['data'])>0:
                        pathway_histogram_df = self.pathway_expression_histogram(pd.DataFrame.from_records(viewport_data[f]['data']),f_values)
                        f_tab_plot = px.bar(
                            data_frame = pathway_histogram_df,
                            x = 'Expression',
                            y = 'Frequency',
                            color = 'Pathway'
                        )
                        f_values = [possible_values.index(i) for i in pathway_histogram_df['Pathway'].unique().tolist()]
                    else:
                        f_tab_plot = None
                
                elif view_type['name'] in pie_chart_views:
                    
                    chart_label = view_type['name'].replace('_',' ')

                    # Getting the counts for each element in each structure
                    data_df = pd.DataFrame.from_records(viewport_data[f]['data'])
                    if view_type['name'] in data_df:
                        if data_df[view_type['name']].dtype=='object':
                            column_sum = pd.DataFrame.from_records(data_df[view_type['name']].tolist()).sum().to_dict()
                        elif data_df[view_type['name']].dtype==np.number:
                            column_sum = data_df.select_dtypes(exclude='object').sum().to_dict()
                        elif data_df[view_type['name']].dtype==str:
                            column_sum = data_df.select_dtypes(exclude='object').value_counts().to_dict()
                        else:
                            column_sum = {}
                    else:
                        column_sum = {}

                    if len(list(column_sum.keys()))==0:
                        continue

                    pie_chart_data = []
                    for key,val in column_sum.items():
                        pie_chart_data.append(
                            {'Cell Type': key, 'Total': val}
                        )
                    
                    pie_chart_df = pd.DataFrame.from_records(pie_chart_data)
                    f_tab_plot = px.pie(
                        data_frame = pie_chart_df,
                        values = 'Total',
                        names = 'Cell Type'
                    )

                    f_tab_plot.update_traces(textposition='inside')
                    f_tab_plot.update_layout(
                        uniformtext_minsize=12,
                        uniformtext_mode='hide'
                    )

                    if view_type['name']=='Main_Cell_Types':
                        # Getting the cell state information
                        top_cell = list(viewport_data[f]['data'][0]['Main_Cell_Types'].keys())[0]
                        second_chart_label = f'{top_cell} Cell States Proportions'
                        
                        pct_states = pd.DataFrame.from_records([i['states'][top_cell] for i in viewport_data[f]['data'] if top_cell in i['states']]).sum(axis=0).to_frame()
                        pct_states = pct_states.reset_index()
                        pct_states.columns = ['Cell State','Proportion']
                        pct_states['Proportion'] = pct_states['Proportion']/(pct_states['Proportion'].sum())
                        state_bar = px.bar(
                            data_frame = pct_states,
                            x = 'Cell State',
                            y = 'Proportion',
                            title = f'Cell State Proportions for: <br><sup>{top_cell} in </sup><br><sup>{f}</sup>'
                        )

                else:

                    chart_label = view_type['name'].replace('_',' ')

                    # These are feature plots. Either a violin plot (single feature), scatter plot (2 features), or UMAP (3+ features)
                    structure_data_df = pd.DataFrame.from_records(viewport_data[f]['data'])
                    n_features = len([i for i in structure_data_df.columns.tolist() if not i in ['label','Hidden']])

                    if n_features==1:
                        f_tab_plot = gen_violin_plot(
                            feature_data = structure_data_df,
                            label_col = 'label',
                            label_name = f_label,
                            feature_col = structure_data_df.columns.tolist()[0],
                            custom_col = 'Hidden' if 'Hidden' in structure_data_df else None
                        )

                        f_tab_plot.update_layout({
                            'yaxis': {
                                'title': {
                                    'text': structure_data_df.columns.tolist()[0]
                                }
                            },
                            'title': {
                                'text': structure_data_df.columns.tolist()[0]
                            }
                        })
                    elif n_features==2:
                        f_names = [i for i in structure_data_df if not i=='Hidden']
                        disable_label_drop = False

                        f_tab_plot = px.scatter(
                            data_frame = structure_data_df,
                            x = f_names[0],
                            y = f_names[1],
                            color = 'label' if 'label' in structure_data_df else None,
                            custom_data = 'Hidden' if 'Hidden' in structure_data_df else None
                        )

                        f_tab_plot.update_layout({
                            'xaxis': {
                                'title': {
                                    'text': f_names[0]
                                }
                            },
                            'yaxis': {
                                'title': {
                                    'text': f_names[1]
                                }
                            }
                        })
                    elif n_features>2:
                        
                        disable_label_drop = False
                        f_umap_data = gen_umap(
                            feature_data = structure_data_df,
                            feature_cols = [i for i in structure_data_df if not i in ['Hidden','label']],
                            label_and_custom_cols = [i for i in structure_data_df if i in ['Hidden','label']]
                        )

                        viewport_data[f]['umap'] = f_umap_data.to_dict('records')

                        f_tab_plot = px.scatter(
                            data_frame = f_umap_data,
                            x = 'UMAP1',
                            y = 'UMAP2',
                            color = 'label' if 'label' in f_umap_data else None,
                            custom_data = 'Hidden' if 'Hidden' in f_umap_data else None
                        )

                # Creating the tab for this structure:
                f_tab = dbc.Tab([
                    dbc.Row([
                        dbc.Col(
                            dbc.Label('Select additonal values: ',html_for={'type': 'viewport-values-drop','index': f_idx}),
                            md = 3, align = 'center'
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                options = [{'label': i, 'value': idx} for idx, i in enumerate(possible_values)],
                                value = f_values,
                                multi = True,
                                id = {'type': 'viewport-values-drop','index': f_idx},
                                disabled = True if len(possible_values)==0 else False
                            ),
                            md = 6, align = 'center'
                        ),
                        dbc.Col(
                            dbc.Button(
                                'Plot it!',
                                className = 'd-grid col-12 mx-auto',
                                n_clicks = 0,
                                style = {'width': '100%'},
                                id = {'type': 'viewport-plot-butt','index': f_idx},
                                disabled = True if len(possible_values)==0 else False
                            ),
                            md = 3
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.Label('Select a label: ',html_for={'type': 'viewport-label-drop','index': f_idx}),
                            md = 3
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                options = [{'label':i,'value': idx} for idx,i in enumerate(possible_values)],
                                value = f_label,
                                id = {'type': 'viewport-label-drop','index': f_idx},
                                disabled = disable_label_drop
                            )
                        )
                    ]),
                    dbc.Row(
                        children = [
                            dbc.Col([
                                dbc.Label(chart_label),
                                dcc.Graph(
                                    id = {'type': 'ftu-cell-pie','index': f_idx},
                                    figure = go.Figure(f_tab_plot)
                                )
                            ],style = {'display': 'none'} if f_tab_plot is None else {'display': 'inline-block'}) 
                        ] if not view_type['name'] == 'Main_Cell_Types' else
                        [
                            dbc.Col([
                                dbc.Label(chart_label),
                                dcc.Graph(
                                    id = {'type': 'ftu-cell-pie','index': f_idx},
                                    figure = go.Figure(f_tab_plot)
                                )
                            ],md=6,style = {'display': 'none'} if f_tab_plot is None else {'display': 'inline-block'}),
                            dbc.Col([
                                dbc.Label(second_chart_label),
                                dcc.Graph(
                                    id = {'type': 'ftu-state-bar','index': f_idx},
                                    figure = go.Figure(state_bar)
                                )
                            ],md=6,style = {'display': 'none'} if state_bar is None else {'display': 'inline-block'})
                        ]
                    )
                ], label = f+f' ({viewport_data[f]["count"]})', tab_id = f'tab_{len(tab_list)}')

                tab_list.append(f_tab)

        viewport_data_components = html.Div([
            dbc.Row([
                dbc.Col(
                    children = [
                        dbc.Label('Select View Type: ')
                    ],
                    md = 4
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options = available_views,
                        value = view_type['name'],
                        multi = False,
                        id = {'type': 'roi-view-data','index': 0}
                    )
                )
            ]),
            html.Hr(),
            dbc.Row(
                dbc.Tabs(
                    tab_list,
                    active_tab = f'tab_{len(tab_list)-1}'
                )
            )
        ])


        return viewport_data_components, viewport_data

    def update_url_style(self, color_options: dict, user_info: dict, slide_info: dict):
        """
        For CODEX slides, updates the color of the overlaid channel by changing the tile_url for that selection

        """
        # Color options is a dict containing "bands" followed by each styled frame
        # Assembling the style dict 
        style_dict = {
            "bands": [
                {
                    "palette": ["rgba(0,0,0,0)",color_options[c]],
                    "framedelta": c
                }
                for c in color_options
            ]
        }

        styled_url = self.girder_handler.gc.urlBase+f'item/{slide_info["slide_info"]["_id"]}/tiles/zxy/'+'/{z}/{x}/{y}?token='+user_info["token"]+'&style='+json.dumps(style_dict)

        return styled_url
    
    def get_rgb_url(self, slide_info, user_info):
        """
        Getting the RGB         
        """
        all_frames = [i['Channel'] for i in slide_info['tiles_metadata']['frames']]
        rgb_style_dict = {
            "bands": [
                {
                    "palette": ["rgba(0,0,0,0)",'rgba('+','.join(['255' if i==c_idx else '0' for i in range(3)]+['0'])+')'],
                    "framedelta": all_frames.index(c)
                }
                for c_idx,c in enumerate(['red','green','blue'])
            ]
        }

        rgb_url = self.girder_handler.gc.urlBase+f'item/{slide_info["slide_info"]["_id"]}/tiles/zxy/'+'/{z}/{x}/{y}?token='+user_info["token"]+'&style='+json.dumps(rgb_style_dict)

        return rgb_style_dict, rgb_url

    def populate_cell_annotation(self, region_selection, current_data):
        """
        #Populating cell annotation tab based on previous/new selections
        """
        # region_selection = whether running for the whole slide or just a specific region (current viewport)
        # current_data = dictionary containing any previous cell annotation information
        # Main component should be a big plot showing dimensional reduction, either select points
        # or run some clustering to pull out groups of points, see violin plots of specific channels, parent structures,
        # and some connectivity/graph-based measurement

        print(f'region_selection: {region_selection}')
        print(f'current_data: {current_data}')

        cell_annotation_components = []

        cell_annotation_components = [html.Div([
                                        dbc.Row('Some header here to describe the tab'),
                                        html.Hr(),
                                        dbc.Row('Then something about either using all data or the current viewport'),
                                        html.Hr(),
                                        dbc.Row('Then something that will show the big plot of all all/select cell features'),
                                        html.Hr(),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Row(
                                                    'Maybe something here where you can specify cell type names or view current ontologies'
                                                )
                                            ],md = 6),
                                            dbc.Col([
                                                dbc.Row(
                                                    'Then something here where you can pick and choose labels for individual cells that were selected'
                                                )
                                            ],md=6)
                                        ]),
                                        html.Hr(),
                                        dbc.Row(
                                            'Should be some kind of progress indicator down here'
                                        )
                                    ])]
        

        return cell_annotation_components

    def generate_annotation_overlays(self, slide_info:dict, style_handler:None, filter_handler:None, color_key:dict):
        """
        Generating overlays for the current map
        """

        slide_annotation_dir = f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}/'
        structures = [i for i in os.listdir(slide_annotation_dir) if 'json' in i]
        overlay_children = []
        if len(slide_info['frame_names'])>0:
            for frame, f_url in zip(slide_info['frame_names'],slide_info['tile_url']):
                overlay_children.append(
                    dl.BaseLayer(
                        dl.TileLayer(
                            url = f_url,
                            tileSize = slide_info['tile_dims'][0],
                            maxNativeZoom = slide_info['zoom_levels']-1,
                            id = {'type':'codex-tile-layer','index': random.randint(0,1000)},
                            bounds = [[0,0],slide_info['map_bounds']]
                        ),
                        name = frame,
                        checked = frame==slide_info['frame_names'][0],
                        id = f'codex-base-layer{random.randint(0,1000)}'
                    )
                )

        if len(structures)>0:
            for st_idx, st in enumerate(structures):

                structure_name = [i for i in slide_info['annotations'] if i['_id']+'.json'==st][0]['annotation']['name']
                overlay_children.append(
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(
                                url = f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}/{st}',
                                id = {'type': 'ftu-bounds','index': st_idx},
                                options = {
                                    'style': style_handler
                                },
                                filter = filter_handler,
                                hideout = {
                                    'color_key': color_key,
                                    'overlay_prop': slide_info['overlay_prop'],
                                    'fillOpacity': slide_info['cell_vis_val'],
                                    'ftu_colors': slide_info['ftu_colors'],
                                    'filter_vals': slide_info['filter_vals']
                                },
                                hoverStyle = arrow_function(
                                    {
                                        'weight': 5,
                                        'color': '#9caf00',
                                        'dashArray':''
                                    }
                                ),
                                zoomToBounds = False,
                                children = [
                                    dl.Popup(
                                        id = {'type': 'ftu-popup','index': st_idx},
                                        autoPan = False,
                                        #maxHeight = 800
                                    )
                                ]
                            )
                        ),
                        name = structure_name, checked = True, id = st.replace('.json','')
                    )
                )

        if len(slide_info['manual_ROIs'])>0:
            for m_idx, m in enumerate(slide_info['manual_ROIs']):
                overlay_children.append(
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(
                                data = m['geojson'],
                                id = {'type': 'ftu-bounds','index': len(overlay_children)},
                                options = {
                                    'style': style_handler
                                },
                                filter = filter_handler,
                                hideout = {
                                    'color_key':color_key,
                                    'overlay_props': slide_info['overlay_prop'],
                                    'fillOpacity': slide_info['cell_vis_val'],
                                    'ftu_colors': slide_info['ftu_colors'],
                                    'filter_vals': slide_info['filter_vals']
                                },
                                hoverStyle = arrow_function(
                                    {
                                        'weight': 5,
                                        'color': '#9caf00',
                                        'dashArray': ''
                                    }
                                ),
                                children = [
                                    dl.Popup(
                                        id = {'type': 'ftu-popup','index': len(overlay_children)},
                                        autoPan = False,
                                        #maxHeight=800
                                    )
                                ]
                            )
                        ),
                        name = f'Manual ROI {m_idx+1}', checked = True, id = slide_info["slide_info"]["_id"]+f'_manual_roi{m_idx}'
                    )
                )

        return overlay_children
    
    def get_properties_list(self,slide_info:dict):
        """
        Get all the visualizable properties for a given slide
        """

        slide_annotation_dir = f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}/'
        structures = [i for i in os.listdir(slide_annotation_dir) if 'json' in i]

        property_list = []
        for st in structures:
            with open(slide_annotation_dir+st,'r') as f:
                structure_data = geojson.load(f)

                f.close()
            
            structure_properties = [i['properties']['user'] for i in structure_data['features'] if 'user' in i['properties']]
            for s_p in structure_properties:
                # This is one structure's properties
                for p in s_p:
                    # This is one property
                    if p in self.visualization_properties:
                        if type(s_p[p])==dict:
                            # Adding sub-properties
                            sub_props = [f'{p} --> {u}' for u in list(s_p[p].keys())]
                        else:
                            sub_props = [p]
                    
                        property_list = list(set(sub_props) | set(property_list))

        if any(['Main_Cell_Types' in i for i in property_list]):
            property_list.append('Max Cell Type')

        return property_list

    def get_overlay_value_list(self, overlay_prop:dict, slide_info:dict):
        """
        Get raw values based on overlay_prop specification (used for heatmap generation and filtering)
        """

        slide_annotation_dir = f'./assets/slide_annotations/{slide_info["slide_info"]["_id"]}/'
        structures = [i for i in os.listdir(slide_annotation_dir) if 'json' in i]

        raw_values_list = []
        for st in structures:
            with open(slide_annotation_dir+st,'r') as f:
                structure_data = geojson.load(f)
                f.close()

            structure_properties = [i['properties']['user'] for i in structure_data['features'] if 'user' in i['properties']]

            raw_values_list.extend(extract_overlay_value(structure_properties, overlay_prop))

        for m in slide_info['manual_ROIs']:
            m_properties = m['geojson']['features'][0]['properties']['user']

            raw_values_list.extend(extract_overlay_value(m_properties,overlay_prop))

        return raw_values_list

    def add_property(self,new_property:Union[list,dict], mode: str, annotation_id:Union[list,str], target_structure:Union[list,Polygon]):
        """
        Add new, named property(ies) to select structures.

        Given new_property = {'property_name': 'property_value'}, mode = 'add', annotation_id = 'blahblah123uuid', target_structure: Polygon([coords]),
        
        Have to ensure that added labels are not applied to multiple people with the same slide open
        """
        assert mode in ['add','remove','modify'], "Mode must be one of the following: 'add', 'remove', or 'modify'"



        pass

    def convert_map_coords(self, coord_list:list,slide_info:dict):
        """
        Converting map coordinates to slide coordinates (pixels)
        coord list in x,y format
        """
        return_coords = []
        for i in coord_list:
            return_coords.append([i[0]/slide_info['scale'][0], i[1]/slide_info['scale'][1]])

        return return_coords

    def convert_slide_coords(self, coord_list:list, slide_info:dict):
        """
        Converting slide coordinates to map coordinates (lat,lng)
        coord list in x,y format
        """        
        return_coords = []
        for i in coord_list:
            return_coords.append([i[0]*slide_info['scale'][0],i[1]*slide_info['scale'][1]])

        return return_coords




