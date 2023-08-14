"""
Initializing FUSION web application for both local and AWS/Web deployment

This file should contain codes to generate:
    - layouts for each page
        - initial visualization layout with default dataset?
    - information dictionaries for available datasets

"""

import os
import sys

import json
import geojson

import pandas as pd
import numpy as np
from glob import glob
import lxml.etree as ET
from geojson import Feature, dump 
import uuid
import zipfile
import shutil
from PIL import Image
from io import BytesIO
import requests
from math import ceil
import base64

import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, ctx, Dash, dash_table
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_leaflet as dl
import dash_mantine_components as dmc

from dash_extensions.enrich import html
from dash_extensions.javascript import arrow_function

from dataclasses import dataclass, field
from typing import Callable, List, Union
from dash.dependencies import handle_callback_args
from dash.dependencies import Input, Output, State

import girder_client


class LayoutHandler:
    def __init__(self,
                 verbose = False):
        
        self.verbose = verbose

        self.validation_layout = []
        self.layout_dict = {}
        self.description_dict = {}

        self.info_button_idx = -1

        self.gen_welcome_layout()
        #self.gen_uploader_layout()

    def gen_info_button(self,text):
        
        self.info_button_idx+=1

        info_button = html.Div([
            html.I(
            className='bi bi-info-circle-fill me-2',
            id={'type':'info-button','index':self.info_button_idx}
            ),
            dbc.Popover(
                text,
                target = {'type':'info-button','index':self.info_button_idx},
                body=True,
                trigger='hover'
            )
        ])

        return info_button

    def gen_vis_layout(self,cell_types, center_point, zoom_levels, map_dict, spot_dict, slide_properties, tile_size, map_bounds, cli_list = None):

        # Main visualization layout, used in initialization and when switching to the viewer

        # Description and instructions card
        vis_description = [
            html.P('FUSION was designed by the members of the CMI Lab at the University of Florida in collaboration with HuBMAP'),
            html.Hr(),
            html.P('We hope that this tool provides users with an immersive visualization method for understanding the roles of specific cell types in combination with different functional tissue units'),
            html.Hr(),
            html.P('As this tool is still under active development, we welcome any and all feedback. Use the "User Survey" link above to provide comments. Thanks!'),
            html.Hr(),
            html.P('Happy fusing!')         
        ]

        # View of WSI
        self.initial_overlays = [
            dl.Overlay(
                dl.LayerGroup(
                    dl.GeoJSON(data = map_dict['FTUs'][struct]['geojson'], id = map_dict['FTUs'][struct]['id'], options = dict(color = map_dict['FTUs'][struct]['color']),
                        hoverStyle = arrow_function(dict(weight=5, color = map_dict['FTUs'][struct]['hover_color'], dashArray = '')),children=[dl.Popup(id = map_dict['FTUs'][struct]['popup_id'])])),
                name = struct, checked = True, id = struct)
        for struct in map_dict['FTUs']
        ] 

        self.initial_overlays+= [
            dl.Overlay(
                dl.LayerGroup(
                    dl.GeoJSON(data = spot_dict['geojson'], id = spot_dict['id'], options = dict(color = spot_dict['color']),
                        hoverStyle = arrow_function(dict(weight=5, color = spot_dict['hover_color'], dashArray = '')),
                        children = [dl.Popup(id=spot_dict['popup_id'])],
                        zoomToBounds=True)),
                name = 'Spots', checked = False, id = 'Spots')
        ]


        map_children = [
            dl.TileLayer(url = map_dict['url'], tileSize = tile_size, id = 'slide-tile'),
            dl.FeatureGroup(id='feature-group',
                            children = [
                                dl.EditControl(id = {'type':'edit_control','index':0},
                                                draw = dict(line=False, circle = False, circlemarker=False))
                            ]),
            dl.LayerGroup(id='mini-label'),
            html.Div(id='colorbar-div',children = [dl.Colorbar(id='map-colorbar')]),
            dl.LayersControl(id='layer-control',children = self.initial_overlays)
        ]

        map_layer = dl.Map(
            center = center_point, zoom = 3, minZoom = 0, maxZoom = zoom_levels, crs='Simple',bounds = map_bounds,
            style = {'width':'100%','height':'80vh','margin':'auto','display':'inline-block'},
            id = 'slide-map',
            children = map_children
        )

        wsi_view = dbc.Card([
            dbc.CardHeader(
                children = [
                    'Whole Slide Image Viewer',
                    self.gen_info_button('Use the mouse to pan and zoom around the slide!')
            ]),
            dbc.Row([
                html.Div(
                    map_layer
                )
            ]),
            dbc.Row([html.Div(id='current-hover')])
        ], style = {'marginBottom':'20px'})

        # Cell type proportions and cell state distributions
        roi_pie = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.P('This tab displays the cell type and state proportions for cell types contained within specific FTU & Spot boundaries')
                ]),
                html.Hr(),
                html.Div(id = 'roi-pie-holder')
            ])
        ])

        # Stylesheet for cyto plot thingy
        cyto_style = [
            {
            'selector':'node',
            'style':{
                'label':'data(label)',
                'width':45,
                'height':45,
                'background-color':'rgba(255,255,255,0)',
                'background-fit':'cover',
                'background-image-opacity':1,
                'background-image':'data(url)',
                }
            },
            {
            'selector':'edge',
            'style':{
                'line-width':15,
                'line-color':'blue'
            }
            }
        ]

        # Creating figure dictionary for nephron diagram
        neph_figure = go.Figure(px.imshow(Image.open('./assets/cell_graphics/nephron_diagram.jpg')))
        neph_figure.update_traces(hoverinfo='none',hovertemplate=None)
        neph_figure.update_xaxes(showticklabels=False)
        neph_figure.update_yaxes(showticklabels=False)
        neph_figure.update_layout(margin={'l':0,'b':0,'r':0,'t':0})

        cell_graphic_tab = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        #dbc.Label('Cell State:',html_for = 'cell-vis-drop'),
                        dcc.Dropdown(['Cell States'],placeholder='Cell States', id = 'cell-vis-drop')
                    ],md=4),
                    dbc.Col([
                        html.Div(id='cell-graphic-name')
                    ],md=8)
                ],align = 'center',style={'marginBottom':'10px'}),
                html.Hr(),
                dbc.Row([
                    html.Div(
                        id = 'cell-vis-graphic',
                        children = [
                            html.Img(
                                id = 'cell-graphic',
                                src = './assets/cell_graphics/default_cell_graphic.png',
                                height = '100%',
                                width = '100%'
                            )
                        ]
                    )
                ],align='center',style={'marginBottom':'10px'})

            ])
        ])

        cell_hierarchy_tab = dbc.Card([
            dbc.CardBody([
                dbc.Row(
                    dbc.Col([
                        html.Div(
                            cyto.Cytoscape(
                                id = 'cell-hierarchy',
                                layout = {'name':'preset'},
                                style = {
                                    'width':'100%',
                                    'height':'400px',
                                    'display':'inline-block',
                                    'margin':'auto',
                                    'background-color':'rgb(221,221,221)'},
                                minZoom=0.5,
                                maxZoom=3,
                                stylesheet=cyto_style,
                                elements = [
                                    {'data': {'id': 'one', 'label': 'Node 1'}, 'position': {'x': 75, 'y': 75}},
                                    {'data': {'id': 'two', 'label': 'Node 2'}, 'position': {'x': 200, 'y': 200}},
                                    {'data': {'source': 'one', 'target': 'two'}}
                                ]
                            )
                        )
                    ],md=12)
                ,align='center'),
                dbc.Row(self.gen_info_button('Pan and click nodes with the mouse for more information!')),
                dbc.Row(html.Div(id = 'label-p')),
                dbc.Row(html.Div(id='id-p')),
                dbc.Row(html.Div(id='notes-p'))
            ])
        ])

        cell_card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.P('Use this tab for graphical selection of specific cell types along the nephron')
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.H2('Nephron Diagram')
                        ]),
                        dbc.Row([
                            dcc.Graph(id='neph-img',figure=neph_figure),
                            dcc.Tooltip(id='neph-tooltip',loading_text='')
                        ])
                    ],md=5),
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(cell_graphic_tab, label = 'Cell Graphic'),
                            dbc.Tab(cell_hierarchy_tab, label = 'Cell Hierarchy')
                        ])
                    ],md=7)
                ],align='center')
            ]),
            dbc.CardFooter(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Link('Derived from ASCT+B Kidney v1.2',href='https://docs.google.com/spreadsheets/d/1NMfu1bEGNFcTYTFT-jCao_lSbFD8n0ti630iIpRj-hw/edit#gid=949267305')
                        ])
                    ])
                ])
            )
        ])

        ftu_list = ['glomerulus','Tubules']
        plot_types = ['TSNE','UMAP']
        labels = ['Cluster','image_id']+cell_types.copy()
        # Cluster viewer tab
        cluster_card = dbc.Card([
            dbc.Row([
                html.P('Use this tab to dynamically view clustering results of morphological properties for select FTUs')
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        id= 'plot-options',
                        children = [
                            dbc.CardHeader(
                                children = [
                                    dbc.Row([
                                        dbc.Col('Plot Options',md=11),
                                        dbc.Col(self.gen_info_button('Select different plot options to update the graph!'),md=1)
                                    ])

                                    ]
                                ),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row([
                                            dbc.Col(dbc.Label('Functional Tissue Unit Type',html_for='ftu-select'),md=11),
                                            dbc.Col(self.gen_info_button("Select a FTU to see that FTU;s morphometrics clustering"),md=1)
                                        ])
                                    ],md=4),
                                    dbc.Col([
                                        html.Div(
                                            dcc.Dropdown(
                                                ftu_list,
                                                ftu_list[0],
                                                id='ftu-select'
                                            )
                                        )],md=8
                                    )
                                ]),
                                html.B(),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row([
                                            dbc.Col(dbc.Label('Type of plot',html_for='plot-select'),md=11),
                                            dbc.Col(self.gen_info_button('Select a method of dimensional reduction to change layout of clustering graph'),md=1)
                                            ])                              
                                        ],md=4),
                                    dbc.Col([
                                        html.Div(
                                            dcc.Dropdown(
                                                plot_types,
                                                plot_types[0],
                                                id='plot-select'
                                            )
                                        )],md=8
                                    )
                                ]),
                                html.B(),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row([
                                            dbc.Col(dbc.Label('Sample Labels',html_for='label-select'),md=11),
                                            dbc.Col(self.gen_info_button('Select a label to overlay onto points in the graph'),md=1)
                                            ])
                                        ],md=4),
                                    dbc.Col([
                                        html.Div(
                                            dcc.Dropdown(
                                                labels,
                                                labels[0],
                                                id='label-select'
                                            )
                                        )],md=8
                                    )
                                ])
                            ])
                        ]
                    )
                ],md=12)
            ]),
            dbc.Row([
                dbc.Col([
                    self.gen_info_button('Click on a point in the graph or select a group of points with the lasso select tool to view the FTU and cell type data at that point'),
                    html.Div(
                        dcc.Graph(id='cluster-graph',figure=go.Figure())
                    )
                ],md=6),
                dbc.Col([
                    dcc.Tabs([
                        dcc.Tab(
                            dbc.Card(
                                id = 'selected-image-card',
                                children = [
                                    dcc.Loading(
                                        id = 'loading-image',
                                        children = [
                                            dcc.Graph(id='selected-image',figure=go.Figure()),
                                            self.gen_info_button('Image(s) extracted from clustering plot')
                                        ]
                                    )
                                ]
                            ),label='Selected Images'),
                        dcc.Tab(
                            dbc.Card(
                                id = 'selected-data-card',
                                children = [
                                    dcc.Loading(
                                        id='loading-data',
                                        children = [
                                            dbc.Row(
                                                children = [
                                                    dbc.Col([
                                                        dcc.Graph(id='selected-cell-types',figure=go.Figure()),
                                                        self.gen_info_button('Click on a section of the pie chart to view cell state proportions')
                                                        ]),
                                                    dbc.Col(dcc.Graph(id='selected-cell-states',figure=go.Figure()))
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),label='Selected Cell Data')
                    ]),
                    html.Div(id='selected-image-info')
                ],md=6),
            ],align='center')
        ])

        # Tools for selecting regions, transparency, and cells

        # Converting the cell_types list into a dictionary to disable some
        disable_list = []
        cell_types_list = []
        for c in slide_properties:
            if c not in disable_list:
                cell_types_list.append({'label':c,'value':c,'disabled':False})
            else:
                cell_types_list.append({'label':c+' (In Progress)','value':c,'disabled':True})

        # Extracting data tab
        self.data_options = [
            {'label':'Annotations','value':'Annotations','disabled':False},
            {'label':'Cell Type and State','value':'Cell Type and State','disabled':False},
            {'label':'Slide Metadata','value':'Slide Metadata','disabled':False},
            {'label':'Selected FTUs and Metadata','value':'Selected FTUs and Metadata','disabled':True},
            {'label':'Manual ROIs','value':'Manual ROIs','disabled':True}
        ]
        extract_card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.P('Use this tab for exporting data from current views')
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Label('Select data for download',html_for = 'data-select'),
                    dcc.Dropdown(self.data_options,placeholder = 'Select Data for Download',multi=True,id='data-select')
                ]),
                dbc.Row([dbc.Label('Download options',html_for = 'data-options')]),
                dbc.Row([
                    html.Div(id='data-options')
                ])
            ])
        ])

        # Test CLI tab
        # Accessing analyses/cli plugins for applying to data in FUSION
        available_clis = cli_list
        cli_tab = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.P('Use this tab for running custom plugins')
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Label('Select CLI to run:',html_for='cli-drop'),
                    html.B(),
                    dcc.Dropdown(available_clis,id='cli-drop')
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Label('CLI Description:',html_for='cli-descrip'),
                    html.Div(id='cli-descrip')
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Label('Current Image Region',html_for='cli-current-image'),
                    html.Div(id = 'cli-current-image')
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Button('Run Job!',color='primary',id='cli-run',disabled=True)
                ]),
                html.Hr(),
                dbc.Row([
                    dcc.Loading(html.Div(id='cli-results'))
                ]),
                html.B(),
                dbc.Row([
                    html.Div(id = 'cli-results-followup')
                ])
            ])
        ])

        # Overlays control tab
        combined_colors_dict = {}
        for f in map_dict['FTUs']:
            combined_colors_dict[f] = {'color':map_dict['FTUs'][f]['color']}
        
        combined_colors_dict['Spots'] = {'color':spot_dict['color']}

        overlays_tab = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.P('Use this tab for controlling properties of FTU & Spot overlays')
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H6("Select Cell for Overlaid Heatmap Viewing",className="cell-select"),
                        self.gen_info_button('Select a cell type or metadata property to change FTU overlay colors'),
                        html.Div(
                            id = 'cell-select-div',
                            children=[
                                dcc.Dropdown(cell_types_list,cell_types_list[0]['value'],id='cell-drop')
                            ]
                        )
                    ])
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Label(
                            "Adjust Transparency of Heatmap",
                            html_for="vis-slider"
                        ),
                        self.gen_info_button('Change transparency of overlaid FTU colors between 0(fully see-through) to 100 (fully opaque)'),
                        dcc.Slider(
                            id='vis-slider',
                            min=0,
                            max=100,
                            step=10,
                            value=50
                        )
                    ])
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Label(
                            "FTUs Filter by Overlay Value",html_for="filter-slider"
                        ),
                        self.gen_info_button('Set a range of values to only view FTUs with that overlay value'),
                        dcc.RangeSlider(
                            id = 'filter-slider',
                            min=0.0,
                            max=1.0,
                            step = 0.01,
                            value = [0.0,1.0],
                            marks=None,
                            tooltip = {'placement':'bottom','always_visible':True},
                            allowCross=False
                        )
                    ])
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Label(
                            'FTU Boundary Color Picker',
                            html_for = 'ftu-bound-opts'
                        )
                    ])
                ]),
                dbc.Row([
                    dbc.Tabs(
                        id = 'ftu-bound-opts',
                        children = [
                            dbc.Tab(
                                children = [
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(
                                                dmc.ColorPicker(
                                                    id =  {'type':'ftu-bound-color','index':idx},
                                                    format = 'hex',
                                                    value = combined_colors_dict[struct]['color'],
                                                    fullWidth=True
                                                ),
                                                style = {'width':'30vh'}
                                            )
                                        ],md=12,align='center')
                                    ],align='center')
                                ], label = struct
                            )
                            for idx,struct in enumerate(list(combined_colors_dict.keys()))
                        ]
                    )
                ])
            ])
        ])

        # List of all tools tabs
        tool_tabs = [
            dbc.Tab(overlays_tab, label = 'Overlays'),
            dbc.Tab(roi_pie, label = "Cell Composition"),
            dbc.Tab(cell_card,label = "Cell Card"),
            dbc.Tab(cluster_card,label = 'Morphological Clustering'),
            dbc.Tab(extract_card,label = 'Download Data'),
            dbc.Tab(cli_tab,label = 'Run Analyses'),
        ]
        
        tools = [
            dbc.Card(
                id='tools-card',
                children=[
                    dbc.CardHeader("Tools"),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Tabs(tool_tabs)
                            ])
                        ])
                    ])
                ]
            )
        ]

        # Separately outputting the functional components of the application for later reference when switching pages
        vis_content = [
            dbc.Row(
                id="app-content",
                children=[
                    dbc.Col(wsi_view,md=6),
                    dbc.Col(tools,md=6)
                ],style={"height":"100vh"}
            )
        ]

        self.current_vis_layout = vis_content
        self.validation_layout.append(vis_content)
        self.layout_dict['vis'] = vis_content
        self.description_dict['vis'] = vis_description

    def gen_builder_layout(self, dataset_handler):

        # This builds the layout for the Dataset Builder functionality, 
        # allowing users to select which datasets/slides are incorporated into their 
        # current viewing instance.

        # Description and instructions card
        builder_description = [
            html.P('Use this page to construct a dataset of slides for a visualization session'),
            html.Hr(),
            html.P('Use the sidebar to return to the visualization page'),
            html.Hr(),
            html.P('Happy fusing!')
        ]

        # Table with metadata for each dataset in dataset_handler
        combined_dataset_dict = []

        # Accessing the folder structure saved in dataset_handler            
        for f in dataset_handler.slide_datasets:
            folder_dict = {}
            folder_dict['Name'] = dataset_handler.slide_datasets[f]['name']
            
            folder_meta_keys = list(dataset_handler.slide_datasets[f]['Metadata'])
            for m in folder_meta_keys:
                folder_dict[m] = dataset_handler.slide_datasets[f]['Metadata'][m]

            combined_dataset_dict.append(folder_dict)

        dataset_df = pd.DataFrame.from_records(combined_dataset_dict)


        # Table with a bunch of filtering and tooltip info
        table_layout = html.Div([
            dash_table.DataTable(
                id = 'dataset-table',
                columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in dataset_df],
                data = dataset_df.to_dict('records'),
                editable = False,
                filter_action='native',
                sort_action = 'native',
                sort_mode = 'multi',
                column_selectable = 'single',
                row_selectable = 'multi',
                row_deletable = False,
                selected_columns = [],
                selected_rows = [],
                page_action='native',
                page_current=0,
                page_size=10,
                style_cell = {
                    'overflow':'hidden',
                    'textOverflow':'ellipsis',
                    'maxWidth':0
                },
                tooltip_data = [
                    {
                        column: {'value':str(value),'type':'markdown'}
                        for column, value in row.items()
                    } for row in dataset_df.to_dict('records')
                ],
                tooltip_duration = None
            )
        ])
        
        builder_layout = [
                    html.H3('Select a Dataset to add slides to current session'),
                    html.Hr(),
                    self.gen_info_button('Click on one of the circles in the far left of the table to load metadata for that dataset. You can also filter/sort the rows using the arrow icons in the column names and the text input in the first row'),
                    table_layout,
                    html.H3('Select Slides to include in current session'),
                    self.gen_info_button('Select/de-select slides to add/remove them from the metadata plot and current viewing session'),
                    html.Hr(),
                    html.Div(id='selected-dataset-slides'),
                    html.Hr(),
                    html.H3('Current Metadata'),
                    self.gen_info_button('Select different metadata options to view the distribution of FTU values within each selected dataset or slide'),
                    dcc.Loading(html.Div(id='slide-metadata-plots'))
                ]

        self.current_builder_layout = builder_layout
        self.validation_layout.append(builder_layout)
        self.layout_dict['dataset-builder'] = builder_layout
        self.description_dict['dataset-builder'] = builder_description

    def gen_uploader_layout(self,dataset_handler):

        # This builds the layout for the Dataset Uploader functionality,
        # allowing users to upload their own data to be incorporated into the 
        # dataset builder or directly to the current viewing instance.

        # Description and instructions card
        uploader_description = [
            html.P('Happy fusing!')
        ]

        # File upload card:
        upload_types = [
            {'label':'10x Visium','value':'Visium','disabled':False},
            {'label':'Co-Detection by Indexing (CODEX)','value':'CODEX','disabled':True},
            {'label':'CosMx','value':'CoxMx','disabled':True}
        ]
        collection_list = [i['name'] for i in dataset_handler.get_collections()]
        collection_list += ['New Collection']
        file_upload_card = dbc.Card([
            dbc.CardHeader('File Uploads'),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            dbc.Label('Select Upload type:',html_for='upload-type'),
                            dcc.Dropdown(upload_types, placeholder = 'Select Spatial -omics method', id = 'upload-type')
                            ]
                        )
                    ),
                    dbc.Col(
                        html.Div(
                            id='upload-requirements',
                            children = []
                        )
                    )
                ],
                align='center')
            ])
        ])

        # Slide QC card:
        slide_qc_card = dbc.Card([
            dbc.CardHeader('Slide Quality Control Results'),
            dbc.CardBody(
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            id='slide-qc-results',
                            children = []
                        )
                    )
                ])
            )
        ])

        # MC model selection card:
        organ_types = [
            {'label':'Kidney','value':'Kidney','disabled':False}
        ]
        mc_model_card = dbc.Card([
            dbc.CardHeader('Multi-Compartment Model Selection'),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            dbc.Label('Select Organ:',html_for='organ-type'),
                            dcc.Dropdown(organ_types,placeholder = 'It better be kidney',id='organ-type',disabled=True)
                        ])
                    )
                ])
            ])
        ])

        # Sub-compartment segmentation card:
        sub_comp_methods_list = [
            {'label':'Manual','value':'Manual','disabled':False},
            {'label':'Use Plugin','value':'plugin','disabled':True}
        ]
        sub_comp_card = dbc.Card([
            dbc.CardHeader('Sub-Compartment Segmentation'),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            dbc.Label('Select FTU',html_for='ftu-select'),
                            dcc.Dropdown(placeholder='FTU Options',id='ftu-select')
                        ]),md=6),
                    dbc.Col(
                        html.Div(
                            id='seg-qc-results',
                            children = []
                        ),md=6)
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            children = [
                                dcc.Graph(figure=go.Figure(),id='ex-ftu-img')
                            ]
                        ),md=8),
                    dbc.Col(
                        html.Div(
                            children = [
                                dbc.Label('Example FTU Segmentation Options',html_for='ex-ftu-opts'),
                                html.Hr(),
                                html.Div(
                                    id='ex-ftu-opts',
                                    children = [
                                        dcc.RadioItems(['Overlaid','Side-by-Side'],value='Overlaid',inline=True,id='ex-ftu-view'),
                                        html.B(),
                                        dbc.Label('Overlaid Mask Transparency:',html_for='ex-ftu-slider'),
                                        dcc.Slider(0,100,5,value=50,marks=None,vertical=False,tooltip={'placement':'bottom'})
                                    ]
                                )
                            ]
                        ),md=4)
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            children = [
                                dbc.Label('Sub-compartment Segmentation Method:',html_for='sub-comp-method')
                            ]
                        ),md=4),
                    dbc.Col(
                        html.Div(
                            children = [
                                dcc.Dropdown(sub_comp_methods_list,placeholder='Available Methods',id='sub-comp-method')
                            ]
                        ),md=8
                    )
                ]),
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            id='sub-comp-tabs',
                            children = []
                        ),md=12
                    )
                )
            ])
        ])

        # Feature extraction card:
        feat_extract_card = dbc.Card([
            dbc.CardHeader('Morphometric Feature Extraction'),
            dbc.CardBody(
                dbc.Row([
                    dbc.Col(html.Div(id='feature-items'))
                ])
            )
        ])

        # Progressbar
        p_bar = dbc.Progress(id='p-bar')

        uploader_layout =[
                html.H1('Dataset Uploader'),
                html.Hr(),
                dbc.Row(
                    children = [
                        dbc.Col(file_upload_card,md=12)
                    ]
                ),
                html.Hr(),
                dbc.Row(
                    children = [
                        dbc.Col(
                            dbc.Row(
                                children = [
                                    dbc.Col(slide_qc_card,md=6),
                                    dbc.Col(mc_model_card,md=6)
                                ]
                            ),md=12
                        )
                    ],
                    align='center',
                    id = 'post-upload-row',
                    style = {'display':'none'}
                ),
                html.Hr(),
                dbc.Row(
                    children = [
                        dbc.Col(
                            html.Div(
                                dbc.Row(
                                    children = [
                                        dbc.Col(html.Div(sub_comp_card),md=6),
                                        dbc.Col(html.Div(feat_extract_card),md=6)
                                    ]
                                )
                            ),md=12
                        )
                    ],
                    align='center',
                    id = 'post-segment-row',
                    style={'display':'none'}
                ),
                html.Hr(),
                dbc.Row(p_bar)

            ]
        self.current_uploader_layout = uploader_layout
        self.validation_layout.append(uploader_layout)
        self.layout_dict['dataset-uploader'] = uploader_layout
        self.description_dict['dataset-uploader'] = uploader_description

    def gen_welcome_layout(self):

        # welcome layout after initialization and information and buttons to go to other areas

        # Description and instructions card
        welcome_description = [
            html.P('This page contains video examples for a variety of things you can do using FUSION'),
            html.Hr(),
            html.P('Select an item from the dropdown menu to watch the video'),
            html.Hr(),
            html.P('Happy fusing!')
        ]

        total_videos = ['General Introduction','Main Window Navigation','Dataset Builder','Cell Type Overlays','Morphological Clustering',
                        'Cell Type and State Proportions','Exporting Data']
        video_names = ['general_introduction','main_window','dataset_builder','cell_overlays','morphological_clustering','cell_types_and_states',[]]
        videos_available = ['General Introduction','Main Window Navigation','Dataset Builder','Cell Type Overlays','Morphological Clustering','Cell Type and State Proportions']
        video_dropdown = []
        for t,n in zip(total_videos,video_names):
            if t in videos_available:
                video_dropdown.append({'label':t,'value':n,'disabled':False})
            else:
                video_dropdown.append({'label':t,'value':n,'disabled':True})
        
        welcome_layout = [
                html.H1('Welcome to FUSION!'),
                html.Hr(),
                html.B(),
                dbc.Row([
                    dbc.Row([dbc.Label('Getting Started: Select a category below to view a tutorial video')]),
                    html.Hr(),
                    html.B(),
                    dcc.Dropdown(video_dropdown,video_dropdown[0],id={'type':'video-drop','index':0}),
                    html.B(),
                    html.Hr(),
                    html.Video(src='./assets/videos/general_introduction.mp4',
                            controls = True,
                            autoPlay = True,
                            preload=True,
                            id = {'type':'video','index':0})
                ])
            ]
        
        self.current_welcome_layout = welcome_layout
        self.validation_layout.append(welcome_layout)
        self.layout_dict['welcome'] = welcome_layout
        self.description_dict['welcome'] = welcome_description

    def gen_initial_layout(self,slide_names):

        # welcome layout after initialization and information and buttons to go to other areas

        # Header
        header = dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.Img(id='logo',src=('./assets/FUSION-LAB_navigator.png'),height='100px'),md='auto'),
                    dbc.Col([
                        html.Div([
                            html.H3('FUSION',style={'color':'rgb(255,255,255)'}),
                            html.P('Functional Unit State Identification and Navigation with WSI',style={'color':'rgb(255,255,255)'})
                        ],id='app-title')
                    ],md=True,align='center')
                ],align='center'),
                dbc.Row([
                    dbc.Col([
                        dbc.NavbarToggler(id='navbar-toggler'),
                        dbc.Collapse(
                            dbc.Nav([
                                dbc.NavItem(
                                    dbc.Button(
                                        'User Survey',
                                        id = 'user-survey-button',
                                        outline = True,
                                        color = 'secondary',
                                        href = ' https://ufl.qualtrics.com/jfe/form/SV_1A0CcKNLhTnFCHI',
                                        style = {'textTransform':'none'}
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "Cell Cards",
                                        id='cell-cards-button',
                                        outline=True,
                                        color="secondary",
                                        href="https://cellcards.org/index.php",
                                        style={"textTransform":"none"}
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "Lab Website",
                                        id='lab-web-button',
                                        outline=True,
                                        color='secondary',
                                        href='https://cmilab.nephrology.medicine.ufl.edu',
                                        style={"textTransform":"none"}
                                    )
                                )
                            ],navbar=True),
                        id="navbar-collapse",
                        navbar=True)
                    ],
                    md=2)
                    ],
                align='center')
                ], fluid=True),
            dark=True,
            color="primary",
            sticky='fixed',
            style={'marginBottom':'20px'}
        )

        # Sidebar
        sider = html.Div([
            dbc.Offcanvas([
                html.Img(id='welcome-logo-side',src=('./assets/FUSION-LAB-FINAL.png'),height='315px',width='250px'),
                dbc.Nav([
                    dbc.NavLink('Welcome',href='/welcome',active='exact'),
                    dbc.NavLink('FUSION Visualizer',href='/vis',active='exact'),
                    dbc.NavLink('Dataset Builder',href='/dataset-builder',active='exact'),
                    dbc.NavLink('Dataset Uploader',href='/dataset-uploader',active='exact')
                ],vertical=True,pills=True)], id={'type':'sidebar-offcanvas','index':0},style={'background-color':"#f8f9fa"}
            )
        ])
        
        # Description and instructions card
        initial_description = [
            html.P('This page contains video examples for a variety of things you can do using FUSION'),
            html.Hr(),
            html.P('Select an item from the dropdown menu to watch the video'),
            html.Hr(),
            html.P('Happy fusing!')
        ]

        # Login popover
        login_popover = dbc.Popover(
            [
                dbc.PopoverHeader('Enter your username and password:'),
                dbc.PopoverBody([
                    dbc.Label('Username:',width='auto'),
                    dbc.Col(
                        dbc.Input(type='text',placeholder='Username',id='username-input')
                    ),
                    dbc.Label('Password',width='auto'),
                    dbc.Col(
                        dbc.Input(type='password',placeholder='Password',id='pword-input')
                    ),
                    dbc.Col(
                        dbc.Button('Submit',color='primary',id='login-submit'),width='auto'
                    )
                ])
            ],
            target = {'type':'login-butt','index':0},
            body=True,
            trigger='click'
        )

        description = dbc.Card(
            children = [
                #dbc.CardHeader("Description and Instructions"),
                dbc.CardBody([
                    dbc.Button('Open Sidebar',id={'type':'sidebar-button','index':0},className='mb-3',color='primary',n_clicks=0,style={'marginRight':'5px'}),
                    dbc.Button("View/Hide Description",id={'type':'collapse-descrip','index':0},className='mb-3',color='primary',n_clicks=0,style={'marginLeft':'5px'}),
                    dbc.Button('Registered User Login',id={'type':'login-butt','index':0},className='mb-3',style = {'marginLeft':'5px'}),
                    login_popover,
                    html.Div(id='logged-in-user'),
                    dbc.Collapse(
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    id = 'descrip',
                                    children = initial_description,
                                    style={'fontSize':10}
                                )
                            )
                        ),id={'type':'collapse-content','index':0},is_open=False
                    )
                ])
            ],style={'marginBottom':'20px'}
        )


        # Slide select row (seeing if keeping it in every layout will let it be updated by actions in other pages)
        # Slide selection
        slide_select = dbc.Card(
            id = 'slide-select-card',
            children = [
                dbc.CardHeader("Select case from dropdown menu"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                id = "slide_select-label",
                                children = [
                                    html.P("Available cases: ")
                                ]
                            ),md=4
                        ),
                        dbc.Col(
                            html.Div(
                                dcc.Dropdown(
                                    slide_names,
                                    slide_names[0],
                                    id = 'slide-select'
                                )
                            ), md=7
                        ),
                        dbc.Col(
                            self.gen_info_button('Click the dropdown menu to select a slide!'),md=1
                        )
                    ],align='center')
                ])
            ],style={'marginBottom':'20px','display':'none'}
        )
        
        welcome_layout = html.Div([
            dcc.Location(id='url'),
            header,
            html.B(),
            dbc.Row(dbc.Col(html.Div(sider))),
            html.B(),
            dbc.Row(
                id = 'descrip-and-instruct',
                children = description,
                align='center'
            ),
            dbc.Row(
                id = 'slide-select-row',
                children = slide_select
            ),
            dbc.Container([
                html.H1('Welcome to FUSION!'),
                ],fluid=True,id='container-content')
        ])

        self.current_initial_layout = welcome_layout
        self.validation_layout.append(welcome_layout)
        self.layout_dict['initial'] = welcome_layout
        self.description_dict['initial'] = initial_description


class GirderHandler:
    def __init__(self,
                apiUrl: str):
        
        self.apiUrl = apiUrl
        self.gc = girder_client.GirderClient(apiUrl = self.apiUrl)
    
    def authenticate(self, username, password):
        # Getting authentication for user
        #TODO: Add some handling here for incorrect username or password
        self.gc.authenticate(username,password)

    def get_token(self):
        # Getting session token for accessing private collections
        user_token = self.gc.get('token/session')['token']

        self.user_token = user_token

        return user_token

    def get_collections(self):
        # Getting collections data 
        # This will be a list of dictionaries with the following keys:
        # _accessLevel, _id, _modelType, created, description, meta, name, public, size, updated
        # of which _id, description, meta, name, and public are probably the most useful
        collections_data = self.gc.get('/collection')

        return collections_data

    def get_resource_id(self,resource):
        # Get unique item id from resource path to file
        item_id = self.gc.get('resource/lookup',parameters={'path':resource})['_id']

        return item_id

    def get_resource_metadata(self,resource):
        # Get metadata associated with resource
        resource_metadata = self.gc.get('resource/lookup',parameters={'path':resource})['meta']

        return resource_metadata

    def get_collection_items(self,collection_path):
        # Get list of items in a collection
        collection_id = self.get_resource_id(collection_path)
        collection_contents = self.gc.get(f'resource/{collection_id}/items',parameters={'type':'collection'})

        return collection_contents

    def get_tile_metadata(self,item_id):
        # tile metadata includes: 'levels', 'magnification', 'mm_x', 'mm_y', 'sizeX', 'sizeY', 'tileHeight', 'tileWidth'
        tile_metadata = self.gc.get(f'item/{item_id}/tiles')

        return tile_metadata
    
    def get_annotations(self,item_id):
        # Getting histomics JSON annotations for an item
        annotations = self.gc.get(f'annotation/item/{item_id}')

        return annotations
    
    def convert_json(self,annotations,image_dims,base_dims,tile_dims):

        # Top left and bottom right should be in (x,y) or (lng,lat) format
        #top_left = [0,tile_dims[0]]
        #bottom_right = [0,tile_dims[1]]

        # Translation step
        #base_x_scale = base_dims[0]/(bottom_right[0]-top_left[0])
        #base_y_scale = base_dims[1]/(bottom_right[1]-top_left[1])
        base_x_scale = base_dims[0]/tile_dims[0]
        base_y_scale = base_dims[1]/tile_dims[1]

        # image bounds [maxX, maxY]
        # bottom_right[0]-top_left[0] --> range of x values in target crs
        # bottom_right[1]-top_left[1] --> range of y values in target crs
        # scaling values so they fall into the current map (pixels)
        #x_scale = (bottom_right[0]-top_left[0])/(image_dims[0])
        #y_scale = (bottom_right[1]-top_left[1])/(image_dims[1])
        x_scale = tile_dims[0]/image_dims[0]
        y_scale = tile_dims[1]/image_dims[1]
        y_scale*=-1
        # y_scale has to be inverted because y is measured upward in these maps

        final_ann = {'type':'FeatureCollection','features':[]}
        for a in annotations:
            if 'elements' in a['annotation']:
                f_name = a['annotation']['name']
                for f in a['annotation']['elements']:
                    f_dict = {'type':'Feature','geometry':{'type':'Polygon','coordinates':[]},'properties':{}}
                    
                    # This is only for polyline type elements
                    if f['type']=='polyline':
                        og_coords = np.squeeze(np.array(f['points']))
                        
                        scaled_coords = og_coords.tolist()
                        scaled_coords = [i[0:-1] for i in scaled_coords]
                        scaled_coords = [[base_x_scale*((i[0]*x_scale)),base_y_scale*((i[1]*y_scale))] for i in scaled_coords]
                        f_dict['geometry']['coordinates'] = [scaled_coords]

                    elif f['type']=='rectangle':
                        width = f['width']
                        height = f['height']
                        center = f['center'][0:-1]
                        # Coords: top left, top right, bottom right, bottom left
                        bbox_coords = [
                            [int(center[0])-int(width/2),int(center[1])-int(height/2)],
                            [int(center[0])+int(width/2),int(center[1])-int(height/2)],
                            [int(center[0])+int(width/2),int(center[1])+int(height/2)],
                            [int(center[0])-int(width/2),int(center[1])+int(height/2)]
                        ]
                        scaled_coords = [[base_x_scale*(i[0]*x_scale),base_y_scale*(i[1]*y_scale)] for i in bbox_coords]
                        f_dict['geometry']['coordinates'] = [scaled_coords]

                    # Who even cares about circles and ellipses??

                    # If any user-provided metadata is provided per element add it to "properties" key                       
                    if 'user' in f:
                        f_dict['properties'] = f['user']

                    f_dict['properties']['name'] = f_name

                    final_ann['features'].append(f_dict)


        return final_ann, base_x_scale*x_scale, base_y_scale*y_scale

    def get_resource_map_data(self,resource):
        # Getting all of the necessary materials for loading a new slide

        # Step 1: get resource item id
        # lol
        if os.sep in resource:
            item_id = self.get_resource_id(resource)
        else:
            item_id = resource

        # Step 2: get resource tile metadata
        tile_metadata = self.get_tile_metadata(item_id)
        print(f'tile_metadata: {tile_metadata}')
        # Step 3: get tile, base, zoom, etc.
        # Number of zoom levels for an image
        zoom_levels = tile_metadata['levels']
        print(f'zoom_levels: {zoom_levels}')
        # smallest level dimensions used to generate initial tiles
        base_dims = [
            tile_metadata['sizeX']/(2**(zoom_levels-1)),
            tile_metadata['sizeY']/(2**(zoom_levels-1))
        ]
        # Getting the tile dimensions (used for all tiles)
        tile_dims = [
            tile_metadata['tileWidth'],
            tile_metadata['tileHeight']
        ]
        # Original image dimensions used for scaling annotations
        image_dims = [
            tile_metadata['sizeX'],
            tile_metadata['sizeY']
        ]

        print(f'base_dims: {base_dims}')
        print(f'tile_dims: {tile_dims}')
        print(f'image_dims: {image_dims}')

        # Step 4: Defining bounds of map
        map_bounds = [[0,image_dims[1]],[0,image_dims[0]]]

        # Step 5: Getting annotations for a resource
        annotations = self.get_annotations(item_id)

        # Step 6: Converting Histomics/large-image annotations to GeoJSON
        geojson_annotations, x_scale, y_scale = self.convert_json(annotations,image_dims,base_dims,tile_dims)

        map_bounds[0][1]*=x_scale
        map_bounds[1][1]*=y_scale

        print(f'map_bounds: {map_bounds}')

        # Step 7: Getting user token and tile url
        user_token = self.get_token()
        tile_url = self.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+user_token

        return map_bounds, base_dims, image_dims, tile_dims[0], zoom_levels-1, geojson_annotations, x_scale, y_scale, tile_url

    def get_cli_list(self):
        # Get a list of possible CLIs available for current user
        #TODO: Find out the format of what is returned from this and reorder

        cli = self.gc.get('/slicer_cli_web/cli')
        self.cli_dict_list = cli

        return cli

    def initialize_folder_structure(self,path,path_type):

        self.current_collection_path = path
        self.current_collection_id = self.gc.get('resource/lookup',parameters={'path':self.current_collection_path})['_id']

        # Getting contents of base collection
        collection_contents = self.gc.get(f'resource/{self.current_collection_id}/items',parameters={'type':path_type})
        # Reducing list to only images
        collection_slides = [i for i in collection_contents if 'largeImage' in i]
        # folderIds for each item (determining ordering of slides)
        slide_folderIds = [i['folderId'] for i in collection_slides]
        # Assigning each slide to a dictionary by shared folderId
        self.slide_datasets = {}
        for f in np.unique(slide_folderIds):
            self.slide_datasets[f] = {}
            folder_name = self.gc.get(f'/folder/{f}')['name']
            self.slide_datasets[f]['name'] = folder_name
        
            folder_slides = [i for i in collection_slides if i['folderId']==f]

            # The item dictionaries for each of the slides will also include metadata, id, etc.
            self.slide_datasets[f]['Slides'] = folder_slides

            # Aggregating non-dictionary metadata
            folder_slide_meta = [i['meta'] for i in folder_slides]
            # Get all the unique keys present in this folder's metadata
            meta_keys = []
            for i in folder_slide_meta:
                meta_keys.extend(list(i.keys()))

            # Not adding dictionaries to the folder metadata
            # Assuming the same type is shared for each item sharing a key 
            #TODO: Include check for types of each member of an item metadata just for safety
            self.slide_datasets[f]['Metadata'] = {}
            for m in meta_keys:
                item_metadata = [item[m] for item in folder_slide_meta if m in item]
                if type(item_metadata[0])==str:
                    self.slide_datasets[f]['Metadata'][m] = ','.join(list(set(item_metadata)))
                elif type(item_metadata[0])==int or type(item_metadata[0])==float:
                    self.slide_datasets[f]['Metadata'][m] = sum(item_metadata)

    def get_image_region(self,item_id,coords_list):

        # Pulling specific region from an image using provided coordinates
        image_region = Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{item_id}/tiles/region?token={self.user_token}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}').content))

        return image_region

    def upload_data(self,data,data_name):

        # Finding the current user's private folder
        user_folder = f'/user/{self.gc.get("/user/me")["login"]}/Private'
        private_folder_id = self.gc.get('/resource/lookup',parameters={'path':user_folder})['_id']
        print(f'user_folder: {user_folder}')
        print(f'private_folder_id: {private_folder_id}')
        # Trying to just upload the entire file at once?
        """
        print('Starting upload')

        # Finding file size
        print(f'len of bytes string: {len(data)}')
        print(f'n_unique characters: {len(set(data))}')
        print(f'type of data: {type(data)}')
        #upload_size = 4*(ceil(len(data)/3))
        upload_size = 8913183
        print(f'upload_size: {upload_size}')

        response = self.gc.post(f'/file?token={self.user_token}',
                     data={'image':data},
                     headers = {
                         'X-HTTP-Metod':'POST',
                         'Content-Type':'image/jpeg'
                     },
                     parameters={
                         'parentType':'folder',
                         'parentId':private_folder_id,
                         'name':data_name,
                         'size':upload_size,
                         'mimeType':'image/jpeg'
                         }
                    )
        self.gc.post(f'/file/completion?token={self.user_token}',
                     parameters={'uploadId':response['_id']})
        print(response)
        print(f'Upload completed: {data_name}')
        """
        return private_folder_id



    """
    def get_cli_input_list(self,cli_id):
        #TODO: figure out how to extract list of expected inputs & types for a given CLI from XML
    
    """
    """
    def post_cli(self,cli_id,inputs):
        #TODO: figure out how to post a specific CLI with expected inputs, keeping track of job status, and returning outputs to FUSION

    """

class DownloadHandler:
    def __init__(self,
                dataset_object,
                verbose = False):
        
        self.dataset_object = dataset_object
        self.verbose = verbose

        # Placeholder for lots of re-formatting and stuff

    def what_data(self,download_options):
        # Figuring out what data to return based on download_options used

        if download_options in ['Aperio XML','Histomics JSON','GeoJSON']:

            return 'annotations'
        
        elif download_options in ['CSV Files','Excel File','RDS File']:

            return 'cell'
        
        elif any([i in ['FTU Properties','Tissue Type','Omics Type','Slide Metadata','FTU Counts'] for i in download_options]):

            return 'metadata'
        
        elif 'man' in download_options:

            return 'manual_rois'
        
        else:

            return 'select_ftus'

    def zip_data(self,download_data_list):

        # Writing temporary directory 
        if not os.path.exists('./assets/FUSION_Download/'):
            os.makedirs('./assets/FUSION_Download/')

        output_file = './assets/FUSION_Download.zip'
        # Writing files to the temp directory
        for d in download_data_list:
            filename = d['filename']
            file_ext = filename.split('.')[-1]

            save_path = './assets/FUSION_Download/'+filename

            if file_ext in ['xml','json','geojson']:

                if file_ext == 'xml':
                    
                    # Writing xml data
                    with open(save_path,'w') as f:
                        f.write(d['content'])
                        f.close()
                else:
                    # Writing JSON data
                    with open(save_path,'w') as f:
                        dump(d['content'],f)

            elif file_ext == 'csv':

                d['content'].to_csv(save_path)
            
            elif file_ext == 'xlsx':
                # If it's supposed to be an excel file, include sheet name as a key
                with pd.ExcelWriter(save_path,mode='a') as writer:
                    d['content'].to_excel(writer,sheet_name=d['sheet'],engine='openpyxl')

        # Writing temporary data to a zip file
        with zipfile.ZipFile(output_file,'w', zipfile.ZIP_DEFLATED) as zip:
            for file in os.listdir('./assets/FUSION_Download/'):
                zip.write('./assets/FUSION_Download/'+file)

        try:
            shutil.rmtree('./assets/FUSION_Download/')
        except OSError as e:
            print(f'OSError removing FUSION_Download directory: {e.strerror}')

    def extract_annotations(self, slide, format):
        
        # Extracting annotations from the current slide object
        annotations = slide.geojson_ftus

        width_scale = slide.x_scale
        height_scale = slide.y_scale

        if format=='GeoJSON':
            
            save_name = slide.slide_name.replace('.'+slide.slide_ext,'.geojson')

            # Have to re-normalize coordinates and only save one or two properties
            final_ann = {'type':'FeatureCollection','features':[]}
            for f in annotations['features']:
                f_dict = {'type':'Feature','geometry':{'type':'Polygon','coordinates':[]}}

                scaled_coords = np.squeeze(np.array(f['geometry']['coordinates']))
                if len(np.shape(scaled_coords))==2:
                    scaled_coords[:,0] *= height_scale
                    scaled_coords[:,1] *= width_scale
                    scaled_coords = scaled_coords.astype(int).tolist()
                    f_dict['geometry']['coordinates'] = scaled_coords

                    f_dict['properties'] = {'label':f['properties']['label'], 'structure':f['properties']['structure']}

                    final_ann['features'].append(f_dict)

        elif format == 'Aperio XML':
            
            save_name = slide.slide_name.replace('.'+slide.slide_ext,'.xml')

            # Initializing xml file
            final_ann = ET.Element('Annotations')

            for l,name in enumerate(list(slide.ftus.keys())):
                # Need some kinda random color here
                ann = ET.SubElement(final_ann,'Annotation',attrib={'Type':'4','Visible':'1','ReadOnly':'0',
                                                            'Incremental':'0','LineColorReadOnly':'0',
                                                            'LineColor':'65280','Id':str(l),'NameReadOnly':'0'})
                
                regions = ET.SubElement(ann,'Regions')

                # Getting the features to add to this layer based on name
                layer_features = [i for i in annotations['features'] if i['properties']['structure']==name]

                for r_idx,f in enumerate(layer_features):

                    region = ET.SubElement(regions,'Region',attrib = {'NegativeROA':'0','ImageFocus':'-1',
                                                                    'DisplayId':'1','InputRegionId':'0',
                                                                    'Analyze':'0','Type':'0','Id':str(r_idx),
                                                                    'Text': f['properties']['label']})

                    verts = ET.SubElement(region,'Vertices')
                    scaled_coords = np.squeeze(np.array(f['geometry']['coordinates']))
                    if len(np.shape(scaled_coords))==2:
                        scaled_coords[:,0] *= height_scale
                        scaled_coords[:,1] *= width_scale
                        scaled_coords = scaled_coords.astype(int).tolist()

                        for v in scaled_coords:
                            ET.SubElement(verts,'Vertex',attrib={'X':str(v[1]),'Y':str(v[0]),'Z':'0'})

                        ET.SubElement(verts,'Vertex',attrib={'X':str(scaled_coords[0][1]),'Y':str(scaled_coords[0][0]),'Z':'0'})

            final_ann = ET.tostring(final_ann,encoding='unicode',pretty_print=True)

        elif format == 'Histomics JSON':
            
            save_name = slide.slide_name.replace('.'+slide.slide_ext,'.json')

            # Following histomics JSON formatting
            final_ann = []

            for ftu_name in list(slide.ftus.keys()):

                output_dict = {'name':ftu_name,'attributes':{},'elements':[]}
                ftu_annotations = [i for i in annotations['features'] if i['properties']['structure']==ftu_name]

                for f in ftu_annotations:
                    scaled_coords = np.squeeze(np.array(f['geometry']['coordinates']))
                    if len(np.shape(scaled_coords))==2:
                        scaled_coords[:,0] *= height_scale
                        scaled_coords[:,1] *= width_scale
                        scaled_coords = scaled_coords.astype(int).tolist()

                        struct_id = uuid.uuid4().hex[:24]
                        struct_dict = {
                            'type':'polyline',
                            'points':[i+[0] for i in scaled_coords],
                            'id':struct_id,
                            'closed':True,
                            'user':{
                                'label':f['properties']['label']
                            }
                        }
                        output_dict['elements'].append(struct_dict)

            final_ann.append(output_dict)

        

        return [{'filename': save_name, 'content':final_ann}]
    
    """
    def extract_metadata(self,slides, include_meta):
    
    """
    def extract_cell(self, intersecting_ftus, file_name):
        # Output here is a dictionary containing Main_Cell_Types and Cell_States for each FTU
        # Main_Cell_Types is a pd.DataFrame with a column for every cell type and an index of the FTU label
        # Cell states is a dictionary of pd.DataFrames with a column for every cell state and an index of the FTU label for each main cell type

        # Formatting for downloads
        download_data = []
        for ftu in list(intersecting_ftus.keys()):

            # Main cell types should just be one file
            if 'xlsx' in file_name:
                main_content = {'filename':file_name,'sheet':ftu,'content':intersecting_ftus[ftu]['Main_Cell_Types']}
            elif 'csv' in file_name:
                main_content = {'filename':file_name,'content':intersecting_ftus[ftu]['Main_Cell_Types']}
            else:
                print('Invalid format (RDS will come later)')
                main_content = {}
            
            download_data.append(main_content)

            # Cell state info
            for mc in intersecting_ftus[ftu]['Cell_States']:
                if 'xlsx' in file_name:
                    state_content = {'filename':file_name.replace('.xlsx','')+f'_{ftu}_Cell_States.xlsx','sheet':mc.replace('/',''),'content':intersecting_ftus[ftu]['Cell_States'][mc]}
                elif 'csv' in file_name:
                    state_content = {'filename':file_name.replace('.csv','')+f'_{ftu}_{mc.replace("/","")}_Cell_States.csv','content':intersecting_ftus[ftu]['Cell_States'][mc]}
                else:
                    print('Invalid format (RDS will come later)')
                    state_content = {}

                download_data.append(state_content)

        return download_data

    """
    def extract_ftu(self, slide, data):
    
    """
    """

    def extract_manual(self, slide, data):
    """


