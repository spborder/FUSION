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
from io import BytesIO, StringIO
import requests
from math import ceil
import base64
from datetime import datetime
from time import time
import tifffile

import plotly.express as px
import plotly.graph_objects as go
#from skimage.draw import polygon_perimeter

from wsi_annotations_kit import wsi_annotations_kit as wak
from shapely import Polygon

from dash import dcc, ctx, Dash, dash_table
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_leaflet as dl
import dash_mantine_components as dmc
import dash_treeview_antd as dta
import dash_draggable as drag

from dash_extensions.enrich import html
from dash_extensions.javascript import arrow_function, assign

import girder_client
from tqdm import tqdm
from timeit import default_timer as timer
import textwrap

from typing_extensions import Union

from scipy import stats
from sklearn.metrics import silhouette_score, silhouette_samples

class LayoutHandler:
    def __init__(self,
                 verbose = False):
        
        self.verbose = verbose

        self.validation_layout = []
        self.layout_dict = {}
        self.description_dict = {}

        self.info_button_idx = -1
        self.cli_list = None

        # Creating figure dictionary for nephron diagram
        neph_figure = go.Figure(px.imshow(Image.open('./assets/cell_graphics/9 april 10.png')))
        neph_figure.update_traces(hoverinfo='none',hovertemplate=None)
        neph_figure.update_xaxes(showticklabels=False, showgrid=False)
        neph_figure.update_yaxes(showticklabels=False, showgrid=False)
        neph_figure.update_layout(margin={'l':0,'b':0,'r':0,'t':0},autosize=True)

        self.neph_figure = neph_figure

        self.gen_welcome_layout()

    def gen_info_button(self,text):
        
        self.info_button_idx+=1

        info_button = html.Div([
            html.I(
            className='bi bi-info-circle-fill me-2',
            id={'type':'info-button','index':self.info_button_idx}
            ),
            dbc.Popover(
                children = [
                    html.Img(src='./assets/fusey_clean.svg',height=20,width=20),
                    text],
                target = {'type':'info-button','index':self.info_button_idx},
                body=True,
                trigger='hover'
            )
        ])

        return info_button

    def gen_vis_layout(self, gene_handler, cli_list = None):

        #cell_types, zoom_levels, map_dict, spot_dict, slide_properties, tile_size, map_bounds,
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

        # This is just to populate these components. This part should never be visible
        #map_url = 'https://placekitten.com/240/240?image={z}'
        map_url = './assets/Initial tiletest.png'
        tile_size = 240
        slide_properties = []
        combined_colors_dict = {}
        zoom_levels = 3
        map_bounds = [tile_size,tile_size]
        
        center_point = [-tile_size,tile_size]

        map_children = [
            html.Div(
                id = 'slide-tile-holder',
                children = [
                    dl.TileLayer(
                        id = f'slide-tile{np.random.randint(0,100)}',
                        url = map_url,
                        tileSize = tile_size,
                        maxNativeZoom=zoom_levels-2
                    )
                ]
            ),
            dl.FullScreenControl(position='topleft'),
            html.Div(
                id = 'edit-control-holder',
                children = [
                    dl.FeatureGroup(
                        id='feature-group',
                        children = [
                            dl.EditControl(
                                id = {'type':'edit-control','index':0},
                                draw = dict(polyline=False, line=False, circle = False, circlemarker=False),
                                position='topleft'
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id='colorbar-div',
                children = []
            ),
            html.Div(
                id = 'layer-control-holder',
                children = [
                    dl.LayersControl(
                        id='layer-control',
                        children = []
                        )
                    ]
            ),
            #dl.EasyButton(icon='fa-solid fa-user-doctor', title='Ask Fusey!',id='fusey-button',position='bottomright'),
            #html.Div(id='ask-fusey-box',style={'visibility':'hidden','position':'absolute','top':'50px','right':'10px','zIndex':'1000'}),
            dl.EasyButton(
                icon = 'fa-solid fa-arrows-to-dot',
                title = 'Re-Center Map',
                id = 'center-map',
                position = 'top-left',
                eventHandlers = {
                    'click': assign('function(e,ctx){ctx.map.flyTo([-120,120],1);}')
                }
            ),
            html.Div(id='marker-add-div',children = []),
            html.Div(
                id = 'dummy-div-user-annotations',
                style={'display': 'none'}
            ),
            html.Div(
                id = 'user-annotations-div',
                style={'display': 'none'}
            )
        ]

        map_layer = dl.Map(
            center = center_point, zoom = 0, minZoom = 0, crs='Simple',
            style = {'width':'100%','height':'90vh','margin':'auto','display':'inline-block'},
            id = 'slide-map',
            zoomDelta = 0.25,
            preferCanvas=True,
            children = map_children,
            eventHandlers={
                'contextmenu': assign('function(e,ctx){console.log(`Clicked coordinates: ${e.latlng}, map center: ${ctx.map.getCenter()}`);}')
            }
        )

        wsi_view = dbc.Card([
            dbc.CardHeader(
                children = [
                    dbc.Col('Whole Slide Image Viewer',md=11),
                    dbc.Col(self.gen_info_button('Use the mouse to pan and zoom around the slide!'),md=1)
            ]),
            dbc.Row([
                html.Div(
                    map_layer
                )
            ])
        ])

        # Cell type proportions and cell state distributions
        roi_pie = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.P('This tab displays the cell type and state proportions for cell types contained within specific FTU & Spot boundaries')
                ]),
                html.Hr(),
                html.Div(
                    dmc.Switch(
                        id = 'viewport-data-update',
                        size = 'lg',
                        onLabel = 'ON',
                        offLabel = 'OFF',
                        checked = True,
                        label = 'Update viewport data',
                        description = 'Update values below when moving around in the image'
                    )
                ),
                html.Hr(),
                html.Div(
                    id = 'roi-pie-holder',
                    children = [
                        'Move around on the slide to initialize cell composition view.'
                    ]
                ),
                html.Div(
                    dcc.Store(
                        id = {'type':'viewport-store-data','index':0},
                        storage_type='memory',
                        data = json.dumps({})
                    )
                )
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
                'line-color':'blue'
            }
            }
        ]

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
                                height = '80%',
                                width = '80%'
                            )
                        ]
                    )
                ],align='center',style={'marginBottom':'10px'})
            ]),
            dbc.CardFooter([
                html.P('Some of the icons in the cell diagrams are informed and inspired by KPMP.org, Biorender.com and "Advances and prospects for the Human BioMolecular Atlas Program (HuBMAP)"')
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
                    dbc.Col(dbc.Label('Select Organ: ',html_for = 'organ-hierarchy-select'),md = 2),
                    dbc.Col(
                        dcc.Dropdown(
                            options = [
                                {'label': i.title(), 'value': i}
                                for i in gene_handler.asct_b['Organ'].tolist() if not i in ['anatomical systems','bone marrow']
                            ],
                            value = 'kidney',
                            id = 'organ-hierarchy-select'
                        ),
                        md = 10
                    ),
                    html.Div(
                        dcc.Store(
                            id = 'organ-hierarchy-store',
                            storage_type = 'memory',
                            data = json.dumps({'organ': None,'table': None,'info': None})
                        )
                    )
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Row([
                                dbc.Col(html.H3('Nephron Diagram'),md=12)
                            ]),
                            dbc.Row(
                                dbc.Col([
                                    html.Div([
                                        dcc.Graph(id='neph-img',figure=self.neph_figure,style={'width':'95%'}),
                                        dcc.Tooltip(id='neph-tooltip',loading_text='')
                                    ])
                                ],md=12)
                            )
                        ], id = 'nephron-diagram'),
                        html.Div(
                            id = 'organ-hierarchy-cell-select-div',
                            children = [
                                html.H2('More Organ Graphics Coming Soon!'),
                                dbc.Col([
                                    dbc.Label('Select a cell type: ',html_for='organ-hierarchy-cell-select')
                                ],md=2),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id = 'organ-hierarchy-cell-select',
                                        options = '',
                                        value = ''
                                    )
                                ],md=10)
                            ],
                            style = {'display': 'none'}
                        )
                    ],md=5,align='center'),
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(cell_graphic_tab, label = 'Cell Graphic',tab_id = 'cell-graphic-tab'),
                            dbc.Tab(cell_hierarchy_tab, label = 'Cell Hierarchy')
                        ],active_tab='cell-graphic-tab')
                    ],md=7)
                ],align='center')
            ]),
            dbc.CardFooter(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Link('Derived from ASCT+B Kidney v1.2',href='https://docs.google.com/spreadsheets/d/1NMfu1bEGNFcTYTFT-jCao_lSbFD8n0ti630iIpRj-hw/edit#gid=949267305',target = '_blank')
                        ])
                    ])
                ])
            )
        ])

        # Cluster viewer tab
        cluster_card = dbc.Card([
            dbc.Row([
                html.P('Use this tab to dynamically view clustering results of morphological properties for select FTUs')
            ]),
            dcc.Store(
                id = 'cluster-store',
                data = json.dumps({'clustering_data':[],'feature_data':[],'umap_df':[]}),
                storage_type = 'memory'
            ),
            html.Hr(),
            dbc.Row(html.Div(id='get-data-div',style={'display':'flex'})),
            html.Hr(),
            dbc.Row([
                html.Div(dbc.Col([
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
                                    dbc.Col('Select Feature(s)',md=11),
                                    dbc.Col(self.gen_info_button('Select one or more features below to plot. Selecting more than 2 features will generate a UMAP'),md=1)
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    html.Div(
                                        dcc.Loading(
                                            dta.TreeView(
                                                id='feature-select-tree',
                                                multiple=True,
                                                checkable=True,
                                                checked = [],
                                                selected = [],
                                                expanded=[],
                                                data = []
                                            )
                                        ),
                                        style={'maxHeight':'250px','overflow':'scroll'}
                                    ),
                                    html.B(),
                                    dcc.RadioItems(
                                        options = [
                                            {'label':html.Span('Include Cell States separately?',style={'padding':'10px'}),'value':'separate'},
                                            {'label':html.Span('Only Main Cell Types Proportions',style={'padding':'10px'}),'value':'main'}
                                            ],
                                        value = 'main',
                                        id = 'cell-states-clustering',
                                        inline=True
                                    )
                                ]),
                                html.Hr(),
                                html.Div(
                                    id = 'label-and-filter-div',
                                    children = [
                                        dbc.Row([
                                            dbc.Col(dbc.Label('Select Filter'),md=11),
                                            dbc.Col(self.gen_info_button('Select specific label items to remove from your plot'),md=1)
                                        ]),
                                        html.Hr(),
                                        dbc.Row([
                                            html.Div(id='filter-info',children = [],style={'marginBottom':'5px'}),
                                            dcc.Loading(
                                                html.Div(
                                                    dta.TreeView(
                                                        id = 'filter-select-tree',
                                                        multiple = True,
                                                        checkable = True,
                                                        checked = [],
                                                        selected = [],
                                                        expanded = [],
                                                        data = []
                                                    ),
                                                    style = {'maxHeight':'250px','overflow':'scroll'}
                                                )
                                            )
                                        ]),
                                        html.Hr(),
                                        dbc.Row([
                                            html.Div(
                                                dbc.Button(
                                                    'Generate Plot!',
                                                    id = 'gen-plot-butt',
                                                    n_clicks = 0,
                                                    disabled = False,
                                                    class_name = 'd-grid col-12 mx-auto'
                                                )
                                            ),
                                            html.Div(id='gen-plot-alert')
                                        ])
                                    ]
                                )
                            ])
                        ]
                    )
                ],md=12),style={'maxHeight':'30vh','overflow':'scroll'})
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col('Select Label',md=11,style={'marginLeft':'5px'}),
                dbc.Col(self.gen_info_button('Select a label for the plot of selected features'),md=1)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        dcc.Dropdown(
                            options = [],
                            id = 'label-select',
                            disabled=True
                        )
                    )
                ],md=8),
                dbc.Col([
                    html.Div(id = 'label-info',children = [])
                ],md=4)
            ],align='center'),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    self.gen_info_button('Click on a point in the graph or select a group of points with the lasso select tool to view the FTU and cell type data at that point'),
                    html.Div(
                        dcc.Loading(dcc.Graph(id='cluster-graph',figure=go.Figure()))
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
                                                        ],md=6),
                                                    dbc.Col(dcc.Graph(id='selected-cell-states',figure=go.Figure()),md=6)
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),label='Selected Cell Data'),
                        dcc.Tab(
                            dbc.Card(
                                id = 'selected-labels-card',
                                children = [
                                    dbc.Row(
                                        children = [
                                            html.Div(id='selected-image-info',children = []),
                                            self.gen_info_button('Distribution of labels in selected samples')
                                        ]
                                    )
                                ]
                            ), label = 'Selected Labels'
                        )
                    ])
                ],md=6),
            ],align='start'),
            html.Hr(),
            dbc.Row([
                dbc.Col(html.H3('Plot Report'),md=11),
                dbc.Col(
                    self.gen_info_button('Generate distribution summaries, simple statistics, and find cluster markers!'),
                    md = 1
                )            
            ]),
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Tabs(
                            id = 'plot-report-tab',
                            active_tab='feat-summ-tab',
                            children = [
                                dbc.Tab(
                                    label = 'Feature Summaries',
                                    tab_id = 'feat-summ-tab'
                                ),
                                dbc.Tab(
                                    label = 'Statistics',
                                    tab_id = 'feat-stat-tab'
                                ),
                                dbc.Tab(
                                    label = 'Cluster Markers',
                                    tab_id = 'feat-cluster-tab'
                                )
                            ]
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            html.Div(id='plot-report-div',children = [],style={'maxHeight':'50vh','overflow':'scroll'})
                            )
                        ])
                    ])
                ]),
            html.Hr(),
            dbc.Row([
                html.Div(
                    dcc.Loading([
                        dbc.Button(
                            'Download Plot Data',
                            id = 'download-plot-butt',
                            disabled=True,
                            className='d-grid col-12 mx-auto'
                        ),
                        dcc.Download(id='download-plot-data')
                    ]),
                    id = 'download-plot-data-div',
                    style={'display':None}
                )
            ])
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
            {'label':'Manual ROIs','value':'Manual ROIs','disabled':True},
            {'label':'FTU User Labels','value':'FTU User Labels','disabled':True}
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
        if cli_list is None:
            if self.cli_list is not None:
                available_clis = self.cli_list
            else:
                available_clis = []
        else:
            available_clis = cli_list
            self.cli_list = cli_list

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
                                dcc.Dropdown(options = cell_types_list, placeholder='Select Property for Overlaid Heatmap',id='cell-drop'),
                                html.Div(id='cell-sub-select-div',children = [],style={'marginTop':'5px'})
                            ]
                        ),
                        html.Div(
                            id = 'special-overlays',
                            children = []
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
                            allowCross=False,
                            disabled = True
                        )
                    ])
                ]),
                dbc.Row([
                    html.Div(
                        id = 'parent-added-filter-div',
                        children = [
                            html.Div(
                                id = {'type':'added-filter-div','index':0},
                                children = [
                                    html.I(
                                        className='bi bi-x-fill fa-2x',
                                        id = {'type':'delete-filter','index':0},
                                        style = {'display':'none'}
                                    )                                    
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        html.I(
                            className='bi bi-filter-circle fa-2x',
                            n_clicks = 0,
                            id='add-filter-button',
                            style = {'display':'inline-block','position':'relative','left':'45%','right':'50%'}
                        ),
                    )
                ],align='center'),
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
                                    dmc.ColorPicker(
                                        id =  {'type':'ftu-bound-color','index':idx},
                                        format = 'hex',
                                        value = combined_colors_dict[struct]['color'],
                                        fullWidth=True
                                    ),
                                ],
                                label = struct
                            )
                            for idx,struct in enumerate(list(combined_colors_dict.keys()))
                        ]
                    )
                ]),
                dbc.Row([
                    dbc.Label(
                        'Structure Hierarchy',
                        html_for = 'structure-hierarchy'
                    ),
                    self.gen_info_button('Update which structures should be considered "contained" within the boundaries of other structures. Click Reset to revert back to the original annotations.')
                ],style={'display':'none'}),
                dbc.Row([
                    dbc.Tabs(
                        id = 'ftu-structure-hierarchy',
                        children = [
                            dbc.Tab(
                                children = [
                                    html.Div(
                                        id = {'type':'structure-tree-div','index':idx},
                                        children = ['blah blah test']
                                    ),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button(
                                                'Reset',
                                                className = 'd-grid col-12 mx-auto',
                                                id = {'type':'reset-structure-hierarchy','index':idx}
                                            ),
                                            md = 6
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                'Update Structure',
                                                className = 'd-grid col-12 mx-auto',
                                                id = {'type': 'update-structure-hierarchy','index':idx}
                                            ),
                                            md = 6
                                        )
                                    ], align = 'center', style = {'marginTop':'5px'})
                                ],
                                label = struct,
                                tab_id = struct
                            )
                            for idx,struct in enumerate(list(combined_colors_dict.keys()))
                        ]
                    )
                ],style={'display':'none'})
            ])
        ])

        # Annotation session tab
        annotation_session_tab = dbc.Card([
            dbc.CardBody([
                html.Div(
                    id = 'annotation-session-div',
                    children = []
                )
            ])
        ])

        # Cell annotation tab
        cell_annotation_tab = dbc.Card([
            dbc.CardBody([
                html.Div(
                    id = 'cell-annotation-div',
                    children = []
                )
            ])
        ])


        # List of all tools tabs
        tool_tabs = [
            dbc.Tab(overlays_tab, label = 'Overlays',tab_id='overlays-tab'),
            dbc.Tab(roi_pie, label = "Cell Compositions",tab_id='cell-compositions-tab'),
            dbc.Tab(cell_card,label = "Cell Graphics",tab_id='cell-graphics-tab'),
            dbc.Tab(cluster_card,label = 'Morphological Clustering',tab_id='clustering-tab'),
            dbc.Tab(extract_card,label = 'Download Data',tab_id='download-tab'),
            dbc.Tab(annotation_session_tab, label = 'Annotation Station', tab_id = 'annotation-tab'),
            dbc.Tab(cell_annotation_tab, label = "Cell Annotation", tab_id = 'cell-annotation-tab',disabled = True,id = 'cell-annotation-tab')
            #dbc.Tab(cli_tab,label = 'Run Analyses',disabled = True,tab_id='analyses-tab'),
        ]
        
        tools = [
            dbc.Card(
                id='tools-card',
                children=[
                    dbc.CardHeader("Tools"),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Tabs(tool_tabs,active_tab = 'overlays-tab',id='tools-tabs')
                            ])
                        ],style={'maxHeight':'90vh','overflow':'scroll'})
                    ])
                ]
            )
        ]

        # Separately outputting the functional components of the application for later reference when switching pages
        use_drag_layout = False
        
        if use_drag_layout:
            vis_content = [
                dbc.Row(
                    id="app-content",
                    children=[
                        html.Div(
                            children = [
                                drag.ResponsiveGridLayout(
                                    children = [
                                        dbc.Col(
                                            wsi_view,
                                            style = {
                                                'min-height':"0",
                                                "flex-grow":"1",
                                                'height':'100%'
                                            }),
                                        dbc.Col(
                                            tools,
                                            style = {
                                                "min-height":"0",
                                                "flex-grow":"1",
                                                'height':'100%'
                                            })
                                    ],
                                    style = {
                                        'height':'100%',
                                        'width':'100%',
                                        'display':'flex',
                                        'flex-direction':'row',
                                        'flex-grow':'0'
                                    }
                                )
                            ]
                        )
                    ],style={"height":"90vh",'marginBottom':'10px'}
                )
            ]

        else:
            vis_content = [
                dbc.Row(
                    id="app-content",
                    children=[
                        dbc.Col(
                            wsi_view,
                            md = 6
                        ),
                        dbc.Col(
                            tools,
                            md = 6
                        )
                    ],
                    style={"height":"90vh",'marginBottom':'10px'}
                )
            ]

        self.current_vis_layout = vis_content
        self.validation_layout.append(vis_content)
        self.layout_dict['vis'] = vis_content
        self.description_dict['vis'] = vis_description

    def gen_report_child(self,feature_data,child_type):

        # Generate new report of feature data        
        feature_data = pd.DataFrame.from_dict(feature_data,orient='index').copy()
        unique_labels = np.unique(feature_data['label'].tolist()).tolist()

        if type(unique_labels[0]) in [int,float]:
            return 'Plot report only implemented for string labels!'

        # Generating a different report depending on the type 
        # one of ['feat-summ-tab','feat-stat-tab','feat-cluster-tab']
        if child_type=='feat-summ-tab':

            # Find summaries for the numeric 
            report_children = []
            for u_l_idx,u_l in enumerate(unique_labels):
                report_children.extend([
                    html.H3(f'Samples labeled: {u_l}'),
                    html.Hr()
                ])

                label_data = feature_data[feature_data['label'].str.match(u_l)]
                data_summ = label_data.describe().round(decimals=4)
                data_summ.reset_index(inplace=True,drop=False)

                report_children.append(
                    dash_table.DataTable(
                        id = {'type':'feat-summ-table','index':u_l_idx},
                        columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in data_summ.columns],
                        data = data_summ.to_dict('records'),
                        editable = False,
                        style_cell = {
                            'overflowX':'auto'
                        },
                        tooltip_data = [
                            {
                                column: {'value':str(value),'type':'markdown'}
                                for column, value in row.items()
                            } for row in data_summ.to_dict('records')
                        ],
                        tooltip_duration = None,
                        style_data_conditional = [
                            {
                                'if':{
                                    'column_id':'index'
                                },
                                'width':'35%'
                            }
                        ]
                    )
                )

                report_children.append(
                    html.Hr()
                )

        elif child_type == 'feat-stat-tab':

            # Generate some preliminary statistics describing data
            # feature_data starts with ['label','Hidden','Main_Cell_Types','Cell_States'] if there's cell type data
            feature_columns = [i for i in feature_data.columns.tolist() if i not in ['label','Hidden','Main_Cell_Types','Cell_States']]
            
            if len(feature_columns)==1:

                # This is a violin plot with a single feature plotted amongst some groups.
                if len(unique_labels)==1:

                    report_children = dbc.Alert(f'Only one label ({unique_labels[0]}) present!', color = 'warning')
                
                elif len(unique_labels)==2:
                    
                    # This is a t-test
                    group_a = feature_data[feature_data['label'].str.match(unique_labels[0])][feature_columns[0]].values
                    group_b = feature_data[feature_data['label'].str.match(unique_labels[1])][feature_columns[0]].values

                    stats_result = stats.ttest_ind(group_a, group_b)
                    t_statistic = stats_result.statistic
                    p_value = stats_result.pvalue
                    confidence_interval = stats_result.confidence_interval(confidence_level=0.95)

                    t_df = pd.DataFrame({
                        't Statistic':t_statistic,
                        'p Value': p_value,
                        '95% Confidence Interval (Lower)': confidence_interval.low,
                        '95% Confidence Interval (Upper)':confidence_interval.high
                    },index=[0]).round(decimals=4)

                    if p_value<0.05:
                        sig_alert = dbc.Alert('Statistically significant! (p<0.05)',color='success')
                    else:
                        sig_alert = dbc.Alert('Not statistically significant (p>=0.05)',color = 'warning')

                    report_children = [
                        sig_alert,
                        html.Hr(),
                        html.Div([
                            html.A('Statistical Test: t-Test',href='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html',target='_blank'),
                            html.P('Tests null hypothesis that two independent samples have identical mean values. Assumes equal variance within groups.')
                        ]),
                        html.Div(
                            dash_table.DataTable(
                                id = 'stats-table',
                                columns = [{'name':i,'id':i} for i in t_df.columns],
                                data = t_df.to_dict('records'),
                                style_cell = {
                                    'overflow':'hidden',
                                    'textOverflow':'ellipsis',
                                    'maxWidth':0
                                },
                                tooltip_data = [
                                    {
                                        column: {'value':str(value),'type':'markdown'}
                                        for column, value in row.items()
                                    } for row in t_df.to_dict('records')
                                ],
                                tooltip_duration = None
                            )
                        )
                    ]

                elif len(unique_labels)>2:

                    # This is a one-way ANOVA (Analysis of Variance)
                    group_data = []
                    for u_l in unique_labels:
                        group_data.append(
                            feature_data[feature_data['label'].str.match(u_l)][feature_columns[0]].values
                        )

                    stats_result = stats.f_oneway(*group_data)
                    f_stat = stats_result.statistic
                    p_value = stats_result.pvalue

                    anova_df = pd.DataFrame({
                        'F Statistic': f_stat,
                        'p Value':p_value
                    },index = [0]).round(decimals=4)

                    # Now performing Tukey's honestly significant difference (HSD) test for pairwise comparisons across different labels
                    # This returns an insane string. Usability score: >:(
                    # Honestly blown away by how idiotic it is to return a "results object" where the only methods are __str__ and updating the confidence interval.
                    # Imagine a program that just said the answer out loud through your speakers.
                    # THE CONFIDENCE INTERVAL DOESN'T EVEN EXIST UNTIL YOU PRINT WHAT THE HECK?? 
                    tukey_result = stats.tukey_hsd(*group_data)
                    _ = tukey_result.confidence_interval(confidence_level = 0.95)

                    # tukey_result is a TukeyHSDResult object, have to assemble the outputs manually because scipy developers are experiencing a gas leak.
                    tukey_data = []
                    for i in range(tukey_result.pvalue.shape[0]):
                        for j in range(tukey_result.pvalue.shape[0]):
                            if i != j:
                                row_dict = {
                                    'Comparison': ' vs. '.join([unique_labels[i],unique_labels[j]]),
                                    'Statistic': f'{tukey_result.statistic[i,j]:>10.3f}',
                                    'p-value': f'{tukey_result.pvalue[i,j]:>10.3f}',
                                    'Lower CI': f'{tukey_result._ci.low[i,j]:>10.3f}',
                                    'Upper CI': f'{tukey_result._ci.high[i,j]:>10.3f}'
                                }
                                tukey_data.append(row_dict)

                    tukey_df = pd.DataFrame(tukey_data).round(decimals=4)
                    

                    if p_value<0.05:
                        sig_alert = dbc.Alert('Statistically significant! (p<0.05)',color='success')
                    else:
                        sig_alert = dbc.Alert('Not statistically significant overall (p<=0.05)',color='warning')
                    
                    report_children = [
                        sig_alert,
                        html.Hr(),
                        html.Div([
                            html.A('Statistical Test: One-Way ANOVA',href='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html',target='_blank'),
                            html.P('Tests null hypothesis that two or more groups have the same population mean. Assumes independent samples from normal, homoscedastic (equal standard deviation) populations')
                        ]),
                        html.Div(
                            dash_table.DataTable(
                                id='stats-table',
                                columns = [{'name':i,'id':i} for i in anova_df.columns],
                                data = anova_df.to_dict('records'),
                                style_cell = {
                                    'overflow':'hidden',
                                    'textOverflow':'ellipsis',
                                    'maxWidth':0
                                },
                                tooltip_data = [
                                    {
                                        column: {'value':str(value),'type':'markdown'}
                                        for column, value in row.items()
                                    } for row in anova_df.to_dict('records')
                                ],
                                tooltip_duration = None
                            )
                        ),
                        html.Hr(),
                        html.Div([
                            html.A("Statistical Test: Tukey's HSD",href='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html',target='_blank'),
                            html.P('Post hoc test for pairwise comparison of means from different groups. Assumes independent samples from normal, equal (finite) variance populations')
                        ]),
                        html.Div(
                            dash_table.DataTable(
                                id='tukey-table',
                                columns = [{'name':i,'id':i} for i in tukey_df.columns],
                                data = tukey_df.to_dict('records'),
                                style_cell = {
                                    'overflow':'hidden',
                                    'textOverflow':'ellipsis',
                                    'maxWidth':0
                                },
                                tooltip_data = [
                                    {
                                        column: {'value':str(value),'type':'markdown'}
                                        for column,value in row.items()
                                    } for row in tukey_df.to_dict('records')
                                ],
                                tooltip_duration = None,
                                style_data_conditional = [
                                    {
                                        'if':{
                                            'column_id':'Comparison'
                                        },
                                        'width':'35%'
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{p-value} <0.05',
                                            'column_id':'p-value',
                                        },
                                        'backgroundColor':'green',
                                        'color':'white'
                                    },
                                    {
                                        'if':{
                                            'filter_query': '{p-value} >= 0.05',
                                            'column_id':'p-value'
                                        },
                                        'backgroundColor':'tomato',
                                        'color':'white'
                                    }
                                ]
                            )
                        )
                    ]

            elif len(feature_columns)==2:
                
                # This is for two dimensional scatter plots without dimensional reduction
                # Calculating Pearson's correlation coefficient for each label
                # Could also do some kind of Fisher Transformation to compare multiple Pearson r values
                report_children = [
                    html.Div([
                        html.A('Statistical Test: Pearson Correlation Coefficient (r)',href='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.pearsonr.html',target='_blank'),
                        html.P('Measures the linear relationship between two datasets. Assumes normally distributed data.')
                    ])
                ]
                pearson_r_list = []
                p_val_list = []
                for u_l in unique_labels:
                    # For each label, generate a new table with r
                    group_data = feature_data[feature_data['label'].str.match(u_l)].values
                    group_r,group_p = stats.mstats.pearsonr(group_data[:,0],group_data[:,1])
                    
                    pearson_r_list.append(group_r)
                    p_val_list.append(group_p)

                pearson_df = pd.DataFrame(data = {'Label': unique_labels, 'Pearson r': pearson_r_list, 'p-value':p_val_list}).round(decimals=4)

                report_children.append(
                    html.Div(
                        dash_table.DataTable(
                            id='pearson-table',
                            columns = [{'name':i,'id':i} for i in pearson_df.columns],
                            data = pearson_df.to_dict('records'),
                            style_cell = {
                                'overflow':'hidden',
                                'textOverflow':'ellipsis',
                                'maxWidth':0
                            },
                            tooltip_data = [
                                {
                                    column: {'value':str(value),'type':'markdown'}
                                    for column,value in row.items()
                                } for row in pearson_df.to_dict('records')
                            ],
                            tooltip_duration = None,
                            style_data_conditional = [
                                {
                                    'if': {
                                        'filter_query': '{p-value} <0.05',
                                        'column_id':'p-value',
                                    },
                                    'backgroundColor':'green',
                                    'color':'white'
                                },
                                {
                                    'if':{
                                        'filter_query': '{p-value} >= 0.05',
                                        'column_id':'p-value'
                                    },
                                    'backgroundColor':'tomato',
                                    'color':'white'
                                }
                            ]
                        )
                    )
                )

            elif len(feature_columns)>2:
                
                if len(unique_labels)>=2:
                    # Clustering scores? Silhouette score/index?
                    report_children = [
                        html.Div([
                            html.A('Clustering Metric: Silhouette Coefficient',href='https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score',target='_blank'),
                            html.P('Quantifies density of distribution for each sample. Values closer to 1 indicate high class clustering. Values closer to 0 indicate mixed clustering between classes. Values closer to -1 indicate highly dispersed distribution for a class.')
                        ])
                    ]

                    # Overall silhouette score for this set of data
                    overall_silhouette = round(silhouette_score(feature_data.values[:,0:2],feature_data['label'].tolist()),4)
                    print(f'overall silhouette score: {overall_silhouette}')
                    if overall_silhouette>=-1 and overall_silhouette<=-0.5:
                        silhouette_alert = dbc.Alert(f'Overall Silhouette Score: {overall_silhouette}',color='danger')
                    elif overall_silhouette>-0.5 and overall_silhouette<=0.5:
                        silhouette_alert = dbc.Alert(f'Overall Silhouette Score: {overall_silhouette}',color = 'primary')
                    elif overall_silhouette>0.5 and overall_silhouette<=1:
                        silhouette_alert = dbc.Alert(f'Overall Silhouette Score: {overall_silhouette}',color = 'success')
                    else:
                        silhouette_alert = dbc.Alert(f'Weird value: {overall_silhouette}')

                    report_children.extend([
                        html.Br(),
                        silhouette_alert,
                        html.Hr()
                    ])

                    samples_silhouette_scores = silhouette_samples(feature_data.values[:,0:2],feature_data['label'].tolist())
                    sil_dict = {'Label':[],'Silhouette Score':[]}
                    for u_l in unique_labels:
                        sil_dict['Label'].append(u_l)
                        sil_dict['Silhouette Score'].append(np.nanmean(samples_silhouette_scores[[i==u_l for i in feature_data['label'].tolist()]]))

                    sil_df = pd.DataFrame(sil_dict).round(decimals=4)

                    report_children.append(
                        html.Div(
                            dash_table.DataTable(
                                id='silhouette-table',
                                columns = [{'name':i,'id':i} for i in sil_df.columns],
                                data = sil_df.to_dict('records'),
                                style_cell = {
                                    'overflow':'hidden',
                                    'textOverflow':'ellipsis',
                                    'maxWidth':0
                                },
                                tooltip_data = [
                                    {
                                        column: {'value':str(value),'type':'markdown'}
                                        for column,value in row.items()
                                    } for row in sil_df.to_dict('records')
                                ],
                                tooltip_duration = None,
                                style_data_conditional = [
                                    {
                                        'if': {
                                            'filter_query': '{Silhouette Score}>0',
                                            'column_id':'Silhouette Score',
                                        },
                                        'backgroundColor':'green',
                                        'color':'white'
                                    },
                                    {
                                        'if':{
                                            'filter_query': '{Silhouette Score}<0',
                                            'column_id':'Silhouette Score'
                                        },
                                        'backgroundColor':'tomato',
                                        'color':'white'
                                    }
                                ]
                            )
                        )
                    )
                else:

                    report_children = [
                        dbc.Alert(f'Only one label: {unique_labels[0]} present!',color='warning')
                    ]

            else:

                # This should never happen lol
                report_children = [
                    'What did you do >:('
                ]

        elif child_type=='feat-cluster-tab':

            # Just return a button that runs cluster marker determination. Have to add this separately somewhere.
            feature_columns = [i for i in feature_data.columns.tolist() if i not in ['label','Hidden','Main_Cell_Types','Cell_States']]
            if len(feature_columns)>2:
                if len(unique_labels)>1:
                    report_children = [
                        dbc.Button(
                            'Find Cluster Markers!',
                            id = {'type':'cluster-markers-butt','index':0},
                            className = 'd-grid col-12 mx-auto'
                        ),
                        html.Div(
                            id = {'type':'cluster-marker-div','index':0}
                        )
                    ]
                else:
                    report_children = [
                        dbc.Alert(f'Only one label ({unique_labels[0]}) present!', color = 'warning')
                    ]
            else:
                report_children = [
                    dbc.Alert(f'Less than 2 features in view! Add more to view cluster markers',color='warning')
                ]

        return report_children
        
    def gen_usability_report(self,user_info,usability_info):

        # Generate a usability report/questionnaire based on the user type
        # If the user is able to click the usability button they are already in the 
        # usability study, it's just a question of whether they are an admin or a 
        # user and then if they are a user, what type of user are they?

        usability_children = []
        if user_info['type']=='admin':
            # Get a report of user responses and also separate by user type
            all_users = usability_info['usability_study_users']

            # Ideally the accordion items would be color-coded, can't get the CSS to stick though
            user_type_color = {
                'pathologist':'rgb(3,252,194)',
                'student':'rgb(252,244,3)',
                'biologist':'rgb(219,3,252)'
            }

            user_data = []
            for u in all_users:
                user_data.append({
                    'Username':u,
                    'User Type':all_users[u]['type'],
                    'Responded?':'Yes' if len(list(all_users[u]['responses'].keys()))>0 else 'No',
                    'Total Participants': 1,
                    'Task Responses':all_users[u]['responses'] if len(list(all_users[u]['responses'].keys()))>0 else 'No Responses'
                })
            
            user_df = pd.DataFrame.from_records(user_data)
            
            # Response chart (sunburst with inner part being total response and outer part being each user type)
            response_burst = px.sunburst(
                data_frame = user_df,
                path = ['User Type','Responded?','Username'],
                values = 'Total Participants'
            )

            usability_children.extend([
                dbc.Row([
                    dbc.Col([
                        html.Div(html.H3(f'Admin View: {user_info["login"]}')),
                        html.Div(
                            dcc.Graph(
                                figure = go.Figure(response_burst)
                            )
                        )
                    ],md=8),
                    dbc.Col([
                        html.Div(html.H3('Individual User Responses')),
                        html.Div(
                            dbc.Accordion([
                                dbc.AccordionItem(
                                    title = f'{all_users[user]["type"]}: {user}',
                                    children = json.dumps(all_users[user]['responses']),
                                    #className = f'{all_users[user]["type"]}-accordion-button'
                                )
                                for user in all_users
                            ]),
                            style = {'maxHeight':'50vh','overflow':'scroll'}
                        ),
                        html.Hr(),
                        html.Div([
                            dbc.Button(
                                'Download User Responses',
                                id = {'type':'download-usability-butt','index':0},
                                className = 'd-grid mx-auto'
                            ),
                            dcc.Download(id = {'type':'usability-download','index':0})
                        ])
                    ])
                ])
            ])

        else:

            # Creating the tutorials and stuff
            tutorials_col = dbc.Col([
                html.H3('Level Tutorials'),
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Tabs(
                            id = {'type':'tutorial-tabs','index':0},
                            active_tab = 'background-tab',
                            children = [
                                dbc.Tab(
                                    tab_id = 'background-tab',
                                    label = 'Background',
                                    activeTabClassName='fw-bold fst-italic'
                                ),
                                dbc.Tab(
                                    tab_id = 'start-tab',
                                    label = 'Start',
                                    activeTabClassName='fw-bold fst-italic'
                                ),
                                dbc.Tab(
                                    tab_id = 'histo-tab',
                                    label = 'Histology',
                                    activeTabClassName='fw-bold fst-italic'
                                ),
                                dbc.Tab(
                                    tab_id = 'omics-tab',
                                    label = 'Spatial -Omics',
                                    activeTabClassName='fw-bold fst-italic'
                                ),
                                dbc.Tab(
                                    tab_id = 'answer-tab',
                                    label = 'Answer Hypothesis',
                                    activeTabClassName='fw-bold fst-italic'
                                ),
                                dbc.Tab(
                                    tab_id = 'generate-tab',
                                    label = 'Generate Hypothesis',
                                    activeTabClassName='fw-bold fst-italic'
                                )
                            ]
                        )
                    ),
                    dbc.CardBody(
                        html.Div(id = {'type':'tutorial-content','index':0},children = [])
                    )
                ])
            ],md=6)

            if user_info['type']=='pathologist':

                questions_col = dbc.Col([
                    html.H3('User Responses'),
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Tabs(
                                id = {'type':'questions-tabs','index':0},
                                active_tab = 'consent-tab',
                                children = [
                                    dbc.Tab(
                                        tab_id = 'consent-tab',
                                        label = 'Study Consent',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':0}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-1-tab',
                                        label = 'Task 1: Histology',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':1}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-2-tab',
                                        label = 'Task 2: Spatial -Omics',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':2}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-3-tab',
                                        label = 'Task 3: Answer Hypothesis',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':3}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-4-tab',
                                        label = 'Task 4: Generate Hypothesis',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':4}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'comments-tab',
                                        label = 'Comments',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':5}
                                    )
                                ]
                            )
                        ),
                        dbc.CardBody(
                            html.Div(id = {'type':'question-div','index':0},
                                children = []
                            )
                        )
                    ])
                ],md=6)
            
            elif user_info['type']=='biologist':

                questions_col = dbc.Col([
                    html.H3('User Responses'),
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Tabs(
                                id = {'type':'questions-tabs','index':0},
                                active_tab = 'consent-tab',
                                children = [
                                    dbc.Tab(
                                        tab_id = 'consent-tab',
                                        label = 'Study Consent',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':0}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-1-tab',
                                        label = 'Task 1: Spatial -Omics',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':1}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-2-tab',
                                        label = 'Task 2: Histology',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':2}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-3-tab',
                                        label = 'Task 3: Answer Hypothesis',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':3}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-4-tab',
                                        label = 'Task 4: Generate Hypothesis',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':4}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'comments-tab',
                                        label = 'Comments',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':5}
                                    )
                                ]
                            )
                        ),
                        dbc.CardBody(
                            html.Div(id = {'type':'question-div','index':0},
                                children = []
                            )
                        )
                    ])
                ],md=6)

            elif user_info['type']=='student':

                questions_col = dbc.Col([
                    html.H3('User Responses'),
                    dbc.Card([
                        dbc.CardHeader(
                            dbc.Tabs(
                                id = {'type':'questions-tabs','index':0},
                                active_tab = 'consent-tab',
                                children = [
                                    dbc.Tab(
                                        tab_id = 'consent-tab',
                                        label = 'Study Consent',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':0}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-1-tab',
                                        label = 'Task 1: Histology',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':1}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-2-tab',
                                        label = 'Task 2: Spatial -Omics',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':2}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-3-tab',
                                        label = 'Task 3: Answer Hypothesis',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':3}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'level-4-tab',
                                        label = 'Task 4: Generate Hypothesis',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':4}
                                    ),
                                    dbc.Tab(
                                        tab_id = 'comments-tab',
                                        label = 'Comments',
                                        activeTabClassName='fw-bold fst-italic',
                                        disabled = True,
                                        id = {'type':'question-tab','index':5}
                                    )
                                ]
                            )
                        ),
                        dbc.CardBody(
                            html.Div(id = {'type':'question-div','index':0},
                                children = []
                            )
                        )
                    ])
                ],md=6)

            usability_children.extend([
                tutorials_col,
                questions_col
            ])


        return dbc.Row(usability_children)

    def gen_special_overlay_opts(self, slide_info):
        """
        Generate special overlay components depending on the WSI spatial omics type and properties
        """
        slide_type = slide_info['slide_type']

        special_overlays_opts = []
        if slide_type=='Visium':
            pass
            """
            if any(['Main_Cell_Types' in i for i in wsi.properties_list]):
                disable_sub_types = False
            else:
                disable_sub_types = True

            special_overlays_opts.extend([
                html.Div(children = [],id = {'type':'gene-info-div','index':0}),
                html.H6('Add Cell Subtypes',style={'marginTop':'5px'}),
                self.gen_info_button('Select a cell type below to add the cell subtypes of that cell type to the list of overlaid visualizations'),
                dbc.Row([
                    dbc.Col(
                        dcc.Loading(dcc.Dropdown(
                            id = {'type':'cell-subtype-drop','index':0},
                            options = [
                                {'label': i.split(' --> ')[-1], 'value': i.split(' --> ')[-1]}
                                for i in wsi.properties_list if 'Main_Cell_Types' in i and not i.replace('Main_Cell_Types','Cell_Subtypes') in wsi.properties_list
                            ],
                            value = [],
                            multi = True,
                            disabled = disable_sub_types
                        )),
                        md = 8
                    ),
                    dbc.Col(
                        dcc.Loading(dbc.Button(
                            'Add Sub-Types!',
                            id = {'type':'cell-subtype-butt','index':0},
                            className = 'd-grid col-12 mx-auto',
                            disabled = disable_sub_types
                        )),
                        md = 4
                    )
                ])
            ])
            """

        elif slide_type in ['CODEX']:

            special_overlays_opts.extend([
                html.H6('Select Additional Channel Overlay(s)'),
                self.gen_info_button('Select Channel and adjust color for combined view of multiple channels.'),
                dcc.Dropdown(
                    id = {'type':'channel-overlay-drop','index':0},
                    options = [
                        {
                            'label': i, 'value': i
                        }
                        for i in slide_info['frame_names']
                        if not i in ['Histology (H&E)','red','green','blue']
                    ],
                    value = [],
                    multi = True,
                    disabled = False
                ),
                html.Div(
                    id = {'type':'channel-overlay-select-div','index':0},
                    children = [],
                    style = {'marginBottom':'5px','marginTop':'5px'}
                ),
                dbc.Button(
                    'Overlay Channels!',
                    id = {'type':'channel-overlay-butt','index':0},
                    className = 'd-grid col-12 mx-auto',
                    disabled = True
                )
            ])
        
        elif slide_type in ['Xenium']:
            # Upload cell anchors/labels
            special_overlays_opts.extend([
                html.H6('Upload cell group labels here'),
                self.gen_info_button('Upload a csv file where one column is "cell_id" and the others are labels you want to apply to the matching cell'),
                html.Div([
                    html.A(dcc.Upload(
                        id={'type':'upload-anchors','index':0},
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                        }
                    )),
                    html.Div(id={'type':'uploaded-anchors-output','index':0}),
                ])
            ])
        
        else:
            special_overlays_opts = []

        return special_overlays_opts

    def gen_wsi_view(self, wsi):

        # Grabbing all the necessary properties to generate the children of the wsi viewer card
        # This should add more robustness to the application as it doesn't have to automatically populate
        # but instead use the ingest_wsi callback. Also may lead to having multiple WSIs at a time.
        if not wsi is None:

            combined_colors_dict = {}
            for f in wsi.map_dict['FTUs']:
                combined_colors_dict[f] = {'color':wsi.map_dict['FTUs'][f]['color']}

            #TODO: Maybe save the slide_annotations to another descriptive name so that other slides don't overlap?
            overlays = [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(url=f'./assets/slide_annotations/{struct}.json', id = wsi.map_dict['FTUs'][struct]['id'], options = {'style':{'color':combined_colors_dict[struct]}},
                            hideout = dict(color_key = {},current_cell=None,fillOpacity=0.5,ftu_color=combined_colors_dict,filter_vals=[0,1]),hoverStyle=arrow_function(dict(weight=5,color=wsi.map_dict['FTUs'][struct]['hover_color'], dashArray = '')),
                            zoomToBounds=True, children = [dl.Popup(id = wsi.map_dict['FTUs'][struct]['popup_id'])])
                    ), name = struct, checked = True, id = struct
                )
                for struct in wsi.map_dict['FTUs']
            ]

            map_url = wsi.map_dict['url']
            tile_size = wsi.tile_dims[0]
            slide_properties = wsi.zoom_levels
            map_bounds = wsi.map_bounds

            center_point = [0.5*(map_bounds[0][0]+map_bounds[1][0]),0.5*(map_bounds[0][1]+map_bounds[1][1])]
            
            map_children = [
                dl.TileLayer(
                    id = 'slide-tile',
                    url = map_url,
                    tileSize = tile_size
                ),
                dl.FullScreenControl(position='topleft'),
                dl.FeatureGroup(
                    id = 'feature-group',
                    children = [
                        dl.EditControl(
                            id = {'type':'edit_control','index':0},
                            draw = dict(polyline=False, line = False, circle = False, circlemarker=False),
                            position='topleft'
                        )
                    ]
                ),
                html.Div(
                    id = 'colorbar-div',
                    children = [
                        dl.Colorbar(id='map-colorbar')
                    ]
                ),
                dl.LayersControl(
                    id = 'layer-control',
                    children = overlays
                ),
                dl.EasyButton(
                    icon = 'fa-solid fa-user-doctor',
                    title = 'Ask Fusey',
                    id = 'fusey-button',
                    position='bottomright'
                ),
                html.Div(
                    id = 'ask-fusey-box',
                    style  = {
                        'visibility':'hidden',
                        'position':'absolute',
                        'top':'50px',
                        'right':'10px',
                        'zIndex':'1000'
                    }
                )
            ]

            map_layer = dl.Map(
                center = center_point,
                zoom = 3,
                minZoom = 0,
                maxZoom = wsi.zoom_levels-1,
                crs = 'Simple',
                bounds = map_bounds,
                style = {
                    'width':'100%',
                    'height':'90vh',
                    'margin':'auto',
                    'display':'inline-block'
                },
                id = 'slide-map',
                preferCanvas=True,
                children = map_children
            )

            return map_children

    def gen_annotation_card(self,dataset_handler, current_ftus, user_info):
        """
        Generate layout for annotation on structures
        """
        """
        1. Current annotation sessions (for the user)
            - Also show trained/training models
            - Either personally created or shared
        """
        # Checking for current annotation sessions:
        current_ann_sessions = dataset_handler.check_user_folder(folder_name='FUSION Annotation Sessions', user_info=user_info)
        if not current_ann_sessions is None:
            # Checking annotation session folder for current sessions
            ann_sessions = dataset_handler.gc.get(f'/folder',parameters={'parentType':'folder','parentId':current_ann_sessions["_id"]})
            ann_session_names = [i['name'] for i in ann_sessions]
            ann_session_meta = [i['meta'] for i in ann_sessions]

            first_session = ann_sessions[0]

            tab_list = [
                dbc.Tab(
                    label = i,
                    id = {'type':'ann-sess-tab','index':idx},
                    tab_id = f'ann-sess-{idx}',
                    activeTabClassName='fw-bold fst-italic',
                    children = []
                )
                for idx,i in enumerate(ann_session_names)
            ]

            session_progress, session_info = dataset_handler.get_annotation_session_progress(first_session['name'],user_info)
            first_session = session_info
            first_tab = self.gen_annotation_content(False,current_ftus, first_session['Annotations'], first_session['Labels'], session_progress)

        else:
            
            tab_list = []
            first_session = ['Create New Session']

            first_tab = self.gen_annotation_content(True, None)

        # Adding "Create New" tab
        tab_list.append(
            dbc.Tab(
                label = 'Create New Session',
                id = {'type':'ann-sess-tab','index':len(tab_list)},
                tab_id = f'ann-sess-{len(tab_list)}',
                activeTabClassName='fw-bold fst-italic',
                children = []
            )
        )

        return tab_list, first_tab, first_session

    def gen_annotation_content(self,new,current_ftus,classes=None,labels=None,ann_progress = None, user_type = 'annotator'):
        """
        Generate annotation content for current ftus
        """
        if not new:

            if user_type=='admin':
                admin_children = [
                    html.Hr(),
                    dbc.Label('Session Admin Components',html_for={'type':'annotation-admin-components','index':0}),
                    html.Div(
                        id = {'type':'annotation-admin-components','index':0},
                        children = [
                            html.Div('Placeholder for annotators progress'),
                            html.Hr(),
                            dbc.Row([
                                dbc.Label('Who should see this annotation session?',html_for = {'type':'annotation-session-users','index':0}),
                                html.Div(
                                    id = {'type':'annotation-session-users-parent','index':0},
                                    children = [
                                        html.Div(
                                            id = {'type':'annotation-add-user-div','index':0},
                                            children = [
                                                html.I(
                                                    className='bi bi-x-fill fa-2x',
                                                    n_clicks = 0,
                                                    id = {'type':'delete-annotation-user','index':0},
                                                    style = {'display':'none'}
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(
                                    html.I(
                                        className = 'bi bi-person-plus fa-2x',
                                        n_clicks = 0,
                                        id = {'type':'add-annotation-user','index':0},
                                        style = {'display':'inline-block','position':'relative','left':'45%','right':'50%'}
                                    ),
                                    style = {'marginBottom':'10px'}
                                )
                            ],align='center',style={'marginLeft':'5px','marginBottom':'20px','marginTop':'10px'})
                        ]
                    )
                ]
            else:
                admin_children = []

            first_tab = html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Row(html.P('Current Structures')),
                        html.Div(
                            children = [
                                dbc.Row(html.A(
                                    id = {'type':'annotation-station-ftu','index':idx},
                                    children = [f'{i}: {len(current_ftus[i])}'],
                                    style = {'display':'inline-block','marginBottom':'15px'}
                                ),align='center')
                            for idx,i in enumerate(current_ftus)
                            ]
                        ),
                        html.Hr(),
                        dbc.Row([
                            dbc.Alert([
                                html.H4('Progress'),
                                html.H6(f'Slides: {ann_progress["slides"]}'),
                                html.Hr(),
                                html.H6(f'Annotations: {ann_progress["annotations"]}'),
                                html.Hr(),
                                html.H6(f'Labels: {ann_progress["labels"]}'),
                                html.Hr(),
                                dbc.Button(
                                    'Download Images and Masks',
                                    id = {'type':'download-ann-session','index': 0},
                                    className = 'd-grid col-12 mx-auto',
                                    n_clicks = 0,
                                    disabled = False if ann_progress['annotations']>0 or ann_progress['labels']>0 else True
                                ),
                                dcc.Download(
                                    id = {'type':'download-ann-session-data','index':0}
                                ),
                                dbc.Modal(
                                    id = {'type':'download-ann-session-modal','index':0},
                                    centered = True,
                                    is_open = False,
                                    size = 'lg'
                                ),
                                dcc.Interval(
                                    id = {'type':'ann-session-interval','index':0},
                                    max_intervals = -1,
                                    disabled=True,
                                    n_intervals = 0
                                )
                            ],
                            color = 'info',
                            style = {'marginTop':'20px','marginLeft':'20px', 'width':'80%'}
                            )
                        ],align='center')
                    ],md = 4),
                    dbc.Col([
                        dbc.Row(html.P('Select a structure to annotate in that structure')),
                        html.Div(
                            id = {'type':'annotation-station-ftu-idx','index':0},
                            children = [
                                "FTU: 0"
                            ],
                            style = {'display':'none'}
                        ),
                        dbc.Row([
                            dbc.Col(
                                dcc.Graph(
                                    id = {'type':'annotation-current-structure','index':0},
                                    figure = go.Figure(
                                        layout = {
                                            'margin': {'l':0,'r':0,'t':0,'b':0},
                                            'xaxis':{'showticklabels':False,'showgrid':False},
                                            'yaxis':{'showticklabels':False,'showgrid':False},
                                            'dragmode':'drawclosedpath'
                                            }
                                    ),
                                    config = {
                                        "modeBarButtonsToAdd": [
                                            "drawopenpath",
                                            "drawclosedpath",
                                            "eraseshape"
                                        ]
                                    }
                                ),
                                md = 9
                            ),
                            dbc.Col([
                                dbc.Row('Annotation Options'),
                                html.Hr(),
                                dbc.Row(dbc.Label('Line Width')),
                                dbc.Row(
                                    dcc.Slider(
                                        id = {'type':'annotation-line-slider','index':0},
                                        min = 1.0,
                                        max = 100.0,
                                        step = 0.1,
                                        vertical=False,
                                        marks=None
                                    ),
                                    style = {'width':'100%','marginBottom':'20px'}
                                ),
                                html.Hr(),
                                dbc.Row(dbc.Label('Class Name')),
                                dbc.Row(
                                    dcc.Dropdown(
                                        placeholder = 'Select annotation class',
                                        options = classes,
                                        value = classes[0]['value'],
                                        id = {'type':'annotation-class-select','index':0}
                                    ),
                                    style = {'width':'100%'}
                                ),
                                dbc.Row(
                                    children = [
                                        dmc.Button(
                                            'Auto-Annotate!',
                                            loaderProps={'type':'dots'},
                                            loading=False,
                                            fullWidth=True,
                                            disabled = True,
                                            style = {'height':'50px'}
                                        )
                                    ],
                                    style = {'marginTop':'15px'}
                                )
                            ])
                        ],style = {'marginBottom':'20px'}),
                        dbc.Row([
                            dcc.Loading(
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Button(
                                            'Previous',
                                            id = {'type':'annotation-previous-button','index':0},
                                            n_clicks = 0,
                                            className = 'd-grid col-12 mx-auto'
                                        ),
                                        md = 3
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            'Save',
                                            id = {'type':'annotation-save-button','index':0},
                                            n_clicks = 0,
                                            className = 'd-grid col-12 mx-auto',
                                            color = 'primary'
                                        ),
                                        md = 6
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            'Next',
                                            id = {'type':'annotation-next-button','index':0},
                                            n_clicks = 0,
                                            className = 'd-grid col-12 mx-auto'
                                        ),
                                        md = 3
                                    )
                                ]
                            ),
                            style = {'width':'100%'})
                        ])
                    ],md = 8)
                ],align='top'),
                dbc.Row(
                    html.Div(
                        dbc.Progress(
                            id = {'type':'annotation-session-progress','index':0},
                            min = 0,
                            max = 100,
                            value = 0
                        )
                    ),
                    align = 'center',
                    style = {'marginTop':'5px'}
                ),
                html.Hr(),
                dbc.Row(
                    html.P('Add a text label for the image')
                ),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            placeholder='Add a class label type',
                            options = labels,
                            id = {'type':'annotation-class-label','index':0},
                            style = {'width':'100%'}
                        ),
                        md = 5
                    ),
                    dbc.Col(
                        dcc.Input(
                            placeholder = 'Label for this image',
                            id = {'type':'annotation-image-label','index':0},
                            style = {'width':'100%'}
                        ),
                        md = 5
                    ),
                    dbc.Col(
                        html.I(
                            id = {'type':'annotation-set-label','index':0},
                            className = 'bi bi-check-circle-fill fa-2x',
                            style = {'color':'rgb(0,255,0)'}
                        ),
                        md = 1
                    ),
                    dbc.Col(
                        html.I(
                            id = {'type':'annotation-delete-label','index':0},
                            className = 'bi bi-x-circle-fill fa-2x',
                            style = {'color':'rgb(255,0,0)'}
                        ),
                        md = 1
                    ),
                ],align = 'center',style = {'marginBottom':'20px'}),
                dbc.Row([
                    html.Div(
                        admin_children
                    )
                ],align = 'center')
                ],
            style = {'maxHeight':'70vh','overflow':'scroll'}
            )
        else:
            first_tab = html.Div([
                dbc.Row([
                    dbc.Col(dbc.Label('Session Name: '),md = 3),
                    dbc.Col(
                        dcc.Input(
                            placeholder = 'Name for new session',
                            id = {'type':'annotation-session-name','index':0},
                            style = {'width':'100%'}
                        ),
                        md = 9
                    )
                ],align='center',style = {'marginBottom':'20px','marginTop':'10px'}),
                dbc.Row(dbc.Label('Session Description'),align = 'center'),
                dbc.Row(
                    dcc.Textarea(
                        id = {'type':'annotation-session-description','index':0},
                        placeholder = 'Write any useful information about the annotation session here including rationale, goals, and considerations for labeling',
                        style = {'width':'100%'},
                        maxLength = 10000
                    ),
                    align = 'center',
                    style = {'marginBottom':'20px'}
                ),
                html.Hr(),
                dbc.Row([
                    dbc.Label('What classes should be annotated in images?',html_for={'type':'annotation-session-classes','index':0}),
                    html.Div(
                        id = {'type':'annotation-classes-parent-div','index':0},
                        children = [
                            html.Div(
                                id = {'type':'annotation-class-div','index':0},
                                children = [
                                    html.I(
                                        className = 'bi-x-fill fa-2x',
                                        id = {'type':'delete-annotation-class','index':0},
                                        style = {'display':'none'}
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id = {'type':'annotation-session-class-div','index':0},
                        children = [
                            html.I(
                                className = 'bi bi-plus-square fa-2x',
                                n_clicks = 0,
                                id = {'type':'add-annotation-class','index':0},
                                style = {'display':'inline-block','position':'relative','left':'45%','right':'50%'}
                            )
                        ],
                        style = {'marginBottom':'15px'}
                    ),
                    dbc.Label('What labels should be assigned to images?',html_for = {'type':'annotation-session-labels','index':0}),
                    html.Div(
                        id = {'type':'annotation-labels-parent-div','index':0},
                        children = [
                            html.I(
                                className = 'bi bi-x-fill fa-2x',
                                n_clicks = 0,
                                id = {'type':'delete-annotation-label','index':0},
                                style = {'display':'none'}
                            )
                        ],
                        style = {'marginBottom':'15px'}
                    ),
                    html.Div(
                        children = [
                            html.I(
                                className = 'bi bi-clipboard-plus fa-2x',
                                n_clicks = 0,
                                id = {'type':'add-annotation-label','index':0},
                                style = {'display':'inline-block','position':'relative','left':'45%','right':'50%'}
                            )
                        ]
                    )
                ],align='center'),
                html.Hr(),
                dbc.Row([
                    dbc.Label('Who should see this annotation session?',html_for = {'type':'annotation-session-users','index':0}),
                    html.Div(
                        id = {'type':'annotation-session-users-parent','index':0},
                        children = [
                            html.Div(
                                id = {'type':'annotation-add-user-div','index':0},
                                children = [
                                    html.I(
                                        className='bi bi-x-fill fa-2x',
                                        n_clicks = 0,
                                        id = {'type':'delete-annotation-user','index':0},
                                        style = {'display':'none'}
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        html.I(
                            className = 'bi bi-person-plus fa-2x',
                            n_clicks = 0,
                            id = {'type':'add-annotation-user','index':0},
                            style = {'display':'inline-block','position':'relative','left':'45%','right':'50%'}
                        ),
                        style = {'marginBottom':'10px'}
                    )
                ],align='center',style={'marginLeft':'5px','marginBottom':'20px','marginTop':'10px'}),
                dbc.Row([
                    dbc.Button(
                        'Create New Session!',
                        id = {'type':'create-annotation-session-button','index':0},
                        className = 'd-grid col-12 mx-auto',
                        n_clicks = 0
                    )
                ],
                align='center',
                style = {'marginBottom':'10px'}
                )
            ],style = {'maxHeight':'70vh','overflow':'scroll'})

        return first_tab

    def get_user_ftu_labels(self,wsi,ftu):

        # Getting user-provided ftu labels entered in the popup input
        ftu_name = ftu['properties']['name']
        ftu_index = ftu['properties']['unique_index']
        if not ftu_name == 'Spots':
            ftu_properties = wsi.ftu_props[ftu_name][ftu_index]
        else:
            ftu_properties = wsi.spot_props[ftu_index]
        
        if 'user_labels' in ftu_properties:
            # Creating div with user-labels
            user_ftu_labels_children = []
            for u_idx,u in enumerate(ftu_properties['user_labels']):
                if not u is None:
                    user_ftu_labels_children.append(
                        dbc.Row([
                            dbc.Col(
                                html.P(textwrap.wrap(u,width=200)),
                                md = 8
                            ),
                            dbc.Col(
                                html.I(
                                    id = {'type':'delete-user-label','index':u_idx},
                                    n_clicks=0,
                                    className='bi bi-x-circle-fill',
                                    style={'color':'rgb(255,0,0)'}
                                ),
                                md = 4
                            )
                        ],align='center')
                    )
        else:
            user_ftu_labels_children = []

        return user_ftu_labels_children

    def gen_builder_layout(self, dataset_handler, user_info, current_slide_dataset = None):

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
        if current_slide_dataset is None:
            slide_datasets = dataset_handler.update_slide_datasets(user_info)
        else:
            if current_slide_dataset['userId']==user_info['_id']:
                slide_datasets = current_slide_dataset
            else:
                slide_datasets = dataset_handler.update_slide_datasets(user_info)

        for f in slide_datasets['slide_dataset']:
            folder_dict = {}
            if 'name' in f:
                folder_dict['Name'] = f['name']
                
                folder_meta_keys = list(f['Aggregated_Metadata'])
                for m in folder_meta_keys:
                    folder_dict[m] = f['Aggregated_Metadata'][m]

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
        
        builder_layout = dbc.Row(
            children = [
                    html.H3('Select a Dataset to add slides to current session'),
                    html.Hr(),
                    self.gen_info_button('Click on one of the circles in the far left of the table to load metadata for that dataset. You can also filter/sort the rows using the arrow icons in the column names and the text input in the first row'),

                    table_layout,
                    html.B(),
                    html.H3('Select Slides to include in current session'),
                    self.gen_info_button('Select/de-select slides to add/remove them from the metadata plot and current viewing session'),
                    html.Hr(),
                    html.Div(id='selected-dataset-slides'),
                    html.Hr(),
                    html.H3('Current Metadata'),
                    self.gen_info_button('Select different metadata options to view the distribution of FTU values within each selected dataset or slide'),
                    dcc.Loading(html.Div(id='slide-metadata-plots'))
                ],
            style = {'maxHeight':'90vh','overflow':'scroll','marginBottom':'10px'}
            )

        self.current_builder_layout = builder_layout
        self.validation_layout.append(builder_layout)
        self.layout_dict['dataset-builder'] = builder_layout
        self.description_dict['dataset-builder'] = builder_description

    def gen_uploader_prep_type(self,upload_type,components_values):

        # Getting specific layouts for different types of pre-processing.
        if upload_type in ['Visium','Regular']:
            # Sub-compartment segmentation card:

            sub_comp_methods_list = [
                {'label':'Manual','value':'Manual','disabled':False},
                {'label':'Use Plugin','value':'plugin','disabled':True}
            ]
            sub_comp_card = dbc.Card([
                dbc.CardHeader([
                    'Sub-Compartment Segmentation',
                    self.gen_info_button('This step allows for specific morphometric calculation for major sub-compartments on all FTUs in a dataset')
                    ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                dbc.Label('Select FTU',html_for={'type':'ftu-select','index':0}),
                                dcc.Dropdown(
                                    options = components_values['ftu_names'],
                                    placeholder='FTU Options',
                                    id={'type':'ftu-select','index':0}
                                )
                            ]),md=12)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                children = [
                                    dcc.Graph(
                                        figure=go.Figure(
                                            data = px.imshow(components_values['image'])['data'],
                                            layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                                        ),
                                        id={'type':'ex-ftu-img','index':0})
                                ]
                            ),md=12)
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                children = [
                                    dbc.Label('Example FTU Segmentation Options',html_for={'type':'ex-ftu-opts','index':0}),
                                    html.Hr(),
                                    html.Div(
                                        id={'type':'ex-ftu-opts','index':0},
                                        children = [
                                            dcc.RadioItems(
                                                [
                                                    {'label':html.Span('Overlaid',style={'marginBottom':'5px','marginLeft':'5px','marginRight':'10px'}),'value':'Overlaid'},
                                                    {'label':html.Span('Side-by-side',style={'marginLeft':'5px'}),'value':'Side-by-side'}
                                                ],
                                                    value='Overlaid',inline=True,id={'type':'ex-ftu-view','index':0}),
                                            html.B(),
                                            dbc.Label('Overlaid Mask Transparency:',html_for={'type':'ex-ftu-slider','index':0},style={'marginTop':'10px'}),
                                            dcc.Slider(0,1,0.05,value=0,marks=None,vertical=False,tooltip={'placement':'bottom'},id={'type':'ex-ftu-slider','index':0}),
                                            html.B(),
                                            dbc.Row([
                                                dbc.Col(dbc.Button('Previous',id={'type':'prev-butt','index':0},outline=True,color='secondary',className='d-grid gap-2 col-6 mx-auto')),
                                                dbc.Col(dbc.Button('Next',id={'type':'next-butt','index':0},outline=True,color='secondary',className='d-grid gap-2 col-6 mx-auto'))
                                            ],style={'marginBottom':'15px','display':'flex'}),
                                            html.Hr(),
                                            dbc.Row([
                                                dbc.Col(dbc.Button('Go to Feature Extraction',id={'type':'go-to-feat','index':0},color='success',className='d-grid gap-2 col-12 mx-auto'))
                                            ])
                                        ]
                                    )
                                ]
                            ),md=12)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                children = [
                                    dbc.Label('Sub-compartment Segmentation Method:',html_for={'type':'sub-comp-method','index':0})
                                ]
                            ),md=4),
                        dbc.Col(
                            html.Div(
                                children = [
                                    dcc.Dropdown(sub_comp_methods_list,placeholder='Available Methods',id={'type':'sub-comp-method','index':0})
                                ]
                            ),md=8
                        ),
                        dbc.Col(
                            self.gen_info_button('Choose whether to use manual thresholds or a pre-loaded sub-compartment segmentation plugin')
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                id='sub-comp-tabs',
                                children = [
                                    dbc.Label('Sub-Compartment Thresholds',html_for={'type':'sub-thresh-slider','index':0}),
                                    dcc.RangeSlider(
                                        id = {'type':'sub-thresh-slider','index':0},
                                        min = 0.0,
                                        max = 255.0,
                                        step = 5.0,
                                        value = [0.0,50.0,120.0],
                                        marks = {
                                            0.0:{'label':'Luminal Space: 0','style':'rgb(0,255,0)'},
                                            50.0:{'label':'Eosinophilic: 50','style':'rgb(255,0,0)'},
                                            120.0:{'label':'Nuclei: 120','style':'rgb(0,0,255)'}
                                        },
                                        tooltip = {'placement':'top','always_visible':False},
                                        allowCross=False
                                    )
                                ]
                            ),md=11, style = {'marginLeft':'20px','marginRight':'20px'}
                        ),
                        dbc.Col(
                            self.gen_info_button('Adjust thresholds here to include/exclude pixels from each sub-compartment'),
                            md = 1
                        )
                    ])
                ])
            ])

            # Feature extraction card:
            feat_extract_card = dbc.Card([
                dbc.CardHeader([
                    'Morphometric Feature Extraction',
                    self.gen_info_button('Select which morphometrics and which FTUs to quantify and then hit the extract features button. These features are used for high-dimensional clustering and visualization.')
                    ]),
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col(html.Div(id={'type':'feature-items','index':0}))
                    ])
                )
            ])

            prep_div_children = dbc.Row([
                dbc.Col(sub_comp_card,md=6),
                dbc.Col(feat_extract_card,md=6)
            ])

        elif upload_type in ['CODEX','Xenium']:

            sub_comp_methods_list = [
                {'label':'Manual','value':'Manual','disabled':False},
                {'label':'Use Plugin','value':'plugin','disabled':True}
            ]
            # Nuclei segmentation card
            nuc_seg_card = dbc.Card([
                dbc.CardHeader([
                    'Nuclei Segmentation',
                    self.gen_info_button('Click on the thumbnail to see different regions of the tissue at high resolution. Select the frame you want to use for nuclei segmentation and optimize parameters below')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                dbc.Label('Select Frame/Channel to use for nuclei segmentation',html_for={'type':'frame-select','index':0}),
                                dcc.Dropdown(
                                    options = components_values['frames'],
                                    placeholder = 'Frame Options',
                                    id = {'type':'frame-select','index':0}
                                )
                            ]),md=12
                        )
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                children = [
                                    dcc.Graph(figure = go.Figure(), id = {'type':'frame-thumbnail','index':0})
                                ]
                            ),md = 6
                        ),
                        dbc.Col(
                            html.Div(
                                children = [
                                    dcc.Graph(figure = go.Figure(), id = {'type':'ex-nuc-img','index':0})
                                ]
                            ), md = 6
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                children = [
                                    dbc.Label('Example Nuclei Segmentation Options',html_for={'type':'ex-nuc-opts','index':0}),
                                    html.Hr(),
                                    html.Div(
                                        id = {'type':'ex-nuc-opts','index':0},
                                        children = [
                                            dcc.RadioItems(
                                                [
                                                    {'label':html.Span('Overlaid',style={'marginBottom':'5px','marginLeft':'5px','marginRight':'10px'}),'value':'Overlaid'},
                                                    {'label':html.Span('Side-by-side',style={'marginLeft':'5px'}),'value':'Side-by-side'}                                                
                                                ],
                                                    value = 'Overlaid',inline = True, id = {'type':'ex-nuc-view','index':0}
                                            ),
                                            html.B(),
                                            dbc.Label('Overlaid Mask Transparency:', html_for = {'type':'ex-nuc-slider','index':0},style = {'marginTop':'10px'}),
                                            dcc.Slider(0,1,0.05, value = 0, marks = None, vertical = False, tooltip = {'placement':'bottom'},id = {'type':'ex-nuc-slider','index':0}),
                                            html.Hr(),
                                            dbc.Row([
                                                dbc.Col(dbc.Button('Go to Feature Extraction',id = {'type':'go-to-feat','index':0},color = 'success',className = 'd-grid gap-2 col-12 mx-auto'))
                                            ])
                                        ]
                                    )
                                ]
                            ),md=12
                        )
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                children = [
                                    dbc.Label('Nuclei Segmentation Method', html_for = {'type':'nuc-seg-method','index':0})
                                ]
                            ), md = 4
                        ),
                        dbc.Col(
                            html.Div(
                                children = [
                                    dcc.Dropdown(
                                        sub_comp_methods_list, placeholder= "Available Methods",id = {'type':'nuc-method','index': 0}
                                    )
                                ]
                            ), md = 8
                        ),
                        self.gen_info_button('Choose whether to set thresholds manually or use another method')
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                id = {'type':'nuc-thresh-div','index':0},
                                children = [
                                    dbc.Label('Set Nuclei Threshold',html_for={'type':'nuc-thresh-slider','index':0}),
                                    dcc.Slider(
                                        id = {'type':'nuc-thresh-slider','index':0},
                                        min = 0.0,
                                        max = 255.0,
                                        step = 5,
                                        value = 128.0,
                                        marks = {
                                            128.0: {'label':'Nuclei: 128','style':'rgb(0,0,255)'}
                                        },
                                        tooltip = {'placement': 'top','always_visible':False}
                                    )
                                ]
                            ),
                            self.gen_info_button('Adjust threshold to include more or fewer pixels in the nuclei segmentation')
                        ])
                    ])
                ])
            ])

            # Feature extraction card:
            feat_extract_card = dbc.Card([
                dbc.CardHeader([
                    'Morphometric Feature Extraction (CODEX)',
                    self.gen_info_button('Select which morphometrics and which FTUs to quantify and then hit the extract features button. These features are used for high-dimensional clustering and visualization.')
                    ]),
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col(html.Div(id={'type':'feature-items','index':0}))
                    ])
                )
            ])

            prep_div_children = dbc.Row([
                dbc.Col(nuc_seg_card,md=6),
                dbc.Col(feat_extract_card,md=6)
            ])

        else:
            raise ValueError

        return prep_div_children

    def gen_uploader_layout(self):

        # This builds the layout for the Dataset Uploader functionality,
        # allowing users to upload their own data to be incorporated into the 
        # dataset builder or directly to the current viewing instance.

        # Description and instructions card
        uploader_description = [
            html.P('Happy fusing!')
        ]

        # File upload card:
        upload_types = [
            {'label':'Regular Histology','value':'Regular','disabled':False},
            {'label':'10x Visium','value':'Visium','disabled':False},
            {'label':'Co-Detection by Indexing (CODEX)','value':'CODEX','disabled':False},
            {'label':'10x Xenium','value':'Xenium','disabled':False},
            {'label':'CosMx','value':'CosMx','disabled':True},
            {'label':'GeoMx','value':'GeoMx','disabled':True}
        ]

        file_upload_card = dbc.Card([
            dbc.CardHeader([
                'File Uploads',
                self.gen_info_button('Select which type of spatial -omics data you are uploading to determine which files are needed for pre-processing steps')
                ]),
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
                        dcc.Loading(
                            html.Div(
                                id='upload-requirements',
                                children = []
                            )
                        )
                    )
                ],
                align='center')
            ])
        ])

        # Slide QC card:
        slide_qc_card = dbc.Card([
            dbc.CardHeader([
                'Slide Quality Control',
                self.gen_info_button('Check uploaded slide thumbnail and add metadata. Add the name of the metadata and the value (e.g. for adding a disease label, type "Disease" in the Metadata Name column and the disease in the Value column).')
                ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            id = 'slide-thumbnail-holder',
                            children = []
                        ),
                        md = 12
                    )
                ],align='center'),
                html.Hr(),
                html.H3('Add Slide-level Metadata'),
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            id='slide-qc-results',
                            children = []
                        )
                    )
                ]),
                dbc.Row([
                    dbc.Button(
                        'Update Metadata',
                        id = {'type':'add-slide-metadata','index':0},
                        className = 'd-grid mx-auto',
                        disabled=False
                    )
                ])
            ])
        ])

        # MC model selection card:
        structures = [
            {'label':'Glomeruli','value':'Glomeruli','disabled':False},
            {'label':'Sclerotic Glomeruli','value':'Sclerotic Glomeruli','disabled':False},
            {'label':'Tubules','value':'Tubules','disabled':False},
            {'label':'Arteries and Arterioles','value':'Arteries and Arterioles','disabled':False},
            {'label':'Cortical interstitium','value':'Cortical interstitium','disabled':False},
            {'label':'Medullary interstitium','value':'Medullary interstitium','disabled':False},
            {'label':'Interstitial Fibrosis and Tubular Atrophy','value':'IFTA','disabled':False},
            {'label':'Peritubular Capillaries','value':'PTC','disabled':False}
        ]
        mc_model_card = dbc.Card([
            dbc.CardHeader([
                'Automated FTU Segmentation',
                self.gen_info_button('Selecting structures here determines which model(s) to use to extract FTUs')
                ]),
            dbc.CardBody([
                html.Div(id = {'type':'codex-segmentation-frame-div','index':0}),
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            dbc.Label('Select Structures:',html_for='structure-type'),
                            dcc.Dropdown(structures,multi=True,id='structure-type',disabled=True),
                            dbc.Button('Start Segmenting!',id='segment-butt'),
                            dcc.Markdown('**Please note**, this process may take some time as the segmentation models and cell deconvolution pipelines run in the backend')
                        ]),md=12
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        html.Div('Do you already have annotations? If so, click here to add them:'),
                        md = 8
                    ),
                    dbc.Col(
                        dbc.Button(
                            'Upload Annotation File(s)',
                            id = {'type':'create-ann-upload','index':0},
                            className = 'd-grid mx-auto',
                            disabled=False
                        )
                    )
                ]),
                dbc.Row([
                    html.Div(id = 'upload-anns-div',children = [])
                ]),
                dbc.Row([
                    html.Div(id = 'seg-woodshed',children = [],style={'maxHeight':'200px','overflow':'scroll'}),
                ]),
                dbc.Row([
                    dbc.Button(
                        'Continue',
                        id = {'type':'seg-continue-butt','index':0},
                        className = 'd-grid col-12 mx-auto',
                        color = 'success',
                        style = {'marginTop':'10px'},
                        disabled=False
                    )
                ])
            ])
        ])

        uploader_layout = dbc.Row(
            children = [
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
                                        html.Div(id = {'type':'prep-div','index':0})
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
            ],
            style = {'height':'90vh','marginBottom':'10px'}
        )
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

        with open('./assets/tutorial_content.json','r') as f:
            self.tutorial_content = json.load(f)
        f.close()

        tutorial_list = [i['name'] for i in self.tutorial_content['categories']]

        welcome_layout = [
                dbc.Row([
                    dbc.Col(html.Img(src='./assets/Welcome Page Banner.svg',style={'width':'100%','height':'20vh'})),
                ],align='center'),
                html.Hr(),
                html.B(),
                dbc.Row([
                    dbc.Row([
                        dbc.Label('Getting Started: Select a category below to view tutorial slides')
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                html.A(dcc.Markdown(f'* {i}'),id={'type':'tutorial-name','index':idx}),
                                html.Div(
                                    id = {'type':'tutorial-parts','index':idx},
                                    children = []
                                ),
                                html.Br()
                            ])
                            for idx,i in enumerate(tutorial_list)
                        ],align='center',md = 2),
                        dbc.Col([
                            html.Div(html.H3('FUSION Introduction'),id='tutorial-name'),
                            html.Br(),
                            html.Div(
                                id = 'welcome-tutorial',
                                children=[
                                    dcc.Loading(
                                        dbc.Carousel(
                                            id = 'welcome-tutorial-slides',
                                            items = [
                                                {
                                                    'key':f'{i+1}',
                                                    'src':f'./static/tutorials/FUSION Introduction/slide_{i}.svg',
                                                    'img_style':{'height':'60vh','width':'80%'}
                                                    }
                                                for i in range(len(os.listdir('./static/tutorials/FUSION Introduction/')))
                                            ],
                                            controls = True,
                                            indicators = True,
                                            variant = 'dark'
                                        )
                                    )
                                ]
                            ),
                        ],align='center',md=10)
                    ])
                ],style={'maxHeight':'70vh','overflow':'scroll'})
            ]
        
        self.current_welcome_layout = welcome_layout
        self.validation_layout.append(welcome_layout)
        self.layout_dict['welcome'] = welcome_layout
        self.description_dict['welcome'] = welcome_description

    def gen_initial_layout(self,slide_names,initial_user,default_slides,available_datasets):

        # welcome layout after initialization and information and buttons to go to other areas
        # Header
        header = dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.Img(id='logo',src=('./assets/Fusion-Logo-Navigator-Color01.png'),height='100px'),md='auto'),
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
                                        color = 'primary',
                                        href = 'https://ufl.qualtrics.com/jfe/form/SV_1A0CcKNLhTnFCHI',
                                        target='_blank',
                                        style = {'textTransform':'none'}
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "Cell Cards",
                                        id='cell-cards-button',
                                        outline=True,
                                        color="primary",
                                        href="https://cellcards.org/index.php",
                                        target='_blank',
                                        style={"textTransform":"none"}
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "Lab Website",
                                        id='lab-web-button',
                                        outline=True,
                                        color='primary',
                                        href='https://cmilab.nephrology.medicine.ufl.edu',
                                        target='_blank',
                                        style={"textTransform":"none"}
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "User Feedback",
                                        id = 'user-feedback-button',
                                        outline=True,
                                        color='primary',
                                        href='https://github.com/SarderLab/FUSION/issues',
                                        target='_blank',
                                        style = {'textTransform':'none'}
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
            color="dark",
            sticky='fixed',
            style={'marginBottom':'20px'}
        )

        # Turning off upload capability if user is "fusionguest"
        if initial_user['login']=='fusionguest':
            upload_disable = True
        else:
            upload_disable = False

        # Sidebar
        sider = html.Div([
            dbc.Offcanvas([
                html.Img(id='welcome-logo-side',src=('./assets/FUSION-LAB-FINAL.png'),height='315px',width='275px'),
                dbc.Nav([
                    dbc.NavLink('Welcome',href='/welcome',active='exact',id='welcome-sidebar'),
                    dbc.NavLink('FUSION Visualizer',href='/vis',active='exact',id='vis-sidebar'),
                    dbc.NavLink('Dataset Builder',href='/dataset-builder',active='exact',id='builder-sidebar'),
                    dbc.NavLink('Dataset Uploader',href='/dataset-uploader',active='exact',id='upload-sidebar',disabled=upload_disable)
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
                        dbc.Input(type='password',placeholder='Password',id='pword-input'),
                        style = {'marginBottom':'5px'}
                    ),
                    html.Div(id = 'create-user-extras',children = []),
                    dbc.Row(
                        children = dcc.Loading(html.Div([
                            dbc.Button('Login',color='primary',id='login-submit'),
                            dbc.Button('Create Account',color='secondary',id='create-user-submit')
                        ],className='d-grid gap-2 d-md-flex'))
                    )
                ])
            ],
            target = {'type':'login-butt','index':0},
            body=True,
            trigger='legacy'
        )

        description = dbc.Card(
            children = [
                #dbc.CardHeader("Description and Instructions"),
                dbc.CardBody([
                    dbc.Button('Menu',id={'type':'sidebar-button','index':0},className='mb-3',color='primary',n_clicks=0),
                    dbc.Button("View/Hide Description",id={'type':'collapse-descrip','index':0},className='mb-3',color='primary',n_clicks=0,style={'marginLeft':'5px','display':'none'}),
                    dbc.Button('Registered User Login',id={'type':'login-butt','index':0},className='mb-3',style = {'marginLeft':'5px'}),
                    login_popover,
                    dbc.Button('Sign up for Usability Study',
                        id = {'type':'usability-sign-up','index':0},
                        className='mb-3',
                        color = 'primary',
                        target = '_blank',
                        n_clicks = 0,
                        href = 'https://ufl.qualtrics.com/jfe/form/SV_ag9QzBmvG5qEce2',
                        style = {'marginLeft':'5px','display':'inline-block'}
                    ),
                    dbc.Button(
                        'Start Usability Study',
                        id = {'type':'usability-butt','index':0},
                        className='mb-3',
                        color = 'primary',
                        n_clicks=0,
                        style={'marginLeft':'5px','display':'none'},
                        disabled=False
                    ),
                    html.Div(id='logged-in-user',children = [
                        f'Welcome, {initial_user["login"]}!',
                        dbc.Badge(
                            html.A('Jobs'),
                            color = 'secondary',
                            id = 'long-plugin-butt'
                        )
                        ]),
                    html.Div(
                        id = 'user-store-div',
                        children = [
                            dcc.Store(
                                id = 'user-store',
                                data = json.dumps(initial_user),
                                storage_type = 'memory'
                            )
                        ]
                    ),
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
                    ),
                    html.Div(
                        children = [
                            dcc.Store(
                                id = {'type': 'available-datasets-store','index':0},
                                data = json.dumps(available_datasets),
                                storage_type='memory'
                            )
                        ]
                    ),
                ]),
                dbc.CardFooter([
                    dbc.Collapse(
                        html.Div(
                            id = 'long-plugin-div',
                            children = []
                        ),
                        id = 'plugin-collapse',
                        is_open = False
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
                                    placeholder = 'Current visualization session',
                                    id = 'slide-select'
                                )
                            ), md=7
                        ),
                        dbc.Col(
                            self.gen_info_button('Click the dropdown menu to select a slide!'),md=1
                        )
                    ],align='center'),
                    html.Div([
                        dbc.Modal(
                            id = 'slide-load-modal',
                            centered = True,
                            is_open = False,
                            size = 'lg'
                        ),
                        dcc.Store(
                            id = 'slide-info-store',
                            data = json.dumps({'overlay_prop': None,'cell_vis_val': 0.5,'current_channels': None, 'filter_vals': None}),
                            storage_type='memory'
                        ),
                        dcc.Store(
                            id = 'visualization-session-store',
                            data = json.dumps(default_slides),
                            storage_type = 'memory'
                        ),
                        dcc.Interval(
                            id = 'slide-load-interval',
                            interval = 1000,
                            n_intervals = 0,
                            max_intervals=-1,
                            disabled = True
                        )
                    ])
                ])
            ],style={'marginBottom':'20px','display':'none'}
        )
        
        welcome_layout = dmc.MantineProvider(
            children = [
                html.Div([
                    dcc.Location(id='url'),
                    html.Div(id='ga-invisible-div', style={'display': 'none'}),
                    html.Div(
                        id = 'fusion-store-div',
                        children = [
                            dcc.Store(
                                id = 'fusion-store',
                                data = {},
                                storage_type = 'memory'
                            )
                        ]
                    ),
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
                        html.Img(
                            src='./assets/Welcome Page Banner.svg',
                            style = {'width':'100%','height':'20vh'}
                        ),
                        ],fluid=True,id='container-content',style = {'height':'100vh'}
                    ),
                    html.Div(id='user-id-div', style={'display': 'none'}),
                    html.Div(id='dummy-div-for-userId', style={'display': 'none'}),
                    html.Div(id='dummy-div-plugin-track', style={'display': 'none'}),
                    html.Div(id='plugin-ga-track', style={'display': 'none'}),
                    html.Footer('“©Copyright 2023 University of Florida Research Foundation, Inc. All Rights Reserved.”')
                ])
            ]
        )

        self.current_initial_layout = welcome_layout
        self.validation_layout.append(welcome_layout)
        self.layout_dict['initial'] = welcome_layout
        self.description_dict['initial'] = initial_description

    def gen_single_page_layout(self,description_text,container_children):

        # Useful for bulk pre-processing and maybe other individual analyses later on
        # Header
        header = dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.Img(id='logo',src=('./assets/Fusion-Logo-Navigator-Color01.png'),height='100px'),md='auto'),
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
                                        color = 'primary',
                                        href = ' https://ufl.qualtrics.com/jfe/form/SV_1A0CcKNLhTnFCHI',
                                        style = {'textTransform':'none'}
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "Cell Cards",
                                        id='cell-cards-button',
                                        outline=True,
                                        color="primary",
                                        href="https://cellcards.org/index.php",
                                        style={"textTransform":"none"}
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "Lab Website",
                                        id='lab-web-button',
                                        outline=True,
                                        color='primary',
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
            color="dark",
            sticky='fixed',
            style={'marginBottom':'20px'}
        )

        # Sidebar
        sider = html.Div([
            dbc.Offcanvas([
                html.Img(id='welcome-logo-side',src=('./assets/FUSION-LAB-FINAL.png'),height='315px',width='275px'),
                'Single-page mode'
                #dbc.Nav([
                #    dbc.NavLink('Welcome',href='/welcome',active='exact',id='welcome-sidebar'),
                #    dbc.NavLink('FUSION Visualizer',href='/vis',active='exact',id='vis-sidebar'),
                #    dbc.NavLink('Dataset Builder',href='/dataset-builder',active='exact',id='builder-sidebar'),
                #    dbc.NavLink('Dataset Uploader',href='/dataset-uploader',active='exact',id='upload-sidebar')
                #],vertical=True,pills=True)], id={'type':'sidebar-offcanvas','index':0},style={'background-color':"#f8f9fa"}
                ]
            )
        ])

        initial_description = [
            html.P(description_text)
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
                    html.Div(id='logged-in-user',children = [f'Welcome, {""}!']),
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

        single_page_layout = html.Div([
            dcc.Location(id='url'),
            html.Div(id='ga-invisible-div', style={'display': 'none'}),
            header,
            html.B(),
            dbc.Row(dbc.Col(html.Div(sider))),
            html.B(),
            dbc.Row(
                id = 'descrip-and-instruct',
                children = description,
                align='center'
            ),
            html.Div(id='user-id-div', style={'display': 'none'}),
            dbc.Container(
                children = container_children,
                fluid=True,id='container-content')
        ])

        return single_page_layout




class GirderHandler:
    def __init__(self,
                apiUrl: str,
                username: str,
                password: str):
        
        self.apiUrl = apiUrl
        self.gc = girder_client.GirderClient(apiUrl = self.apiUrl)
        self.base_path = None
        self.base_path_type = None

        # Initializing usability study users/admins list
        self.usability_users = {
            'usability_study_users':{},
            'usability_study_admins':[]
        }

        self.usability_group = '65e5e9ceadb89a58fea146ba'

        self.user_info, self.user_details = self.authenticate(username, password)

        # Name of plugin used for fetching clustering/plotting metadata
        self.get_cluster_data_plugin = 'samborder2256_get_cluster_data_latest/clustering_data'
        self.cached_annotation_ids = []

        self.padding_pixels = 50

        # Initializing blank annotation metadata cache to prevent multiple dsa requests
        self.cached_annotation_metadata = {}

    def authenticate(self, username, password):
        # Getting authentication for user
        #TODO: Add some handling here for incorrect username or password
        try:
            self.username = username.lower()
            self.password = password
            
            self.user_details = self.gc.authenticate(username.lower(),password)

            user_info = self.check_usability(self.username)
            self.get_token()
            self.user_details['token'] = self.user_token

            if not self.base_path is None:
                self.initialize_folder_structure(self.username)

            return user_info, self.user_details
        
        except girder_client.AuthenticationError:
            self.user_details = self.gc.authenticate(self.username.lower(),self.password)

            user_info = self.check_usability(self.username.lower())
            self.get_token()
            self.user_details['token'] = self.user_token

            if not self.base_path is None:
                self.initialize_folder_structure()

            return user_info, self.user_details

    def create_user(self,username,password,email,firstName,lastName):

        # Creating new user from username/password combo
        self.username = username.lower()
        self.password = password

        self.gc.post('/user',
                     parameters = {
                         'login':username.lower(),
                         'password':password,
                         'email':email,
                         'firstName':firstName,
                         'lastName':lastName
                     })
        
        user_info, user_details = self.authenticate(self.username,self.password)

        if self.username in self.usability_users['usability_study_users'] or self.username in self.usability_users['usability_study_admins']:
            # Adding user to usability study group to enable write access to response document:
            print('Checking if user is in group')
            self.gc.post(f'/slicer_cli_web/samborder2256_auto_group_add_latest/GroupAdd/run',
                            parameters={
                                'user_id': self.user_details["_id"],
                                'group_id': self.usability_group,
                                'girderApiUrl': self.apiUrl
                            })


        return user_info, user_details

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

    def get_item_info(self,item_id:str):
        """
        Get information for a given itemId, return None if invalid id or bad permissions
        """
        try:
            # Getting the information for an item from it's unique id
            item_info = self.gc.get(f'item/{item_id}')
        except girder_client.HttpError:
            item_info = None

        return item_info
    
    def get_available_annotation_ids(self,item_id:str):
        """
        Get all available annotation id's and associated info for an item
        """
        try:
            annotation_info = self.gc.get(f'/annotation',parameters= {'itemId':item_id,'limit':0})
        except girder_client.HttpError:
            annotation_info = None
        
        return annotation_info

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

    def get_cli_list(self):
        # Get a list of possible CLIs available for current user
        #TODO: Find out the format of what is returned from this and reorder

        cli = self.gc.get('/slicer_cli_web/cli')
        self.cli_dict_list = cli

        return cli

    def update_slide_datasets(self,user_info):
        """
        Grabbing available collections as well as user public folders
        outputs list of accessible folders' info (folders that are immediate parents of slides)
        """

        slide_datasets = []
        
        # This is all collections in the DSA instance
        all_collections = self.gc.get('/collection',parameters={'limit': 0})

        # Checking user access (only include public collections)
        all_collections = [i for i in all_collections if i['public']]

        # Adding in user public folders
        user_folder_path = f'/user/{user_info["login"]}/Public'
        user_public_folder = self.gc.get('/resource/lookup',parameters={'path':user_folder_path})

        all_collections += [user_public_folder]         

        for c in all_collections:

            # Get all large image objects in each collection that are not in histoqc outputs
            collection_items = []
            try:
                if not c["_id"]==user_public_folder["_id"]:
                        collection_items = self.gc.get(f'/resource/{c["_id"]}/items',parameters={'limit': 1000,'type':'collection'})
                else:
                    collection_items = self.gc.get(f'/resource/{c["_id"]}/items',parameters={'limit': 1000,'type':'folder'})
            except json.JSONDecodeError:
                print(f'JSONDecodeError encountered')

            image_items = [i for i in collection_items if 'largeImage' in i and not 'png' in i['name']]

            folder_ids = np.unique([i['folderId'] for i in image_items]).tolist()
            folder_info = [self.gc.get(f'/folder/{i}') for i in folder_ids]
            # Don't want to see histoqc outputs (which should all be pngs) or anything in the annotation sessions (Masks are tiff, Images are pngs)
            folder_info = [i for i in folder_info if not i['name'] in ['histoqc_outputs','Masks','Images']]

            # Aggregating slide metadata in each folder
            for f in folder_info:
                folder_slides = [i for i in image_items if i['folderId']==f["_id"]]
                f['Aggregated_Metadata'] = {}
                
                folder_slides_meta = [i['meta'] for i in folder_slides]

                meta_keys = []
                for f_s_m in folder_slides_meta:
                    meta_keys.extend(list(f_s_m.keys()))

                for m in meta_keys:
                    items_meta = [i['meta'][m] for i in folder_slides if m in i['meta']]
                    
                    if all([type(i)==str for i in items_meta]):
                        f['Aggregated_Metadata'][m] = ', '.join(list(set(items_meta)))
                    elif all([type(i)==int or type(i)==float for i in items_meta]):
                        f['Aggregated_Metadata'][m] = sum(items_meta)
                
                slide_datasets.append(f)


        return_dict = {
            'userId': user_info["_id"],
            'slide_dataset': slide_datasets
        }

        return return_dict

    def get_folder_slides(self,folder_id):
        """
        Get all the slides in a given folder id        
        """

        # Get all large image objects in each collection that are not in histoqc outputs
        try:
            collection_items = self.gc.get(f'/resource/{folder_id}/items',parameters={'limit': 1000,'type':'collection'})
        except girder_client.HttpError:
            collection_items = self.gc.get(f'/resource/{folder_id}/items',parameters={'limit': 1000,'type':'folder'})
        
        image_items = [i for i in collection_items if 'largeImage' in i and not 'png' in i['name']]

        return image_items
    
    def get_folder_name(self,folder_id):
        """
        Return the name for a folder_id (checks if it's either a "collection" or a "folder")
        """

        try:
            folder_info = self.gc.get(f'/folder/{folder_id}')
        except girder_client.HttpError:
            folder_info = self.gc.get(f'/collection/{folder_id}')
        
        return folder_info['name']

    def clean_old_annotations(self, days = 1):
        """
        Clear cached annotations with access times greater than 1 day
        """
        annotations_path = './assets/slide_annotations/'
        if os.path.exists(annotations_path):
            item_annotations = os.listdir(annotations_path)

    def set_default_slides(self,default_slide_list):
        # Setting default slides with name and item information

        if len(default_slide_list)>0:
            self.default_slides = default_slide_list
        else:
            self.default_slides = []
        
    def get_collection_annotation_meta(self,select_ids:list):

        if len(select_ids)>0:
            print(f'Getting annotation metadata for: {select_ids}')
            # Running get_cluster_data plugin 
            job_response = self.gc.post(f'/slicer_cli_web/{self.get_cluster_data_plugin}/run',
                                        parameters = {
                                            'girderApiUrl':self.apiUrl,
                                            'girderToken':self.user_token,
                                            'add_ids':','.join(select_ids),
                                            'remove_ids':''
                                        })

        else:
            job_response = []
        
        return job_response

    def get_image_region(self,item_id,user_info,coords_list,frame_index=None):

        # Checking to make sure coords are within the slide boundaries
        slide_metadata = self.gc.get(f'/item/{item_id}/tiles')
        slide_bounds = np.array([0,0,slide_metadata['sizeX'],slide_metadata['sizeY']])
        coords_list = coords_list[0:2]+np.minimum(np.array(coords_list[2:]),slide_bounds[2:]).tolist()
        coords_list[0] = np.maximum(0,coords_list[0])
        coords_list[1] = np.maximum(0,coords_list[1])

        user_token = user_info['token']

        if frame_index is None:
            # Pulling specific region from an image using provided coordinates            
            try:
                if not 'frames' in slide_metadata:
                    image_region = Image.open(
                        BytesIO(
                            requests.get(
                                self.gc.urlBase+f'/item/{item_id}/tiles/region?token={user_token}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}'
                            ).content
                        )
                    )
                else:

                    style_string = '&style={"bands": [{"frame":0,"palette":"#f00"},{"frame":1,"palette":"#0f0"},{"frame":2,"palette":"#00f"}]}'
                    image_region = Image.open(
                        BytesIO(
                            requests.get(
                                self.gc.urlBase+f'/item/{item_id}/tiles/region?token={user_token}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}'+style_string
                            ).content
                        )
                    )
            
            except Exception as e:
                print('-------------------------------------------------')
                print(f'Error: {e}')
                print(f'Error reading image region from item: {item_id}')
                print(f'Provided coordinates: {coords_list}')
                print(f'------------------------------------------------')

                return np.zeros((100,100))

        else:
            # Hoping that all the frames of each image are the same size. They should be.
            # coords_list is organized: [min_x, min_y, max_x, max_y]
            if frame_index<len(slide_metadata['frames']):
                try:
                    image_region = Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{item_id}/tiles/region?token={user_token}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}&frame={frame_index}').content))
                except:
                    print('-------------------------------------------------')
                    print(f'Error reading image region from item: {item_id}')
                    print(f'Provided coordinates: {coords_list}')
                    print(f'------------------------------------------------')

                    return np.zeros((100,100))
            else:
                # This is if the requested frame index is greater than the number of frames for this image
                # Hopefully we never come here.
                return np.zeros((int(coords_list[3]-coords_list[1]),int(coords_list[2]-coords_list[0])))

        return image_region

    def get_annotation_image(self,slide_id,user_info,bounding_box):

        min_x = bounding_box[0]
        min_y = bounding_box[1]
        max_x = bounding_box[2]
        max_y = bounding_box[3]

        image_region = np.array(self.get_image_region(slide_id,user_info,[min_x,min_y,max_x,max_y]))

        return image_region

    def get_user_folder_id(self,folder_name:str,username:str):

        # Finding current user's private folder and returning the parent ID
        user_folder = f'/user/{username}/{folder_name}'
        print(f'user_folder: {user_folder}')
        try:
            folder_id = self.gc.get('/resource/lookup',parameters={'path':user_folder})['_id']
        except girder_client.HttpError:
            # This is if the folder doesn't exist yet (only works with one nest so a folder in an already existing folder)
            parent_id = self.gc.get('/resource/lookup',parameters={'path':'/'.join(user_folder.split('/')[0:-1])})['_id']
            
            self.gc.post('/folder',parameters={'parentId':parent_id,'name':folder_name.split('/')[-1],'description':'Folder created by FUSION'})
            folder_id = self.gc.get('/resource/lookup',parameters={'path':user_folder})['_id']


        return folder_id

    def get_new_upload_id(self,parent_folder:str):

        # Getting the items in the specified folder and then getting  the id of the newest item
        folder_items = self.gc.get(f'resource/{parent_folder}/items',parameters={'type':'folder'})

        if len(folder_items)>0:
            # Getting all the updated datetime strings
            updated_list = [datetime.fromisoformat(i['updated']) for i in folder_items]
            # Getting latest updated file
            latest_idx = np.argmax(updated_list)

            new_upload_id = folder_items[latest_idx]['_id']

            return new_upload_id
        else:
            return None

    def get_user_jobs(self, user_id):
        """
        Returns list of job ids for a user. 
        """

        user_job_list = self.gc.get('/job',
                                    parameters = {
                                        'userId': user_id,
                                        'limit': 10
                                    })

        return user_job_list

    def get_job_status(self,job_id: Union[str,None]):
        """
        Returns status of job (3=complete, 2 = started/in-progress), and most recent log (if applicable)
        """
        if not job_id is None:
            job_info = self.gc.get(f'/job/{job_id}')
            if 'log' in job_info:
                #print(f"most recent log: {job_info['log'][-1]}")
                most_recent_log = job_info['log'][-1]
            else:
                most_recent_log = ''
            return job_info['status'], most_recent_log
        else:
            # Return complete, no logs
            return 3, ''
    
    def get_slide_thumbnail(self,item_id:str):

        #thumbnail = Image.open(BytesIO(self.gc.get(f'/item/{item_id}/tiles/thumbnail?token={self.user_token}')))
        try:
            thumbnail = Image.open(BytesIO(requests.get(f'{self.gc.urlBase}/item/{item_id}/tiles/thumbnail?width=200&height=200&token={self.user_token}').content))
            return thumbnail
        except:
            return np.zeros((200,200))

    def run_histo_qc(self,folder_id:str):

        print('Running HistoQC')
        response = self.gc.post(f'/folder/{folder_id}/histoqc')
        print('Done!')
        return response

    def get_asset_items(self,assets_path):

        # Key items to grab from assets:
        # cell_graphics_key
        # morphometrics_reference
        # asct+b table
        # usability study usernames

        self.fusion_assets = assets_path
        self.usability_users = self.update_usability()

        # Downloading JSON resource
        with open('./assets/graphic_reference.json','r') as f:
            self.cell_graphics_key = json.load(f)

        self.cell_names = []
        for ct in self.cell_graphics_key:
            self.cell_names.append(self.cell_graphics_key[ct]['full'])

        with open('./assets/morphometrics_reference.json','r') as f:
            self.morphometrics_reference = json.load(f)
        
        self.morpho_names = []
        for mo in self.morphometrics_reference["Morphometrics"]:
            mo_name = mo['name']
            if not '{}' in mo_name:
                self.morpho_names.append(mo_name)
            else:
                if not mo['subcompartments']=='All':
                    for sc in mo['subcompartments']:
                        self.morpho_names.append(mo_name.replace('{}',sc))
                else:
                    for sc in ['Nuclei','Luminal Space','Eosinophilic']:
                        self.morpho_names.append(mo_name.replace('{}',sc))

        # Getting asct+b table
        self.asct_b_table = pd.read_csv('./assets/Kidney_v1.2 - Kidney_v1.2.csv',skiprows=list(range(10)))

        # Generating plot feature selection dictionary
        self.generate_feature_dict(self.default_slides)

    def update_usability(self, updated_info = None):
        
        # Checking usability study usernames in FUSION Assets folder
        # Running at startup and then when pages change so we can update this file without restarting FUSION
        try:
            usability_usernames_id = self.gc.get('resource/lookup',parameters={'path':self.fusion_assets+'usability_study_information/usability_study_usernames.json'})
            if updated_info is None:
                usability_info = self.gc.get(f'/item/{usability_usernames_id["_id"]}/download')
                self.usability_users = usability_info
                return usability_info
            else:
                
                item_files = self.gc.get(f'/item/{usability_usernames_id["_id"]}/files',parameters={'limit':1000})
                put_response = self.gc.put(f'/file/{item_files[0]["_id"]}/contents',
                    parameters={'size':len(json.dumps(updated_info).encode('utf-8'))},
                    data = json.dumps(updated_info)
                )

                post_response = self.gc.post(f'/file/chunk',
                    parameters={
                        'size':len(json.dumps(updated_info).encode('utf-8')),
                        'offset':0,
                        'uploadId':put_response['_id']
                        },
                    data = json.dumps(updated_info)
                )

                self.usability_users = updated_info
        except girder_client.HttpError:
            self.usability_users = {}

    def check_usability(self,username):

        # Checking if a given username is involved in the usability study. Returning info.
        user_info = None
        if username in self.usability_users['usability_study_admins']:
            user_info = {
                'type':'admin'
            }
        elif username in list(self.usability_users['usability_study_users'].keys()):
            user_info = self.usability_users['usability_study_users'][username]

        return user_info

    def generate_feature_dict(self,slide_list):
        
        # Given a list of slides (output of GET /item/{item_id}), generate label options, feature options, and filter options
        slide_folders = np.unique([i['folderId'] for i in slide_list]).tolist()

        folder_names = [self.gc.get(f'/folder/{i}')['name'] for i in slide_folders]
        slide_names = [i['name'] for i in slide_list]

        # Default labels are FTU, Slide Name, Cell Type, and Morphometric
        self.label_dict = [
            {'label':'FTU','value':'FTU','disabled':False},
            {'label':'Slide Name','value':'Slide Name','disabled':False},
            {'label':'Folder Name','value':'Folder Name','disabled':False},
            {'label':'Cell Type','value':'Cell Type','disabled':True},
            {'label':'Morphometric','value':'Morphometric','disabled':True}
        ]
        
        # Change this if you ever want to filter slides based on metadata values (Spatial Omics Type might be one)
        self.filter_keys = []
        l_i = -1
        label_filter_children = []
        """
        # Adding labels according to current slide-dataset metadata
        meta_labels = []
        for f in slide_folders:
            slides_in_f = [i for i in slide_list if i['folderId']==f]
            for s in slides_in_f:
                meta_labels.extend(list(s['meta'].keys()))
        # Adding only unique labels
        meta_labels = np.unique(meta_labels).tolist()
        for m in meta_labels:
            if m=='FTUs':
                self.label_dict.append({
                    'label': m,
                    'value': m,
                    'disabled':False
                })
        
        # Creating filter label dict for subsetting plot data
        self.filter_keys = []
        label_filter_children = []
        for l_i,l in enumerate(self.label_dict):
            if not l['disabled']:
                
                l_dict = {
                    'title':l['label'],
                    'key':f'0-{l_i}',
                    'children':[]
                }
                # Finding the different possible values for each of those labels
                l_vals = []
                for f in slide_folders:
                    if l['label'] in list(self.slide_datasets[f]['Metadata'].keys()):
                        if not type(self.slide_datasets[f]['Metadata'][l['label']])==int:
                            if ',' in self.slide_datasets[f]['Metadata'][l['label']]:
                                l_vals.extend(self.slide_datasets[f]['Metadata'][l['label']].split(','))
                            else:
                                l_vals.append(self.slide_datasets[f]['Metadata'][l['label']])
                        else:
                            l_vals.append(self.slide_datasets[f]['Metadata'][l['label']])
                # Getting unique label values
                u_l_vals = np.unique(l_vals).tolist()
                for u_l_i,u_l in enumerate(u_l_vals):
                    l_dict['children'].append({
                        'title': u_l,
                        'key':f'0-{l_i}-{u_l_i}'
                    })
                    self.filter_keys.append({'title':u_l,'key':f'0-{l_i}-{u_l_i}'})
                
                # Only add the labels that have sub-values, if they don't then it isn't used for any of the current slides
                if len(l_vals)>0:
                    label_filter_children.append(l_dict)
                    self.filter_keys.append({'title':l['label'],'key':f'0-{l_i}'})
        """

        # Adding slide names to label_filter_children
        slide_names_children = {
            'title':'Slide Names',
            'key':f'0-{l_i+1}',
            'children':[]
        }
        for s_i,s in enumerate(slide_names):
            slide_names_children['children'].append(
                {
                    'title':s,
                    'key':f'0-{l_i+1}-{s_i}',
                }
            )
            self.filter_keys.append({'title':s,'key':f'0-{l_i+1}-{s_i}'})
        
        self.filter_keys.append({'title':'Slide Names','key':f'0-{l_i+1}'})
        label_filter_children.append(slide_names_children)

        # Adding folder names to label_filter_children
        folder_names_children = {
            'title':'Folder Names',
            'key':f'0-{l_i+2}',
            'children':[]
        }
        for f_i, f in enumerate(folder_names):
            folder_names_children['children'].append(
                {
                    'title':f,
                    'key':f'0-{l_i+2}-{f_i}'
                }
            )
            self.filter_keys.append({'title':f,'key':f'0-{l_i+2}-{f_i}'})

        self.filter_keys.append({'title':'Folder Names','key':f'0-{l_i+2}'})
        label_filter_children.append(folder_names_children)

        self.label_filter_dict = {
            'title':'Filter Labels',
            'key':'0',
            'children': label_filter_children
        }

        # Dictionary defining plotting items in hierarchy
        morphometrics_children = []
        self.feature_keys = []
        for g_i,g in enumerate(np.unique([i['group'] for i in self.morphometrics_reference['Morphometrics']]).tolist()):
            group_dict = {
                'title':g,
                'key':f'0-0-{g_i}',
                'children':[]
            }
            sub_comp_offset = 0

            for f_i,f in enumerate([i['name'] for i in self.morphometrics_reference['Morphometrics'] if i['group']==g]):

                if '{}' in f:
                    for sc_i,sub_comp in enumerate(['Eosinophilic','Luminal Space','Nuclei']):
                        group_dict['children'].append({
                            'title':f.replace('{}',sub_comp),
                            'key':f'0-0-{g_i}-{sub_comp_offset}'
                        })
                        self.feature_keys.append({'title':f.replace('{}',sub_comp),'key':f'0-0-{g_i}-{sub_comp_offset}'})
                        sub_comp_offset+=1

                else:
                    group_dict['children'].append({
                        'title':f,
                        'key':f'0-0-{g_i}-{sub_comp_offset}'
                    })
                    self.feature_keys.append({'title':f,'key':f'0-0-{g_i}-{sub_comp_offset}'})
                    sub_comp_offset+=1

            morphometrics_children.append(group_dict)

        cell_comp_children = []
        for c_i,c in enumerate(np.unique([self.cell_graphics_key[i]['structure'][0] for i in self.cell_graphics_key]).tolist()):

            structure_children = {
                'title':c,
                'key':f'0-1-{c_i}',
                'children':[]
            }
            for sc_i,sc in enumerate([self.cell_graphics_key[i]['full'] for i in self.cell_graphics_key if self.cell_graphics_key[i]['structure'][0]==c]):
                
                structure_children['children'].append({
                    'title':sc,
                    'key':f'0-1-{c_i}-{sc_i}'
                })
                self.feature_keys.append({'title':sc,'key':f'0-1-{c_i}-{sc_i}'})

            cell_comp_children.append(structure_children)

        self.plotting_feature_dict = {
            'title':'All Features',
            'key': '0',
            'children': [
                {
                    'title':'Morphometrics',
                    'key':'0-0',
                    'children': morphometrics_children
                },
                {
                    'title':'Cellular Composition',
                    'key':'0-1',
                    'children': cell_comp_children
                }
            ]
        }

    def load_clustering_data(self,user_info):
        # Grabbing feature clustering info from user's public folder
        try:
            public_folder_id = self.gc.get('/resource/lookup',parameters={'path':f'/user/{user_info["login"]}/Public'})['_id']
            public_folder_contents = self.gc.get(f'/resource/{public_folder_id}/items?token={user_info["token"]}',parameters={'type':'folder','limit':10000})

            public_folder_names = [i['name'] for i in public_folder_contents]
            if 'FUSION_Clustering_data.json' in public_folder_names:
                print('Found clustering data')
                cluster_data_id = public_folder_contents[public_folder_names.index('FUSION_Clustering_data.json')]['_id']

                cluster_json = json.loads(requests.get(f'{self.gc.urlBase}/item/{cluster_data_id}/download?token={user_info["token"]}').content)
                try:
                    cluster_data = pd.DataFrame.from_dict(cluster_json)

                    # Fixing columns that say PAS
                    cluster_data.columns = [i.replace('PAS','Eosinophilic') for i in cluster_data.columns.tolist()]
                    print('Clustering data loaded')
                    cluster_data = cluster_data.loc[:,~cluster_data.columns.duplicated()].copy()

                except ValueError:
                    cluster_data = pd.DataFrame()    

                return cluster_data
            else:
                print('No clustering data found')
                return pd.DataFrame()
        except girder_client.HttpError:
            # Maybe just load the default clustering data if there's an error?
            return pd.DataFrame()

    def save_to_user_folder(self,save_object, user_info, output_path = None):

        if not os.path.exists('/tmp'):
            os.makedirs('/tmp')

        if output_path is None:
            output_path = f'/user/{user_info["login"]}/Public'
        output_path_id = self.gc.get('/resource/lookup',parameters={'path':output_path})['_id']

        # Removing file if there's a duplicate
        current_items = self.gc.get(f'/resource/{output_path_id}/items?token={user_info["token"]}',parameters={'type':'folder','limit':10000})
        current_items_names = [i['name'] for i in current_items]
        if save_object['filename'] in current_items_names:
           self.gc.delete(f'/item/{current_items[current_items_names.index(save_object["filename"])]["_id"]}')

        # Saving dataframe
        if 'csv' in save_object['filename']:
            save_object['content'].to_csv(f'/tmp/{save_object["filename"]}')

        elif 'xlsx' in save_object['filename']:
            with pd.ExcelWriter(f'/tmp/{save_object["filename"]}') as writer:
                save_object['content'].to_excel(writer,engine='openpyxl')
        
        elif 'png' in save_object['filename'] or 'tiff' in save_object['filename']:
            if 'png' in save_object['filename']:
                # Should be a PIL.Image object
                Image.fromarray(save_object['content']).save(f'/tmp/{save_object["filename"]}')
            elif 'tiff' in save_object['filename']:
                tifffile.imwrite(
                    f'/tmp/{save_object["filename"]}',
                    save_object['content'],
                    shape = save_object['content'].shape,
                    metadata={'axes':'YXC'}
                    )

        print(f'Uploading: {save_object["filename"]} to {output_path_id}')
        upload_file_response = self.gc.uploadFileToFolder(
            folderId = output_path_id,
            filepath = f'/tmp/{save_object["filename"]}'
        )

        # Adding metadata to uploaded object
        if 'metadata' in save_object:
            self.add_slide_metadata(upload_file_response['itemId'],save_object['metadata'])


        return upload_file_response

    def grab_from_user_folder(self,filename,username,folder = None):
        
        if folder is None:
            # Checking user public folder by default
            public_folder_path = f'/user/{username}/Public'
        else:
            public_folder_path = f'/user/{username}/Public/{folder}'
        
        public_folder_id = self.gc.get('/resource/lookup',parameters={'path':public_folder_path})['_id']
        public_folder_items = self.gc.get(f'/resource/{public_folder_id}/items?token={self.user_token}',parameters= {'type':'folder','limit':10000})

        public_folder_names = [i['name'] for i in public_folder_items]

        if filename in public_folder_names:
            if 'csv' in filename:
                user_folder_file = pd.read_csv(BytesIO(requests.get(f'{self.apiUrl}/item/{public_folder_items[public_folder_names.index(filename)]["_id"]}/download?token={self.user_token}').content))
            elif 'json' in filename:
                user_folder_file = json.loads(requests.get(f'{self.apiUrl}/item/{public_folder_items[public_folder_names.index(filename)]["_id"]}/download?token={self.user_token}').content)
            elif 'png' in filename:
                user_folder_file = Image.open(BytesIO(requests.get(f'{self.apiUrl}/item/{public_folder_items[public_folder_names.index(filename)]["_id"]}/download?token={self.user_token}').content))            
            else:
                print(f'File format: {filename} not implemented yet!')
                user_folder_file = None
        else:
            print(f'File: {filename} not found! :(')
            user_folder_file = None


        return user_folder_file

    def check_user_folder(self,folder_name, user_info, subfolder = None, sub_sub_folder = None):
        """
        Checking if a folder with a specific name is in the user's public folder
        """
        if subfolder is None:
            public_folder_path = f'/user/{user_info["login"]}/Public'
        else:
            public_folder_path = f'/user/{user_info["login"]}/Public/{folder_name}'

            if not sub_sub_folder is None:
                public_folder_path = f'/user/{user_info["login"]}/Public/{folder_name}/{subfolder}'
                folder_name = sub_sub_folder
            else:
                folder_name = subfolder

        public_folder_id = self.gc.get('/resource/lookup',parameters={'path':public_folder_path})['_id']

        # Getting all folders in the folder
        all_folders = self.gc.get(f'/folder',parameters = {'parentType':'folder','parentId':public_folder_id,'limit':100000})
        folder_names = [i['name'] for i in all_folders]

        if folder_name in folder_names:
            return all_folders[folder_names.index(folder_name)]
        else:
            return None

    def get_annotation_session_progress(self, session_name, user_info):
        """
        Getting slide and image count (both with annotations and with labels)
        """
        session_progress = {
            'slides': 0,
            'annotations': 0,
            'labels': 0
        }

        session_info = {
            'name':'',
            'Annotations':[],
            'Labels':[],
            'user_type': 'annotator'
        }

        path_to_ann_sessions = f'/user/{user_info["login"]}/Public/FUSION Annotation Sessions/{session_name}'
        folder_id = self.gc.get('/resource/lookup',parameters={'path':path_to_ann_sessions})

        session_info['name'] = folder_id['name']
        session_info['folder_id'] = folder_id['_id']
        session_info['Annotations'] = folder_id['meta']['Annotations']
        session_info['Labels'] = folder_id['meta']['Labels']

        if 'Users' in folder_id['meta']:
            session_info['user_type'] = 'admin'

        # Getting slide count
        all_folders = self.gc.get(f'/folder',parameters = {'parentType':'folder','parentId':folder_id["_id"],'limit':100000})
        session_progress['slides'] += len(all_folders)
        
        # Getting annotation and label count
        for a in all_folders:
            folders_in_slide = self.gc.get(f'folder',parameters = {'parentType':'folder','parentId':a["_id"],'limit':10000})
            folder_names = [i['name'] for i in folders_in_slide]
            if 'Images' in folder_names:
                image_folder_id = folders_in_slide[folder_names.index('Images')]['_id']
                image_contents = self.gc.get(f'/folder/{image_folder_id}/details')
                session_progress['annotations'] += image_contents['nItems']
            
            if 'Images with Labels' in folder_names:
                label_folder_id = folders_in_slide[folder_names.index('Images with Labels')]["_id"]
                label_contents = self.gc.get(f'/folder/{label_folder_id}/details')
                session_progress['labels'] += label_contents['nItems']


        return session_progress, session_info

    def create_user_folder(self, parent_path, folder_name, metadata = None):
        """
        Creating a folder in user's public folder
        """
        public_folder_path = parent_path
        public_folder_id = self.gc.get('/resource/lookup',parameters={'path':public_folder_path})['_id']

        # Creating folder
        if not metadata is None:
            new_folder = self.gc.loadOrCreateFolder(
                folderName = folder_name,
                parentId = public_folder_id,
                parentType = 'folder',
                metadata = metadata
            )
        else:
            new_folder = self.gc.loadOrCreateFolder(
                folderName = folder_name,
                parentId = public_folder_id,
                parentType = 'folder'
            )

        return new_folder

    def add_slide_metadata(self,item_id,metadata_dict):

        # Adding slide-level metadata from upload page
        self.gc.put(f'/item/{item_id}/metadata',parameters={'metadata':json.dumps(metadata_dict)})



class DownloadHandler:
    def __init__(self,
                dataset_handler,
                verbose = False):
        
        self.dataset_handler = dataset_handler
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

    def zip_data(self,download_data_list,folder = False):
        """
        Adding downloads to a zip file in order to only have one download for multiple files
        """

        output_file = './assets/FUSION_Download.zip'

        if not folder:
            # Writing temporary directory 
            if not os.path.exists('./assets/FUSION_Download/'):
                os.makedirs('./assets/FUSION_Download/')

            # Writing files to the temp directory

            # Checking if there are any xlsx files first
            excel_files = [i for i in download_data_list if i['filename'].split('.')[-1]=='xlsx']
            if len(excel_files)>0:
                # Getting unique filenames
                unique_filenames = np.unique([i['filename'] for i in excel_files]).tolist()
                for u_f in unique_filenames:
                    save_path = './assets/FUSION_Download/'+u_f
                    with pd.ExcelWriter(save_path) as writer:
                        for e in excel_files:
                            if e['filename']==u_f:
                                e['content'].to_excel(writer,sheet_name=e['sheet'],engine='openpyxl')

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
                
                elif file_ext in ['tiff','png']:
                    
                    # Maybe this will work?
                    #TODO: test this for larger ROIs (& multi-frame)
                    Image.fromarray(d['content']).save(save_path)

        # Writing temporary data to a zip file
        with zipfile.ZipFile(output_file,'w', zipfile.ZIP_DEFLATED) as zip:
            for path,subdirs,files in os.walk('./assets/FUSION_Download/'):
                for name in files:
                    zip.write(os.path.join(path,name))


        try:
            shutil.rmtree('./assets/FUSION_Download/')
        except OSError as e:
            print(f'OSError removing FUSION_Download directory: {e.strerror}')

    def extract_annotations(self, slide, box_poly, format):
        
        # Extracting annotations from the current slide object
        intersecting_annotations = wak.Annotation()
        intersecting_annotations.add_names(slide.ftu_names)
        width_scale = slide.x_scale
        height_scale = slide.y_scale
        for ftu in slide.ftu_names:
        
            # Finding which members of a specfied ftu group intersect with the provided box_poly
            if not ftu=='Spots':
                ftu_intersect_idx = [i for i in range(0,len(slide.ftu_polys[ftu])) if slide.ftu_polys[ftu][i].intersects(box_poly)]
                intersecting_polys = [slide.ftu_polys[ftu][i] for i in ftu_intersect_idx]
                intersecting_props = [slide.ftu_props[ftu][i] for i in ftu_intersect_idx]
            else:
                ftu_intersect_idx = [i for i in range(0,len(slide.spot_polys)) if slide.spot_polys[i].intersects(box_poly)]
                intersecting_polys = [slide.spot_polys[i] for i in ftu_intersect_idx]
                intersecting_props = [slide.spot_polys[i] for i in ftu_intersect_idx]

            if len(intersecting_polys)>0:

                # Adjusting coordinates for polygons based on width and height scale
                #scaled_polys = []
                for i_p_idx,i_p in enumerate(intersecting_polys):
                    og_coords = list(i_p.exterior.coords)
                    scaled_coords = [(i[1]/height_scale,i[0]/width_scale) for i in og_coords]

                    intersecting_annotations.add_shape(
                        poly = Polygon(scaled_coords),
                        box_crs=[0,0],
                        structure = ftu,
                        name = f'{ftu}_{ftu_intersect_idx[i_p_idx]}',
                        properties=intersecting_props[i_p_idx]
                    )

        if format=='GeoJSON':
            
            save_name = slide.slide_name.replace('.'+slide.slide_ext,'.geojson')
            final_ann = wak.GeoJSON(intersecting_annotations,verbose=False).geojson

        elif format == 'Aperio XML':
            
            save_name = slide.slide_name.replace('.'+slide.slide_ext,'.xml')

            final_ann = wak.AperioXML(intersecting_annotations,verbose=False).xml
            final_ann = ET.tostring(final_ann,encoding='unicode',pretty_print=True)

        elif format == 'Histomics JSON':
            
            save_name = slide.slide_name.replace('.'+slide.slide_ext,'.json')
            final_ann = wak.Histomics(intersecting_annotations,verbose=False).json       

        return [{'filename': save_name, 'content':final_ann}]
    
    """
    def extract_metadata(self,slides, include_meta):
    
    """
    def extract_cell(self, intersecting_ftus, file_name):
        # Output here is a dictionary containing Main_Cell_Types and Cell_States for each FTU
        # Main_Cell_Types is a pd.DataFrame with a column for every cell type and an index of the FTU label
        # Cell states is a dictionary of pd.DataFrames with a column for every cell state and an index of the FTU label for each main cell type

        #TODO: Update for CODEX

        # Formatting for downloads
        download_data = []
        for ftu in list(intersecting_ftus.keys()):
            
            # intersecting_ftus is a dictionary containing each FTU and a list of intersecting properties
            ftu_cell_types_df = pd.DataFrame.from_records([i['Main_Cell_Types'] for i in intersecting_ftus[ftu] if 'Main_Cell_Types' in i])

            # Main cell types should just be one file
            if 'Excel' in file_name:
                main_content = {'filename':'Main_Cell_Types.xlsx','sheet':ftu,'content':ftu_cell_types_df}
            elif 'CSV' in file_name:
                main_content = {'filename':f'{ftu}_Main_Cell_Types.csv','content':ftu_cell_types_df}
            else:
                print('Invalid format (RDS will come later)')
                main_content = {}
            
            download_data.append(main_content)

            # Cell state info
            cell_states_list = [i['Cell_States'] for i in intersecting_ftus[ftu] if 'Cell_States' in i]
            main_cell_types = np.unique([list(i.keys()) for i in cell_states_list]).tolist()
            for mc in main_cell_types:

                ftu_cell_states_df = pd.DataFrame.from_records([i[mc] for i in cell_states_list if mc in i])

                if 'Excel' in file_name:
                    state_content = {'filename':f'{ftu}_Cell_States.xlsx','sheet':mc.replace('/',''),'content':ftu_cell_states_df}
                elif 'CSV' in file_name:
                    state_content = {'filename':f'{ftu}_{mc.replace("/","")}_Cell_States.csv','content':ftu_cell_states_df}
                else:
                    print('Invalid format (RDS will come later)')
                    state_content = {}

                download_data.append(state_content)

        return download_data

    def extract_manual_rois(self, current_slide, options, user_info):

        download_data = []

        # Initializing bounding box
        if current_slide.spatial_omics_type=='Visium':
            if 'Image' in options:
                bounding_box = {'minx':0,'miny':0,'maxx':0,'maxy':0}

                for m in current_slide.manual_rois:

                    # Getting geojson property for manual ROI
                    m_geojson = m['geojson']

                    # Scaling coordinates to slide coordinates
                    slide_coordinates = current_slide.convert_map_coords(m_geojson['geometry']['coordinates'])
                    
                    # Checking with current bounding box (pretty sure the geojson is X,Y)
                    slide_coord_array = np.array(slide_coordinates)
                    bounding_box['minx'] = np.minimum(bounding_box['minx'],np.min(slide_coord_array[:,0]))
                    bounding_box['miny'] = np.minimum(bounding_box['miny'],np.min(slide_coord_array[:,1]))
                    bounding_box['maxx'] = np.maximum(bounding_box['maxx'],np.max(slide_coord_array[:,0]))
                    bounding_box['maxy'] = np.maximum(bounding_box['maxy'],np.max(slide_coord_array[:,1]))

                    # Replacing coordinates in m_geojson
                    #m_geojson['geometry']['coordinates'] = slide_coordinates
                    #total_geojson['features'].append(m_geojson)

                # Now extracting that image region from the bounding box
                full_image_region = self.dataset_handler.get_image_region(
                    current_slide.item_id,
                    user_info,
                    [bounding_box['minx'],bounding_box['miny'],bounding_box['maxx'],bounding_box['maxy']]
                )

                download_data.append({
                    'filename':f'{current_slide.slide_name.replace(current_slide.slide_ext,"")}_Manual_ROIs.tiff',
                    'content': full_image_region
                })

            if 'Cell' in options:

                # Getting cell type info from the manual ROI geojsons
                cell_filename = f'{current_slide.slide_name.replace(current_slide.slide_ext,"")}_Cell.xlsx'
                cell_data = []
                cell_state_data = []
                for m_idx,m in enumerate(current_slide.manual_rois):
                    if 'Main_Cell_Types' in m['geojson']['features'][0]['properties']:
                        cell_types_data = m['geojson']['features'][0]['properties']['Main_Cell_Types']
                        cell_types_data['Name'] = f'Manual_ROI_{m_idx+1}'

                        cell_state_data.append(m['geojson']['features'][0]['properties']['Cell_States'])

                        cell_data.append(cell_types_data)

                main_cell_types_df = pd.DataFrame.from_records(cell_data)
                download_data.append({
                    'filename': cell_filename,
                    'sheet':'Main_Cell_Types',
                    'content': main_cell_types_df
                })

                # Now getting cell state info for each cell type
                for m in main_cell_types_df.columns.tolist():
                    if not m=='Name':

                        # Pulling that main cell types cell state info from each manual ROI
                        # This should be a list of dictionaries for each manual ROI 
                        m_cell_state = [i[m] for i in cell_state_data]
                        
                        m_cell_state_df = pd.DataFrame.from_records(m_cell_state)
                        m_cell_state_df['Name'] = [f'Manual_ROI_{i+1}' for i in range(0,len(current_slide.manual_rois))]

                        download_data.append({
                            'filename': cell_filename,
                            'sheet': m,
                            'content': m_cell_state_df
                        })

        elif current_slide.spatial_omics_type=='Regular':
            if 'Image' in options:
                bounding_box = {'minx':0,'miny':0,'maxx':0,'maxy':0}

                for m in current_slide.manual_rois:

                    # Getting geojson property for manual ROI
                    m_geojson = m['geojson']

                    # Scaling coordinates to slide coordinates
                    slide_coordinates = current_slide.convert_map_coords(m_geojson['geometry']['coordinates'])
                    
                    # Checking with current bounding box (pretty sure the geojson is X,Y)
                    slide_coord_array = np.array(slide_coordinates)
                    bounding_box['minx'] = np.minimum(bounding_box['minx'],np.min(slide_coord_array[:,0]))
                    bounding_box['miny'] = np.minimum(bounding_box['miny'],np.min(slide_coord_array[:,1]))
                    bounding_box['maxx'] = np.maximum(bounding_box['maxx'],np.max(slide_coord_array[:,0]))
                    bounding_box['maxy'] = np.maximum(bounding_box['maxy'],np.max(slide_coord_array[:,1]))

                    # Replacing coordinates in m_geojson
                    #m_geojson['geometry']['coordinates'] = slide_coordinates
                    #total_geojson['features'].append(m_geojson)

                # Now extracting that image region from the bounding box
                full_image_region = self.dataset_handler.get_image_region(
                    current_slide.item_id,
                    user_info,
                    [bounding_box['minx'],bounding_box['miny'],bounding_box['maxx'],bounding_box['maxy']]
                )

                download_data.append({
                    'filename':f'{current_slide.slide_name.replace(current_slide.slide_ext,"")}_Manual_ROIs.tiff',
                    'content': full_image_region
                })

        elif current_slide.spatial_omics_type=='CODEX':
            if 'Image' in options:
                bounding_box = {'minx':0,'miny':0,'maxx':0,'maxy':0}

                for m in current_slide.manual_rois:

                    # Getting geojson property for manual ROI
                    m_geojson = m['geojson']

                    # Scaling coordinates to slide coordinates
                    slide_coordinates = current_slide.convert_map_coords(m_geojson['geometry']['coordinates'])
                    
                    # Checking with current bounding box (pretty sure the geojson is X,Y)
                    slide_coord_array = np.array(slide_coordinates)
                    bounding_box['minx'] = np.minimum(bounding_box['minx'],np.min(slide_coord_array[:,0]))
                    bounding_box['miny'] = np.minimum(bounding_box['miny'],np.min(slide_coord_array[:,1]))
                    bounding_box['maxx'] = np.maximum(bounding_box['maxx'],np.max(slide_coord_array[:,0]))
                    bounding_box['maxy'] = np.maximum(bounding_box['maxy'],np.max(slide_coord_array[:,1]))

                    # Replacing coordinates in m_geojson
                    #m_geojson['geometry']['coordinates'] = slide_coordinates
                    #total_geojson['features'].append(m_geojson)

                # Now extracting that image region from the bounding box (getting each frame for CODEX images)
                frame_list = []
                for frame in range(current_slide.n_frames):

                    # Getting each frame
                    full_image_region = self.dataset_handler.get_image_region(
                        current_slide.item_id,
                        user_info,
                        [bounding_box['minx'],bounding_box['miny'],bounding_box['maxx'],bounding_box['maxy']],
                        frame_index = frame
                    )

                    frame_list.append(full_image_region)

                download_data.append({
                    'filename':f'{current_slide.slide_name.replace(current_slide.slide_ext,"")}_Manual_ROIs.tiff',
                    'content': frame_list
                })

            if 'Cell' in options:
                
                #TODO: Update where it says "Main_Cell_Types" here as needed
                # Getting cell type info from the manual ROI geojsons
                cell_filename = f'{current_slide.slide_name.replace(current_slide.slide_ext,"")}_Cell.xlsx'
                cell_data = []
                cell_state_data = []
                for m_idx,m in enumerate(current_slide.manual_rois):
                    if 'Main_Cell_Types' in m['geojson']['features'][0]['properties']:
                        cell_types_data = m['geojson']['features'][0]['properties']['Main_Cell_Types']
                        cell_types_data['Name'] = f'Manual_ROI_{m_idx+1}'


                main_cell_types_df = pd.DataFrame.from_records(cell_data)
                download_data.append({
                    'filename': cell_filename,
                    'sheet':'Main_Cell_Types',
                    'content': main_cell_types_df
                })


        return download_data

    def extract_select_ftus(self,current_slide,options,user_info):

        #TODO: Update this for CODEX images
        # Extracting select ftus image/cell info
        download_data = []
        for s_idx, s in enumerate(current_slide.marked_ftus):

            if current_slide.spatial_omics_type == 'Visium':
                # Checking if including image data
                if 'Image' in options:
                    # Pulling out the coordinates from the geojson

                    s_coords = current_slide.convert_map_coords(s['geojson']['features'][0]['geometry']['coordinates'])
                    s_coords_array = np.array(s_coords)

                    # Getting bounding box for this structure
                    min_x = np.min(s_coords_array[:,0])
                    min_y = np.min(s_coords_array[:,1])
                    max_x = np.max(s_coords_array[:,0])
                    max_y = np.max(s_coords_array[:,1])

                    # Scaling coordinates (for saving boundary mask)
                    #TODO: Saving bounding box image
                    scaled_coords = [
                        [i[0]-min_x,i[1]-min_y]
                        for i in s_coords
                    ]

                    image = self.dataset_handler.get_image_region(
                        current_slide.item_id,
                        user_info,
                        [min_x,min_y,max_x,max_y]
                    )

                    download_data.append({
                        'filename': f'Marked_FTU_{s_idx+1}.png',
                        'content': image
                    })

                if 'Cell' in options:
                    # Pulling out cell type info for marked ftus
                    cell_filename = f'{current_slide.slide_name.replace(current_slide.slide_ext,"")}_Cell.xlsx'
                    cell_data = []
                    cell_state_data = []
                    for m_idx,m in enumerate(current_slide.marked_ftus):
                        if 'Main_Cell_Types' in m['geojson']['features'][0]['properties']:
                            cell_types_data = m['geojson']['features'][0]['properties']['Main_Cell_Types']
                            cell_types_data['Name'] = f'Marked_FTU_{m_idx+1}'

                            cell_state_data.append(m['geojson']['features'][0]['properties']['Cell_States'])

                            cell_data.append(cell_types_data)

                    main_cell_types_df = pd.DataFrame.from_records(cell_data)
                    download_data.append({
                        'filename': cell_filename,
                        'sheet':'Main_Cell_Types',
                        'content': main_cell_types_df
                    })

                    # Now getting cell state info for each cell type
                    for m in main_cell_types_df.columns.tolist():
                        if not m=='Name':

                            # Pulling that main cell types cell state info from each manual ROI
                            # This should be a list of dictionaries for each manual ROI 
                            m_cell_state = [i[m] for i in cell_state_data]
                            
                            m_cell_state_df = pd.DataFrame.from_records(m_cell_state)
                            m_cell_state_df['Name'] = [f'Marked_FTU_{i+1}' for i in range(0,len(current_slide.manual_rois))]

                            download_data.append({
                                'filename': cell_filename,
                                'sheet': m,
                                'content': m_cell_state_df
                            })

            elif current_slide.spatial_omics_type == 'Regular':
                # Checking if including image data
                if 'Image' in options:
                    # Pulling out the coordinates from the geojson

                    s_coords = current_slide.convert_map_coords(s['geojson']['features'][0]['geometry']['coordinates'])
                    s_coords_array = np.array(s_coords)

                    # Getting bounding box for this structure
                    min_x = np.min(s_coords_array[:,0])
                    min_y = np.min(s_coords_array[:,1])
                    max_x = np.max(s_coords_array[:,0])
                    max_y = np.max(s_coords_array[:,1])

                    # Scaling coordinates (for saving boundary mask)
                    #TODO: Saving bounding box image
                    scaled_coords = [
                        [i[0]-min_x,i[1]-min_y]
                        for i in s_coords
                    ]

                    image = self.dataset_handler.get_image_region(
                        current_slide.item_id,
                        user_info,
                        [min_x,min_y,max_x,max_y]
                    )

                    download_data.append({
                        'filename': f'Marked_FTU_{s_idx+1}.png',
                        'content': image
                    })

            elif current_slide.spatial_omics_type == 'CODEX':
                # Checking if including image data
                if 'Image' in options:
                    # Pulling out the coordinates from the geojson

                    s_coords = current_slide.convert_map_coords(s['geojson']['features'][0]['geometry']['coordinates'])
                    s_coords_array = np.array(s_coords)

                    # Getting bounding box for this structure
                    min_x = np.min(s_coords_array[:,0])
                    min_y = np.min(s_coords_array[:,1])
                    max_x = np.max(s_coords_array[:,0])
                    max_y = np.max(s_coords_array[:,1])

                    # Scaling coordinates (for saving boundary mask)
                    #TODO: Saving bounding box image
                    scaled_coords = [
                        [i[0]-min_x,i[1]-min_y]
                        for i in s_coords
                    ]

                    # Getting each frame for this image region
                    image_frame_list = []
                    for frame in range(current_slide.n_frames):
                        image = self.dataset_handler.get_image_region(
                            current_slide.item_id,
                            user_info,
                            [min_x,min_y,max_x,max_y]
                        )
                        image_frame_list.append(image)

                    download_data.append({
                        'filename': f'Marked_FTU_{s_idx+1}.png',
                        'content': image_frame_list
                    })

                #TODO: Update "Main_Cell_Types" to whatever it should be
                if 'Cell' in options:
                    # Pulling out cell type info for marked ftus
                    cell_filename = f'{current_slide.slide_name.replace(current_slide.slide_ext,"")}_Cell.xlsx'
                    cell_data = []
                    cell_state_data = []
                    for m_idx,m in enumerate(current_slide.marked_ftus):
                        if 'Main_Cell_Types' in m['geojson']['features'][0]['properties']:
                            cell_types_data = m['geojson']['features'][0]['properties']['Main_Cell_Types']
                            cell_types_data['Name'] = f'Marked_FTU_{m_idx+1}'

                    main_cell_types_df = pd.DataFrame.from_records(cell_data)
                    download_data.append({
                        'filename': cell_filename,
                        'sheet':'Main_Cell_Types',
                        'content': main_cell_types_df
                    })


        return download_data

    def extract_annotation_session(self, ann_session_id):
        """
        Grabbing contents of annotation session folder (save both image and metadata)
        """
        # Getting images and masks:
        download_output = self.dataset_handler.gc.downloadResource(
            ann_session_id,
            "./assets/FUSION_Download/",
            'folder'    
        )

        # Getting image metadata
        folder_items = self.dataset_handler.gc.get(
            f'/resource/{ann_session_id}/items',
            parameters = {
                'type': 'folder',
                'limit':1000000
            }
        )

        session_metadata = [{'name': i['name']} | i['meta'] for i in folder_items if 'png' in i['name']]
        metadata_df = pd.DataFrame.from_records(session_metadata)
        metadata_df.to_csv('./assets/FUSION_Download/Session_Metadata.csv')

        self.zip_data(None,folder=True)
        #zip_file_path = './assets/FUSION_Download.zip'





class GeneHandler:
    """
    Class made to get more information on genes from mygene.info
    """
    def __init__(self):

        self.info_url = 'https://mygene.info/v3/'
        self.hra_url = 'https://grlc.io/api-git/hubmapconsortium/ccf-grlc/subdir/fusion//?endpoint=https://lod.humanatlas.io/sparql'

        self.asct_b = pd.read_csv('./assets/asctb_release7.csv')


    def get_layout(self, gene_id:str):
        """
        Returns list of layout components (buttons and divs) when someone selects an overlay that contains "Gene Counts"
        """
        # The part after the "." is just the version number for that gene
        gene_info = self.get_gene_info(gene_id.split('Gene Counts --> ')[-1].split('.')[0])

        if not gene_info is None:
            if "alias" in gene_info:
                if type(gene_info["alias"])==list:
                    alias = ','.join(gene_info['alias'])
                elif type(gene_info['alias'])==str:
                    alias = gene_info['alias']
            else:
                alias = 'None Specified'

            if "summary" in gene_info:
                summary = gene_info['summary'].replace('[','').replace(']','')
            else:
                summary = "No summary available"

            if "HGNC" in gene_info:
                hgnc = gene_info['HGNC']
            else:
                hgnc = 'No HGNC found'

            gene_info_components = [
                html.H6('Gene Information',style = {'marginTop':'5px'}),
                #layout_handler.gen_info_button('Information on current gene selected, derived from mygene.info'),
                dbc.Row([
                    dbc.Col(html.P(f'HGNC Id: {hgnc}',id = {'type':'hgnc-id','index':0}))
                ]),
                dbc.Row([
                    dcc.Markdown(f'''
                                **Alias**: {alias}

                                **Summary**: {summary}
                                 ''')
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            'Get Anatomical Structures and Cell Types',
                            className = 'd-grid col-12 mx-auto',
                            id = {'type':'get-asct-butt','index':0}
                        )
                    )
                ]),
                dbc.Row([
                    html.Div(
                        children = [],
                        id = {'type':'asct-gene-table','index':0}
                    )
                ])
            ]
        else:
            gene_info_components = []

        return gene_info_components

    def get_gene_info(self, id:Union[str,list]):
        """
        Get information about a given gene id or list of ids.
        By default returns HGNC, alias, and summary
        Can be expanded to include go, pubmed articles, etc.
        """
        if isinstance(id,str):
            request_response = requests.get(f'{self.info_url}gene/{id}?fields=HGNC,alias,summary&dotfield=false&size=5')
            if request_response.ok:
                return request_response.json()
            else:
                return None
            
        elif isinstance(id,list):
            return_list = []
            for i in id:
                request_response = requests.get(f'{self.info_url}gene/{i}?fields=HGNC,alias,summary&dotfield=false&size=5')
                if request_response.ok:
                    return_list.append(request_response.json())
            
            return return_list

    def get_asct(self, id:Union[str,int]):
        """
        Get Anatomical Structure & Cell Type associated with a given HGNC Id.
        """
        # Have to add on the whole iri here:
        # HGNC id is a number but should be interpreted as a string
        id = f'http://identifiers.org/hgnc/{id}'
        request_response = requests.get(
            f'{self.hra_url.replace("fusion//","fusion//asct_by_biomarker")}&biomarker={id}',
            headers={'Accept':'application/json','Content-Type':'application/json'}
        )
        if request_response.ok:
            return pd.json_normalize(request_response.json()['results']['bindings'],max_level=1)
        else:
            print(f'{self.hra_url.replace("fusion//","fusion//asct_by_biomarker")}&biomarker={id}')
            print('Request not ok!')
            return None
        
    def get_cell(self, id:str):
        """
        Get all the cell types available within a given anatomical structure
        Input has to be an UBERON id "UBERON_######..."
        """

        # Modifiying input id
        id = f'http://purl.odolibrary.org/obo/{id}'
        request_response = requests.get(
            f'{self.hra_url.replace("fusion//","fusion//cell_by_location")}&location={id}',
            headers={'Accept':'application/json','Content-type':'application/json'}
        )

        print(request_response)
        if request_response.ok:
            print(request_response.content)
            return pd.DataFrame(request_response.content)
        else:
            return None

    def get_table(self,organ_selection: str):
        """
        Getting csv table containing asct+b info for a given organ
        """

        if organ_selection in self.asct_b['Organ'].tolist():

            csv_location = self.asct_b['csv'].tolist()[self.asct_b['Organ'].tolist().index(organ_selection)]

            new_table_req = requests.get(
                csv_location
            )

            if new_table_req.ok:
                new_table = pd.read_csv(BytesIO(new_table_req.content))
                new_table_content = pd.read_csv(BytesIO(new_table_req.content),skiprows=list(range(10)))
                new_table_info = new_table.iloc[1:9,:1]

            else:
                new_table_content = None
                new_table_info = None
        else:
            new_table_content = None
            new_table_info = None

        return new_table_content, new_table_info




















