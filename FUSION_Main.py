"""

Whole Slide Image heatmap generation and viewer for specific cell types using Ground Truth Spatial Transcriptomics

"""

import os
import sys
import pandas as pd
import numpy as np
import json

from glob import glob

from PIL import Image
import requests
from io import BytesIO

try:
    import openslide
except:
    import tiffslide as openslide

import lxml.etree as ET

from tqdm import tqdm

import shapely
from shapely.geometry import Polygon, Point, shape, box
from skimage.draw import polygon
from skimage.transform import resize
import geojson
import random

import requests
import girder_client

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import cm

from dash import dcc, ctx, Dash, MATCH, ALL, ALLSMALLER, dash_table, exceptions, callback_context,no_update
#from dash.long_callback import DiskcacheLongCallbackManager

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, Namespace, arrow_function
from dash_extensions.enrich import DashProxy, html, Input, Output, MultiplexerTransform, State
from dash_extensions import Download

from timeit import default_timer as timer
#import diskcache

from FUSION_WSI import WholeSlide, DSASlide
from Initialize_FUSION import DatasetHandler, LayoutHandler, DownloadHandler, GirderHandler


class SlideHeatVis:
    def __init__(self,
                app,
                layout_handler,
                dataset_handler,
                download_handler,
                wsi,
                cell_graphics_key,
                asct_b_table,
                cluster_metadata,
                slide_info_dict=None,
                run_type = None,
                ga_tag = None
                ):
                
        # Saving all the layouts for this instance
        self.layout_handler = layout_handler
        self.current_page = 'welcome'

        self.dataset_handler = dataset_handler
        self.current_overlays = self.layout_handler.initial_overlays

        self.download_handler = download_handler

        # Setting some app-related things
        self.app = app
        self.app.title = "FUSION"
        self.app.layout = self.layout_handler.current_initial_layout
        self.app._favicon = './assets/favicon.ico'
        self.app.validation_layout = html.Div(self.layout_handler.validation_layout)

        # If provided with a Google Analytics tag
        if not ga_tag is None:
            print(ga_tag)
            print(ga_tag is None)
            self.app.index_string = ga_tag

        self.run_type = run_type

        # clustering related properties (and also cell types, cell states, image_ids, etc.)
        self.metadata = cluster_metadata

        # Only use this slide_info_dict for non-dsa backend deployments
        if not self.run_type == 'dsa': 
            self.slide_info_dict = slide_info_dict
            self.current_slides = [
                {
                'Slide Names': i,
                'Dataset':self.dataset_handler.get_slide_dataset(i),
                'included':True
                }
                for i in list(self.slide_info_dict.keys())
            ]
        self.wsi = wsi

        if type(cell_graphics_key)==str:
            self.cell_graphics_key = json.load(open(cell_graphics_key))
        else:
            self.cell_graphics_key = cell_graphics_key

        # Inverting the graphics key to get {'full_name':'abbreviation'}
        self.cell_names_key = {}
        for ct in self.cell_graphics_key:
            self.cell_names_key[self.cell_graphics_key[ct]['full']] = ct

        # Number of main cell types to include in pie-charts
        self.plot_cell_types_n = 5

        # ASCT+B table for cell hierarchy generation
        self.table_df = asct_b_table    

        # FTU settings
        self.ftus = self.wsi.ftu_names
        self.ftu_colors = {
            'Glomeruli':'#390191',
            'Tubules':'#e71d1d',
            'Arterioles':'#b6d7a8',
            'Spots':'#dffa00'
        }

        self.current_ftu_layers = self.wsi.ftu_names+['Spots']
        self.current_ftus = self.wsi.ftu_names+['Spots']
        self.pie_ftu = self.current_ftu_layers[-1]
        self.pie_chart_order = self.current_ftu_layers.copy()

        # Specifying available properties with visualizations implemented
        # TODO:Add cell states in there later, need to figure out how to view proportions of different cell states in the same overlay
        self.visualization_properties = [
            'Area', 'Arterial Area', 'Average Cell Thickness', 'Average TBM Thickness', 'Cluster',
            'Luminal Fraction','Main_Cell_Types','Mesangial Area','Mesangial Fraction'
        ]

        self.current_chart_coords = [0,0]

        # Initializing some parameters
        self.current_cell = 'PT'
        self.current_barcodes = []

        # Cell Hierarchy related properties
        self.node_cols = {
            'Anatomical Structure':{'abbrev':'AS','x_start':50,'y_start':75,
                                    'base_url':'https://www.ebi.ac.uk/ols/ontologies/uberon/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FUBERON_'},
            'Cell Types':{'abbrev':'CT','x_start':250,'y_start':0,
                          'base_url':'https://www.ebi.ac.uk/ols/ontologies/uberon/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FCL_'},
            'Genes':{'abbrev':'BGene','x_start':450,'y_start':75,
                     'base_url':'https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/'}
        }

        # Colormap settings (customize later)
        self.color_map = cm.get_cmap('jet',255)
        self.cell_vis_val = 0.5
        self.ftu_style_handle = assign("""function(feature,context){
            const {color_key,current_cell,fillOpacity,ftu_colors} = context.props.hideout;
            
            if (current_cell==='cluster'){
                if (current_cell in feature.properties){
                    var cell_value = feature.properties.Cluster;
                    cell_value = (Number(cell_value)).toFixed(1);
                } else {
                    cell_value = Number.Nan;
                }
            } else if (current_cell==='max'){
                // Extracting all the cell values for a given FTU/Spot
                var cell_values = feature.properties.Main_Cell_Types;
                // Initializing some comparison values
                var cell_value = 0.0;
                var use_cell_value = 0.0;
                var cell_idx = -1.0;
                // Iterating through each cell type in cell values
                for (var key in cell_values){
                    cell_idx += 1.0;
                    var test_val = cell_values[key];
                    // If the test value is greater than the cell_value, replace cell value with that test value
                    if (test_val > cell_value) {
                        cell_value = test_val;
                        use_cell_value = cell_idx;
                    }
                }
                cell_value = (use_cell_value).toFixed(1);

            } else if (current_cell in feature.properties.Main_Cell_Types){
                var cell_value = feature.properties.Main_Cell_Types[current_cell];
                
                if (cell_value==1) {
                    cell_value = (cell_value).toFixed(1);
                } else if (cell_value==0) {
                    cell_value = (cell_value).toFixed(1);
                }
            } else if (current_cell in feature.properties){
                var cell_value = feature.properties[current_cell];

            } else {
                var cell_value = Number.Nan;
            }

            var style = {};
            if (cell_value == cell_value) {
                const fillColor = color_key[cell_value];

                style.fillColor = fillColor;
                style.fillOpacity = fillOpacity;

            } else {
                style.fillOpacity = 0.0;
            }

            return style;
            }
            """
        )

        # Adding callbacks to app
        self.vis_callbacks()
        self.all_layout_callbacks()
        self.builder_callbacks()
        self.welcome_callbacks()
        self.upload_callbacks()

        # Comment out this line when running on the web
        if self.run_type == 'local':
            self.app.run_server(debug=False,use_reloader=False,port=8000)

        elif self.run_type == 'AWS' or self.run_type=='dsa':
            self.app.run_server(host = '0.0.0.0',debug=False,use_reloader=False,port=8000)

    def view_instructions(self,n,is_open):
        if n:
            return not is_open
        return is_open
    
    def view_sidebar(self,n,is_open):
        if n:
            return not is_open
        return is_open

    def update_page(self,pathname):
        
        print(f'Navigating to {pathname}')
        if pathname.replace('/','') in self.layout_handler.layout_dict:
            self.current_page = pathname.replace('/','')

        else:
            self.current_page = 'welcome'

        container_content = self.layout_handler.layout_dict[self.current_page]
        description = self.layout_handler.description_dict[self.current_page]

        if self.current_page == 'vis':
            slide_style = {'marginBottom':'20px','display':'inline-block'}
        else:
            slide_style = {'display':'none'}

        return container_content, description, slide_style

    def all_layout_callbacks(self):

        # Adding callbacks for items in every page
        self.app.callback(
            [Output('container-content','children'),
             Output('descrip','children'),
             Output('slide-select-card','style')],
             Input('url','pathname'),
             prevent_initial_call = True
        )(self.update_page)

        self.app.callback(
            Output({'type':'collapse-content','index':MATCH},'is_open'),
            Input({'type':'collapse-descrip','index':MATCH},'n_clicks'),
            [State({'type':'collapse-content','index':MATCH},'is_open')],
            prevent_initial_call=True
        )(self.view_instructions)

        self.app.callback(
            Output({'type':'sidebar-offcanvas','index':MATCH},'is_open'),
            Input({'type':'sidebar-button','index':MATCH},'n_clicks'),
            [State({'type':'sidebar-offcanvas','index':MATCH},'is_open')],
            prevent_initial_call=True
        )(self.view_sidebar)

        self.app.callback(
            [Output('login-submit','color'),
             Output('login-submit','children'),
             Output('logged-in-user','children')],
            [Input('username-input','value'),
             Input('pword-input','value'),
             Input('login-submit','n_clicks')],
             prevent_initial_call=True
        )(self.girder_login)

    def vis_callbacks(self):

        self.app.callback(
            [Output('layer-control','children'),Output('colorbar-div','children')],
            [Input('cell-drop','value'),Input('vis-slider','value')],
            prevent_initial_call=True
        )(self.update_cell)

        self.app.callback(
            Output('roi-pie-holder','children'),
            [Input('slide-map','zoom'),Input('slide-map','viewport')],
            State('slide-map','bounds'),
        )(self.update_roi_pie)      

        self.app.callback(
            [Output('cell-graphic','src'),
             Output('cell-hierarchy','elements'),
             Output('cell-vis-drop','options'),
             Output('cell-graphic-name','children')],
            Input('neph-img','clickData'),
        )(self.update_cell_hierarchy)

        self.app.callback(
            [Output('neph-tooltip','show'),
             Output('neph-tooltip','bbox'),
             Output('neph-tooltip','children')],
             Input('neph-img','hoverData')
        )(self.get_neph_hover)

        self.app.callback(
            Output({'type':'ftu-state-bar','index':MATCH},'figure'),
            Input({'type':'ftu-cell-pie','index':MATCH},'clickData'),
            prevent_initial_call = True
        )(self.update_state_bar)

        """
        # Commenting out hover for now, not really useful
        self.app.callback(
            Output('current-hover','children'),
            Input({'type':'ftu-bounds','index':ALL},'hover_feature'),
            prevent_initial_call=True
        )(self.get_hover)
        """

        self.app.callback(
            Output('mini-label','children'),
            [Input({'type':'ftu-bounds','index':ALL},'click_feature'),
            Input('mini-drop','value')],
            prevent_initial_call=True
        )(self.get_click)

        self.app.callback(
            [Output('slide-tile','url'),
             Output('layer-control','children'),
             Output('slide-map','center'),
             Output('slide-map','bounds'),
             Output('slide-tile','tileSize'),
             Output('feature-group','children'),
             Output('cell-drop','options')],
            Input('slide-select','value'),
            prevent_initial_call=True,
            suppress_callback_exceptions=True
        )(self.ingest_wsi)
        
        self.app.callback(
            [Output('label-p','children'),
            Output('id-p','children'),
            Output('notes-p','children')],
            Input('cell-hierarchy','tapNodeData'),
            prevent_initial_call=True
        )(self.get_cyto_data)

        self.app.callback(
            [Input('ftu-select','value'),
            Input('plot-select','value'),
            Input('label-select','value')],
            [Output('cluster-graph','figure'),
            Output('label-select','options')],
        )(self.update_graph)

        self.app.callback(
            [Input('cluster-graph','clickData'),
            Input('cluster-graph','selectedData')],
            [Output('selected-image','figure'),
            Output('selected-cell-types','figure'),
            Output('selected-cell-states','figure')],
        )(self.update_selected)

        self.app.callback(
            Input('selected-cell-types','clickData'),
            Output('selected-cell-states','figure'),
            prevent_initial_call=True
        )(self.update_selected_state_bar)

        self.app.callback(
            Input({'type':'edit_control','index':ALL},'geojson'),
            [Output('layer-control','children'),
             Output('data-select','options')],
            prevent_initial_call=True
        )(self.add_manual_roi)

        # Callbacks for data download
        self.app.callback(
            Input('data-select','value'),
            Output('data-options','children'),
            prevent_initial_call = True
        )(self.update_download_options)

        self.app.callback(
            [Input({'type':'download-opts','index':MATCH},'value'),
            Input({'type':'download-butt','index':MATCH},'n_clicks')],
            Output({'type':'download-data','index':MATCH},'data'),
            prevent_initial_call = True
        )(self.download_data)

        # Callbacks for CLI application
        self.app.callback(
            [Input('cli-drop','value'),
             Input('cli-run','n_clicks')],
            [Output('cli-descrip','children'),
             Output('cli-run','disabled'),
             Output('cli-results','children')],
             prevent_initial_call = True
        )(self.run_analysis)

    def builder_callbacks(self):

        self.app.callback(
            Input('dataset-table','selected_rows'),
            [Output('selected-dataset-slides','children'),
             Output('slide-metadata-plots','children'),
             Output('slide-select','options')],
             prevent_initial_call=True
        )(self.initialize_metadata_plots)

        self.app.callback(
            [Input({'type':'meta-drop','index':ALL},'value'),
             Input({'type':'cell-meta-drop','index':ALL},'value'),
             Input({'type':'agg-meta-drop','index':ALL},'value'),
             Input({'type':'slide-dataset-table','index':ALL},'selected_rows')],
             [Output('slide-select','options'),
             Output({'type':'meta-plot','index':ALL},'figure'),
             Output({'type':'cell-meta-drop','index':ALL},'disabled'),
             Output({'type':'current-slide-count','index':ALL},'children')],
             prevent_initial_call = True
        )(self.update_metadata_plot)

    def welcome_callbacks(self):

        self.app.callback(
            Input({'type':'video-drop','index':ALL},'value'),
            Output({'type':'video','index':ALL},'src'),
            prevent_initial_call = True
        )(self.get_video)

    def upload_callbacks(self):

        # Creating upload components depending on omics type
        self.app.callback(
            [Input('collect-select','value'),
             Input('new-collect-entry','value'),
             Input('upload-type','value')],
             [Output('upload-requirements','children'),
              Output('new-collect-entry','disabled')],
             prevent_initial_call=True
        )(self.update_upload_requirements)

        # Enabling components after upload is complete

    def get_video(self,tutorial_category):
        tutorial_category = tutorial_category[0]
        print(f'tutorial_category: {tutorial_category}')

        # Pull tutorial videos from DSA 
        if not self.run_type == 'dsa':
            video_src = [f'./assets/videos/{tutorial_category}.mp4']

        return video_src

    def initialize_metadata_plots(self,selected_dataset_list):

        # Extracting metadata from selected datasets and plotting
        all_metadata_labels = []
        all_metadata = []
        slide_dataset_dict = []

        for d in selected_dataset_list:

            # Pulling dataset metadata for non-DSA backend deployment
            if not self.run_type =='dsa':
                d_name = self.dataset_handler.dataset_names[d]
                metadata_available = self.dataset_handler.get_dataset(d_name)['metadata']
                slides_list = self.dataset_handler.get_dataset(d_name)['slide_info']
            else:
                # Dataset name in this case is associated with a folder or collection
                dataset_ids = list(self.dataset_handler.slide_datasets.keys())
                d_name = [self.dataset_handler.slide_datasets[i]['name'] for i in dataset_ids][d]
                d_id = dataset_ids[d]
                metadata_available = self.dataset_handler.slide_datasets[d_id]['Metadata']
                # This will store metadata, id, name, etc. for every slide in that dataset
                slides_list = [i for i in self.dataset_handler.slide_datasets[d_id]['Slides']]
                slide_dataset_dict.extend([{'Slide Names':s['name'],'Dataset':d_name} for s in slides_list])

            if not self.run_type =='dsa':
                for s in slides_list:
                    
                    self.slide_info_dict[s['name']] = s
                    
                    ftu_props = []
                    if 'geojson' in s['metadata_path']:

                        slide_dataset_dict.append({'Slide Names':s['name'],'Dataset':d_name})

                        with open(s['metadata_path']) as f:
                            meta_json = geojson.load(f)

                        for f in meta_json['features']:
                            f['properties']['dataset'] = d_name
                            f['properties']['slide_name'] = s['name']
                            ftu_props.append(f['properties'])

                    all_metadata.extend(ftu_props)

                all_metadata_labels.extend(metadata_available)
            else:
                # Grabbing dataset-level metadata
                metadata_available['FTU Expression Statistics'] = []

                all_metadata.append(metadata_available)
                all_metadata_labels.extend(list(metadata_available.keys()))

        #self.metadata = all_metadata
        all_metadata_labels = np.unique(all_metadata_labels)
        slide_dataset_df = pd.DataFrame.from_records(slide_dataset_dict)
        self.current_slides = []
        for i in slide_dataset_dict:
            i['included'] = True
            self.current_slides.append(i)

        # Different procedure for local vs. DSA-backend 
        if not self.run_type=='dsa':
            cell_type_dropdowns = [
                dbc.Col(dcc.Dropdown(all_metadata_labels,id={'type':'meta-drop','index':0})),
                dbc.Col(dcc.Dropdown([self.cell_names_key[c] for c in list(self.cell_names_key.keys())],id={'type':'cell-meta-drop','index':0},disabled=True))
            ]
        else:
            cell_type_dropdowns = [
                dbc.Col(dcc.Dropdown(all_metadata_labels,id={'type':'meta-drop','index':0}),md=6),
                dbc.Col(dcc.Dropdown(['Main Cell Types','Cell States'],'Main Cell Types',id={'type':'cell-meta-drop','index':0},disabled=True),md=2),
                dbc.Col(dcc.Dropdown(['Mean','Median','Sum','Standard Deviation','Nonzero'],'Mean',id={'type':'cell-meta-drop','index':1},disabled=True),md=2),
                dbc.Col(dcc.Dropdown(list(self.cell_names_key.keys()),list(self.cell_names_key.keys())[0],id={'type':'cell-meta-drop','index':2},disabled=True),md=2)
            ]

        if not all_metadata==[]:
            drop_div = html.Div([
                dash_table.DataTable(
                    id = {'type':'slide-dataset-table','index':0},
                    columns = [{'name':i,'id':i,'deletable':False,'selectable':False} for i in slide_dataset_df],
                    data = slide_dataset_df.to_dict('records'),
                    editable = False,
                    filter_action='native',
                    sort_action = 'native',
                    sort_mode = 'multi',
                    column_selectable = 'single',
                    row_selectable = 'multi',
                    row_deletable = False,
                    selected_columns = [],
                    selected_rows = list(range(0,len(slide_dataset_df))),
                    page_action='native',
                    page_current=0,
                    page_size=5,
                    style_cell = {
                        'overflow':'hidden',
                        'textOverflow':'ellipsis',
                        'maxWidth':0
                    },
                    tooltip_data = [
                        {
                            column: {'value':str(value),'type':'markdown'}
                            for column, value in row.items()
                        } for row in slide_dataset_df.to_dict('records')
                    ],
                    tooltip_duration = None
                ),
                html.Div(id={'type':'current-slide-count','index':0},children=[html.P(f'Included Slide Count: {len(slide_dataset_dict)}')]),
                html.P('Select a Metadata feature for plotting'),
                html.B(),
                dbc.Row(
                    id = {'type':'meta-row','index':0},
                    children = cell_type_dropdowns
                ),
                html.B(),
                html.P('Select whether to separate by dataset or slide'),
                dbc.Row(
                    id = {'type':'lower-row','index':0},
                    children = [
                        dbc.Col(dcc.Dropdown(['By Dataset','By Slide'],'By Dataset',id={'type':'agg-meta-drop','index':0}))
                    ]
                )
            ])

            self.selected_meta_df = pd.DataFrame.from_records(all_metadata)

            plot_div = html.Div([
                dcc.Graph(id = {'type':'meta-plot','index':0},figure = go.Figure())
            ])

        else:
            drop_div = html.Div()
            plot_div = html.Div()

        slide_rows = [list(range(len(slide_dataset_dict)))]
        current_slide_count, slide_select_options = self.update_current_slides(slide_rows)

        return drop_div, plot_div, slide_select_options

    def update_metadata_plot(self,new_meta,sub_meta,group_type,slide_rows):
        
        if not self.run_type == 'dsa':
            cell_types_turn_off = True
        else:
            if not new_meta == ['FTU Expression Statistics']:
                cell_types_turn_off = (True,True,True)
            else:
                cell_types_turn_off = (False, False, False)


        current_slide_count, slide_select_options = self.update_current_slides(slide_rows)

        print(f'new_meta: {new_meta}')
        if type(new_meta)==list:
            new_meta = new_meta[0]

        if not self.run_type=='dsa':
            if type(sub_meta) == list:
                sub_meta = sub_meta[0]
        
        if type(group_type)==list:
            group_type = group_type[0]


        if not self.run_type == 'dsa':

            if not new_meta is None:
                if group_type == 'By Dataset':
                    group_bar = 'dataset'   
                elif group_type == 'By Slide':
                    group_bar = 'slide_name'
                else:
                    group_bar = 'dataset'

                plot_data = self.selected_meta_df.dropna(subset=[new_meta]).convert_dtypes()

                # Filtering out de-selected slides
                present_slides = [s['Slide Names'] for s in self.current_slides]
                included_slides = [t for t in present_slides if self.current_slides[present_slides.index(t)]['included']]
                plot_data = plot_data[plot_data['slide_name'].isin(included_slides)]

                if plot_data[new_meta].dtypes == float:
                    # Making violin plots 
                    fig = go.Figure(px.violin(plot_data[plot_data[new_meta]>0],x=group_bar,y=new_meta))
                
                elif plot_data[new_meta].dtypes == object:

                    if new_meta == 'Main_Cell_Types':

                        all_cell_types_df = pd.DataFrame.from_records(plot_data[new_meta].tolist())
                        cell_types_present = all_cell_types_df.columns.to_list()
                        cell_types_turn_off = False
                        
                        all_cell_types_df[group_bar] = plot_data[group_bar].tolist()

                        if not sub_meta is None:
                            fig = go.Figure(px.violin(all_cell_types_df,x=group_bar,y=sub_meta))
                        else:
                            fig = go.Figure()

                    if new_meta == 'Cell_States':
                        all_cell_states_df = pd.DataFrame.from_records(plot_data[new_meta].tolist())

                        cell_types_present = all_cell_states_df.columns.to_list()
                        cell_types_turn_off = False

                        if not sub_meta is None:
                            specific_cell_states_df = pd.DataFrame.from_records(all_cell_states_df[sub_meta].tolist())

                            groups_present = plot_data[group_bar].unique()
                            count_df = pd.DataFrame()
                            for g in groups_present:
                                g_df = specific_cell_states_df[plot_data[group_bar]==g]
                                g_counts = g_df.sum(axis=0).to_frame()
                                g_counts[group_bar] = [g]*g_counts.shape[0]

                                if count_df.empty:
                                    count_df = g_counts
                                else:
                                    count_df = pd.concat([count_df,g_counts],axis=0,ignore_index=False)

                            count_df = count_df.reset_index()
                            count_df.columns = ['Cell State','Abundance',group_bar]

                            fig = go.Figure(px.bar(count_df,x=group_bar,y='Abundance',color='Cell State'))
                        else:
                            fig = go.Figure()

                else:
                    # Finding counts of each unique value present
                    groups_present = plot_data[group_bar].unique()
                    count_df = pd.DataFrame()
                    for g in groups_present:
                        g_df = plot_data[plot_data[group_bar]==g]
                        g_counts = g_df[new_meta].value_counts().to_frame()
                        g_counts[group_bar] = [g]*g_counts.shape[0]

                        if count_df.empty:
                            count_df = g_counts
                        else:
                            count_df = pd.concat([count_df,g_counts],axis=0,ignore_index=False)

                    count_df = count_df.reset_index()
                    count_df.columns = [new_meta,'counts',group_bar]

                    fig = go.Figure(px.bar(count_df,x = group_bar, y = 'counts',color=new_meta))

                return [slide_select_options, [fig], [cell_types_turn_off], [current_slide_count]]
            else:
                return [slide_select_options, [go.Figure()], [cell_types_turn_off], [current_slide_count]]

        else:
            # For DSA-backend deployment
            if not new_meta is None:
                # Filtering out de-selected slides
                present_slides = [s['Slide Names'] for s in self.current_slides]
                included_slides = [t for t in present_slides if self.current_slides[present_slides.index(t)]['included']]                    
                #print(f'included_slides: {included_slides}')

                dataset_metadata = []
                for d_id in self.dataset_handler.slide_datasets:
                    slide_data = self.dataset_handler.slide_datasets[d_id]['Slides']
                    dataset_name = self.dataset_handler.slide_datasets[d_id]['name']
                    d_include = [s for s in slide_data if s['name'] in included_slides]

                    if len(d_include)>0:
                        
                        # Adding new_meta value to dataset dictionary
                        if not new_meta=='FTU Expression Statistics':
                            d_dict = {'Dataset':[],'Slide Name':[],new_meta:[]}

                            for s in d_include:
                                if new_meta in s['meta']:
                                    d_dict['Dataset'].append(dataset_name)
                                    d_dict['Slide Name'].append(s['name'])
                                    d_dict[new_meta].append(s['meta'][new_meta])
                        
                        else:
                            # Whether it's Mean, Median, Sum, Standard Deviation, or Nonzero
                            #print(f'sub_meta: {sub_meta}')
                            ftu_expression_feature = sub_meta[1]+' '
                            # Whether it's Main Cell Types or Cell States
                            ftu_expression_feature+=sub_meta[0]
                            # Getting cell type abbreviation from full name
                            cell_type = self.cell_names_key[sub_meta[2]]

                            # Getting FTU specific expression values
                            d_dict = {'Dataset':[],'Slide Name':[],'FTU':[],new_meta:[]}
                            if sub_meta[0]=='Cell States':
                                d_dict['State'] = []

                            for d_i in d_include:
                                slide_meta = d_i['meta']
                                slide_name = d_i['name']
                                ftu_expressions = [i for i in list(slide_meta.keys()) if 'Expression' in i]
                                for f in ftu_expressions:
                                    expression_stats = slide_meta[f][ftu_expression_feature]
                                    for ct in list(expression_stats.keys()):
                                        if cell_type in ct:
                                            d_dict[new_meta].append(expression_stats[ct])
                                            d_dict['FTU'].append(f.replace(' Expression Statistics',''))
                                            d_dict['Dataset'].append(dataset_name)
                                            d_dict['Slide Name'].append(slide_name)
                                            if sub_meta[0]=='Cell States':
                                                d_dict['State'].append(ct.split('_')[-1])



                        dataset_metadata.append(d_dict)
                        #print(f'd_dict: {d_dict}')

                # Converting to dataframe
                plot_data = pd.concat([pd.DataFrame.from_dict(i) for i in dataset_metadata],ignore_index=True)
                plot_data = plot_data.dropna(subset=[new_meta]).convert_dtypes()
                
                # Assigning grouping variable
                if group_type=='By Dataset':
                    group_bar = 'Dataset'
                elif group_type=='By Slide':
                    group_bar = 'Slide Name'

                # Checking if new_meta is a number or a string
                # This is dumb, c'mon pandas
                if plot_data[new_meta].dtype.kind in 'biufc':
                    if group_bar == 'Dataset':
                        print('Generating violin plot')
                        if not new_meta == 'FTU Expression Statistics':
                            fig = go.Figure(px.violin(plot_data,x=group_bar,y=new_meta,hover_data=['Slide Name']))
                        else:
                            if sub_meta[0]=='Main Cell Types':
                                fig = go.Figure(px.violin(plot_data,x=group_bar,y=new_meta,hover_data=['Slide Name'],color='FTU'))
                            else:
                                fig = go.Figure(px.violin(plot_data,x=group_bar,y=new_meta,hover_data=['Slide Name','State'],color='FTU'))
                    else:
                        print('Generating bar plot')
                        if not new_meta=='FTU Expression Statistics':
                            fig = go.Figure(px.bar(plot_data,x=group_bar,y=new_meta,hover_data = ['Dataset']))
                        else:
                            if sub_meta[0]=='Main Cell Types':
                                fig = go.Figure(px.bar(plot_data,x=group_bar,y=new_meta,hover_data=['Dataset'],color='FTU'))
                            else:
                                fig = go.Figure(px.bar(plot_data,x=group_bar,y=new_meta,hover_data=['Dataset','State'],color='FTU'))
                else:
                    print(f'Generating bar plot')
                    groups_present = plot_data[group_bar].unique()
                    count_df = pd.DataFrame()
                    for g in groups_present:
                        g_df = plot_data[plot_data[group_bar]==g]
                        g_counts = g_df[new_meta].value_counts().to_frame()
                        g_counts[group_bar] = [g]*g_counts.shape[0]

                        if count_df.empty:
                            count_df = g_counts
                        else:
                            count_df = pd.concat([count_df,g_counts],axis=0,ignore_index=False)
                    
                    count_df = count_df.reset_index()
                    count_df.columns = [new_meta, 'counts', group_bar]

                    fig = go.Figure(px.bar(count_df, x = group_bar, y = 'counts', color = new_meta))

                return [slide_select_options, [fig], cell_types_turn_off,[current_slide_count]]
            else:
                raise exceptions.PreventUpdate

    def update_current_slides(self,slide_rows):

        # Updating the current slides
        slide_options = []
        if len(slide_rows)>0:
            #slide_rows = slide_rows[0]
            if type(slide_rows[0])==list:
                slide_rows = slide_rows[0]
            for s in range(0,len(self.current_slides)):
                if s in slide_rows:
                    self.current_slides[s]['included'] = True
                else:
                    self.current_slides[s]['included'] = False
                    
            slide_options = [{'label':i['Slide Names'],'value':i['Slide Names']} for i in self.current_slides if i['included']]

        if slide_options == []:
            slide_options = [{'label':'blah','value':'blah'}]

        return [html.P(f'Included Slide Count: {len(slide_rows)}')], slide_options

    def update_roi_pie(self,zoom,viewport,bounds):
        
        #print(f'bounds:{bounds}')
        # Making a box-poly from the bounds
        if len(bounds)==2:
            bounds_box = shapely.geometry.box(bounds[0][1],bounds[0][0],bounds[1][1],bounds[1][0])
        else:
            bounds_box = shapely.geometry.box(*bounds)

        # Getting a dictionary containing all the intersecting spots with this current ROI
        intersecting_ftus = {}
        if 'Spots' in self.current_ftu_layers:
            intersecting_spots = self.wsi.find_intersecting_spots(bounds_box)
            intersecting_ftus['Spots'] = intersecting_spots

        for ftu in self.current_ftu_layers:
            if not ftu=='Spots':
                intersecting_ftus[ftu] = self.wsi.find_intersecting_ftu(bounds_box,ftu)

        for m_idx,m_ftu in enumerate(self.wsi.manual_rois):
            if not self.run_type=='dsa':
                intersecting_ftus[f'Manual ROI: {m_idx}'] = {
                    'polys':[shape(m_ftu['geojson']['features'][0]['geometry'])],
                    'main_counts':[m_ftu['geojson']['features'][0]['properties']['Main_Cell_Types']],
                    'states':[m_ftu['geojson']['features'][0]['properties']['Cell_States']]
                }
            else:
                intersecting_ftus[f'Manual ROI: {m_idx}'] = [m_ftu['geojson']['features'][0]['properties']]
                
        self.current_ftus = intersecting_ftus
        # Now we have main cell types, cell states, by ftu

        included_ftus = list(intersecting_ftus.keys())
        #print(f'included_ftus: {included_ftus}')
        if not self.run_type == 'dsa':
            included_ftus = [i for i in included_ftus if len(intersecting_ftus[i]['polys'])>0]
        else:
            included_ftus = [i for i in included_ftus if len(intersecting_ftus[i])>0]


        if len(included_ftus)>0:

            tab_list = []
            #counts_data = pd.DataFrame()
            for f_idx,f in enumerate(included_ftus):
                counts_data = pd.DataFrame()
                if not self.run_type == 'dsa':
                    counts_data = pd.DataFrame(intersecting_ftus[f]['main_counts']).sum(axis=0).to_frame()
                else:
                    counts_dict_list = [i['Main_Cell_Types'] for i in intersecting_ftus[f] if 'Main_Cell_Types' in i]
                    if len(counts_dict_list)>0:
                        counts_data = pd.DataFrame.from_records(counts_dict_list).sum(axis=0).to_frame()

                if not counts_data.empty:
                    counts_data.columns = [f]

                    # Normalizing to sum to 1
                    counts_data[f] = counts_data[f]/counts_data[f].sum()
                    # Only getting top n
                    counts_data = counts_data.sort_values(by=f,ascending=False).iloc[0:self.plot_cell_types_n,:]
                    counts_data = counts_data.reset_index()

                    f_pie = px.pie(counts_data,values=f,names='index')

                    top_cell = counts_data['index'].tolist()[0]
                    if not self.run_type == 'dsa':
                        pct_states = pd.DataFrame([i[top_cell] for i in intersecting_ftus[f]['states']]).sum(axis=0).to_frame()
                    else:
                        pct_states = pd.DataFrame.from_records([i['Cell_States'][top_cell] for i in intersecting_ftus[f]if 'Cell_States' in i]).sum(axis=0).to_frame()
                    
                    pct_states = pct_states.reset_index()
                    pct_states.columns = ['Cell State','Proportion']
                    pct_states['Proportion'] = pct_states['Proportion']/pct_states['Proportion'].sum()

                    state_bar = px.bar(pct_states,x='Cell State',y = 'Proportion', title = f'Cell State Proportions for:<br><sup>{self.cell_graphics_key[top_cell]["full"]} in:</sup><br><sup>{f}</sup>')

                    f_tab = dbc.Tab(
                        dbc.Row([
                            dbc.Col([
                                dbc.Label(f'{f} Cell Type Proportions'),
                                dcc.Graph(
                                    id = {'type':'ftu-cell-pie','index':f_idx},
                                    figure = go.Figure(f_pie)
                                )
                            ],md=6),
                            dbc.Col([
                                dbc.Label(f'{f} Cell State Proportions'),
                                dcc.Graph(
                                    id = {'type':'ftu-state-bar','index':f_idx},
                                    figure = go.Figure(state_bar)
                                )
                            ],md=6)
                        ]),label = f,tab_id = f'tab_{f_idx}'
                    )

                    tab_list.append(f_tab)

            return dbc.Tabs(tab_list,active_tab = 'tab_0')
        else:

            return html.P('No FTUs in current view')

    def update_state_bar(self,cell_click):
        
        if not cell_click is None:
            self.pie_cell = cell_click['points'][0]['label']

            self.pie_ftu = list(self.current_ftus.keys())[ctx.triggered_id['index']]
            if not self.run_type == 'dsa':
                pct_states = pd.DataFrame([i[self.pie_cell] for i in self.current_ftus[self.pie_ftu]['states']]).sum(axis=0).to_frame()
            else:
                pct_states = pd.DataFrame.from_records([i['Cell_States'][self.pie_cell] for i in self.current_ftus[self.pie_ftu]]).sum(axis=0).to_frame()
    
            pct_states = pct_states.reset_index()
            pct_states.columns = ['Cell State', 'Proportion']
            pct_states['Proportion'] = pct_states['Proportion']/pct_states['Proportion'].sum()

            state_bar = go.Figure(px.bar(pct_states,x='Cell State', y = 'Proportion', title = f'Cell State Proportions for:<br><sup>{self.cell_graphics_key[self.pie_cell]["full"]} in:</sup><br><sup>{self.pie_ftu}</sup>'))

            return state_bar
        else:
            return go.Figure()
    
    def update_hex_color_key(self,color_type):
        
        # Iterate through all structures (and spots) in current wsi,
        # concatenate all of their proportions of specific cell types together
        # scale it with self.color_map (make sure to multiply by 255 after)
        # convert uint8 RGB colors to hex
        # create look-up table for original value --> hex color
        # add that as get_color() function in style dict (fillColor) along with fillOpacity
        raw_values_list = []
        if color_type == 'cell_value':
            # iterating through current ftus
            if not self.run_type=='dsa':
                for f in self.wsi.ftus:
                    for g in self.wsi.ftus[f]:
                        # get main counts for this ftu
                        ftu_counts = pd.DataFrame(g['main_counts'])[self.current_cell].tolist()
                        #raw_values_list.extend([float('{:.4f}'.format(round(i,4))) for i in ftu_counts])
                        raw_values_list.extend(ftu_counts)

                for g in self.wsi.spots:
                    spot_counts = pd.DataFrame(g['main_counts'])[self.current_cell].tolist()
                    raw_values_list.extend(spot_counts)

            else:
                for f in self.wsi.ftu_props:
                    for g in self.wsi.ftu_props[f]:
                        # Getting main counts for this ftu
                        if 'Main_Cell_Types' in g:
                            ftu_counts = g['Main_Cell_Types'][self.current_cell]
                            raw_values_list.append(ftu_counts)

                for f in self.wsi.spot_props:
                    # Getting main counts for spots
                    spot_counts =f['Main_Cell_Types'][self.current_cell]
                    raw_values_list.append(spot_counts)

            for f in self.wsi.manual_rois:
                manual_counts = f['geojson']['features'][0]['properties']['Main_Cell_Types'][self.current_cell]
                raw_values_list.append(manual_counts)

        elif color_type == 'max_cell':

            if not self.run_type=='dsa':
                # iterating through current ftus
                for f in self.wsi.ftus:
                    for g in self.wsi.ftus[f]:
                        all_cell_type_counts = [np.argmax(list(i.values())) for i in g['main_counts']]
                        raw_values_list.extend([float(i) for i in np.unique(all_cell_type_counts)])
            else:
                # Iterating through current ftus
                for f in self.wsi.ftu_props:
                    for g in self.wsi.ftu_props[f]:
                        if 'Main_Cell_Types' in g:
                            all_cell_type_counts = float(np.argmax(list(g['Main_Cell_Types'].values())))
                            raw_values_list.append(all_cell_type_counts)

                #TODO: For slides without any ftu's, search through the spots for cell names
                # Also search through manual ROIs
                for s in self.wsi.spot_props:
                    for g in self.wsi.spot_props[s]:
                        if 'Main_Cell_Types' in g:
                            all_cell_type_counts = float(np.argmax(list(g['Main_Cell_Types'].values())))
                            raw_values_list.append(all_cell_type_counts)

        elif color_type == 'cluster':
            # iterating through current ftus
            if not self.run_type=='dsa':
                for f in self.wsi.ftus:
                    for g in self.wsi.ftus[f]:
                        if 'Cluster' in g:
                            cluster_label = g['Cluster']
                            raw_values_list.extend([float(i) for i in cluster_label])
            else:
                for f in self.wsi.ftu_props:
                    for g in self.wsi.ftu_props[f]:
                        if 'Cluster' in g:
                            cluster_label = g['Cluster']
                            raw_values_list.append(cluster_label)
        else:
            # For specific morphometrics
            if not self.run_type=='dsa':
                for f in self.wsi.ftus:
                    for g in self.wsi.ftus[f]:
                        if color_type in g:
                            morpho_value = g[color_type]
                            raw_values_list.extend([float(i) for i in morpho_value if float(i)>0])
            else:
                for f in self.wsi.ftu_props:
                    for g in self.wsi.ftu_props[f]:
                        if color_type in g:
                            morpho_value = g[color_type]
                            raw_values_list.append(morpho_value)

        raw_values_list = np.unique(raw_values_list)
        
        # Converting to RGB
        if max(raw_values_list)<=1:
            rgb_values = np.uint8(255*self.color_map(np.uint8(255*raw_values_list)))[:,0:3]
        else:
            scaled_values = [(i-min(raw_values_list))/max(raw_values_list) for i in raw_values_list]
            rgb_values = np.uint8(255*self.color_map(scaled_values))[:,0:3]

        hex_list = []
        for row in range(rgb_values.shape[0]):
            hex_list.append('#'+"%02x%02x%02x" % (rgb_values[row,0],rgb_values[row,1],rgb_values[row,2]))

        self.hex_color_key = {i:j for i,j in zip(raw_values_list,hex_list)}
        #print(f'hex color key: {self.hex_color_key}')

    def update_cell(self,cell_val,vis_val):
        
        # Update for sub-properties
        m_prop = None
        if '-->' in cell_val:
            cell_val_parts = cell_val.split(' --> ')
            m_prop = cell_val_parts[0]
            cell_val = cell_val_parts[1]

        if not cell_val is None:
            # Updating current cell prop
            if cell_val in self.cell_names_key:
                if m_prop == 'Main_Cell_Types':
                    self.current_cell = self.cell_names_key[cell_val]
                    self.update_hex_color_key('cell_value')

                    color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')
                elif m_prop == 'Cell_States':
                    self.current_cell = self.cell_names_key[cell_val]
                    self.update_hex_color_key('cell_state')

                    color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')

            elif cell_val == 'Max Cell Type':
                self.update_hex_color_key('max_cell')
                self.current_cell = 'max'

                cell_types = list(self.wsi.geojson_ftus['features'][0]['properties']['Main_Cell_Types'].keys())
                color_bar = dlx.categorical_colorbar(categories = cell_types, colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')
            
            elif cell_val == 'Morphometrics Clusters':
                self.update_hex_color_key('cluster')
                self.current_cell = 'cluster'

                color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')
            
            else:
                # Used for morphometrics values
                self.current_cell = cell_val
                self.update_hex_color_key(cell_val)

                color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')

            self.cell_vis_val = vis_val/100

            vis_val = vis_val/100

            # More print statements:
            """
            print(f'Slide name in update_cell: {self.wsi.slide_name}')
            print(f'Image URL in update_cell: {self.wsi.image_url}')
            print(f'FTU path in update_cell: {self.wsi.ftu_path}')
            """
            # Changing fill and fill-opacity properties for structures and adding that as a property

            if  not self.run_type == 'dsa':
                map_dict = {
                    'url':self.wsi.image_url,
                    'FTUs':{
                        struct : {
                            'geojson':{'type':'FeatureCollection','features':[i for i in self.wsi.geojson_ftus['features'] if i['properties']['structure']==struct]},
                            'id':{'type':'ftu-bounds','index':self.wsi.ftu_names.index(struct)},
                            'color':'',
                            'hover_color':''
                        }
                        for struct in self.wsi.ftu_names
                    }
                }

                spot_dict = {
                    'geojson':self.wsi.geojson_spots,
                    'id': {'type':'ftu-bounds','index':len(self.wsi.ftu_names)},
                    'color': '#dffa00',
                    'hover_color':'#9caf00'
                }

                new_children = [
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = map_dict['FTUs'][struct]['geojson'], id = map_dict['FTUs'][struct]['id'], options = dict(style=self.ftu_style_handle),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = vis_val, ftu_colors=self.ftu_colors),
                                    hoverStyle = arrow_function(dict(weight=5, color = map_dict['FTUs'][struct]['hover_color'], dashArray = '')))),
                            name = struct, checked = True, id = self.wsi.slide_info_dict['key_name']+'_'+struct)
                    for struct in map_dict['FTUs']
                    ]
                
                new_children += [
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = spot_dict['geojson'], id = spot_dict['id'], options = dict(style = self.ftu_style_handle),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = vis_val, ftu_colors= self.ftu_colors),
                                    hoverStyle = arrow_function(dict(weight=5, color = spot_dict['hover_color'], dashArray = '')))),
                            name = 'Spots', checked = False, id = self.wsi.slide_info_dict['key_name']+'_Spots')
                    ]
                
                for m_idx,man in enumerate(self.wsi.manual_rois):
                    new_children.append(
                        dl.Overlay(
                            dl.LayerGroup(
                                dl.GeoJSON(data = man['geojson'], id = man['id'], options = dict(style = self.ftu_style_handle),
                                        hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = vis_val, ftu_colors = self.ftu_colors),
                                        hoverStyle = arrow_function(dict(weight=5, color = man['hover_color'], dashArray = '')))),
                                name = f'Manual ROI {m_idx}', checked = True, id = self.wsi.slide_info_dict['key_name']+f'_manual_roi{m_idx}'
                            )
                        )
                
            else:

                map_dict = {
                    'url':self.wsi.tile_url,
                    'FTUs':{
                        struct: {
                            'geojson':{'type':'FeatureCollection','features':[i for i in self.wsi.geojson_ftus['features'] if i['properties']['name']==struct]},
                            'id': {'type':'ftu-bounds','index':self.wsi.ftu_names.index(struct)},
                            'color':'',
                            'hover_color':''
                        }
                        for struct in self.wsi.ftu_names
                    }
                }

                spot_dict = {
                    'geojson':self.wsi.geojson_spots,
                    'id':{'type':'ftu-bounds','index':len(self.wsi.ftu_names)},
                    'color':'#dffa00',
                    'hover_color':'#9caf00'
                }

                new_children = [
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = map_dict['FTUs'][struct]['geojson'], id = map_dict['FTUs'][struct]['id'], options = dict(style=self.ftu_style_handle),
                                       hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity=vis_val, ftu_colors = self.ftu_colors),
                                       hoverStyle = arrow_function(dict(weight=5, color = map_dict['FTUs'][struct]['hover_color'],dashArray='')))
                        ), name = struct, checked = True, id = self.wsi.item_id+'_'+struct
                    )
                    for struct in map_dict['FTUs']
                ]
                new_children += [
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = spot_dict['geojson'], id = spot_dict['id'], options = dict(style = self.ftu_style_handle),
                                       hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = vis_val, ftu_colors = self.ftu_colors),
                                       hoverStyle = arrow_function(dict(weight=5,color=spot_dict['hover_color'],dashArray='')))
                        ),name = 'Spots', checked = False, id = self.wsi.item_id+'_Spots'
                    )
                ]

                for m_idx,man in enumerate(self.wsi.manual_rois):
                    new_children.append(
                        dl.Overlay(
                            dl.LayerGroup(
                                dl.GeoJSON(data = man['geojson'], id = man['id'], options = dict(style = self.ftu_style_handle),
                                           hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = vis_val, ftu_colors = self.ftu_colors),
                                           hoverStyle = arrow_function(dict(weight=5,color=man['hover_color'],dashArray='')))
                            ), name = f'Manual ROI {m_idx}', checked = True, id = self.wsi.item_id+f'_manual_roi{m_idx}'
                        )
                    )


            self.current_overlays = new_children
                        
            return new_children, color_bar

    def update_cell_hierarchy(self,cell_clickData):
        # Loading the cell-graphic and hierarchy image
        cell_graphic = './assets/cell_graphics/default_cell_graphic.png'
        cell_hierarchy = [
                        {'data': {'id': 'one', 'label': 'Node 1'}, 'position': {'x': 75, 'y': 75}},
                        {'data': {'id': 'two', 'label': 'Node 2'}, 'position': {'x': 200, 'y': 200}},
                        {'data': {'source': 'one', 'target': 'two'}}
                    ]
        cell_state_droptions = []
        cell_name = html.P('Default Cell')

        # Getting cell_val from the clicked location in the nephron diagram
        if not cell_clickData is None:
            
            pt = cell_clickData['points'][0]
            click_point = Point(pt['x'],pt['y'])

            # Checking if the clicked point is inside any of the cells bounding boxes
            possible_cells = [i for i in self.cell_graphics_key if len(self.cell_graphics_key[i]['bbox'])>0]
            intersecting_cell = [i for i in possible_cells if click_point.intersects(box(*self.cell_graphics_key[i]['bbox']))]
            
            if len(intersecting_cell)>0:
                cell_val = self.cell_graphics_key[intersecting_cell[0]]['full']
                cell_name = html.P(cell_name)
                if self.cell_names_key[cell_val] in self.cell_graphics_key:
                    cell_graphic = self.cell_graphics_key[self.cell_names_key[cell_val]]['graphic']
                    cell_hierarchy = self.gen_cyto(self.cell_names_key[cell_val])
                    cell_state_droptions = np.unique(self.cell_graphics_key[self.cell_names_key[cell_val]]['states'])

        return cell_graphic, cell_hierarchy, cell_state_droptions, cell_name

    def get_neph_hover(self,neph_hover):

        if neph_hover is None:
            return False, no_update, no_update
        
        pt = neph_hover['points'][0]
        tool_bbox = pt['bbox']

        hover_point = Point(pt['x'],pt['y'])

        possible_cells = [i for i in self.cell_graphics_key if len(self.cell_graphics_key[i]['bbox'])>0]
        intersecting_cell = [i for i in possible_cells if hover_point.intersects(box(*self.cell_graphics_key[i]['bbox']))]
        if len(intersecting_cell)>0:
            cell_name = self.cell_graphics_key[intersecting_cell[0]]['full']
            tool_children = [
                html.Div([
                    cell_name
                ])
            ]
        else:
            tool_children = []


        return True, tool_bbox, tool_children

    """
    def get_hover(self,ftu_hover):
        
        hover_text = []

        if self.current_cell in self.cell_graphics_key:
            for f in ftu_hover:
                if f is not None:
                    hover_text.append(f'{self.current_cell}:{round(f["properties"]["Main_Cell_Types"][self.current_cell],3)}')
        else:
            print(f'current_cell: {self.current_cell} not in cell_names_key')        

        return hover_text
    """
    def get_click(self,ftu_click,mini_specs):
        
        print(f'Number of ftu_bounds to get_click: {len(ftu_click)}')
        if not mini_specs == 'None' and len(ftu_click)>0:
            click_data = [i for i in ftu_click if i is not None]
            
            if len(click_data)>0:
                mini_chart_list = []
                for c in click_data:
                    chart_coords = np.mean(np.squeeze(c['geometry']['coordinates']),axis=0)
                    chart_dict_data = c['properties']

                    if mini_specs == 'All Main Cell Types':
                        chart_dict_data = chart_dict_data['Main_Cell_Types']
                    elif mini_specs == 'Cell States for Current Cell Type':
                        chart_dict_data = chart_dict_data['Cell_States'][self.current_cell]
                    else:
                        # For clusters and morphometrics just show main cell types
                        chart_dict_data = chart_dict_data['Main_Cell_Types']
                    

                    chart_labels = list(chart_dict_data.keys())
                    chart_data = [chart_dict_data[j] for j in chart_labels]

                    if not all([i==j for i,j in zip(chart_coords,self.current_chart_coords)]):
                        self.current_chart_coords = chart_coords

                        mini_pie_chart = dl.Minichart(
                            data = chart_data, 
                            labels = chart_labels, 
                            lat=chart_coords[1],
                            lon=chart_coords[0],
                            height=100,
                            width=100,
                            labelMinSize=2,
                            type='pie',id=f'pie_click{random.randint(0,1000)}')

                        mini_chart_list.append(mini_pie_chart)

                return mini_chart_list
            else:
                #return []
                raise exceptions.PreventUpdate
        else:
            #return []
            raise exceptions.PreventUpdate
                    
    def gen_cyto(self,cell_val):

        cyto_elements = []

        # Getting cell sub-types under that main cell
        cell_subtypes = self.cell_graphics_key[cell_val]['subtypes']

        # Getting all the rows that contain these sub-types
        table_data = self.table_df.dropna(subset = ['CT/1/ABBR'])
        cell_data = table_data[table_data['CT/1/ABBR'].isin(cell_subtypes)]

        # cell type
        cyto_elements.append(
            {'data':{'id':'Main_Cell',
                     'label':cell_val,
                     'url':'./assets/cell.png'},
            'classes': 'CT',
            'position':{'x':self.node_cols['Cell Types']['x_start'],'y':self.node_cols['Cell Types']['y_start']},
                     }
        )

        # Getting the anatomical structures for this cell type
        an_structs = cell_data.filter(regex=self.node_cols['Anatomical Structure']['abbrev']).dropna(axis=1)

        an_start_y = self.node_cols['Anatomical Structure']['y_start']
        col_vals = an_structs.columns.values.tolist()
        col_vals = [i for i in col_vals if 'LABEL' in i]

        for idx,col in enumerate(col_vals):
            cyto_elements.append(
                {'data':{'id':col,
                         'label':an_structs[col].tolist()[0],
                         'url':'./assets/kidney.png'},
                'classes':'AS',
                'position':{'x':self.node_cols['Anatomical Structure']['x_start'],'y':an_start_y}
                         }
            )
            
            if idx>0:
                cyto_elements.append(
                    {'data':{'source':col_vals[idx-1],'target':col}}
                )
            an_start_y+=75
        
        last_struct = col
        cyto_elements.append(
            {'data':{'source':last_struct,'target':'Main_Cell'}}
        )
        
        cell_start_y = self.node_cols['Cell Types']['y_start']
        gene_start_y = self.node_cols['Genes']['y_start']
        for idx_1,c in enumerate(cell_subtypes):

            matching_rows = table_data[table_data['CT/1/ABBR'].str.match(c)]

            if not matching_rows.empty:
                cell_start_y+=75

                cyto_elements.append(
                    {'data':{'id':f'ST_{idx_1}',
                             'label':c,
                             'url':'./assets/cell.png'},
                    'classes':'CT',
                    'position':{'x':self.node_cols['Cell Types']['x_start'],'y':cell_start_y}}
                )
                cyto_elements.append(
                    {'data':{'source':'Main_Cell','target':f'ST_{idx_1}'}}
                )

                # Getting genes
                genes = matching_rows.filter(regex=self.node_cols['Genes']['abbrev']).dropna(axis=1)
                col_vals = genes.columns.values.tolist()
                col_vals = [i for i in col_vals if 'LABEL' in i]

                for idx,col in enumerate(col_vals):
                    cyto_elements.append(
                        {'data':{'id':col,
                                 'label':genes[col].tolist()[0],
                                 'url':'./assets/gene.png'},
                        'classes':'G',
                        'position':{'x':self.node_cols['Genes']['x_start'],'y':gene_start_y}}
                    )

                    cyto_elements.append(
                        {'data':{'source':col,'target':f'ST_{idx_1}'}}
                    )
                    gene_start_y+=75

        return cyto_elements

    def get_cyto_data(self,clicked):

        if not clicked is None:
            if 'ST' in clicked['id']:
                table_data = self.table_df.dropna(subset=['CT/1/ABBR'])
                table_data = table_data[table_data['CT/1/ABBR'].str.match(clicked['label'])]

                label = clicked['label']
                try:
                    id = table_data['CT/1/ID'].tolist()[0]
                    # Modifying base url to make this link to UBERON
                    base_url = self.node_cols['Cell Types']['base_url']
                    new_url = base_url+id.replace('CL:','')

                except IndexError:
                    print(table_data['CT/1/ID'].tolist())
                    id = ''
                
                try:
                    notes = table_data['CT/1/NOTES'].tolist()[0]
                except:
                    print(table_data['CT/1/NOTES'])
                    notes = ''

            elif 'Main_Cell' not in clicked['id']:
                
                table_data = self.table_df.dropna(subset=[clicked['id']])
                table_data = table_data[table_data[clicked['id']].str.match(clicked['label'])]

                base_label = '/'.join(clicked['id'].split('/')[0:-1])
                label = table_data[base_label+'/LABEL'].tolist()[0]

                id = table_data[base_label+'/ID'].tolist()[0]
                
                if self.node_cols['Anatomical Structure']['abbrev'] in clicked['id']:
                    base_url = self.node_cols['Anatomical Structure']['base_url']

                    new_url = base_url+id.replace('UBERON:','')
                else:
                    base_url = self.node_cols['Genes']['base_url']

                    new_url = base_url+id.replace('HGNC:','')

                try:
                    notes = table_data[base_label+'/NOTES'].tolist()[0]
                except KeyError:
                    notes = ''

            else:
                label = ''
                id = ''
                notes = ''
                new_url = ''
        else:
            label = ''
            id = ''
            notes = ''
            new_url = ''

        return f'Label: {label}', dcc.Link(f'ID: {id}', href = new_url), f'Notes: {notes}'
    
    def ingest_wsi(self,slide_name):

        print(f'Slide selected: {slide_name}')
        if not self.run_type == 'dsa':
            new_slide_key_name = self.slide_info_dict[slide_name]['key_name']
            old_slide_key_name = self.wsi.slide_info_dict['key_name']

            new_dataset_key_name = self.dataset_handler.get_dataset(self.dataset_handler.get_slide_dataset(slide_name))['key_name']
            old_dataset_key_name = self.dataset_handler.get_dataset(self.dataset_handler.get_slide_dataset(self.wsi.slide_name))['key_name']

            new_url = self.wsi.image_url.replace(old_slide_key_name,new_slide_key_name).replace(old_dataset_key_name,new_dataset_key_name)

            old_ftu_path = self.wsi.ftu_path
            old_spot_path = self.wsi.spot_path

            new_ext = slide_name.split('.')[-1]

            new_ftu_path = old_ftu_path.replace(self.wsi.slide_name.replace('.'+self.wsi.slide_ext,''),slide_name.replace('.'+new_ext,''))
            new_spot_path = old_spot_path.replace(self.wsi.slide_name.replace('.'+self.wsi.slide_ext,''),slide_name.replace('.'+new_ext,''))
            new_slide = WholeSlide(new_url,slide_name,self.slide_info_dict[slide_name],new_ftu_path,new_spot_path)

            map_dict = {
                'url':self.wsi.image_url,
                'FTUs':{
                    struct : {
                        'geojson':{'type':'FeatureCollection','features':[i for i in self.wsi.geojson_ftus['features'] if i['properties']['structure']==struct]},
                        'id':{'type':'ftu-bounds','index':list(self.wsi.ftus.keys()).index(struct)},
                        'color':'',
                        'hover_color':''
                    }
                    for struct in self.wsi.ftu_names
                }
            }

            spot_dict = {
                'geojson':self.wsi.geojson_spots,
                'id': {'type':'ftu-bounds','index':len(list(self.wsi.ftus.keys()))},
                'color': '#dffa00',
                'hover_color':'#9caf00'
            }
            new_children = [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = map_dict['FTUs'][struct]['geojson'], id = map_dict['FTUs'][struct]['id'], options = dict(style = self.ftu_style_handle),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val),
                                    hoverStyle = arrow_function(dict(weight=5, color = map_dict['FTUs'][struct]['hover_color'], dashArray = '')))),
                        name = struct, checked = True, id = self.wsi.slide_info_dict['key_name']+'_'+struct)
                for struct in map_dict['FTUs']
                ]
            
            new_children += [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = spot_dict['geojson'], id = spot_dict['id'], options = dict(style = self.ftu_style_handle),
                                hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val),
                                hoverStyle = arrow_function(dict(weight=5, color = spot_dict['hover_color'], dashArray = '')))),
                        name = 'Spots', checked = False, id = self.wsi.slide_info_dict['key_name']+'_Spots')
                ]

            for m_idx,man in enumerate(self.wsi.manual_rois):
                new_children.append(
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = man['geojson'], id = man['id'], options = dict(style = self.ftu_style_handle),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors),
                                    hoverStyle = arrow_function(dict(weight=5, color = man['hover_color'], dashArray = '')))),
                            name = f'Manual ROI {m_idx}', checked = True, id = self.wsi.slide_info_dict['key_name']+f'_manual_roi{m_idx}'
                        )
                    )

            new_url = self.wsi.image_url

            center_point = [(self.wsi.slide_bounds[1]+self.wsi.slide_bounds[3])/2,(self.wsi.slide_bounds[0]+self.wsi.slide_bounds[2])/2]

        else:
            # Find folder containing this slide
            for d in self.dataset_handler.slide_datasets:
                d_slides = [i['name'] for i in self.dataset_handler.slide_datasets[d]['Slides']]
                if slide_name in d_slides:
                    # Getting slide item id
                    slide_id = self.dataset_handler.slide_datasets[d]['Slides'][d_slides.index(slide_name)]['_id']
                    # Getting all the mapping info
                    map_bounds, base_dims, image_dims, tile_size, geojson_annotations, tile_url = self.dataset_handler.get_resource_map_data(slide_id)

            new_slide = DSASlide(slide_name,slide_id,tile_url,geojson_annotations,image_dims,base_dims)

            pre_slide_properties = new_slide.properties_list

            # Updating available slide_properties to the ones that are implemented as visualizations
            slide_properties = []
            for p in pre_slide_properties:
                if p in self.visualization_properties:
                    slide_properties.append(p)
                elif '-->' in p:
                    main_p = p.split(' --> ')[0]
                    cell_abbrev = p.split(' --> ')[1]
                    if main_p in self.visualization_properties:
                        slide_properties.append(p.replace(cell_abbrev,self.cell_graphics_key[cell_abbrev]['full']))

            self.wsi = new_slide

            if not self.current_cell in ['max','cluster']:
                self.update_hex_color_key('cell_value')
            elif self.current_cell == 'max':
                self.update_hex_color_key('max_cell')
            elif self.current_cell == 'cluster':
                self.update_hex_color_key('cluster')
            else:
                self.update_hex_color_key(self.current_cell)

            # map_dict contains ftu geojson information
            map_dict = {
                'url': tile_url,
                'FTUs':{
                    struct: {
                        'geojson':{'type':'FeatureCollection','features':[i for i in new_slide.geojson_ftus['features'] if i['properties']['name']==struct]},
                        'id':{'type':'ftu-bounds','index':new_slide.ftu_names.index(struct)},
                        'color':'',
                        'hover_color':''
                    }
                    for struct in new_slide.ftu_names
                }
            }

            # spot_dict contains spot geojson information
            spot_dict = {
                'geojson': new_slide.geojson_spots,
                'id':{'type':'ftu-bounds','index':len(new_slide.ftu_names)},
                'color': '#dffa00',
                'hover_color':'#9caf00'
            }

            new_children = [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = map_dict['FTUs'][struct]['geojson'], id = map_dict['FTUs'][struct]['id'], options = dict(style = self.ftu_style_handle),
                                   hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val),
                                   hoverStyle = arrow_function(dict(weight=5, color = map_dict['FTUs'][struct]['hover_color'], dashArray = '')))
                    ), name = struct, checked = True, id = new_slide.item_id+'_'+struct
                )
                for struct in map_dict['FTUs']
            ]

            new_children += [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = spot_dict['geojson'], id = spot_dict['id'], options = dict(style = self.ftu_style_handle),
                                   hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val),
                                   hoverStyle = arrow_function(dict(weight=5, color = spot_dict['hover_color'], dashArray='')))),
                    name = 'Spots', checked = False, id = new_slide.item_id+'_Spots'
                )
            ]

            # Now iterating through manual ROIs
            for m_idx, man in enumerate(new_slide.manual_rois):
                new_children.append(
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = man['geojson'], id = man['id'], options = dict(style = self.ftu_style_handle),
                                       hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors),
                                       hoverStyle = arrow_function(dict(weight=5,color=man['hover_color'], dashArray='')))
                        ),
                        name = f'Manual ROI {m_idx}', checked = True, id = new_slide.item_id+f'_manual_roi{m_idx}'
                    )
                )

            new_url = tile_url
            center_point = [0.5*(map_bounds[0][0]+map_bounds[1][0]),-0.5*(map_bounds[0][1]+map_bounds[1][1])]


        self.current_ftus = self.wsi.ftu_names+['Spots']
        self.current_ftu_layers = self.wsi.ftu_names+['Spots']

        # Adding the layers to be a property for the edit_control callback
        self.current_overlays = new_children

        # Adding fresh edit-control to the outputs
        new_edit_control = dl.EditControl(
            id={'type':'edit_control','index':np.random.randint(0,1000)},
            draw = dict(line=False,circle=False,circlemarker=False)
            )

        return new_url, new_children, center_point, map_bounds, tile_size, new_edit_control, slide_properties

    def update_graph(self,ftu,plot,label):
        
        self.current_ftu = ftu
        # Filtering by selected FTU
        #current_data = self.metadata[self.metadata['ftu_type'].str.match(ftu)]

        # Getting the labels that can be applied to the cluster plot
        cell_types = list(self.wsi.geojson_ftus['features'][0]['properties']['Main_Cell_Types'].keys())
        if self.current_ftu=='glomerulus':
            available_labels = ['Cluster','image_id','Area','Mesangial Area','Mesangial Fraction']+cell_types
        elif self.current_ftu == 'Tubules':
            available_labels = ['Cluster','image_id','Average TBM Thickness','Average Cell Thickness','Luminal Fraction']+cell_types

        current_data = []
        for f in self.metadata:
            if 'ftu_type' in f:
                if f['ftu_type'] == ftu:
                    current_data.append(f)

        if plot=='TSNE':
            plot_data_x = [i['x_tsne'] for i in current_data if 'x_tsne' in i]
            plot_data_y = [i['y_tsne'] for i in current_data if 'y_tsne' in i]

        elif plot=='UMAP':
            plot_data_x = [i['x_umap'] for i in current_data if 'x_umap' in i]
            plot_data_y = [i['y_umap'] for i in current_data if 'y_umap' in i]

        custom_data = [i['ftu_name'] for i in current_data if 'ftu_name' in i]
        # If the label is image_id or cluster
        try:
            label_data = [i[label] for i in current_data]
        except:
            # If the label is a main cell type or cell states of a main cell type
            try:
                label_data = [i['Main_Cell_Types'][label] for i in current_data]
            except:
                label_data = []
                for i in current_data:
                    if label in i:
                        label_data.append(i[label])
                    else:
                        label_data.append(np.nan)

        graph_df = pd.DataFrame({'x':plot_data_x,'y':plot_data_y,'ID':custom_data,'Label':label_data})

        graph_df = graph_df.dropna()

        cluster_graph = go.Figure(px.scatter(graph_df,x='x',y='y',custom_data=['ID'],color='Label',title=f'{plot} Plot of:<br><sup>{ftu} Morphometrics</sup><br><sup>Labeled by {label}</sup>'))
        cluster_graph.update_layout(
            margin=dict(l=0,r=0,t=80,b=0)
        )

        return cluster_graph, available_labels

    def grab_image(self,sample_info):

        img_list = []
        for idx,s in enumerate(sample_info):
            # openslide needs min_x, min_y, width, height
            min_x = int(s['Min_x_coord'])
            min_y = int(s['Min_y_coord'])
            max_x = int(s['Max_x_coord'])
            max_y = int(s['Max_y_coord'])

            if not self.run_type == 'dsa':
                # Normalize coordinates to be within slide bounds
                # width, height
                wsi_dims = self.slide_info_dict[s['image_id']+'.svs']['wsi_dims']
                # min_long, min lat, max_long, max lat
                slide_bounds = self.slide_info_dict[s['image_id']+'.svs']['bounds']

                min_slide_x = slide_bounds[0]+(slide_bounds[2]-slide_bounds[0])*(min_x/wsi_dims[0])
                min_slide_y = slide_bounds[1]+(slide_bounds[3]-slide_bounds[1])*(min_y/wsi_dims[1])
                max_slide_x = slide_bounds[0]+(slide_bounds[2]-slide_bounds[0])*(max_x/wsi_dims[0])
                max_slide_y = slide_bounds[1]+(slide_bounds[3]-slide_bounds[1])*(max_y/wsi_dims[1])
                #test_merc = mercantile.bounding_tile(max_slide_x,min_slide_y,min_slide_x,max_slide_y)
                test_merc = mercantile.tile((max_slide_x+min_slide_x)/2,(max_slide_y+min_slide_y)/2,17)

                # Use requests to pull out image data from URL
                dataset_name = self.dataset_handler.get_slide_dataset(s['image_id']+'.svs')
                dataset = self.dataset_handler.get_dataset(dataset_name)
                
                slide_key_name = self.slide_info_dict[s['image_id']+'.svs']['key_name']
                merc_tiles = mercantile.neighbors(test_merc)
                tile_mask = np.zeros((3*256,3*256,3))
                for t in merc_tiles:
                    tile_x, tile_y, tile_z = t.x, t.y, t.z

                    url = f'http://192.168.86.39:5000/rgb/{dataset["key_name"]}/{slide_key_name}/{tile_z}/{tile_x}/{tile_y}.png?r=B04&r_range=[46,168]&g=B03&g_range=[46,168]&b=B02&b_range=[46,168]&'

                    response = requests.get(url,verify=False,timeout=10)
                    img = Image.open(BytesIO(response.content))

                    # It says that there's no guarantee of the order so have to figure something out for that
                    tile_x_diff = tile_x-test_merc.x
                    tile_y_diff = tile_y-test_merc.y
                    tile_mask[256*(1+tile_y_diff):256+256*(1+tile_y_diff),256*(1+tile_x_diff):256+256*(1+tile_x_diff),:] = np.uint8(np.array(img))
                
                # Now grabbing the middle image
                url = f'http://192.168.86.39:5000/rgb/{dataset["key_name"]}/{slide_key_name}/{test_merc.z}/{test_merc.x}/{test_merc.y}.png?r=B04&r_range=[46,168]&g=B03&g_range=[46,168]&b=B02&b_range=[46,168]&'

                response = requests.get(url,verify=False,timeout=10)
                img = Image.open(BytesIO(response.content))
                
                tile_mask[256:512,256:512,:] += np.uint8(np.array(img))

                tile_mask = resize(tile_mask,output_shape=(512,512,3))
                img_list.append(tile_mask)

            else:

                # Pulling image region using provided coordinates
                # Have to find the slide id for this particular sample
                image_region = self.dataset_handler.get_image_region(s['item_id'],[min_x,min_y,max_x,max_y])
                img_list.append(resize(np.array(image_region),output_shape=(512,512,3)))


        return img_list        

    def update_selected(self,hover,selected):

        if hover is not None:
            print(f'triggered_prop_ids: {ctx.triggered_prop_ids.keys()}')
            if 'cluster-graph.selectedData' in list(ctx.triggered_prop_ids.keys()):
                sample_ids = [i['customdata'][0] for i in selected['points']]
                sample_info = []
                for f in self.metadata:
                    if 'ftu_name' in f:
                        if f['ftu_name'] in sample_ids:
                            sample_info.append(f)
            else:
                if hover is not None:
                    sample_id = hover['points'][0]['customdata'][0]
                    sample_info = []
                    for f in self.metadata:
                        if 'ftu_name' in f:
                            if f['ftu_name']==sample_id:
                                sample_info.append(f)
                else:
                    sample_info = [self.metadata[0]]

            self.current_selected_samples = sample_info

            current_image = self.grab_image(sample_info)
            print(f'length: {len(current_image)}')
            if len(current_image)==1:
                selected_image = go.Figure(px.imshow(current_image[0]))
            elif len(current_image)>1:
                selected_image = go.Figure(px.imshow(np.stack(current_image,axis=0),animation_frame=0,binary_string=True,labels=dict(animation_frame=self.current_ftu)))
            else:
                print(f'No images found')
                print(f'hover: {hover}')
                print(f'selected:{selected}')
                print(f'self.current_selected_samples: {self.current_selected_samples}')


            selected_image.update_layout(
                margin=dict(l=0,r=0,t=0,b=0)
            )

            # Preparing figure containing cell types + cell states info
            counts_data = pd.DataFrame([i['Main_Cell_Types'] for i in sample_info]).sum(axis=0).to_frame()
            counts_data.columns = ['Selected Data Points']
            counts_data = counts_data.reset_index()
            # Normalizing to sum to 1
            counts_data['Selected Data Points'] = counts_data['Selected Data Points']/counts_data['Selected Data Points'].sum()
            # Only getting the top-5
            counts_data = counts_data.sort_values(by='Selected Data Points',ascending=False)
            counts_data = counts_data[counts_data['Selected Data Points']>0]
            f_pie = px.pie(counts_data,values='Selected Data Points',names='index')

            # Getting initial cell state info
            first_cell = counts_data['index'].tolist()[0]
            state_data = pd.DataFrame([i['Cell_States'][first_cell] for i in sample_info]).sum(axis=0).to_frame()
            state_data = state_data.reset_index()
            state_data.columns = ['Cell States',f'Cell States for {first_cell}']

            state_data[f'Cell States for {first_cell}'] = state_data[f'Cell States for {first_cell}']/state_data[f'Cell States for {first_cell}'].sum()

            s_bar = px.bar(state_data, x='Cell States', y = f'Cell States for {first_cell}', title = f'Cell States for:<br><sup>{self.cell_graphics_key[first_cell]["full"]} in:</sup><br><sup>selected points</sup>')
            
            selected_cell_types = go.Figure(f_pie)
            selected_cell_states = go.Figure(s_bar)

            selected_cell_states.update_layout(
                margin=dict(l=0,r=0,t=85,b=0)
            )
            selected_cell_types.update_layout(
                margin=dict(l=0,r=0,t=0,b=0),
                showlegend=False
            )

            return selected_image, selected_cell_types, selected_cell_states
        else:
            return go.Figure(), go.Figure(), go.Figure()
    
    def update_selected_state_bar(self, selected_cell_click):
        #print(f'Selected cell click: {selected_cell_click}')
        if not selected_cell_click is None:
            cell_type = selected_cell_click['points'][0]['label']

            state_data = pd.DataFrame([i['Cell_States'][cell_type] for i in self.current_selected_samples]).sum(axis=0).to_frame()
            state_data = state_data.reset_index()
            state_data.columns = ['Cell States',f'Cell States for {cell_type}']
            state_data[f'Cell States for {cell_type}'] = state_data[f'Cell States for {cell_type}']/state_data[f'Cell States for {cell_type}'].sum()

            s_bar = px.bar(state_data, x='Cell States', y = f'Cell States for {cell_type}', title = f'Cell States for:<br><sup>{self.cell_graphics_key[cell_type]["full"]} in:</sup><br><sup>selected points</sup>')
            s_bar = go.Figure(s_bar)

            s_bar.update_layout(
                margin=dict(l=0,r=0,t=85,b=0)
            )

            return s_bar
        else:
            return go.Figure()

    def add_manual_roi(self,new_geojson):
        
        print(f'triggered_id for add_manual: {ctx.triggered_id}')
        print(f'triggered prop ids for add_manual: {ctx.triggered_prop_ids}')
        print(f'type of ctx.triggered_id: {type(ctx.triggered_id)}')
        triggered_id = ctx.triggered_id['type']

        print(f'triggered_id: {triggered_id}')

        if triggered_id == 'edit_control':
            print(f'manual_roi:{new_geojson}')
            if not new_geojson is None:
                if type(new_geojson)==list:
                    new_geojson = new_geojson[0]
                if not new_geojson is None:
                    if len(new_geojson['features'])>0:
                        
                        if not new_geojson['features'][len(self.wsi.manual_rois)]['properties']['type']=='marker':
                            # Only getting the most recent to add
                            new_geojson = {'type':'FeatureCollection','features':[new_geojson['features'][len(self.wsi.manual_rois)]]}
                            print(f'new_geojson: {new_geojson}')

                            # New geojson has no properties which can be used for overlays or anything so we have to add those
                            # Step 1, find intersecting spots:
                            if not self.run_type=='dsa':
                                overlap_dict = self.wsi.find_intersecting_spots(shape(new_geojson['features'][0]['geometry']))
                                main_counts_data = pd.DataFrame(overlap_dict['main_counts']).sum(axis=0).to_frame()
                            else:

                                overlap_spot_props = self.wsi.find_intersecting_spots(shape(new_geojson['features'][0]['geometry']))
                                main_counts_data = pd.DataFrame.from_records([i['Main_Cell_Types'] for i in overlap_spot_props if 'Main_Cell_Types' in i]).sum(axis=0).to_frame()

                            main_counts_data = (main_counts_data/main_counts_data.sum()).fillna(0.000).round(decimals=18)

                            main_counts_data[0] = main_counts_data[0].map('{:.19f}'.format)
                            
                            main_counts_dict = main_counts_data.astype(float).to_dict()[0]

                            agg_cell_states = {}
                            for m_c in list(main_counts_dict.keys()):

                                if not self.run_type=='dsa':
                                    cell_states = pd.DataFrame([i[m_c] for i in overlap_dict['states']]).sum(axis=0).to_frame()
                                else:
                                    cell_states = pd.DataFrame.from_records([i['Cell_States'][m_c] for i in overlap_spot_props]).sum(axis=0).to_frame()


                                cell_states = (cell_states/cell_states.sum()).fillna(0.000).round(decimals=18)

                                cell_states[0] = cell_states[0].map('{:.19f}'.format)

                                agg_cell_states[m_c] = cell_states.astype(float).to_dict()[0]
                            

                            new_geojson['features'][0]['properties']['Main_Cell_Types'] = main_counts_dict
                            new_geojson['features'][0]['properties']['Cell_States'] = agg_cell_states

                            print(f'Length of wsi.manual_rois: {len(self.wsi.manual_rois)}')
                            print(f'Length of current_overlays: {len(self.current_overlays)}')

                            self.wsi.manual_rois.append(
                                {
                                    'geojson':new_geojson,
                                    'id':{'type':'ftu-bounds','index':len(self.current_overlays)},
                                    'hover_color':'#32a852'
                                }
                            )

                            print(f'Length of wsi.manual_rois: {len(self.wsi.manual_rois)}')
                            # Updating the hex color key with new values
                            self.update_hex_color_key('cell_value')

                            new_child = dl.Overlay(
                                dl.LayerGroup(
                                    dl.GeoJSON(data = new_geojson, id = {'type':'ftu-bounds','index':len(self.current_overlays)}, options = dict(style=self.ftu_style_handle),
                                        hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val),
                                        hoverStyle = arrow_function(dict(weight=5, color = '#32a852',dashArray = ''))
                                    )
                                ), name = f'Manual ROI {len(self.wsi.manual_rois)}', checked = True, id = self.wsi.item_id+f'_manual_roi{len(self.wsi.manual_rois)}'
                            )

                            self.current_overlays.append(new_child)
                            print(f'Length of current_overlays: {len(self.current_overlays)}')

                            # Updating data download options
                            if len(self.wsi.manual_rois)>0:
                                data_select_options = self.layout_handler.data_options
                                data_select_options[4]['disabled'] = False
                            else:
                                data_select_options = self.layout_handler.data_options
                            
                            if len(self.wsi.marked_ftus)>0:
                                data_select_options[3]['disabled'] = False

                            return self.current_overlays, data_select_options
                        
                        elif new_geojson['features'][len(self.wsi.manual_rois)]['properties']['type']=='marker':
                            # Find the ftu that this marker is included in if there is one otherwise 
                            new_geojson = {'type':'FeatureCollection','features':[new_geojson['features'][len(self.wsi.manual_rois)]]}

                            # TODO: placeholder for adding the marked FTU to the slide's list of marked FTUs
                            print(new_geojson['features'][len(self.wsi.manual_rois)])

                            # Find the ftu that intersects with this marker
                            overlap_dict = self.wsi.find_intersecting_ftu(shape(new_geojson['features'][0]['geometry']))
                            print(f'Intersecting FTUs with marker: {overlap_dict}')
                            if len(overlap_dict['polys'])>0:
                                self.wsi.marked_ftus.append(overlap_dict)

                            # Updating data download options
                            if len(self.wsi.manual_rois)>0:
                                data_select_options = self.layout_handler.data_options
                                data_select_options[4]['disabled'] = False
                            else:
                                data_select_options = self.layout_handler.data_options
                            
                            if len(self.wsi.marked_ftus)>0:
                                data_select_options[3]['disabled'] = False


                            return self.current_overlays, data_select_options
                    else:
                        raise exceptions.PreventUpdate
                else:
                    raise exceptions.PreventUpdate
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate    

    def update_download_options(self,selected_data):
        print(f'selected_data: {selected_data}')
        new_children = []
        tab_labels = []

        options_idx = 0
        for d in selected_data:
            if d == 'Annotations':
                # After selecting annotations, users can select whether they want the annotations in
                # JSON, GeoJSON, or Aperio XML format (eventually also OME-TIFF but that might be too large).
                # They can also select whether they want the cell types/states info to be included with the annotations or
                # saved as a spreadsheet with some row labels for each FTU.
                child = dbc.Card([
                    dbc.Label('Format for annotations:'),
                    dbc.Row(
                        dcc.RadioItems(
                            [{'label':html.Div(['Aperio XML'],style = {'padding-left':'50px','padding-right':'10px'}),'value':'Aperio XML'},
                             {'label':html.Div(['Histomics JSON'],style={'padding-left':'50px','padding-right':'10px'}),'value':'Histomics JSON'},
                             {'label':html.Div(['GeoJSON'],style={'padding-left':'50px','padding-right':'10px'}),'value':'GeoJSON'}],
                            value = 'Aperio XML',
                            inline=True,
                            id = {'type':'download-opts','index':options_idx},
                            labelStyle={'display':'flex'}),
                            style = {'marginBottom':'20px'}
                        ),
                    html.Hr(),
                    html.B(),
                    dcc.Loading(
                        children = html.Div([
                            dbc.Button('Download Annotations',color='primary',id={'type':'download-butt','index':options_idx}),
                            Download(id={'type':'download-data','index':options_idx})
                            ])
                    )                    
                ])

                new_children.append(child)
                tab_labels.append(d)
                options_idx+=1

            if d == 'Slide Metadata':
                # After selecting slide metadata, users select which labels they want to keep slide-level metadata
                # names of slides, disease label (if there), numbers and names of FTUs, tissue type, omics type, per-FTU properties

                child = dbc.Card([
                    dbc.Label('Select per-slide properties:'),
                    dbc.Row(
                        dcc.Dropdown(
                            ['FTU Properties', 'Tissue Type','Omics Type','Slide Metadata', 'FTU Counts'],
                            ['FTU Properties', 'Tissue Type','Omics Type','Slide Metadata', 'FTU Counts'],
                            multi=True,
                            id = {'type':'download-opts','index':options_idx} 
                        ),style = {'marginBottom':'20px'}
                    ),
                    html.Hr(),
                    html.B(),
                    dcc.Loading(
                        children = [
                            dbc.Button('Download Slide Data',color='primary',id={'type':'download-butt','index':options_idx}),
                            Download(id={'type':'download-data','index':options_idx})
                        ]
                    )                    
                ])

                new_children.append(child)
                tab_labels.append(d)
                options_idx+=1

            if d == 'Cell Type and State':
                # Outputting cell type and state info in different formats
                cell_type_items = [
                    {'label':html.Div(['CSV Files'],style = {'padding-left':'50px','padding-right':'10px'}),'value':'CSV Files'},
                    {'label':html.Div(['Excel File'],style = {'padding-left':'50px','padding-right':'10px'}),'value':'Excel File'},
                    {'label':html.Div(['RDS File'],style = {'padding-left':'50px','padding-right':'10px'}),'value':'RDS File','disabled':True}
                ]
                child = dbc.Card([
                    dbc.Label('Format for Cell Types and States:'),
                    dbc.Row(
                        dcc.RadioItems(cell_type_items,
                        value = 'CSV Files',
                        inline=True,
                        id = {'type':'download-opts','index':options_idx},
                        labelStyle={'display':'flex'}),
                        style = {'marginBottom':'20px'}
                    ),
                    html.Hr(),
                    html.B(),
                    dcc.Loading(
                        children = [
                            dbc.Button('Download Cell Type Data',color='primary',id={'type':'download-butt','index':options_idx}),
                            Download(id={'type':'download-data','index':options_idx})
                        ]
                    )                    
                ])

                new_children.append(child)
                tab_labels.append(d)
                options_idx+=1

            if d == 'Selected FTUs and Metadata':
                # Saving selected FTU image regions and cell type/state info
                include_opts = ['Image & Cell Type/State Information','Image Only','Cell Type/State Only']
                select_ftu_list = []
                for i in include_opts:
                    select_ftu_list.append(
                        {'label':html.Div([i],style={'padding-left':'50px','padding-right':'10px'}),'value':i}
                    )          

                child = dbc.Card([
                    dbc.Label('Selected FTU Data to Save'),
                    dbc.Row(
                        dcc.RadioItems(select_ftu_list,
                        include_opts[0],
                        inline = True,
                        id = {'type':'download-opts','index':options_idx},
                        labelStyle={'display':'flex'}),
                        style = {'marginBottom':'20px'}
                    ),
                    html.Hr(),
                    html.B(),
                    dcc.Loading(
                        children = [
                            dbc.Button('Download Selected FTUs Data',color='primary',id={'type':'download-butt','index':options_idx}),
                            Download(id={'type':'download-data','index':options_idx})
                        ]
                    )
                ])

                new_children.append(child)
                tab_labels.append(d)
                options_idx+=1

            if d == 'Manual ROIs':
                # Saving manually generated ROIs and cell type/state info
                include_opts = ['Image & Cell Type/State Information','Image Only','Cell Type/State Only']
                select_ftu_list = []
                for i in include_opts:
                    select_ftu_list.append(
                        {'label':html.Div([i],style={'padding-left':'50px','padding-right':'10px'}),'value':i+'_man'}
                    )          

                child = dbc.Card([
                    dbc.Label('Manual ROI Data to Save'),
                    dbc.Row(
                        dcc.RadioItems(select_ftu_list,
                        include_opts[0],
                        inline=True,
                        id = {'type':'download-opts','index':options_idx},
                        labelStyle={'display':'flex'}),
                        style = {'marginBottom':'20px'}
                    ),
                    html.Hr(),
                    html.B(),
                    dcc.Loading(
                        children = [
                            dbc.Button('Download Manual ROI Data', color = 'primary', id = {'type':'download-butt','index':options_idx}),
                            Download(id={'type':'download-data','index':options_idx})
                        ]
                    )                    
                ])    

                new_children.append(child)
                tab_labels.append(d)
                options_idx+=1

        tab_data = []
        id_count = 0
        for t,l in zip(new_children,tab_labels):
            tab_data.append(dbc.Tab(t,label=l,id=f'tab_{id_count}'))
            id_count+=1
        
        new_children = dbc.Tabs(tab_data,active_tab = 'tab_0')

        return new_children

    def download_data(self,options,button_click):
        print(ctx.triggered_id)
        print(options)
        if button_click:
            if ctx.triggered_id['type'] == 'download-butt':
                # Download data has to be a dictionary with content and filename keys. The filename extension will vary

                try:
                    os.remove('./assets/FUSION_Download.zip')
                except OSError:
                    print('No previous download zip file to remove')

                print(f'Download type: {self.download_handler.what_data(options)}')
                download_type = self.download_handler.what_data(options)
                if download_type == 'annotations':
                    download_list = self.download_handler.extract_annotations(self.wsi,options)
                elif download_type == 'cell':
                    download_list = self.download_handler.extract_cell(self.current_ftus,options)
                else:
                    print('Working on it!')
                    download_list = []

                self.download_handler.zip_data(download_list)
                
                return dcc.send_file('./assets/FUSION_Download.zip')

            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate

    def run_analysis(self,cli_name,cli_butt):

        cli_dict = [i for i in self.dataset_handler.cli_dict_list if i['name']==cli_name][0]
        cli_description = dcc.Markdown(cli_dict['description'])
        cli_id = cli_dict['_id']
        cli_butt_disable = False

        # Printing xml to find inputs
        cli_xml = self.dataset_handler.gc.get(f'slicer_cli_web/cli/{cli_id}')['xml']
        print('cli XML')
        print(cli_xml)

        if ctx.triggered_id=='cli-drop':

            # Get description for cli
            cli_results = 'Click "Run Job!" to do the damn thing!'
        
        elif ctx.triggered_id=='cli-run':
            
            # Running job:
            cli_results = 'And then the job would run'
        
        return cli_description, cli_butt_disable, cli_results

    def update_upload_requirements(self,collection,new_collect,upload_type):
        
        input_disabled = True
        # Creating an upload div specifying which files are needed for a given upload type
        if ctx.triggered_id=='upload-type':
            if not collection=='New Collection':

                # Getting the collection id
                collection_id = self.dataset_handler.get_resource_id(f'/collection/{collection}')

            else:
                if new_collect is not None:

                    # Creating a new collection with this name
                    self.dataset_handler.gc.post('/collection',parameters={'name':new_collect})
                    collection_id = self.dataset_handler.get_resource_id(f'/collection/{new_collect}')
            
            upload_style = {
                'width':'100%',
                'height':'40px',
                'lineHeight':'40px',
                'borderWidth':'1px',
                'borderStyle':'dashed',
                'borderRadius':'5px',
                'textAlign':'center',
                'margin':'10px'
            }

            if upload_type=='Visium':
                upload_reqs = html.Div([
                    dbc.Row([
                        dcc.Upload(
                            id={'type':'wsi-upload','index':0},
                            children = html.Div([
                                'Drag and Drop or ',
                                html.A('Select WSI File')
                            ]),
                            style = upload_style,
                            multiple=False
                        ),
                        html.Div(id={'type':'wsi-upload-contents','index':0})
                    ]),
                    dbc.Row([
                        dcc.Upload(
                            id={'type':'omics-upload','index':0},
                            children = html.Div([
                                'Drag and Drop or ',
                                html.A('Select Omics File')
                            ]),
                            style = upload_style,
                            multiple = False
                        ),
                        html.Div(id={'type':'omics-upload-contents','index':0})
                    ])
                ])
            
            elif upload_type=='CODEX':

                upload_reqs = html.Div([
                    dbc.Row([
                        dcc.Upload(
                            id={'type':'wsi-upload','index':1},
                            children = html.Div([
                                'Drag and Drop or ',
                                html.A('Select qptiff File')
                            ]),
                            style = upload_style,
                            multiple = False
                        ),
                        html.Div(id={'type':'wsi-upload-contents','index':1})
                    ])
                ])

            else:
                upload_reqs = html.Div(
                    'You should not have done that'
                )
            
            return upload_reqs, input_disabled
        
        elif ctx.triggered_id=='collect_select':
            if collection=='New Collection':
                input_disabled = False
            else:
                input_disabled = True
            
            return html.Div(),input_disabled
        else:
            raise exceptions.PreventUpdate

    def girder_login(self,username,pword,p_butt):

        if ctx.triggered_id=='login-submit':

            try:
                self.dataset_handler.authenticate(username,pword)

                button_color = 'success'
                button_text = 'Success!'
                logged_in_user = f'Welcome: {username}'

            except girder_client.AuthenticationError:

                button_color = 'warning'
                button_text = 'Login Failed'
                logged_in_user = ''

            return button_color, button_text, logged_in_user
        else:
            raise exceptions.PreventUpdate

#if __name__ == '__main__':
def app(*args):

    run_type = 'dsa'

    try:
        run_type = os.environ['RUNTYPE']
    except:
        print(f'Using {run_type} run type')

    if run_type == 'local':
        # For local testing
        base_dir = '/mnt/c/Users/Sam/Desktop/HIVE/SpotNet_NonEssential_Files/WSI_Heatmap_Viewer_App/assets/slide_info/'

        # Loading initial dataset
        dataset_reference_path = 'dataset_reference.json'
        dataset_handler = DatasetHandler(dataset_reference_path)

        dataset_info_dict = dataset_handler.get_dataset('Indiana U.')

        slide_names = []
        slide_info_dict = {}
        for slide in dataset_info_dict['slide_info']:
            slide_info_dict[slide['name']] = slide
            slide_names.append(slide['name'])

        slide_name = slide_names[0]
        slide_extension = slide_name.split('.')[-1]
        dataset_key = dataset_info_dict['key_name']

        slide_url = 'http://localhost:5000/rgb/'+dataset_key+'/'+slide_info_dict[slide_name]["key_name"]+'/{z}/{x}/{y}.png?r=B04&r_range=[46,168]&g=B03&g_range=[46,168]&b=B02&b_range=[46,168]&'
        ftu_path = base_dir+slide_name.replace('.'+slide_extension,'_scaled.geojson')
        spot_path = base_dir+slide_name.replace('.'+slide_extension,'_Spots_scaled.geojson')
        cell_graphics_path = 'graphic_reference.json'
        asct_b_path = 'Kidney_v1.2 - Kidney_v1.2.csv'

        metadata_paths = [base_dir+s.replace('.'+slide_extension,'_scaled.geojson') for s in slide_names]

    elif run_type == 'web' or run_type=='AWS':

        base_dir = os.getcwd()
        slide_info_path = base_dir+'assets/slide_info/'

        # Loading initial dataset
        dataset_reference_path = 'dataset_reference.json'
        dataset_handler = DatasetHandler(dataset_reference_path)

        dataset_info_dict = dataset_handler.get_dataset('Indiana U.')

        slide_names = []
        slide_info_dict = {}
        for slide in dataset_info_dict['slide_info']:
            slide_info_dict[slide['name']] = slide
            slide_names.append(slide['name'])

        slide_name = slide_names[0]
        slide_extension = slide_name.split('.')[-1]
        dataset_key = dataset_info_dict['key_name']

        slide_url = f'{os.environ.get("TILE_SERVER_HOST")}/rgb/{dataset_key}/{slide_info_dict[slide_name["key_name"]]}'+'/{z}/{x}/{y}.png?r=B04&r_range=[46,168]&g=B03&g_range=[46,168]&b=B02&b_range=[46,168]&'
        ftu_path = base_dir+slide_name.replace('.'+slide_extension,'_scaled.geojson')
        spot_path = base_dir+slide_name.replace('.'+slide_extension,'_Spots_scaled.geojson')

        cell_graphics_path = base_dir+'graphic_reference.json'
        asct_b_path = base_dir+'Kidney_v1.2 - Kidney_v1.2.csv'

        metadata_paths = [base_dir+s.replace('.'+slide_extension,'_scaled.geojson') for s in slide_names]

    elif run_type == 'dsa':
        
        # Using DSA as base directory for storage and accessing files
        dsa_url = 'http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/'

        # Different kind of dataset handler
        dataset_handler = GirderHandler(apiUrl=dsa_url)

        # Using environment variables for login TODO: Add login to main page
        #try:
        username = os.environ.get('DSA_USER')
        p_word = os.environ.get('DSA_PWORD')
        dataset_handler.authenticate(username,p_word)
        #except:
        #    print('Get a load of this guy, no login!')

        # Initial collection TODO: get some default image as a placeholder in the visualization
        initial_collection = '/collection/10X_Visium/FFPE Cohort/Histology/Reference'
        path_type = 'folder'
        #setattr(dataset_handler,'current_collection_path',initial_collection)
        print(f'initial collection: {initial_collection}')
        initial_collection_id = dataset_handler.gc.get('resource/lookup',parameters={'path':initial_collection})
        #setattr(dataset_handler,'current_collection_id',initial_collection_id)
        
        # Saving & organizing relevant id's in GirderHandler
        dataset_handler.initialize_folder_structure(initial_collection,path_type)

        print(f'found initial collection: {initial_collection_id}')
        # Contents of folder
        initial_collection_contents = dataset_handler.gc.get(f'resource/{initial_collection_id["_id"]}/items',parameters={'type':path_type})
        #print(f'contents: {initial_collection_contents}')
        initial_collection_contents = [i for i in initial_collection_contents if 'largeImage' in i]

        # Loading metadata for initial collection
        print('Loading Metadata')
        metadata = []
        for i in initial_collection_contents:
            item_annotations = dataset_handler.gc.get(f'/annotation/item/{i["_id"]}')
            for g in tqdm(item_annotations):
                for e in g['annotation']['elements']:
                    if 'user' not in e:
                        e['user'] = {}
                    e['user']['item_id'] = i['_id']
                    metadata.append(e['user'])

    if not run_type=='dsa':
        # Reading dictionary containing paths for specific cell types
        cell_graphics_key = cell_graphics_path
        cell_graphics_json = json.load(open(cell_graphics_key))
        cell_names = []
        for ct in cell_graphics_json:
            cell_names.append(cell_graphics_json[ct]['full'])

        # Compiling info for clustering metadata
        metadata = []
        for m in metadata_paths:
            # Reading geojson file
            with open(m) as f:
                current_geojson = geojson.load(f)
            
            # Iterating through features and adding properties to metadata
            # This will not include any of the geometries data which would be the majority
            for f in current_geojson['features']:
                metadata.append(f['properties'])
            
        # Adding ASCT+B table to files
        asct_b_table = pd.read_csv(asct_b_path,skiprows=list(range(10)))

        # Separate class for WSIs when running outside of DSA
        wsi = WholeSlide(slide_url,slide_name,slide_info_dict[slide_name],ftu_path,spot_path)

        # Calculating center point for initial layout
        current_slide_bounds = slide_info_dict[slide_name]['bounds']
        center_point = [(current_slide_bounds[1]+current_slide_bounds[3])/2,(current_slide_bounds[0]+current_slide_bounds[2])/2]
        map_bounds = None
        tile_size = 256

        if dataset_info_dict['ftu']['annotation_type']=='GeoJSON':
            map_dict = {
                'url':wsi.image_url,
                'FTUs':{
                    struct : {
                        'geojson':{'type':'FeatureCollection','features':[i for i in wsi.geojson_ftus['features'] if i['properties']['structure']==struct]},
                        'id':{'type':'ftu-bounds','index':list(wsi.ftus.keys()).index(struct)},
                        'color':'',
                        'hover_color':''
                    }
                    for struct in list(wsi.ftus.keys())
                }
            }

        if dataset_info_dict['omics_type']=='Visium':
            spot_dict = {
                'geojson':wsi.geojson_spots,
                'id': {'type':'ftu-bounds','index':len(list(wsi.ftus.keys()))},
                'color': '#dffa00',
                'hover_color':'#9caf00'
            }

        cli_list = None

    else:

        # Getting graphics_reference.json from the FUSION Assets folder
        assets_path = '/collection/FUSION Assets/'

        cell_graphics_key = dataset_handler.gc.get('resource/lookup',parameters={'path':assets_path+'cell_graphics/graphic_reference.json'})
        # Downloading cell_graphics_key to get json file contents
        cell_graphics_key = dataset_handler.gc.get(f'item/{cell_graphics_key["_id"]}/download')
        cell_names = []
        for ct in cell_graphics_key:
            cell_names.append(cell_graphics_key[ct]['full'])

        # Getting asct+b table
        asct_b_table_id = dataset_handler.gc.get('resource/lookup',parameters={'path':assets_path+'asct_b/Kidney_v1.2 - Kidney_v1.2.csv'})['_id']
        token = dataset_handler.get_token()
        asct_b_table = pd.read_csv(dsa_url+f'item/{asct_b_table_id}/download?token={token}',skiprows=list(range(10)))

        print(f'first slide: {initial_collection_contents[0]["name"]}')
        map_bounds, base_dims, image_dims, tile_size, geojson_annotations, slide_url = dataset_handler.get_resource_map_data(resource=initial_collection_contents[0]['_id'])
        print(f'map_bounds: {map_bounds}')
        print(f'base_dims: {base_dims}')
        print(f'image_dims: {image_dims}')

        # Getting the slide data for DSASlide()

        slide_name = initial_collection_contents[0]['name']
        slide_item_id = initial_collection_contents[0]['_id']
        slide_names = [i['name'] for i in initial_collection_contents if 'largeImage' in i]
        slide_info_dict = None

        wsi = DSASlide(slide_name,slide_item_id,slide_url,geojson_annotations,image_dims,base_dims)
        center_point = [0,0]

        pre_slide_properties = wsi.properties_list
        slide_properties = []
        visualization_properties = [
            'Area', 'Arterial Area', 'Average Cell Thickness', 'Average TBM Thickness', 'Cluster',
            'Luminal Fraction','Main_Cell_Types','Mesangial Area','Mesangial Fraction'
        ]
        for p in pre_slide_properties:
            if p in visualization_properties:
                slide_properties.append(p)
            elif '-->' in p:
                main_p = p.split(' --> ')[0]
                cell_abbrev = p.split(' --> ')[1]
                if main_p in visualization_properties:
                    slide_properties.append(p.replace(cell_abbrev,cell_graphics_key[cell_abbrev]['full']))

        map_dict = {
            'url':slide_url,
            'FTUs':{
                struct : {
                    'geojson':{'type':'FeatureCollection','features':[i for i in wsi.geojson_annotations['features'] if i['properties']['name']==struct]},
                    'id':{'type':'ftu-bounds','index':wsi.ftu_names.index(struct)},
                    'color':'',
                    'hover_color':''
                }
                for struct in wsi.ftu_names
            }
        }
        spot_dict = {
            'geojson':{'type':'FeatureCollection','features':[i for i in wsi.geojson_annotations['features'] if i['properties']['name']=='Spots']},
            'id':{'type':'ftu-bounds','index':len(wsi.ftu_names)},
            'color':'#dffa00',
            'hover_color':'#9caf00'
        }

        # Getting list of available CLIs in DSA instance
        # This dict will contain all the info for the CLI's, have to reduce it to names
        cli_dict_list = dataset_handler.get_cli_list()
        cli_list = []
        for c in cli_dict_list:
            cli_dict = {'label':c['name'],'value':c['name']}

            # Constraining to only the dsarchive ones just for convenience:
            if 'dsarchive' in c['image']:
                cli_dict['disabled'] = False
            else:
                cli_dict['disabled'] = True
            
            cli_list.append(cli_dict)


    external_stylesheets = [dbc.themes.LUX,dbc.icons.BOOTSTRAP]

    layout_handler = LayoutHandler()
    layout_handler.gen_initial_layout(slide_names)
    layout_handler.gen_vis_layout(cell_names,center_point,map_dict,spot_dict,slide_properties,run_type,tile_size,map_bounds,cli_list)
    layout_handler.gen_builder_layout(dataset_handler,run_type)
    layout_handler.gen_uploader_layout(dataset_handler)

    download_handler = DownloadHandler(dataset_handler)

    main_app = DashProxy(__name__,external_stylesheets=external_stylesheets,transforms = [MultiplexerTransform()])
    vis_app = SlideHeatVis(
        main_app,
        layout_handler,
        dataset_handler,
        download_handler,
        wsi,
        cell_graphics_key,
        asct_b_table,
        metadata,
        slide_info_dict,
        run_type
    )

    if run_type=='web':
        return vis_app.app

# Comment this portion out for web running
if __name__=='__main__':
    app()
