"""

Whole Slide Image heatmap generation and viewer for specific cell types using Ground Truth Spatial Transcriptomics

"""

import os
import sys
import pandas as pd
import numpy as np
import json

from PIL import Image
from io import BytesIO

from tqdm import tqdm

import shapely
from shapely.geometry import Point, shape, box
from skimage.transform import resize
import random

import girder_client

import plotly.express as px
import plotly.graph_objects as go
from matplotlib import colormaps

from dash import dcc, ctx, MATCH, ALL, dash_table, exceptions, callback_context, no_update

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, arrow_function
from dash_extensions.enrich import DashProxy, html, Input, Output, MultiplexerTransform, State
import dash_mantine_components as dmc
import dash_uploader as du

from timeit import default_timer as timer

from FUSION_WSI import DSASlide
from FUSION_Handlers import LayoutHandler, DownloadHandler, GirderHandler
from FUSION_Prep import PrepHandler


class FUSION:
    def __init__(self,
                app,
                layout_handler,
                dataset_handler,
                download_handler,
                prep_handler,
                wsi,
                cluster_metadata,
                ga_tag = None
                ):
                
        # Saving all the layouts for this instance
        self.layout_handler = layout_handler
        self.current_page = 'welcome'

        self.dataset_handler = dataset_handler
        self.current_overlays = self.layout_handler.initial_overlays

        self.download_handler = download_handler
        self.prep_handler = prep_handler

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

        # clustering related properties (and also cell types, cell states, image_ids, etc.)
        self.metadata = cluster_metadata
        self.wsi = wsi
        self.cell_graphics_key = self.dataset_handler.cell_graphics_key

        # Inverting the graphics key to get {'full_name':'abbreviation'}
        self.cell_names_key = {}
        for ct in self.cell_graphics_key:
            self.cell_names_key[self.cell_graphics_key[ct]['full']] = ct

        # Number of main cell types to include in pie-charts
        self.plot_cell_types_n = len(list(self.cell_names_key.keys()))

        # ASCT+B table for cell hierarchy generation
        self.table_df = self.dataset_handler.asct_b_table    

        # FTU settings
        self.ftus = self.wsi.ftu_names
        self.ftu_colors = self.wsi.ftu_colors

        self.current_ftu_layers = self.wsi.ftu_names+['Spots']
        self.current_ftus = self.wsi.ftu_names+['Spots']
        self.pie_ftu = self.current_ftu_layers[-1]

        # Specifying available properties with visualizations implemented
        # TODO:Add cell states in there later, need to figure out how to view proportions of different cell states in the same overlay
        self.visualization_properties = [
            'Area', 'Arterial Area', 'Average Cell Thickness', 'Average TBM Thickness', 'Cluster',
            'Luminal Fraction','Main_Cell_Types','Mesangial Area','Mesangial Fraction'
        ]

        # Initializing some parameters
        self.current_cell = 'PT'

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
        self.color_map = colormaps['jet']
        self.cell_vis_val = 0.5
        self.filter_vals = [0,1]

        self.ftu_style_handle = assign("""function(feature,context){
            const {color_key,current_cell,fillOpacity,ftu_color,filter_vals} = context.props.hideout;
            
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
                style.color = ftu_color;

            } else {
                style.display = none;
                style.fillOpacity = 0.0;
            }           
                                       
            return style;
            }
            """
        )

        self.ftu_filter = assign("""function(feature,context){
                const {color_key,current_cell,fillOpacity,ftu_color,filter_vals} = context.props.hideout;
                                 
                if (current_cell in feature.properties.Main_Cell_Types){
                    var cell_value = feature.properties.Main_Cell_Types[current_cell];
                    if (cell_value >= filter_vals[0]){
                        if (cell_value <= filter_vals[1]){
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else if (current_cell in feature.properties){
                    var cell_value = feature.properties[current_cell];
                    if (cell_value >= filter_vals[0]){
                        if (cell_value <= filter_vals[1]){
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return true;
                }
            }
            """
        )
        
        # Adding callbacks to app
        self.vis_callbacks()
        self.all_layout_callbacks()
        self.builder_callbacks()
        self.welcome_callbacks()
        self.upload_callbacks()

        # Running server
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

        # Updating GeoJSON fill/color/filter
        self.app.callback(
            [Output('layer-control','children'),Output('colorbar-div','children'),
             Output('filter-slider','max'),Output('filter-slider','disabled')],
            [Input('cell-drop','value'),Input('vis-slider','value'),
             Input('filter-slider','value'),Input({'type':'ftu-bound-color','index':ALL},'value')],
            prevent_initial_call = True
        )(self.update_overlays)

        # Updating Cell Composition pie charts
        self.app.callback(
            Output('roi-pie-holder','children'),
            [Input('slide-map','zoom'),Input('slide-map','viewport')],
            State('slide-map','bounds'),
        )(self.update_roi_pie)      

        # Updating cell hierarchy data
        self.app.callback(
            [Output('cell-graphic','src'),
             Output('cell-hierarchy','elements'),
             Output('cell-vis-drop','options'),
             Output('cell-graphic-name','children')],
            Input('neph-img','clickData'),
        )(self.update_cell_hierarchy)

        # Getting nephron hover (cell type label)
        self.app.callback(
            [Output('neph-tooltip','show'),
             Output('neph-tooltip','bbox'),
             Output('neph-tooltip','children')],
             Input('neph-img','hoverData')
        )(self.get_neph_hover)

        # Updating cell state bar plot based on clicked cell type in cell composition tab
        self.app.callback(
            Output({'type':'ftu-state-bar','index':MATCH},'figure'),
            Input({'type':'ftu-cell-pie','index':MATCH},'clickData'),
            prevent_initial_call = True
        )(self.update_state_bar)

        # Getting GeoJSON popup div with data
        self.app.callback(
            Output({'type':'ftu-popup','index':MATCH},'children'),
            Input({'type':'ftu-bounds','index':MATCH},'click_feature'),
            prevent_initial_call = True
        )(self.get_click_popup)

        # Loading new WSI from dropdown selection
        self.app.callback(
            [Output('slide-tile','url'),
             Output('layer-control','children'),
             Output({'type':'edit_control','index':ALL},'editToolbar'),
             Output('slide-map','center'),
             Output('slide-map','bounds'),
             Output('slide-tile','tileSize'),
             Output('slide-map','maxZoom'),
             Output('cell-drop','options'),
             Output('ftu-bound-opts','children')],
            Input('slide-select','value'),
            prevent_initial_call=True,
            suppress_callback_exceptions=True
        )(self.ingest_wsi)
        
        # Updating cytoscapes plot for cell hierarchy
        self.app.callback(
            [Output('label-p','children'),
            Output('id-p','children'),
            Output('notes-p','children')],
            Input('cell-hierarchy','tapNodeData'),
            prevent_initial_call=True
        )(self.get_cyto_data)

        # Updating morphometric cluster plot parameters
        self.app.callback(
            [Input('ftu-select','value'),
            Input('plot-select','value'),
            Input('label-select','value')],
            [Output('cluster-graph','figure'),
            Output('label-select','options')],
        )(self.update_graph)

        # Grabbing image(s) from morphometric cluster plot
        self.app.callback(
            [Input('cluster-graph','clickData'),
            Input('cluster-graph','selectedData')],
            [Output('selected-image','figure'),
            Output('selected-cell-types','figure'),
            Output('selected-cell-states','figure')],
        )(self.update_selected)

        # Updating cell states bar chart from within the selected point(s) in morphometrics cluster plot
        self.app.callback(
            Input('selected-cell-types','clickData'),
            Output('selected-cell-states','figure'),
            prevent_initial_call=True
        )(self.update_selected_state_bar)

        # Adding manual ROIs using EditControl
        self.app.callback(
            Input({'type':'edit_control','index':ALL},'geojson'),
            [Output('layer-control','children'),
             Output('data-select','options')],
            prevent_initial_call=True
        )(self.add_manual_roi)

        # Updating options for downloads
        self.app.callback(
            Input('data-select','value'),
            Output('data-options','children'),
            prevent_initial_call = True
        )(self.update_download_options)

        # Downloading data
        self.app.callback(
            [Input({'type':'download-opts','index':MATCH},'value'),
            Input({'type':'download-butt','index':MATCH},'n_clicks')],
            Output({'type':'download-data','index':MATCH},'data'),
            prevent_initial_call = True
        )(self.download_data)

        # Selecting and running CLIs
        self.app.callback(
            [Input('cli-drop','value'),
             Input('cli-run','n_clicks')],
            [Output('cli-descrip','children'),
             Output('cli-run','disabled'),
             Output('cli-current-image','children'),
             Output('cli-results','children'),
             Output('cli-results-followup','children')],
             prevent_initial_call = True
        )(self.run_analysis)

    def builder_callbacks(self):

        # Initializing plot after selecting dataset(s)
        self.app.callback(
            Input('dataset-table','selected_rows'),
            [Output('selected-dataset-slides','children'),
             Output('slide-metadata-plots','children'),
             Output('slide-select','options')],
             prevent_initial_call=True
        )(self.initialize_metadata_plots)

        # Updating metadata plot based on selected parameters
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
        
        # Selecting a specific tutorial video
        self.app.callback(
            Input({'type':'video-drop','index':ALL},'value'),
            Output({'type':'video','index':ALL},'src'),
            prevent_initial_call = True
        )(self.get_video)

    def upload_callbacks(self):

        # Creating upload components depending on omics type
        self.app.callback(
            Input('upload-type','value'),
            Output('upload-requirements','children'),
            prevent_initial_call=True
        )(self.update_upload_requirements)

        # Uploading data to DSA collection
        self.app.callback(
            [Input({'type':'wsi-upload','index':ALL},'contents'),
             Input({'type':'omics-upload','index':ALL},'contents')],
             [State({'type':'wsi-upload','index':ALL},'filename'),
              State({'type':'omics-upload','index':ALL},'filename')],
              [Output('slide-qc-results','children'),
               Output('slide-thumbnail-holder','children'),
               Output({'type':'wsi-upload','index':ALL},'disabled'),
               Output({'type':'omics-upload','index':ALL},'disabled'),
               Output('organ-type','disabled'),
               Output('post-upload-row','style')],
            prevent_initial_call=True
        )(self.upload_data)

        # Executing segmentation according to model selection
        self.app.callback(
            [Input('organ-type','value')],
            [Output('post-segment-row','style'),
             Output('organ-type','disabled'),
             Output('ftu-select','options'),
             Output('ftu-select','value'),
             Output('sub-comp-method','value'),
             Output('ex-ftu-img','figure')],
            prevent_initial_call=True
        )(self.apres_segmentation)

        # Updating sub-compartment segmentation
        self.app.callback(
            [Input('ftu-select','value'),
             Input('prev-butt','n_clicks'),
             Input('next-butt','n_clicks'),
             Input('go-to-feat','n_clicks'),
             Input('ex-ftu-view','value'),
             Input('ex-ftu-slider','value'),
             Input('sub-thresh-slider','value'),
             Input('sub-comp-method','value')],
             State('go-to-feat','disabled'),
             [Output('ex-ftu-img','figure'),
              Output('sub-thresh-slider','marks'),
              Output('feature-items','children'),
              Output('sub-thresh-slider','disabled'),
              Output('sub-comp-method','disabled'),
              Output('go-to-feat','disabled')],
            prevent_initial_call = True
        )(self.update_sub_compartment)

        # Enabling components after upload is complete

    def get_video(self,tutorial_category):
        tutorial_category = tutorial_category[0]
        print(f'tutorial_category: {tutorial_category}')

        # Pull tutorial videos from DSA 
        video_src = [f'./assets/videos/{tutorial_category}.mp4']

        return video_src

    def update_plotting_metadata(self):

        # Populating metadata based on current slides selection
        select_ids = [i['_id'] for i in self.current_slides if i['included']]

        metadata = self.dataset_handler.get_collection_annotation_meta(select_ids)

        return metadata

    def initialize_metadata_plots(self,selected_dataset_list):

        # Extracting metadata from selected datasets and plotting
        all_metadata_labels = []
        all_metadata = []
        slide_dataset_dict = []
        full_slides_list = []

        for d in selected_dataset_list:

            # Pulling dataset metadata
            # Dataset name in this case is associated with a folder or collection
            dataset_ids = list(self.dataset_handler.slide_datasets.keys())
            d_name = [self.dataset_handler.slide_datasets[i]['name'] for i in dataset_ids][d]
            d_id = dataset_ids[d]
            metadata_available = self.dataset_handler.slide_datasets[d_id]['Metadata']
            # This will store metadata, id, name, etc. for every slide in that dataset
            slides_list = [i for i in self.dataset_handler.slide_datasets[d_id]['Slides']]
            full_slides_list.extend(slides_list)

            slide_dataset_dict.extend([{'Slide Names':s['name'],'Dataset':d_name} for s in slides_list])

            # Grabbing dataset-level metadata
            metadata_available['FTU Expression Statistics'] = []

            all_metadata.append(metadata_available)
            all_metadata_labels.extend(list(metadata_available.keys()))

        #self.metadata = all_metadata
        all_metadata_labels = np.unique(all_metadata_labels)
        slide_dataset_df = pd.DataFrame.from_records(slide_dataset_dict)
        self.current_slides = []
        for i in full_slides_list:
            i['included'] = True
            self.current_slides.append(i)

        # Updating annotation metadata
        print(f'Getting plot metadata')
        self.metadata = self.update_plotting_metadata()
        print(f'Done')

        # Defining cell_type_dropdowns
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
                        dbc.Col(dcc.Dropdown(['By Dataset','By Slide'],'By Dataset',id={'type':'agg-meta-drop','index':0})),
                        dbc.Col(
                            dbc.Button('Go to Visualization!',id={'type':'go-to-vis-butt','index':0},color='success',href='/vis')
                        )
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
        
        if not new_meta == ['FTU Expression Statistics']:
            cell_types_turn_off = (True,True,True)
        else:
            cell_types_turn_off = (False, False, False)

        current_slide_count, slide_select_options = self.update_current_slides(slide_rows)

        print(f'new_meta: {new_meta}')
        if type(new_meta)==list:
            new_meta = new_meta[0]
        
        if type(group_type)==list:
            group_type = group_type[0]

        if type(ctx.triggered_id)==dict:
            if ctx.triggered_id['type']=='slide-dataset-table':
                self.metadata = self.update_plotting_metadata()

        # For DSA-backend deployment
        if not new_meta is None:
            # Filtering out de-selected slides
            present_slides = [s['name'] for s in self.current_slides]
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
                    
            slide_options = [{'label':i['name'],'value':i['name']} for i in self.current_slides if i['included']]

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

        #print(f'current viewport bounds: {bounds}')

        # Storing current slide boundaries
        self.current_slide_bounds = bounds_box

        # Getting a dictionary containing all the intersecting spots with this current ROI
        intersecting_ftus = {}
        if 'Spots' in self.current_ftu_layers:
            intersecting_spots = self.wsi.find_intersecting_spots(bounds_box)
            intersecting_ftus['Spots'] = intersecting_spots

        for ftu in self.current_ftu_layers:
            if not ftu=='Spots':
                intersecting_ftus[ftu] = self.wsi.find_intersecting_ftu(bounds_box,ftu)

        for m_idx,m_ftu in enumerate(self.wsi.manual_rois):
            intersecting_ftus[f'Manual ROI: {m_idx+1}'] = [m_ftu['geojson']['features'][0]['properties']]
                
        self.current_ftus = intersecting_ftus
        # Now we have main cell types, cell states, by ftu
        included_ftus = list(intersecting_ftus.keys())

        included_ftus = [i for i in included_ftus if len(intersecting_ftus[i])>0]


        if len(included_ftus)>0:

            tab_list = []
            #counts_data = pd.DataFrame()
            for f_idx,f in enumerate(included_ftus):
                counts_data = pd.DataFrame()

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
            for f in self.wsi.ftu_props:
                for g in self.wsi.ftu_props[f]:
                    if 'Cluster' in g:
                        cluster_label = g['Cluster']
                        raw_values_list.append(cluster_label)
        else:
            # For specific morphometrics
            for f in self.wsi.ftu_props:
                for g in self.wsi.ftu_props[f]:
                    if color_type in g:
                        morpho_value = g[color_type]
                        raw_values_list.append(morpho_value)

        raw_values_list = np.unique(raw_values_list)
        
        # Converting to RGB
        if len(raw_values_list)>0:
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
        else:
            self.hex_color_key = {}

    def update_overlays(self,cell_val,vis_val,filter_vals,ftu_color):

        m_prop = None

        if not ftu_color is None:
            # Getting these to align with the ftu-colors property order
            current_ftu_colors = list(self.ftu_colors.values())
            ftu_list = list(self.ftu_colors.keys())

            # Index of which ftu is different
            new_color = [i for i in range(len(current_ftu_colors)) if current_ftu_colors[i] not in ftu_color]
            if len(new_color)>0:
                print(new_color)
                for n in new_color:
                    check_for_new = [i for i in ftu_color if i not in current_ftu_colors]
                    if len(check_for_new)>0:
                        print(check_for_new)
                        self.ftu_colors[ftu_list[n]] = check_for_new[0]
                    
            self.filter_vals = filter_vals

        if not cell_val is None:
            # Extracting cell val if there are sub-properties
            if '-->' in cell_val:
                cell_val_parts = cell_val.split(' --> ')
                m_prop = cell_val_parts[0]
                cell_val = cell_val_parts[1]

            # Updating current_cell property
            if cell_val in self.cell_names_key:
                if m_prop == 'Main_Cell_Types':
                    self.current_cell = self.cell_names_key[cell_val]
                    self.update_hex_color_key('cell_value')

                    color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')
                    
                    filter_max_val = np.max(list(self.hex_color_key.keys()))
                    filter_disable = False
                
                elif m_prop == 'Cell_States':
                    self.current_cell = self.cell_names_key[cell_val]
                    self.update_hex_color_key('cell_state')

                    color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')

                    filter_max_val = np.max(list(self.hex_color_key.keys()))
                    filter_disable = False

            elif cell_val == 'Max Cell Type':
                self.current_cell = 'max'
                self.update_hex_color_key('max_cell')

                cell_types = list(self.wsi.geojson_ftus['features'][0]['properties']['Main_Cell_Types'].keys())
                color_bar = dlx.categorical_colorbar(categories = cell_types, colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')

                filter_max_val = 1.0
                filter_disable = True

            elif cell_val == 'Morphometrics Clusters':
                self.current_cell = 'cluster'
                self.update_hex_color_key('cluster')

                #TODO: This should probably be a categorical colorbar
                color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')

                filter_max_val = 1.0
                filter_disable = True
            
            else:
                # For other morphometric properties
                self.current_cell = cell_val
                self.update_hex_color_key(cell_val)

                color_bar = dl.Colorbar(colorscale = list(self.hex_color_key.values()),width=600,height=10,position='bottomleft',id=f'colorbar{random.randint(0,100)}')

                filter_max_val = np.max(list(self.hex_color_key.keys()))
                filter_disable = False

            self.cell_vis_val = vis_val/100

            new_children = [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = self.wsi.map_dict['FTUs'][struct]['geojson'], id = self.wsi.map_dict['FTUs'][struct]['id'], options = dict(style=self.ftu_style_handle,filter = self.ftu_filter),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity=self.cell_vis_val, ftu_color = self.ftu_colors[struct],filter_vals = self.filter_vals),
                                    hoverStyle = arrow_function(dict(weight=5, color = self.wsi.map_dict['FTUs'][struct]['hover_color'],dashArray='')),
                                    children = [dl.Popup(id=self.wsi.map_dict['FTUs'][struct]['popup_id'])])
                    ), name = struct, checked = True, id = self.wsi.item_id+'_'+struct
                )
                for struct in self.wsi.map_dict['FTUs']
            ]
            new_children += [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = self.wsi.spot_dict['geojson'], id = self.wsi.spot_dict['id'], options = dict(style = self.ftu_style_handle,filter = self.ftu_filter),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_color = self.ftu_colors['Spots'],filter_vals = self.filter_vals),
                                    hoverStyle = arrow_function(dict(weight=5,color=self.wsi.spot_dict['hover_color'],dashArray='')),
                                    children = [dl.Popup(id=self.wsi.spot_dict['popup_id'])],
                                    zoomToBounds=False)
                    ),name = 'Spots', checked = False, id = self.wsi.item_id+'_Spots'
                )
            ]

            for m_idx,man in enumerate(self.wsi.manual_rois):
                new_children.append(
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = man['geojson'], id = man['id'], options = dict(style = self.ftu_style_handle, filter = self.ftu_filter),
                                        hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_color = 'white',filter_vals = self.filter_vals),
                                        hoverStyle = arrow_function(dict(weight=5,color=man['hover_color'],dashArray='')),
                                        children = [dl.Popup(id=man['popup_id'])])
                        ), name = f'Manual ROI {m_idx+1}', checked = True, id = self.wsi.item_id+f'_manual_roi{m_idx}'
                    )
                )

            self.current_overlays = new_children
                        
            return new_children, color_bar, filter_max_val, filter_disable
        else:
            raise exceptions.PreventUpdate

    def update_cell_hierarchy(self,cell_clickData):
        # Loading the cell-graphic and hierarchy image
        cell_graphic = './assets/cell_graphics/default_cell_graphic.png'
        cell_hierarchy = [
                        {'data': {'id': 'one', 'label': 'Node 1'}, 'position': {'x': 75, 'y': 75}},
                        {'data': {'id': 'two', 'label': 'Node 2'}, 'position': {'x': 200, 'y': 200}},
                        {'data': {'source': 'one', 'target': 'two'}}
                    ]
        cell_state_droptions = []
        cell_name = html.H3('Default Cell')

        # Getting cell_val from the clicked location in the nephron diagram
        if not cell_clickData is None:
            
            pt = cell_clickData['points'][0]
            click_point = Point(pt['x'],pt['y'])

            # Checking if the clicked point is inside any of the cells bounding boxes
            possible_cells = [i for i in self.cell_graphics_key if len(self.cell_graphics_key[i]['bbox'])>0]
            intersecting_cell = [i for i in possible_cells if click_point.intersects(box(*self.cell_graphics_key[i]['bbox']))]
            
            if len(intersecting_cell)>0:
                cell_val = self.cell_graphics_key[intersecting_cell[0]]['full']
                cell_name = html.H3(cell_val)
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

    def get_click_popup(self,ftu_click):

        if not ftu_click is None:
            self.clicked_ftu = ftu_click

            if 'Main_Cell_Types' in ftu_click['properties']:

                # Getting the main cell type data (only using top-n)
                main_cell_types = ftu_click['properties']['Main_Cell_Types']
                chart_data = [main_cell_types[i] for i in main_cell_types]

                if not len(chart_data)==0:

                    # Only keeping the first self.plot_cell_n
                    top_idx = np.argsort(chart_data)[::-1][0:self.plot_cell_types_n]
                    chart_data = [chart_data[i] for i in top_idx]
                    chart_labels = [list(main_cell_types.keys())[i] for i in top_idx]
                    chart_full_labels = [self.cell_graphics_key[i]['full'] for i in chart_labels]

                    # Getting the cell state info for one of the cells and getting the names of the cells for a dropdown menu
                    cell_states = ftu_click['properties']['Cell_States']
                    cells_for_cell_states = list(cell_states.keys())

                    # Checking for non-zero cell states
                    non_zero_list = []
                    for cs in cells_for_cell_states:
                        if sum(list(cell_states[cs].values()))>0:
                            non_zero_list.append(cs)

                    cs_df_list = []
                    for nz_cs in non_zero_list:
                        cell_state_info = cell_states[nz_cs]
                        cell_state_df = pd.DataFrame({'States':list(cell_state_info.keys()),'Values':list(cell_state_info.values())})
                        cs_df_list.append(cell_state_df)

                    # Getting other FTU/Spot properties
                    all_properties = list(ftu_click['properties'].keys())
                    all_properties = [i for i in all_properties if not type(ftu_click['properties'][i])==dict]
                    all_props_dict = {'Property':all_properties,'Value':[ftu_click['properties'][i] for i in all_properties]}
                    all_properties_df = pd.DataFrame(all_props_dict)

                    # popup div
                    main_cells_df = pd.DataFrame.from_dict({'Values':chart_data,'Labels':chart_labels,'Full':chart_full_labels})
                    popup_div = html.Div([
                        dbc.Accordion([
                            dbc.AccordionItem([
                                html.Div([
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(
                                                figure=go.Figure(
                                                    data = [
                                                        go.Pie(
                                                            name = '',
                                                            values = main_cells_df['Values'],
                                                            labels = main_cells_df['Labels'],
                                                            customdata = main_cells_df['Full'],
                                                            hovertemplate = "Cell: %{customdata}: <br>Proportion: %{value}</br>"
                                                        )],
                                                    layout = {'autosize':True,'margin':{'t':0,'b':0,'l':0,'r':0},'showlegend':False}
                                                )),
                                            md='auto')
                                        ],style={'height':'100%','width':'100%'})
                                    ],style = {'height':'250px','width':'250px','display':'inline-block'})
                                ], title = 'Main Cell Types'),
                            dbc.AccordionItem([
                                dbc.Tabs([
                                    dbc.Tab(
                                        html.Div(
                                            dcc.Graph(
                                                figure = go.Figure(
                                                    data = [
                                                        go.Pie(
                                                            name = '',
                                                            values = cs_df['Values'],
                                                            labels = cs_df['States'],
                                                            hovertemplate = "State: %{label} <br>Proportion: %{value}</br>"
                                                        )
                                                    ],
                                                    layout = {'autosize':True,'margin':{'t':0,'b':0,'l':0,'r':0},'showlegend':False}
                                                )
                                            )
                                        ), label = cs
                                    )
                                    for cs,cs_df in zip(non_zero_list,cs_df_list)
                                ])
                            ], title = 'Cell States'),
                            dbc.AccordionItem([
                                html.Div([
                                    dash_table.DataTable(
                                        id = 'popup-table',
                                        columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in all_properties_df],
                                        data = all_properties_df.to_dict('records'),
                                        editable=False,                                        sort_mode='multi',
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
                                            } for row in all_properties_df.to_dict('records')
                                        ],
                                        tooltip_duration = None
                                    )
                                ])
                            ],title = 'Other Properties'),
                            dbc.AccordionItem([
                                html.Div([
                                    dcc.Input(type='text',placeholder='Notes',id={'type':'popup-notes','index':0})
                                ])
                            ],title = 'Custom Properties')
                        ])
                    ],style={'height':'300px','width':'300px','display':'inline-block'})

                    return popup_div
                else:
                    return html.Div([html.P('No cell type information')])
            else:
                return html.Div([html.P('No intersecting spots')])
        else:
            raise exceptions.PreventUpdate
        
    def update_state_popup(self,cell_type):
        print(f'cell_type: {cell_type}')
        if not cell_type is None:
            # Getting the cell state info for one of the cells and getting the names of the cells for a dropdown menu
            cell_states = self.clicked_ftu['properties']['Cell_States']
            cells_for_cell_states = list(cell_states.keys())
            initial_cell_states = cell_states[cells_for_cell_states[0]]
            cell_states_df = pd.DataFrame({'States':list(initial_cell_states.keys()),'Values':list(initial_cell_states.values())})

            cell_states_fig = dcc.Graph(
                figure = go.Figure(
                            data = [
                                go.Pie(
                                    name = '',
                                    values = cell_states_df['Values'],
                                    labels = cell_states_df['States']
                                )                    
                            ],
                            layout = {'autosize':True,'margin':{'t':0,'b':0,'l':0,'r':0}}
                        )      
            )      

            return cell_states_fig
        else:
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
        # Find folder containing this slide
        for d in self.dataset_handler.slide_datasets:
            d_slides = [i['name'] for i in self.dataset_handler.slide_datasets[d]['Slides']]
            if slide_name in d_slides:
                # Getting slide item id
                slide_id = self.dataset_handler.slide_datasets[d]['Slides'][d_slides.index(slide_name)]['_id']

        #TODO: Check for previous manual ROIs or marked FTUs
        new_slide = DSASlide(slide_name,slide_id,self.dataset_handler,self.ftu_colors,manual_rois=[],marked_ftus=[])
        self.wsi = new_slide

        # Updating overlays colors according to the current cell
        self.update_hex_color_key(self.current_cell)

        new_children = [
            dl.Overlay(
                dl.LayerGroup(
                    dl.GeoJSON(data = self.wsi.map_dict['FTUs'][struct]['geojson'], id = self.wsi.map_dict['FTUs'][struct]['id'], options = dict(style = self.ftu_style_handle, filter = self.ftu_filter),
                                hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, filter_vals = self.filter_vals),
                                hoverStyle = arrow_function(dict(weight=5, color = self.wsi.map_dict['FTUs'][struct]['hover_color'], dashArray = '')),
                                children = [dl.Popup(id=self.wsi.map_dict['FTUs'][struct]['popup_id'])])
                ), name = struct, checked = True, id = new_slide.item_id+'_'+struct
            )
            for struct in self.wsi.map_dict['FTUs']
        ]

        new_children += [
            dl.Overlay(
                dl.LayerGroup(
                    dl.GeoJSON(data = self.wsi.spot_dict['geojson'], id = self.wsi.spot_dict['id'], options = dict(style = self.ftu_style_handle,filter = self.ftu_filter),
                                hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, filter_vals = self.filter_vals),
                                hoverStyle = arrow_function(dict(weight=5, color = self.wsi.spot_dict['hover_color'], dashArray='')),
                                children = [dl.Popup(id=self.wsi.spot_dict['popup_id'])],
                                zoomToBounds=True),
                ), name = 'Spots', checked = False, id = new_slide.item_id+'_Spots'
            )
        ]

        # Now iterating through manual ROIs
        for m_idx, man in enumerate(self.wsi.manual_rois):
            new_children.append(
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = man['geojson'], id = man['id'], options = dict(style = self.ftu_style_handle,filter = self.ftu_filter),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors,filter_vals = self.filter_vals),
                                    hoverStyle = arrow_function(dict(weight=5,color=man['hover_color'], dashArray='')),
                                    children = [dl.Popup(id=man['popup_id'])])
                    ),
                    name = f'Manual ROI {m_idx}', checked = True, id = new_slide.item_id+f'_manual_roi{m_idx}'
                )
            )

        print(f'length of new_children: {len(new_children)}')

        new_url = self.wsi.tile_url
        center_point = [0.5*(self.wsi.map_bounds[0][0]+self.wsi.map_bounds[1][0]),0.5*(self.wsi.map_bounds[0][1]+self.wsi.map_bounds[1][1])]

        self.current_ftus = self.wsi.ftu_names+['Spots']
        self.current_ftu_layers = self.wsi.ftu_names+['Spots']

        # Adding the layers to be a property for the edit_control callback
        self.current_overlays = new_children

        # Removes manual ROIs added via dl.EditControl
        remove_old_edits = [{
            'mode':'remove',
            'n_clicks':0,
            'action':'clear all'
        }]

        # Populating FTU boundary options:
        combined_colors_dict = {}
        for f in self.wsi.map_dict['FTUs']:
            combined_colors_dict[f] = {'color':self.wsi.map_dict['FTUs'][f]['color']}
        
        combined_colors_dict['Spots'] = {'color':self.wsi.spot_dict['color']}

        boundary_options_children = [
            dbc.Tab(
                children = [
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                dmc.ColorPicker(
                                    id = {'type':'ftu-bound-color','index':idx},
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
            for idx, struct in enumerate(list(combined_colors_dict.keys()))
        ]

        return new_url, new_children, remove_old_edits, center_point, self.wsi.map_bounds, self.wsi.tile_dims[0], self.wsi.zoom_levels-1, self.wsi.properties_list, boundary_options_children

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
            print(f's: {s}')
            image_region = self.dataset_handler.get_annotation_image(s['slide_id'],s['layer_id'],s['annotation_id'])
            
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
                selected_image = go.Figure(
                    data = px.imshow(current_image[0])['data'],
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                    )
            elif len(current_image)>1:
                selected_image = go.Figure(
                    data = px.imshow(np.stack(current_image,axis=0),animation_frame=0,binary_string=True,labels=dict(animation_frame=self.current_ftu)),
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                    )
            else:
                print(f'No images found')
                print(f'hover: {hover}')
                print(f'selected:{selected}')
                print(f'self.current_selected_samples: {self.current_selected_samples}')

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
        
        triggered_id = ctx.triggered_id['type']
        print(f'triggered_id for add_manual_roi: {triggered_id}')
        print(new_geojson)
        if type(new_geojson)==list:
            new_geojson = new_geojson[0]

        if triggered_id == 'edit_control':
            if not new_geojson is None:
                if len(new_geojson['features'])>0:
                    if not new_geojson['features'][-1]['properties']['type']=='marker':
                        # Only getting the most recent to add
                        new_geojson = {'type':'FeatureCollection','features':[new_geojson['features'][len(self.wsi.manual_rois)]]}

                        # New geojson has no properties which can be used for overlays or anything so we have to add those
                        # Step 1, find intersecting spots:
                        overlap_spot_props = self.wsi.find_intersecting_spots(shape(new_geojson['features'][0]['geometry']))
                        
                        # Adding Main_Cell_Types from intersecting spots data
                        main_counts_data = pd.DataFrame.from_records([i['Main_Cell_Types'] for i in overlap_spot_props if 'Main_Cell_Types' in i]).sum(axis=0).to_frame()
                        main_counts_data = (main_counts_data/main_counts_data.sum()).fillna(0.000).round(decimals=18)
                        main_counts_data[0] = main_counts_data[0].map('{:.19f}'.format)
                        main_counts_dict = main_counts_data.astype(float).to_dict()[0]

                        # Aggregating cell state information from intersecting spots
                        agg_cell_states = {}
                        for m_c in list(main_counts_dict.keys()):

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
                                'popup_id':{'type':'ftu-popup','index':len(self.current_overlays)},
                                'color':'white',
                                'hover_color':'#32a852'
                            }
                        )

                        print(f'Length of wsi.manual_rois: {len(self.wsi.manual_rois)}')
                        # Updating the hex color key with new values
                        self.update_hex_color_key('cell_value')

                        new_child = dl.Overlay(
                            dl.LayerGroup(
                                dl.GeoJSON(data = new_geojson, id = {'type':'ftu-bounds','index':len(self.current_overlays)}, options = dict(style=self.ftu_style_handle),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, filter_vals = self.filter_vals),
                                    hoverStyle = arrow_function(dict(weight=5, color = '#32a852',dashArray = '')),
                                    children = [dl.Popup(id={'type':'ftu-popup','index':len(self.current_overlays)})]
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
                    
                    elif new_geojson['features'][-1]['properties']['type']=='marker':
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
                            dcc.Download(id={'type':'download-data','index':options_idx})
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
                            dcc.Download(id={'type':'download-data','index':options_idx})
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
                            dcc.Download(id={'type':'download-data','index':options_idx})
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
                            dcc.Download(id={'type':'download-data','index':options_idx})
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
                            dcc.Download(id={'type':'download-data','index':options_idx})
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

            # Getting current image region:
            wsi_coords = np.array(self.wsi.convert_map_coords(list(self.current_slide_bounds.exterior.coords)))
            min_x = np.min(wsi_coords[:,0])
            min_y = np.min(wsi_coords[:,1])
            max_x = np.max(wsi_coords[:,0])
            max_y = np.max(wsi_coords[:,1])
            image_region = self.dataset_handler.get_image_region(self.wsi.item_id,[min_x,min_y,max_x,max_y])

            current_image_region = html.Div(
                dcc.Graph(
                    figure = go.Figure(
                                data = px.imshow(image_region)['data'],
                                layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                                )
                    )
            )

            cli_results_followup = html.Div()

        elif ctx.triggered_id=='cli-run':
            
            # Running job:
            cli_results = 'And then the job would run'

            cli_results_followup = html.Div(
                dbc.Button('And this would tell you what to do next')
            )

            current_image_region = html.Div()
        
        return cli_description, cli_butt_disable, current_image_region, cli_results, cli_results_followup

    def update_upload_requirements(self,upload_type):
        
        input_disabled = True
        # Creating an upload div specifying which files are needed for a given upload type
        # Getting the collection id
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
        
            self.upload_check = {'WSI':False,'Omics':False}

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

            self.upload_check = {'WSI':False}

        else:
            upload_reqs = html.Div(
                'You should not have done that'
            )
        
        return upload_reqs, input_disabled

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

    def upload_data(self,wsi_file,omics_file,wsi_name,omics_name):

        # Posting contents based on the ctx.triggered_id
        print(f'wsi_file: {wsi_name}')
        print(f'omics_file: {omics_name}')
        wsi_file = wsi_file[0]
        omics_file = omics_file[0]
        wsi_name = wsi_name[0]
        omics_name = omics_name[0]

        wsi_disabled = False
        omics_disabled = False

        if ctx.triggered_id['type']=='wsi-upload' and not wsi_file is None:
            self.upload_item_id = self.dataset_handler.upload_data(wsi_file,wsi_name)

            self.upload_check['WSI'] = True
            wsi_disabled = True

        elif ctx.triggered_id['type']=='omics-upload' and not omics_file is None:

            self.upload_item_id = self.dataset_handler.upload_data(omics_file,omics_name)
            
            self.upload_check['Omics'] = True
            omics_disabled = True
        else:
            print(f'ctx.triggered_id["type"]: {ctx.triggered_id["type"]}')

        print(self.upload_check)
        # Checking the upload check
        if all([self.upload_check[i] for i in self.upload_check]):
            print('All set!')
            wsi_disabled = True
            omics_disabled = True

            slide_thumbnail, slide_qc_table = self.slide_qc(self.upload_item_id)
            print(slide_qc_table)

            thumb_fig = dcc.Graph(
                figure=go.Figure(
                    data = px.imshow(slide_thumbnail)['data'],
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0},'height':200,'width':200}
                )
            )

            slide_qc_results = dash_table.DataTable(
                id = {'type':'slide-qc-table','index':0},
                columns = [{'name':i, 'id': i, 'deletable':False, 'selectable':True} for i in slide_qc_table],
                data = slide_qc_table.to_dict('records'),
                editable=False,
                filter_action='native',
                sort_action='native',
                sort_mode='multi',
                column_selectable = 'single',
                row_selectable = 'multi',
                row_deletable = False,
                selected_columns = [],
                selected_rows = [],
                page_action = 'native',
                page_current = 0,
                page_size = 10,
                style_cell = {
                    'overflow':'hidden',
                    'textOverflow':'ellipsis',
                    'maxWidth':0
                },
                tooltip_data = [
                    {
                        column:{'value':str(value), 'type':'markdown'}
                        for column,value in row.items()
                    } for row in slide_qc_table.to_dict('records')
                ],
                tooltip_duration = None
            )

            organ_type_disabled = False
            post_upload_style = {'display':'flex'}

            return slide_qc_results, thumb_fig, [wsi_disabled], [omics_disabled],organ_type_disabled, post_upload_style
        
        else:
            return no_update, no_update, [wsi_disabled], [omics_disabled],no_update, no_update

    def slide_qc(self, upload_id):

        #try:
        #collection_contents = self.dataset_handler.get_collection_items(upload_id)
        thumbnail = self.dataset_handler.get_slide_thumbnail(upload_id)
        collection_contents = self.dataset_handler.gc.get(f'/item/{upload_id}')
        print(collection_contents)

        #TODO: Activate the HistoQC plugin from here and return some metrics
        histo_qc_output = pd.DataFrame(collection_contents)

        return thumbnail, histo_qc_output

    def apres_segmentation(self,organ_selection):

        if not organ_selection is None:

            # Executing segmentation CLI for organ/model/FTU selections
            sub_comp_style = {'display':'flex'}
            disable_organ = True

            print(f'Running segmentation!')
            try:
                segmentation_info = self.prep_handler.segment_image(self.upload_item_id,organ_selection)
            except girder_client.HttpError:
                print('Error running job')

            # Extract annotation and initial sub-compartment mask
            self.upload_annotations = self.dataset_handler.get_annotations(self.upload_item_id)

            # Populate with default sub-compartment parameters
            self.sub_compartment_params = self.prep_handler.initial_segmentation_parameters

            # Adding options to FTU Options dropdown menu
            ftu_names = []
            for idx,i in enumerate(self.upload_annotations):
                if 'annotation' in i:
                    if 'elements' in i['annotation']:
                        if not 'interstitium' in i['annotation']['name']:
                            if len(i['annotation']['elements'])>0:
                                ftu_names.append({
                                    'label':i['annotation']['name'],
                                    'value':idx,
                                    'disabled':False
                                })
                            else:
                                ftu_names.append({
                                    'label':i['annotation']['name']+' (None detected in slide)',
                                    'value':idx,
                                    'disabled':True
                                })
                        else:
                            ftu_names.append({
                                'label':i['annotation']['name']+' (Not implemented for interstitium)',
                                'value':idx,
                                'disabled':True
                            })

            # Initializing layer and annotation idxes (starting with the first one that isn't disabled)
            self.layer_ann = {
                'current_layer':[i['value'] for i in ftu_names if not i['disabled']][0],
                'current_annotation':0,
                'previous_annotation':0,
                'max_layers':[len(i['annotation']['elements']) for i in self.upload_annotations if 'annotation' in i]
            }
            
            self.feature_extract_ftus = ftu_names
            image, mask = self.prep_handler.get_annotation_image_mask(self.upload_item_id,self.upload_annotations,self.layer_ann['current_layer'],self.layer_ann['current_annotation'])

            self.layer_ann['current_image'] = image
            self.layer_ann['current_mask'] = mask

            image_figure = go.Figure(
                data = px.imshow(image)['data'],
                layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                )

            return sub_comp_style, disable_organ, ftu_names, ftu_names[self.layer_ann['current_layer']],'Manual',image_figure
        
        else:
            raise exceptions.PreventUpdate
        
    def update_sub_compartment(self,select_ftu,prev,next,go_to_feat,ex_ftu_view,ftu_slider,thresh_slider,sub_method,go_to_feat_state):

        new_ex_ftu = go.Figure()
        feature_extract_children = []
        go_to_feat_disabled = go_to_feat_state
        disable_slider = go_to_feat_state
        disable_method = go_to_feat_state

        slider_marks = {
            val:{'label':f'{sub_comp["name"]}: {val}','style':{'color':sub_comp["marks_color"]}}
            for val,sub_comp in zip(thresh_slider[::-1],self.sub_compartment_params)
        }

        for idx,ftu,thresh in zip(list(range(len(self.sub_compartment_params))),self.sub_compartment_params,thresh_slider[::-1]):
            ftu['threshold'] = thresh
            self.sub_compartment_params[idx] = ftu

        if ctx.triggered_id=='next-butt':
            # Moving to next annotation in current layer
            self.layer_ann['previous_annotation'] = self.layer_ann['current_annotation']

            if self.layer_ann['current_annotation']+1>=self.layer_ann['max_layers'][self.layer_ann['current_layer']]:
                self.layer_ann['current_annotation'] = 0
            else:
                self.layer_ann['current_annotation'] += 1

        elif ctx.triggered_id=='prev-butt':
            # Moving back to previous annotation in current layer
            self.layer_ann['previous_annotation'] = self.layer_ann['current_annotation']

            if self.layer_ann['current_annotation']==0:
                self.layer_ann['current_annotation'] = self.layer_ann['max_layers'][self.layer_ann['current_layer']]-1
            else:
                self.layer_ann['current_annotation'] -= 1
        
        elif ctx.triggered_id=='ftu-select':
            # Moving to next annotation layer, restarting annotation count
            if type(select_ftu)==dict:
                self.layer_ann['current_layer']=select_ftu['value']
            elif type(select_ftu)==int:
                self.layer_ann['current_layer'] = select_ftu

            self.layer_ann['current_annotation'] = 0
            self.layer_ann['previous_annotation'] = self.layer_ann['max_layers'][self.layer_ann['current_layer']]

        if ctx.triggered_id not in ['go-to-feat','ex-ftu-slider','sub-comp-method']:
            
            new_image, new_mask = self.prep_handler.get_annotation_image_mask(self.upload_item_id,self.upload_annotations,self.layer_ann['current_layer'],self.layer_ann['current_annotation'])
            self.layer_ann['current_image'] = new_image
            self.layer_ann['current_mask'] = new_mask

        if ctx.triggered_id not in ['go-to-feat']:
            
            sub_compartment_image = self.prep_handler.sub_segment_image(self.layer_ann['current_image'],self.layer_ann['current_mask'],self.sub_compartment_params,ex_ftu_view,ftu_slider)

            new_ex_ftu = go.Figure(
                data = px.imshow(sub_compartment_image)['data'],
                layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
            )
        else:
            go_to_feat_disabled = True
            disable_slider = True
            disable_method = True

            new_ex_ftu = go.Figure(
                data = px.imshow(self.prep_handler.current_sub_comp_image)['data'],
                layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
            )

            feature_extract_children = self.prep_handler.gen_feat_extract_card(self.feature_extract_ftus)

        if go_to_feat_state:
            return new_ex_ftu, slider_marks, no_update, disable_slider, disable_method, go_to_feat_disabled
        else:
            return new_ex_ftu, slider_marks, feature_extract_children, disable_slider, disable_method, go_to_feat_disabled

    

def app(*args):
    
    # Using DSA as base directory for storage and accessing files
    dsa_url = 'http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/'
    username = os.environ.get('DSA_USER')
    p_word = os.environ.get('DSA_PWORD')

    # Initializing GirderHandler
    dataset_handler = GirderHandler(apiUrl=dsa_url,username=username,password=p_word)

    # Initial collection
    initial_collection = '/collection/10X_Visium'
    path_type = 'collection'
    print(f'initial collection: {initial_collection}')
    initial_collection_id = dataset_handler.gc.get('resource/lookup',parameters={'path':initial_collection})

    print(f'loading initial slide(s)')
    # Contents of folder (used for testing to initialize with one slide)
    initial_collection_contents = dataset_handler.gc.get(f'resource/{initial_collection_id["_id"]}/items',parameters={'type':path_type})
    initial_collection_contents = [i for i in initial_collection_contents if 'largeImage' in i]

    # For testing, setting initial slide
    initial_collection_contents = initial_collection_contents[0:2]
    
    # Saving & organizing relevant id's in GirderHandler
    print('Getting initial items metadata')
    dataset_handler.initialize_folder_structure(initial_collection,path_type)
    metadata = dataset_handler.get_collection_annotation_meta([i['_id'] for i in initial_collection_contents])

    # Getting graphics_reference.json from the FUSION Assets folder
    print(f'Getting asset items')
    assets_path = '/collection/FUSION Assets/'
    dataset_handler.get_asset_items(assets_path)

    # Getting the slide data for DSASlide()
    slide_name = initial_collection_contents[0]['name']
    slide_item_id = initial_collection_contents[0]['_id']
    slide_names = [i['name'] for i in initial_collection_contents if 'largeImage' in i]

    # Initializing FTU Colors
    ftu_colors = {
        'Glomeruli':'#390191',
        'Tubules':'#e71d1d',
        'Arterioles':'#b6d7a8',
        'Spots':'#dffa00'
    }

    print(f'Initializing DSA Slide: {slide_name}')
    wsi = DSASlide(slide_name,slide_item_id,dataset_handler,ftu_colors)

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

    # Adding functionality that is specifically implemented in FUSION
    fusion_cli = ['Segment Anything Model (SAM)','Contrastive Language-Image Pre-training (CLIP)']

    external_stylesheets = [dbc.themes.LUX,dbc.icons.BOOTSTRAP]

    print(f'Generating layouts')
    layout_handler = LayoutHandler()
    layout_handler.gen_initial_layout(slide_names)
    layout_handler.gen_vis_layout(wsi,cli_list)
    layout_handler.gen_builder_layout(dataset_handler)
    layout_handler.gen_uploader_layout(dataset_handler)

    download_handler = DownloadHandler(dataset_handler)

    prep_handler = PrepHandler(dataset_handler)

    main_app = DashProxy(__name__,external_stylesheets=external_stylesheets,transforms = [MultiplexerTransform()])
    vis_app = FUSION(
        main_app,
        layout_handler,
        dataset_handler,
        download_handler,
        prep_handler,
        wsi,
        metadata
    )

# Comment this portion out for web running
if __name__=='__main__':
    app()
