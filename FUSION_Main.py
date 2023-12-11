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
from datetime import datetime

from tqdm import tqdm

import shapely
from shapely.geometry import Point, shape, box
from skimage.transform import resize
import random

from umap import UMAP

from uuid import uuid4
import textwrap

import girder_client

import plotly.express as px
import plotly.graph_objects as go
from matplotlib import colormaps

from dash import dcc, ctx, MATCH, ALL, dash_table, exceptions, callback_context, no_update, DiskcacheManager

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, arrow_function
from dash_extensions.enrich import DashProxy, html, Input, Output, MultiplexerTransform, State
import dash_mantine_components as dmc

from timeit import default_timer as timer
import time

from FUSION_WSI import DSASlide
from FUSION_Handlers import LayoutHandler, DownloadHandler, GirderHandler
from FUSION_Prep import PrepHandler

from upload_component import UploadComponent

from waitress import serve


class FUSION:
    def __init__(self,
                app,
                layout_handler,
                dataset_handler,
                download_handler,
                prep_handler
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

        # clustering related properties (and also cell types, cell states, image_ids, etc.)
        self.cell_graphics_key = self.dataset_handler.cell_graphics_key

        # Inverting the graphics key to get {'full_name':'abbreviation'}
        self.cell_names_key = {}
        for ct in self.cell_graphics_key:
            self.cell_names_key[self.cell_graphics_key[ct]['full']] = ct

        # Getting morphometrics reference from dataset_handler
        self.morphometrics_reference = self.dataset_handler.morphometrics_reference["Morphometrics"]
        self.morphometrics_names = self.dataset_handler.morpho_names

        # Load clustering data
        self.clustering_data = pd.DataFrame()
        self.filter_labels = []
        self.umap_df = None
        self.reports_generated = {}

        # Number of main cell types to include in pie-charts (currently set to all cell types)
        self.plot_cell_types_n = len(list(self.cell_names_key.keys()))

        # ASCT+B table for cell hierarchy generation
        self.table_df = self.dataset_handler.asct_b_table    

        # FTU settings
        self.wsi = None
        self.current_slides = self.dataset_handler.default_slides
        for i in self.current_slides:
            i['included'] = True
        if not self.wsi is None:
            self.ftus = self.wsi.ftu_names
            self.ftu_colors = self.wsi.ftu_colors

            self.current_ftu_layers = self.wsi.ftu_names
        else:
            # Initialization of these properties
            self.ftu_colors = {
                'Glomeruli':'#390191',
                'Tubules':'#e71d1d',
                'Arterioles':'#b6d7a8',
                'Spots':'#dffa00'
            }
            self.ftus = []
            self.current_ftu_layers = []

        # Specifying available properties with visualizations implemented
        self.visualization_properties = [
            'Area', 'Arterial Area', 'Average Cell Thickness', 'Average TBM Thickness', 'Cluster',
            'Luminal Fraction','Main_Cell_Types','Mesangial Area','Mesangial Fraction','Max Cell Type'
        ]

        # Initializing some parameters
        self.current_cell = None

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
        self.hex_color_key = {}

        # JavaScript functions for controlling annotation properties
        self.ftu_style_handle = assign("""function(feature,context){
            const {color_key,current_cell,fillOpacity,ftu_colors,filter_vals} = context.hideout;
            if (current_cell){
                if (current_cell==='cluster'){
                    if (current_cell in feature.properties){
                        var cell_value = feature.properties.Cluster;
                        cell_value = (Number(cell_value)).toFixed(1);
                    } else {
                        cell_value = Number.Nan;
                    }
                } else if (current_cell==='max'){
                    // Extracting all the cell values for a given FTU/Spot
                    if ("Main_Cell_Types" in feature.properties){       
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
                    } else {
                        cell_value = Number.Nan;
                    }
                } else if ('Main_Cell_Types' in feature.properties){
                    if (current_cell in feature.properties.Main_Cell_Types){
                                        
                        var cell_value = feature.properties.Main_Cell_Types[current_cell];
                        if (cell_value==1) {
                            cell_value = (cell_value).toFixed(1);
                        } else if (cell_value==0) {
                            cell_value = (cell_value).toFixed(1);
                        }                                                    
                    } else if (current_cell in feature.properties){
                        var cell_value = feature.properties[current_cell];
                    } else if (current_cell.includes('_')){
                        
                        var split_cell_value = current_cell.split("_");
                        var main_cell_value = split_cell_value[0];
                        var sub_cell_value = split_cell_value[1];
                        
                        var cell_value = feature.properties.Main_Cell_Types[main_cell_value];
                        cell_value *= feature.properties.Cell_States[main_cell_value][sub_cell_value];
                        
                        if (cell_value==1) {
                            cell_value = (cell_value).toFixed(1);
                        } else if (cell_value==0) {
                            cell_value = (cell_value).toFixed(1);
                        } 
                    } else {
                        var cell_value = Number.Nan;
                    }
                } else {
                    var cell_value = Number.Nan;
                }
            } else {
                var cell_value = Number.Nan;
            }

            var style = {};
            if (cell_value == cell_value) {
                const fillColor = color_key[cell_value];

                style.fillColor = fillColor;
                style.fillOpacity = fillOpacity;
                if (feature.properties.name in ftu_colors){
                    style.color = ftu_colors[feature.properties.name];
                } else {
                    style.color = 'white';
                }

            } else {
                if (feature.properties.name in ftu_colors){
                    style.color = ftu_colors[feature.properties.name];
                } else {
                    style.color = 'white';
                }
            }           
                                                                              
            return style;
            }
            """
        )

        self.ftu_filter = assign("""function(feature,context){
                const {color_key,current_cell,fillOpacity,ftu_colors,filter_vals} = context.hideout;
                
                if (current_cell){
                    if ("Main_Cell_Types" in feature.properties){
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
                        } else if (current_cell.includes('_')) {
                            var split_cell_value = current_cell.split("_");
                            var main_cell_value = split_cell_value[0];
                            var sub_cell_value = split_cell_value[1];
                            
                            var cell_value = feature.properties.Main_Cell_Types[main_cell_value];
                            cell_value *= feature.properties.Cell_States[main_cell_value][sub_cell_value];
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
                } else {
                    return true;
                }
            }
            """
        )

        self.render_marker_handle = assign("""function(feature,latlng,context){
            const p = feature.properties;
            if (p.type === 'marker') {
                return L.marker(latlng);
            } else {
                return true;
            }
        }
        """)

        # Adding callbacks to app
        self.vis_callbacks()
        self.all_layout_callbacks()
        self.builder_callbacks()
        self.welcome_callbacks()
        self.upload_callbacks()

        # Running server
        #self.app.run_server(host = '0.0.0.0',debug=False,use_reloader=False,port=8000)
        serve(self.app.server,host='0.0.0.0',port=8000)

    def view_instructions(self,n,n2,is_open):
        # Opening collapse and populating internal div 
        if ctx.triggered_id['type']=='collapse-descrip':
            collapse_children = self.layout_handler.description_dict[self.current_page]
        elif ctx.triggered_id['type']=='usability-butt':
            user_info = self.dataset_handler.check_usability(self.dataset_handler.username)
            collapse_children = [
                html.P(f'Your user type is: {user_info["type"]}')
            ]

        if n or n2:
            return [not i for i in is_open], collapse_children
        return [i for i in is_open], collapse_children
    
    def view_sidebar(self,n,is_open):
        if n:
            return not is_open
        return is_open

    def update_page(self,pathname):
        
        pathname = pathname[1:]
        print(f'Navigating to {pathname}')

        slide_select_value = ''

        self.dataset_handler.update_usability()

        if pathname in self.layout_handler.layout_dict:
            self.current_page = pathname

            if not pathname=='vis':
                if not pathname=='dataset-builder':
                    if pathname=='dataset-uploader':
                        # Double-checking that a user is logged in before giving access to dataset-uploader
                        if self.dataset_handler.username=='fusionguest':
                            self.current_page = 'welcome'
                    container_content = self.layout_handler.layout_dict[self.current_page]
                else:
                    # Checking if there was any new slides added via uploader (or just added externally)
                    self.dataset_handler.update_folder_structure()
                    self.layout_handler.gen_builder_layout(self.dataset_handler)

                    container_content = self.layout_handler.layout_dict[self.current_page]

            else:
                self.wsi = None
                self.filter_vals = [0,1]
                self.layout_handler.gen_vis_layout(
                    self.wsi
                    )
                self.clustering_data = pd.DataFrame()

                container_content = self.layout_handler.layout_dict[self.current_page]

        else:
            self.current_page = 'welcome'
                
            container_content = self.layout_handler.layout_dict[self.current_page]

        if self.current_page == 'vis':
            slide_style = {'marginBottom':'20px','display':'inline-block'}
        else:
            slide_style = {'display':'none'}

        return container_content, slide_style, slide_select_value

    def open_nav_collapse(self,n,is_open):
        if n:
            return not is_open
        return is_open

    def all_layout_callbacks(self):

        # Adding callbacks for items in every page
        self.app.callback(
            [Output('container-content','children'),
             Output('slide-select-card','style'),
             Output('slide-select','value')],
             Input('url','pathname'),
             prevent_initial_call = True
        )(self.update_page)

        self.app.callback(
            [Output({'type':'collapse-content','index':ALL},'is_open'),
             Output('descrip','children')],
            [Input({'type':'collapse-descrip','index':ALL},'n_clicks'),
             Input({'type':'usability-butt','index':ALL},'n_clicks')],
            [State({'type':'collapse-content','index':ALL},'is_open')],
            prevent_initial_call=True
        )(self.view_instructions)

        self.app.callback(
            Output('navbar-collapse','is_open'),
            Input('navbar-toggler','n_clicks'),
            State('navbar-collapse','is_open')
        )(self.open_nav_collapse)

        self.app.callback(
            Output({'type':'sidebar-offcanvas','index':MATCH},'is_open'),
            Input({'type':'sidebar-button','index':MATCH},'n_clicks'),
            [State({'type':'sidebar-offcanvas','index':MATCH},'is_open')],
            prevent_initial_call=True
        )(self.view_sidebar)

        self.app.callback(
            [Output('login-submit','color'),
             Output('login-submit','children'),
             Output('logged-in-user','children'),
             Output('upload-sidebar','disabled'),
             Output('create-user-extras','children'),
             Output({'type':'usability-butt','index':ALL},'disabled')],
            [Input('login-submit','n_clicks'),
             Input('create-user-submit','n_clicks')],
            [State('username-input','value'),
             State('pword-input','value'),
             State({'type':'email-input','index':ALL},'value'),
             State({'type':'first-name-input','index':ALL},'value'),
             State({'type':'last-name-input','index':ALL},'value')],
             prevent_initial_call=True
        )(self.girder_login)

    def vis_callbacks(self):

        # Updating GeoJSON fill/color/filter
        self.app.callback(
            [Output({'type':'ftu-bounds','index':ALL},'hideout'),Output('colorbar-div','children'),
             Output('filter-slider','max'),Output('filter-slider','disabled'),
             Output('cell-sub-select-div','children')],
            [Input('cell-drop','value'),Input('vis-slider','value'),
             Input('filter-slider','value'),Input({'type':'ftu-bound-color','index':ALL},'value'),
             Input({'type':'cell-sub-drop','index':ALL},'value')],
            prevent_initial_call = True
        )(self.update_overlays)

        # Updating Cell Composition pie charts
        self.app.callback(
            Output('roi-pie-holder','children'),
            Input('slide-map','bounds'),
            State('tools-tabs','active_tab')
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
            Input({'type':'ftu-bounds','index':MATCH},'clickData'),
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
            [Input('gen-plot-butt','n_clicks'),
             Input('label-select','value')],
            [Output('cluster-graph','figure'),
             Output('label-info','children'),
             Output('filter-info','children'),
             Output('plot-report-tab','active_tab'),
             Output('download-plot-butt','disabled')],
            [State('feature-select-tree','checked'),
             State('filter-select-tree','checked'),
             State('cell-states-clustering','value')],
            prevent_initial_call=True
        )(self.update_graph)

        # Grabbing image(s) from morphometric cluster plot
        self.app.callback(
            [Input('cluster-graph','clickData'),
            Input('cluster-graph','selectedData')],
            [Output('selected-image','figure'),
            Output('selected-cell-types','figure'),
            Output('selected-cell-states','figure'),
            Output('selected-image-info','children')],
            prevent_initial_call=True
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

        # Add histology marker from clustering/plot
        self.app.callback(
            Input({'type':'add-mark-cluster','index':ALL},'n_clicks'),
            [Output({'type':'edit_control','index':ALL},'geojson'),
             Output('marker-add-div','children')],
            prevent_initial_call=True
        )(self.add_marker_from_cluster)

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
        """
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
        """

        # Callback for Ask Fusey
        self.app.callback(
            [Output('ask-fusey-box','children'),
             Output('ask-fusey-box','style')],
            Input('fusey-button','n_clicks'),
            State('ask-fusey-box','style'),
            prevent_initial_call = True
        )(self.ask_fusey)

        # Adding labels from popup boxes
        self.app.callback(
            Output({'type':'added-labels-div','index':ALL},'children'),
            [Input({'type':'add-popup-note','index':ALL},'n_clicks'),
             Input({'type':'delete-user-label','index':ALL},'n_clicks')],
            State({'type':'popup-notes','index':ALL},'value'),
            prevent_initial_call = True
        )(self.add_label)

        # Checking if there is aligned clustering data currently available
        self.app.callback(
            [Output('get-data-div','children'),
             Output('feature-select-tree','data'),
             Output('label-select','disabled'),
             Output('label-select','options'),
             Output('label-select','value'),
             Output('filter-select-tree','data'),
             Output('download-plot-data-div','style')],
            [Input('tools-tabs','active_tab'),
             Input({'type':'get-clustering-butt','index':ALL},'n_clicks')],
            prevent_initial_call=True
        )(self.populate_cluster_tab)

        # Updating contents of plot-report tabs when switched
        self.app.callback(
            Output('plot-report-div','children'),
            Input('plot-report-tab','active_tab'),
            prevent_initial_call = True
        )(self.update_plot_report)

        # Downloading data in the current plot
        self.app.callback(
            Output('download-plot-data','data'),
            Input('download-plot-butt','n_clicks'),
            prevent_initial_call = True
        )(self.download_plot_data)

        # Find cluster markers button clicked and return dcc.Interval object
        self.app.callback(
            [Output({'type':'cluster-marker-div','index':ALL},'children'),
             Output({'type':'cluster-markers-butt','index':ALL},'disabled')],
            Input({'type':'cluster-markers-butt','index':ALL},'n_clicks'),
            prevent_initial_call = True
        )(self.start_cluster_markers)

        # Updating logs from cluster markers job
        self.app.callback(
            [Output({'type':'markers-interval','index':ALL},'disabled'),
             Output({'type':'marker-logs-div','index':ALL},'children')],
            Input({'type':'markers-interval','index':ALL},'n_intervals')
        )(self.update_cluster_logs)

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
             Output({'type':'cell-meta-drop','index':ALL},'options'),
             Output({'type':'cell-meta-drop','index':ALL},'disabled'),
             Output({'type':'current-slide-count','index':ALL},'children')],
             prevent_initial_call = True
        )(self.update_metadata_plot)

    def welcome_callbacks(self):
        
        # Selecting a specific tutorial video
        self.app.callback(
            Input({'type':'video-drop','index':ALL},'value'),
            [Output({'type':'video','index':ALL},'src'),
             Output('tutorial-content','children')],
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
            [Input({'type':'wsi-upload','index':ALL},'uploadComplete'),
             Input({'type':'omics-upload','index':ALL},'uploadComplete'),
             Input({'type':'wsi-upload','index':ALL},'fileTypeFlag'),
             Input({'type':'omics-upload','index':ALL},'fileTypeFlag')],
            [Output('slide-qc-results','children'),
             Output('slide-thumbnail-holder','children'),
             Output({'type':'wsi-upload-div','index':ALL},'children'),
             Output({'type':'omics-upload-div','index':ALL},'children'),
             Output('structure-type','disabled'),
             Output('post-upload-row','style'),
             Output('upload-type','disabled')],
            prevent_initial_call=True
        )(self.upload_data)

        # Starting segmentation for selected structures
        self.app.callback(
            output = [
                Output('seg-woodshed','children'),
                Output('structure-type','disabled'),
                Output('segment-butt','disabled')
            ],
            inputs = [
                Input('structure-type','value'),
                Input('segment-butt','n_clicks')
            ],
            prevent_initial_call = True
        )(self.start_segmentation)

        #TODO: Add callback here for uploading annotations
        # Uploading annotations from text file
        self.app.callback(
            Output('upload-anns-div','children'),
            Input({'type':'create-ann-upload','index':ALL},'n_clicks'),
            prevent_initial_call=True
        )(self.create_seg_upload)

        # Adding new upload to item annotations
        self.app.callback(
            Output({'type':'seg-file-accordion','index':ALL},'children'),
            Input({'type':'seg-file-upload','index':ALL},'contents'),
            [State({'type':'seg-file-accordion','index':ALL},'children'),
             State({'type':'seg-file-upload','index':ALL},'filename')],
            prevent_initial_call = True
        )(self.new_seg_upload)

        # Updating log output for segmentation
        self.app.callback(
            output = [
                Output({'type':'seg-logs','index':ALL},'children'),
                Output({'type':'seg-log-interval','index':ALL},'disabled')
            ],
            inputs = [
                Input({'type':'seg-log-interval','index':ALL},'n_intervals')
            ],
            prevent_initial_call = True
        )(self.update_logs)

        # Populating the post-segmentation sub-compartment segmentation and feature extraction row
        self.app.callback(
            output = [
                Output('post-segment-row','style'),
                Output('structure-type','disabled'),
                Output('ftu-select','options'),
                Output('ftu-select','value'),
                Output('sub-comp-method','value'),
                Output('ex-ftu-img','figure')
            ],
            inputs = [
                Input({'type':'seg-log-interval','index':ALL},'disabled')
            ],
            prevent_initial_call = True
        )(self.post_segmentation)

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

        # Running feature extraction plugin
        self.app.callback(
            Input({'type':'start-feat','index':ALL},'n_clicks'),
            Output({'type':'feat-logs','index':ALL},'children'),
            prevent_initial_call=True
        )(self.run_feature_extraction)

        # Updating logs for feature extraction
        self.app.callback(
            output = [
                Output({'type':'feat-interval','index':ALL},'disabled'),
                Output({'type':'feat-log-output','index':ALL},'children')
            ],
            inputs = [
                Input({'type':'feat-interval','index':ALL},'n_intervals')
            ],
            prevent_initial_call = True
        )(self.update_feat_logs)

    def get_video(self,tutorial_category):
        tutorial_category = tutorial_category[0]
        print(f'tutorial_category: {tutorial_category}')

        # Pull tutorial videos from DSA 
        video_src = [f'./assets/videos/{tutorial_category}.mp4']

        #TODO: Add some markdown tutorial/Readme-like thing for each of the videos below
        tutorial_children = [
            tutorial_category
        ]

        return video_src, tutorial_children

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
            metadata_available['FTU Morphometrics'] = []

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
        #self.update_plotting_metadata()

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
            if not new_meta == ['FTU Morphometrics']:
                # This is for expression statistics
                cell_types_turn_off = (True,True,True)
            else:
                # This is for morphometrics statistics
                cell_types_turn_off = (True,False,False)
        else:
            # This is for everything else
            cell_types_turn_off = (False, False, False)

        try:
            if not ctx.triggered_id['type']=='meta-drop':
                cell_types_options = (no_update,no_update,no_update)
            else:
                print(new_meta)
                if new_meta==['FTU Expression Statistics']:
                    # These are the options for expression statistics
                    cell_types_options = (
                        ['Main Cell Types','Cell States'],
                        ['Mean','Median','Sum','Standard Deviation','Nonzero'],
                        list(self.cell_names_key.keys())
                    )
                elif new_meta==['FTU Morphometrics']:
                    # These are the options for morphometrics
                    cell_types_options = (
                        [],
                        ['Maximum','Mean','Median','Minimum','Standard Deviation','Sum'],
                        self.morphometrics_names
                    )
                else:
                    cell_types_options = (no_update,no_update,no_update)
        except:
            cell_types_options = (no_update,no_update,no_update)

        current_slide_count, slide_select_options = self.update_current_slides(slide_rows)

        print(f'new_meta: {new_meta}')
        if type(new_meta)==list:
            if len(new_meta)>0:
                new_meta = new_meta[0]
        
        if type(group_type)==list:
            if len(group_type)>0:
                group_type = group_type[0]

        # ctx.triggered_id here is type <dash._utils.AttributeDict> not dict
        #try:
        #    if ctx.triggered_id['type']=='slide-dataset-table':
        #        #self.metadata = self.update_plotting_metadata()
        #        #self.update_plotting_metadata()
        #except:
        #    pass

        #print(f'length of updated metadata: {len(self.metadata)}')
        # For DSA-backend deployment
        if not new_meta is None:
            if not len(new_meta)==0:
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
                        if not new_meta=='FTU Expression Statistics' and not new_meta=='FTU Morphometrics':
                            d_dict = {'Dataset':[],'Slide Name':[],new_meta:[]}

                            for s in d_include:
                                if new_meta in s['meta']:
                                    d_dict['Dataset'].append(dataset_name)
                                    d_dict['Slide Name'].append(s['name'])
                                    d_dict[new_meta].append(s['meta'][new_meta])
                        
                        else:
                            if new_meta=='FTU Expression Statistics':
                                if not ctx.triggered_id['type']=='meta-drop':
                                    # Whether it's Mean, Median, Sum, Standard Deviation, or Nonzero
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
                                else:
                                    d_dict = {}
                            
                            elif new_meta=='FTU Morphometrics':
                                
                                if not ctx.triggered_id['type']=='meta-drop':
                                    ftu_morphometric_feature = sub_meta[2]
                                    stat_agg = sub_meta[1]

                                    # Getting FTU Specific expression values
                                    d_dict = {'Dataset':[],'Slide Name':[],'FTU':[],new_meta:[]}
                                    
                                    for d_i in d_include:
                                        slide_meta = d_i['meta']
                                        slide_name = d_i['name']
                                        ftu_morphometrics = [i for i in list(slide_meta.keys()) if 'Morphometrics' in i]
                                        if len(ftu_morphometrics)>0:
                                            for f in ftu_morphometrics:
                                                if ftu_morphometric_feature+'_'+stat_agg in slide_meta[f]:
                                                    morpho_stat = slide_meta[f][ftu_morphometric_feature+'_'+stat_agg]
                                                    d_dict[new_meta].append(morpho_stat)
                                                    d_dict['FTU'].append(f.replace('_Morphometrics',''))
                                                    d_dict['Dataset'].append(dataset_name)
                                                    d_dict['Slide Name'].append(slide_name)
                                else:
                                    d_dict = {}

                        dataset_metadata.append(d_dict)

                # Converting to dataframe
                plot_data = pd.concat([pd.DataFrame.from_dict(i) for i in dataset_metadata],ignore_index=True)
                
                if not plot_data.empty:
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
                            if not new_meta=='FTU Expression Statistics' and not new_meta=='FTU Morphometrics':
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
                else:

                    fig = go.Figure()

                return [slide_select_options, [fig], cell_types_options, cell_types_turn_off,[current_slide_count]]
            else:
                raise exceptions.PreventUpdate
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

    def update_roi_pie(self,bounds,current_tab):
        
        # Making a box-poly from the bounds
        if current_tab=='cell-compositions-tab':
            if not self.wsi is None:
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

                for marked_idx, marked_ftu in enumerate(self.wsi.marked_ftus):
                    intersecting_ftus[f'Marked FTUs: {marked_idx+1}'] = [i['properties'] for i in marked_ftu['geojson']['features']]
                        
                self.current_ftus = intersecting_ftus
                # Now we have main cell types, cell states, by ftu
                included_ftus = list(intersecting_ftus.keys())
                included_ftus = [i for i in included_ftus if len(intersecting_ftus[i])>0]

                if len(included_ftus)>0:

                    tab_list = []
                    self.fusey_data = {}
                    #counts_data = pd.DataFrame()
                    for f_idx,f in enumerate(included_ftus):
                        counts_data = pd.DataFrame()

                        counts_dict_list = [i['Main_Cell_Types'] for i in intersecting_ftus[f] if 'Main_Cell_Types' in i]
                        if len(counts_dict_list)>0:
                            counts_data = pd.DataFrame.from_records(counts_dict_list).sum(axis=0).to_frame()

                        if not counts_data.empty:
                            counts_data.columns = [f]
                            
                            # Storing some data for Fusey to use :3
                            structure_number = len(counts_dict_list)
                            normalized_counts = counts_data[f]/counts_data[f].sum()

                            # Normalizing to sum to 1
                            counts_data[f] = counts_data[f]/counts_data[f].sum()
                            # Only getting top n
                            counts_data = counts_data.sort_values(by=f,ascending=False).iloc[0:self.plot_cell_types_n,:]
                            counts_data = counts_data.reset_index()

                            f_pie = px.pie(counts_data,values=f,names='index')
                            f_pie.update_traces(textposition='inside')
                            f_pie.update_layout(uniformtext_minsize=12,uniformtext_mode='hide')

                            top_cell = counts_data['index'].tolist()[0]

                            pct_states = pd.DataFrame.from_records([i['Cell_States'][top_cell] for i in intersecting_ftus[f]if 'Cell_States' in i]).sum(axis=0).to_frame()
                            
                            pct_states = pct_states.reset_index()
                            pct_states.columns = ['Cell State','Proportion']
                            pct_states['Proportion'] = pct_states['Proportion']/pct_states['Proportion'].sum()

                            # Fusey data
                            self.fusey_data[f] = {
                                'structure_number':structure_number,
                                'normalized_counts':normalized_counts,
                                'pct_states':pct_states,
                                'top_cell':top_cell
                            }
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
                                ]),label = f+f' ({len(counts_dict_list)})',tab_id = f'tab_{f_idx}'
                            )

                            tab_list.append(f_tab)

                    return dbc.Tabs(tab_list,active_tab = 'tab_0')
                else:
                    return html.P('No FTUs in current view')
            else:
                return html.P('Select a slide to get started!')
        else:
            raise exceptions.PreventUpdate
    
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

        elif color_type =='cell_sub_value':
            # Visualizing a main cell type + cell state combination
            main_cell = self.current_cell.split('_')[0]
            sub_cell = self.current_cell.split('_')[1]
            # Iterating through current ftus
            for f in self.wsi.ftu_props:
                for g in self.wsi.ftu_props[f]:
                    # Getting main counts for this ftu
                    if 'Main_Cell_Types' in g:
                        ftu_counts = g['Main_Cell_Types'][main_cell]
                        cell_sub_values = g['Cell_States'][main_cell][sub_cell]
                        raw_values_list.append(ftu_counts*cell_sub_values)


            # Iterating through spots
            for f in self.wsi.spot_props:
                spot_counts = f['Main_Cell_Types'][main_cell]
                spot_sub_counts = f['Cell_States'][main_cell][sub_cell]
                raw_values_list.append(spot_counts*spot_sub_counts)

            # Iterating through manual ROIs
            for f in self.wsi.manual_rois:
                if 'Main_Cell_Types' in f['geojson']['features'][0]['properties']:
                    manual_counts = f['geojson']['features'][0]['properties']['Main_Cell_Types'][main_cell]
                    manual_subcounts = f['geojson']['features'][0]['properties']['Cell_States'][main_cell][sub_cell]
                    raw_values_list.append(manual_counts*manual_subcounts)

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
                if 'Main_Cell_Types' in s:
                    all_cell_type_counts = float(np.argmax(list(s['Main_Cell_Types'].values())))
                    raw_values_list.append(all_cell_type_counts)

        elif color_type == 'cluster':
            # iterating through current ftus
            for f in self.wsi.ftu_props:
                for g in self.wsi.ftu_props[f]:
                    if 'Cluster' in g:
                        cluster_label = int(g['Cluster'])
                        raw_values_list.append(cluster_label)
            
        elif color_type is None:
            raw_values_list = []
        
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
        else:
            self.hex_color_key = {}

    def update_overlays(self,cell_val,vis_val,filter_vals,ftu_color,cell_sub_val):

        print(f'Updating overlays for current slide: {self.wsi.slide_name}, {cell_val}')

        m_prop = None
        cell_sub_select_children = no_update

        color_bar_style = {
            'visibility':'visible',
            'background':'white',
            'background':'rgba(255,255,255,0.8)',
            'box-shadow':'0 0 15px rgba(0,0,0,0.2)',
            'border-radius':'10px',
            'width':'',
            'padding':'0px 0px 0px 25px',

        }

        if not type(cell_sub_val) is list:
            cell_sub_val = [cell_sub_val]
        if len(cell_sub_val)==0:
            cell_sub_val = [None]

        if not ftu_color is None:
            # Getting these to align with the ftu-colors property order
            current_ftu_colors = list(self.ftu_colors.values())
            ftu_list = list(self.ftu_colors.keys())

            # Index of which ftu is different
            new_color = [i for i in range(len(current_ftu_colors)) if current_ftu_colors[i] not in ftu_color]
            if len(new_color)>0:
                for n in new_color:
                    check_for_new = [i for i in ftu_color if i not in current_ftu_colors]
                    if len(check_for_new)>0:
                        self.ftu_colors[ftu_list[n]] = check_for_new[0]
                    
            self.filter_vals = filter_vals

        # Extracting cell val if there are sub-properties
        if not cell_val is None:
            if '-->' in cell_val:
                cell_val_parts = cell_val.split(' --> ')
                m_prop = cell_val_parts[0]
                cell_val = cell_val_parts[1]

            # Updating current_cell property
            if cell_val in self.cell_names_key:
                if m_prop == 'Main_Cell_Types':

                    if ctx.triggered_id == 'cell-drop':
                        
                        cell_sub_val= [None]
                        self.current_cell = self.cell_names_key[cell_val]
                        # Getting all possible cell states for this cell type:
                        possible_cell_states = np.unique(self.cell_graphics_key[self.current_cell]['states'])
                        # Creating dropdown for cell states
                        cell_sub_select_children = [
                            dcc.Dropdown(
                                options = [{'label':p,'value':p,'disabled':False} for p in possible_cell_states]+[{'label':'All','value':'All','disabled':False}],
                                placeholder = 'Select A Cell State Value',
                                id = {'type':'cell-sub-drop','index':0}
                            )
                        ]
                    else:
                        cell_sub_select_children = no_update

                    if cell_sub_val[0] is None:

                        self.current_cell = self.cell_names_key[cell_val]
                        self.update_hex_color_key('cell_value')
                        color_bar_style['width'] = '350px'

                        color_bar = dl.Colorbar(
                            colorscale = list(self.hex_color_key.values()),
                            width=300,height=10,position='bottomleft',
                            id=f'colorbar{random.randint(0,100)}',
                            style = color_bar_style)
                        
                        filter_max_val = np.max(list(self.hex_color_key.keys()))
                        filter_disable = False
                    
                    elif cell_sub_val[0]=='All':
                        self.current_cell = self.cell_names_key[cell_val]
                        self.update_hex_color_key('cell_value')
                        color_bar_style['width'] = '350px'

                        color_bar = dl.Colorbar(
                            colorscale = list(self.hex_color_key.values()),
                            width=300,height=10,position='bottomleft',
                            id=f'colorbar{random.randint(0,100)}',
                            style = color_bar_style)
                        
                        filter_max_val = np.max(list(self.hex_color_key.keys()))
                        filter_disable = False
                    
                    else:
                        # Visualizing a sub-property of a main cell type
                        self.current_cell = self.cell_names_key[cell_val]+'_'+cell_sub_val[0]
                        self.update_hex_color_key('cell_sub_value')
                        color_bar_style['width'] = '350px'

                        color_bar = dl.Colorbar(
                            colorscale = list(self.hex_color_key.values()),
                            width=300,height=10,position='bottomleft',
                            id=f'colorbar{random.randint(0,100)}',
                            style = color_bar_style)

                        filter_max_val = np.max(list(self.hex_color_key.keys()))
                        filter_disable = False
                
                elif m_prop == 'Cell_States':
                    self.current_cell = self.cell_names_key[cell_val]
                    self.update_hex_color_key('cell_state')
                    color_bar_style['width'] = '350px'

                    color_bar = dl.Colorbar(
                        colorscale = list(self.hex_color_key.values()),
                        width=300,height=10,position='bottomleft',
                        id=f'colorbar{random.randint(0,100)}',
                        style = color_bar_style)

                    filter_max_val = np.max(list(self.hex_color_key.keys()))
                    filter_disable = False

            elif cell_val == 'Max Cell Type':
                self.current_cell = 'max'
                self.update_hex_color_key('max_cell')
                color_bar_style['width'] = '650px'

                cell_sub_select_children = []

                #cell_types = list(self.wsi.geojson_ftus['features'][0]['properties']['Main_Cell_Types'].keys())
                cell_types = sorted(list(self.cell_graphics_key.keys()))
                color_bar = dlx.categorical_colorbar(
                    categories = cell_types,
                    colorscale = list(self.hex_color_key.values()),
                    width=600,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                filter_max_val = 1.0
                filter_disable = True

            elif cell_val == 'Morphometrics Clusters':
                self.current_cell = 'cluster'
                self.update_hex_color_key('cluster')

                cell_sub_select_children = []

                #TODO: This should probably be a categorical colorbar
                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width=300,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                filter_max_val = 1.0
                filter_disable = True
                
            elif cell_val == 'FTU Label':
                self.current_cell = 'name'
                self.update_hex_color_key(cell_val)

                cell_sub_select_children = []

                ftu_types = list(self.hex_color_key.keys())
                color_bar = dlx.categorical_colorbar(
                    categories = ftu_types,
                    colorscale = list(self.hex_color_key.values()),
                    width = 600, height = 10, position = 'bottom left',
                    id = f'colorbar{random.randint(0,100)}',
                    style = color_bar_style
                )

                filter_max_val = 1.0
                filter_disable = True
            else:
                # For other morphometric properties
                print(cell_val)
                self.current_cell = cell_val
                self.update_hex_color_key(cell_val)

                cell_sub_select_children = []

                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width=300,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                filter_max_val = np.max(list(self.hex_color_key.keys()))
                filter_disable = False

        else:
            self.current_cell = cell_val
            self.update_hex_color_key(cell_val)

            cell_sub_select_children = []

            color_bar = no_update
            filter_disable = True
            filter_max_val = 1

        self.cell_vis_val = vis_val/100
        print(callback_context.outputs_list)
        n_layers = len(callback_context.outputs_list[0])
        print(f'n_layers to update: {n_layers}')
        geojson_hideout = [
            {
                'color_key':self.hex_color_key,
                'current_cell':self.current_cell,
                'fillOpacity': self.cell_vis_val,
                'ftu_colors':self.ftu_colors,
                'filter_vals':self.filter_vals
            }
            for i in range(0,n_layers)
        ]

        """
        new_children = [
            dl.Overlay(
                dl.LayerGroup(
                    dl.GeoJSON(url = f'./assets/slide_annotations/{struct}.json', id = self.wsi.map_dict['FTUs'][struct]['id'], options = dict(style=self.ftu_style_handle,filter = self.ftu_filter),
                                hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity=self.cell_vis_val, ftu_color = self.ftu_colors[struct],filter_vals = self.filter_vals),
                                hoverStyle = arrow_function(dict(weight=5, color = self.wsi.map_dict['FTUs'][struct]['hover_color'],dashArray='')),
                                children = [dl.Popup(id=self.wsi.map_dict['FTUs'][struct]['popup_id'])])
                ), name = struct, checked = True, id = self.wsi.item_id+'_'+struct
            )
            for struct in self.wsi.map_dict['FTUs']
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

        for mark_idx,mark in enumerate(self.wsi.marked_ftus):
            new_children.append(
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(data = mark['geojson'], id = mark['id'], options = dict(style = self.ftu_style_handle, filter = self.ftu_filter),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_color = 'white', filter_vals = self.filter_vals),
                                    hoverStyle = arrow_function(dict(weight=5,color = mark['hover_color'],dashArray='')),
                                    children = [])
                    ), name = f'Marked FTUs {mark_idx+1}', checked = True, id = self.wsi.item_id+f'_marked_ftus{mark_idx}'
                )
            )

        self.current_overlays = new_children
        """

        return geojson_hideout, color_bar, filter_max_val, filter_disable, cell_sub_select_children

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
            if 'unique_index' in ftu_click['properties']:
                ftu_idx = ftu_click['properties']['unique_index']

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

                    main_cells_df = pd.DataFrame.from_dict({'Values':chart_data,'Labels':chart_labels,'Full':chart_full_labels})
                    f_pie = go.Figure(
                        data = [
                            go.Pie(
                                name = '',
                                values = main_cells_df['Values'],
                                labels = main_cells_df['Labels'],
                                customdata = main_cells_df['Full'],
                                hovertemplate = "Cell: %{customdata}: <br>Proportion: %{value}</br>"
                            )],
                        layout = {'autosize':True, 'margin':{'t':0,'b':0,'l':0,'r':0},'showlegend':False,'uniformtext_minsize':12,'uniformtext_mode':'hide'}
                    )
                    f_pie.update_traces(textposition='inside')

                    # popup divs
                    if 'unique_index' in ftu_click['properties']:
                        add_labels_children = self.layout_handler.get_user_ftu_labels(self.wsi,ftu_click)
                        accordion_children = [
                            dbc.AccordionItem([
                                html.Div([
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(figure = f_pie),
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
                                                    layout = {'autosize':True,'margin':{'t':0,'b':0,'l':0,'r':0},'showlegend':False,
                                                              'uniformtext_minsize':12,'uniformtext_mode':'hide'}
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
                                    dbc.Row([
                                        dbc.Col(html.Div(
                                                id={'type':'added-labels-div','index':ftu_idx},
                                                children = add_labels_children
                                            ), md=11
                                        ),
                                        dbc.Col(self.layout_handler.gen_info_button('Add your own labels for each structure by typing your label in the "Notes" field and clicking the green check mark'),md=1)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dcc.Input(placeholder='Notes',id={'type':'popup-notes','index':ftu_idx}),md=8),
                                        dbc.Col(html.I(className='bi bi-check-circle-fill me-2',style = {'color':'rgb(0,255,0)'},id={'type':'add-popup-note','index':ftu_idx}),md=4)
                                    ])
                                ])
                            ],title = 'Custom Properties')
                        ]
                    else:
                        accordion_children = [
                            dbc.AccordionItem([
                                html.Div([
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(figure = f_pie),
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
                                                    layout = {'autosize':True,'margin':{'t':0,'b':0,'l':0,'r':0},'showlegend':False,
                                                              'uniformtext_minsize':12,'uniformtext_mode':'hide'}
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
                            ],title = 'Other Properties')
                        ]
                    
                    popup_div = html.Div([
                        dbc.Accordion(
                            children = accordion_children
                        )
                    ],style={'height':'300px','width':'300px','display':'inline-block'})

                    return popup_div
                else:
                    return html.Div([html.P('No cell type information')])
            else:
                return html.Div([html.P('No intersecting spots')])
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
        
        if not slide_name=='':
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

            # Updating in the case that an FTU isn't in the previous set of ftu_colors
            self.ftu_colors = self.wsi.ftu_colors

            # Updating overlays colors according to the current cell
            self.update_hex_color_key(self.current_cell)

            new_children = [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(url = f'./assets/slide_annotations/{struct}.json', id = self.wsi.map_dict['FTUs'][struct]['id'], options = dict(style = self.ftu_style_handle, filter = self.ftu_filter),
                                    hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors, filter_vals = self.filter_vals),
                                    hoverStyle = arrow_function(dict(weight=5, color = self.wsi.map_dict['FTUs'][struct]['hover_color'], dashArray = '')),
                                    zoomToBounds=True,children = [dl.Popup(id=self.wsi.map_dict['FTUs'][struct]['popup_id'])])
                    ), name = struct, checked = True, id = new_slide.item_id+'_'+struct
                )
                for struct in self.wsi.map_dict['FTUs']
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

            self.current_ftus = self.wsi.ftu_names
            self.current_ftu_layers = self.wsi.ftu_names

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

        else:

            raise exceptions.PreventUpdate

    def update_graph_label_children(self,leftover_labels):

        if len(leftover_labels)>0:
            unique_labels = np.unique(leftover_labels).tolist()
            u_l_data = []
            for u in unique_labels:
                one_u_l = {'label':u, 'count': leftover_labels.count(u)}
                u_l_data.append(one_u_l)

            label_pie_data = pd.DataFrame.from_records(u_l_data)
            label_pie = px.pie(
                data_frame = label_pie_data,
                names='label',
                values='count',
                title = '<br>'.join(
                    textwrap.wrap('Count of samples in each label category',width=20)
                )
            )
            label_pie.update_traces(textposition='inside')
            label_pie.update_layout(
                uniformtext_minsize=12,
                uniformtext_mode='hide',
                showlegend=False,
                autosize=True,
                margin={'b':0,'l':0,'r':0}
                )

            label_info_children = [
                dbc.Button(
                    'See Label Distribution',
                    id='label-dist',
                    className='d-grid mx-auto',
                    n_clicks=0
                ),
                dbc.Popover(
                    dbc.PopoverBody(
                        html.Div(
                            dcc.Graph(
                                figure = go.Figure(label_pie)
                            )
                        )
                    ),
                    trigger='click',
                    target = 'label-dist'
                )
            ]

            filter_info_children = []

        else:

            label_info_children = []
            filter_info_children = [
                dbc.Alert('Remove some filtered out labels to see a plot!',color='warning')

            ]

        return label_info_children, filter_info_children

    def update_graph(self,gen_plot_butt,label,checked_feature,filter_labels,states_option):
        
        # Grabbing current metadata from user private folder        
        # Finding features checked:
        if ctx.triggered_id=='gen-plot-butt':

            self.reports_generated = {}
            report_active_tab = 'feat-summ-tab'

            # Enabling download plot data button
            download_plot_disable = False

            if self.clustering_data.empty:
                print(f'Getting new clustering data')
                self.clustering_data = self.dataset_handler.load_clustering_data()

            feature_names = [i['title'] for i in self.dataset_handler.feature_keys if i['key'] in checked_feature]
            cell_features = [i for i in feature_names if i in self.cell_names_key]
            feature_data = pd.DataFrame()
            if len(cell_features)>0:
                feature_data = self.clustering_data.loc[:,[i for i in feature_names if i in self.clustering_data.columns]]
                feature_data = feature_data.reset_index(drop=True)

                print(f'shape of feature_data: {feature_data.shape}')
                if 'Main_Cell_Types' in self.clustering_data.columns:
                    cell_values = self.clustering_data['Main_Cell_Types'].tolist()
                    state_values = self.clustering_data['Cell_States'].tolist()
                    if states_option=='main':
                        for c in cell_features:
                            cell_abbrev = self.cell_names_key[c]

                            specific_cell_values = []
                            for i in cell_values:
                                if not i is None:
                                    if cell_abbrev in i:
                                        specific_cell_values.append(i[cell_abbrev])
                                    else:
                                        specific_cell_values.append(0)
                                else:
                                    specific_cell_values.append(0)
                            
                            feature_data[c] = specific_cell_values
                    elif states_option=='separate':
                        for c in cell_features:
                            cell_abbrev = self.cell_names_key[c]

                            cell_and_states = []
                            cell_state_names = []
                            for i,j in zip(cell_values,state_values):
                                if not i is None:
                                    if cell_abbrev in i and cell_abbrev in j:
                                        main_val = i[cell_abbrev]
                                        states = j[cell_abbrev]
                                    else:
                                        main_val = 0.0
                                        states = {c_state:0.0 for c_state in np.unique(self.cell_graphics_key[cell_abbrev]['states'])}
                                else:
                                    main_val = 0.0
                                    states = {c_state:0.0 for c_state in np.unique(self.cell_graphics_key[cell_abbrev]['states'])}

                                states_dict = {}
                                for c_s in states:
                                    states_dict[c+'_'+c_s] = main_val*states[c_s]
                                    cell_state_names.append(c+'_'+c_s)

                                cell_and_states.append(states_dict)

                            if feature_data.empty:
                                feature_data = pd.DataFrame.from_records(cell_and_states)
                            else:
                                feature_data = pd.concat([feature_data,pd.DataFrame.from_records(cell_and_states)],axis=1,ignore_index=False)
                            feature_names[feature_names.index(c):feature_names.index(c)+1] = tuple(np.unique(cell_state_names).tolist())
            else:
                feature_data = self.clustering_data.loc[:,[i for i in feature_names if i in self.clustering_data.columns]]
                feature_data = feature_data.reset_index(drop=True)

            # Coercing dtypes of columns in feature_data
            for f in feature_data.columns.tolist():
                feature_data[f] = pd.to_numeric(feature_data[f],errors='coerce')

            # Getting the label data
            if label in self.clustering_data.columns:
                label_data = self.clustering_data[label].tolist()
            else:
                sample_ids = [i['Slide_Id'] for i in self.clustering_data['Hidden'].tolist()]
                unique_ids = np.unique(sample_ids).tolist()

                if label=='Slide Name':
                    # Getting the name of each slide:
                    slide_names = []
                    for u in unique_ids:
                        item_data = self.dataset_handler.gc.get(f'/item/{u}')
                        slide_names.append(item_data['name'])
                    
                    label_data = [slide_names[unique_ids.index(i)] for i in sample_ids]
                
                elif label=='Folder Name':
                    # Getting the folder name for each slide
                    folder_names = []
                    for u in unique_ids:
                        # Already stored the folder names in the dataset_handler so don't need another request
                        item_data = self.dataset_handler.gc.get(f'/item/{u}')
                        folderId = item_data['folderId']

                        folder_names.append(self.dataset_handler.slide_datasets[folderId]['name'])
                    
                    label_data = [folder_names[unique_ids.index(i)] for i in sample_ids]

                elif label=='Cell Type':
                    #TODO: Get this selectable as a label
                    label_data = ['tba' for i in range(len(sample_ids))]
                
                elif label == 'Morphometric':
                    #TODO: Get this selectable as a label
                    label_data = ['tba' for i in range(len(sample_ids))]
                
                else:
                    # Used for slide-level metadata keys
                    slide_meta = []
                    for u in unique_ids:
                        item_data = self.dataset_handler.gc.get(f'/item/{u}')
                        if label in item_data['meta']:
                            slide_meta.append(item_data['meta'][label])
                        else:
                            # This one will get removed from the final df after .dropna()
                            slide_meta.append(None)
                    
                    label_data = [slide_meta[unique_ids.index(i)] for i in sample_ids]
            
            # Now filtering labels according to any selected filter labels
            filter_label_names = [i['title'] for i in self.dataset_handler.filter_keys if i['key'] in filter_labels]
            filter_idx = []
            if len(filter_label_names)>0:
                # Now have to find the parents for each of these
                filter_label_parents = ['-'.join(i.split('-')[0:-1]) for i in filter_labels]
                filter_label_parent_names = [i['title'] for i in self.dataset_handler.filter_keys if i['key'] in filter_label_parents]

                unique_parent_filters = np.unique(filter_label_parent_names).tolist()
                if 'FTUs' in unique_parent_filters:
                    # Removing specific FTUs by name
                    ftu_labels = self.clustering_data['FTU'].tolist()
                    filter_idx.extend([i for i in range(len(ftu_labels)) if ftu_labels[i] in filter_label_names])
                    unique_parent_filters = [i for i in unique_parent_filters if not i=='FTUs']
                    
                if len(unique_parent_filters)>0:
                    # Grab slide metadata
                    sample_ids = [i['Slide_Id'] for i in self.clustering_data['Hidden'].tolist()]
                    unique_ids = np.unique(sample_ids).tolist()

                    for u_id in unique_ids:
                        item_data = self.dataset_handler.gc.get(f'/item/{u_id}')
                        item_meta = item_data['meta']
                        item_name = item_data['name']
                        item_folder = self.dataset_handler.slide_datasets[item_data['folderId']]['name']
                        
                        # If this slide is filtered out then we don't need to check if it's filtered out for any other reason
                        if item_name in filter_label_names or item_folder in filter_label_names:
                            filter_idx.extend([i for i in range(len(sample_ids)) if sample_ids[i]==u_id and i not in filter_idx])
                        
                        # Checking if this item has a filter parent name in it's metadata
                        elif len(list(set(filter_label_parent_names) & set(list(item_meta.keys()))))>0:
                            overlap_keys = list(set(filter_label_parent_names) & set(list(item_meta.keys())))
                            for o in overlap_keys:
                                slide_value = item_meta[o]
                                if slide_value in filter_label_names:
                                    # Add the row indices to the filter_idx list
                                    filter_idx.extend([i for i in range(len(sample_ids)) if sample_ids[i]==u_id and i not in filter_idx])

            filter_idx = np.unique(filter_idx).tolist()

            # Generating appropriate plot
            if len(feature_names)==1:
                print(f'Generating violin plot for {feature_names}')

                # Adding "Hidden" column with image grabbing info
                feature_data['Hidden'] = self.clustering_data['Hidden'].tolist()       
                feature_data['label'] = label_data
                if 'Main_Cell_Types' in self.clustering_data.columns:
                    feature_data['Main_Cell_Types'] = self.clustering_data['Main_Cell_Types'].tolist()
                    feature_data['Cell_States'] = self.clustering_data['Cell_States'].tolist()

                self.feature_data = feature_data
                
                # Dropping filtered out rows
                if len(filter_idx)>0:
                    self.feature_data = self.feature_data.iloc[[int(i) for i in range(self.feature_data.shape[0]) if int(i) not in filter_idx],:]

                self.feature_data = self.feature_data.dropna(subset=[i for i in feature_names if i in self.feature_data.columns]+['label','Hidden'])

                # Generating labels_info_children
                labels_left = self.feature_data['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)

                if self.feature_data.shape[0]>0:

                    # Adding unique point index to hidden data
                    hidden_data = self.feature_data['Hidden'].tolist()
                    for h_i,h in enumerate(hidden_data):
                        h['Index'] = h_i

                    self.feature_data.loc[:,'Hidden'] = hidden_data

                    figure = go.Figure(data = go.Violin(
                        x = self.feature_data['label'],
                        y = self.feature_data[feature_names[0]],
                        customdata = self.feature_data['Hidden'],
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
                                f'{feature_names[0]}',
                                width=30
                            )
                        ),
                        yaxis_title = dict(
                            text = '<br>'.join(
                                textwrap.wrap(
                                    f'{feature_names[0]}',
                                    width=15
                                )
                            ),
                            font = dict(size = 10)
                        ),
                        xaxis_title = dict(
                            text = '<br>'.join(
                                textwrap.wrap(
                                    label,
                                    width=15
                                )
                            ),
                            font = dict(size = 10)
                        ),
                        margin = {'r':0,'b':25}
                    )

                else:
                    figure = go.Figure()

            elif len(feature_names)==2:
                print(f'Generating a scatter plot')
                feature_columns = feature_names

                # Adding "Hidden" column with image grabbing info
                feature_data['Hidden'] = self.clustering_data['Hidden'].tolist()     
                feature_data['label'] = label_data
                if 'Main_Cell_Types' in self.clustering_data.columns:
                    feature_data['Main_Cell_Types'] = self.clustering_data['Main_Cell_Types'].tolist()
                    feature_data['Cell_States'] = self.clustering_data['Cell_States'].tolist()

                self.feature_data = feature_data
                
                # Dropping filtered out rows
                if len(filter_idx)>0:
                    self.feature_data = self.feature_data.iloc[[int(i) for i in range(self.feature_data.shape[0]) if int(i) not in filter_idx],:]
                
                self.feature_data = self.feature_data.dropna(subset=[i for i in feature_names if i in self.feature_data.columns]+['label','Hidden'])
                # Adding point index to hidden data
                hidden_data = self.feature_data['Hidden'].tolist()
                for h_i,h in enumerate(hidden_data):
                    h['Index'] = h_i

                self.feature_data.loc[:,'Hidden'] = hidden_data
                # Generating labels_info_children and filter_info_children
                labels_left = self.feature_data['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)

                figure = go.Figure(data = px.scatter(
                    data_frame=self.feature_data,
                    x = feature_columns[0],
                    y = feature_columns[1],
                    color = 'label',
                    custom_data = 'Hidden',
                    title = '<br>'.join(
                        textwrap.wrap(
                            f'Scatter plot of {feature_names[0]} and {feature_names[1]} labeled by {label}',
                            width = 30
                            )
                        )
                ))

                figure.update_layout(
                    legend = dict(
                        orientation='h',
                        y = 0,
                        yanchor='top',
                        xanchor='left'
                    ),
                    margin = {'r':0,'b':25}
                )

            elif len(feature_names)>2:
                print(f'Running UMAP and returning a scatter plot')

                # Scaling and reducing feature data using UMAP
                feature_data['Hidden'] = self.clustering_data['Hidden'].tolist()
                feature_data['label'] = label_data
                if 'Main_Cell_Types' in self.clustering_data.columns:
                    feature_data['Main_Cell_Types'] = self.clustering_data['Main_Cell_Types'].tolist()
                    feature_data['Cell_States'] = self.clustering_data['Cell_States'].tolist()

                self.feature_data = feature_data
                # Dropping filtered out rows
                if len(filter_idx)>0:
                    self.feature_data = self.feature_data.iloc[[int(i) for i in range(self.feature_data.shape[0]) if int(i) not in filter_idx],:]

                self.feature_data = self.feature_data.dropna(subset=[i for i in feature_names if i in self.feature_data.columns]+['label','Hidden'])
                hidden_col = self.feature_data['Hidden'].tolist()

                # Adding point index to hidden_col
                for h_i,h in enumerate(hidden_col):
                    h['Index'] = h_i

                label_col = self.feature_data['label'].tolist()
                main_cell_types_col = self.feature_data['Main_Cell_Types'].tolist()
                cell_states_col = self.feature_data['Cell_States'].tolist()
                feature_data = self.feature_data.loc[:,[i for i in feature_names if i in self.feature_data.columns]].values
                # Scaling feature_data
                feature_data_means = np.nanmean(feature_data,axis=0)
                feature_data_stds = np.nanstd(feature_data,axis=0)

                scaled_data = (feature_data-feature_data_means)/feature_data_stds
                scaled_data[np.isnan(scaled_data)] = 0.0
                scaled_data[~np.isfinite(scaled_data)] = 0.0

                umap_reducer = UMAP()
                embeddings = umap_reducer.fit_transform(scaled_data)

                umap_df = pd.DataFrame(data = embeddings, columns = ['UMAP1','UMAP2'])
                
                # Saving this so we can update the label separately without re-running scaling or reduction
                self.umap_df = umap_df

                self.umap_df['Hidden'] = hidden_col
                self.umap_df['label'] = label_col
                self.umap_df['Main_Cell_Types'] = main_cell_types_col
                self.umap_df['Cell_States'] = cell_states_col
                
                # Generating labels_info_children and filter_info_children
                labels_left = self.umap_df['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)

                figure = go.Figure(data = px.scatter(
                    data_frame = self.umap_df,
                    x = 'UMAP1',
                    y = 'UMAP2',
                    color = 'label',
                    custom_data = 'Hidden',
                    title = '<br>'.join(
                        textwrap.wrap(
                            f'UMAP of selected features labeled with {label}',
                            width=30
                        )
                    )
                ))

                figure.update_layout(
                    legend = dict(
                        orientation='h',
                        y = 0,
                        yanchor='top',
                        xanchor='left'
                    ),
                    margin = {'r':0,'b':25}
                )

            else:
                # No features selected
                figure = go.Figure()
                label_info_children = []
                filter_info_children = no_update

            return figure, label_info_children, filter_info_children, report_active_tab, download_plot_disable
        
        elif ctx.triggered_id=='label-select':
            self.reports_generated = {}
            report_active_tab = 'feat-summ-tab'

            # Enabling download plot data button
            download_plot_disable = False

            if not self.feature_data is None:
                # Getting the label data
                if label in self.clustering_data.columns:
                    label_data = self.clustering_data[label].tolist()
                else:
                    sample_ids = [i['Slide_Id'] for i in self.clustering_data['Hidden'].tolist()]
                    unique_ids = np.unique(sample_ids).tolist()

                    if label=='Slide Name':
                        # Getting the name of each slide:
                        slide_names = []
                        for u in unique_ids:
                            item_data = self.dataset_handler.gc.get(f'/item/{u}')
                            slide_names.append(item_data['name'])
                        
                        label_data = [slide_names[unique_ids.index(i)] for i in sample_ids]
                    
                    elif label=='Folder Name':
                        # Getting the folder name for each slide
                        folder_names = []
                        for u in unique_ids:
                            # Already stored the folder names in the dataset_handler so don't need another request
                            item_data = self.dataset_handler.gc.get(f'/item/{u}')
                            folderId = item_data['folderId']

                            folder_names.append(self.dataset_handler.slide_datasets[folderId]['name'])
                        
                        label_data = [folder_names[unique_ids.index(i)] for i in sample_ids]

                    elif label=='Cell Type':
                        #TODO: Get this selectable as a label
                        label_data = ['tba' for i in range(len(sample_ids))]
                    
                    elif label == 'Morphometric':
                        #TODO: Get this selectable as a label
                        label_data = ['tba' for i in range(len(sample_ids))]
                    
                    else:
                        # Used for slide-level metadata keys
                        slide_meta = []
                        for u in unique_ids:
                            item_data = self.dataset_handler.gc.get(f'/item/{u}')
                            if label in item_data['meta']:
                                slide_meta.append(item_data['meta'][label])
                            else:
                                # This one will get removed from the final df after .dropna()
                                slide_meta.append(None)
                        
                        label_data = [slide_meta[unique_ids.index(i)] for i in sample_ids]
                
                # Needs an alignment step going from the full self.clustering_data to the na-dropped and filtered self.feature_data
                # self.feature_data contains the features included in the current plot, label, Hidden, Main_Cell_Types, and Cell_States
                # So for a violin plot the shape should be nX5
                self.feature_data.loc[:,'label'] = [label_data[i] for i in list(self.feature_data.index)]

                # Generating labels_info_children
                labels_left = self.feature_data['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)

                feature_number = len([i for i in self.feature_data.columns.tolist() if i not in ['label','Hidden','Main_Cell_Types','Cell_States']])

                if feature_number==1:
                    feature_names = self.feature_data.columns.tolist()
                    figure = go.Figure(data = go.Violin(
                        x = self.feature_data['label'],
                        y = self.feature_data[feature_names[0]],
                        customdata = self.feature_data['Hidden'],
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
                                f'{feature_names[0]}',
                                width=30
                            )
                        ),
                        yaxis_title = dict(
                            text = '<br>'.join(
                                textwrap.wrap(
                                    f'{feature_names[0]}',
                                    width=15
                                )
                            ),
                            font = dict(size = 10)
                        ),
                        xaxis_title = dict(
                            text = '<br>'.join(
                                textwrap.wrap(
                                    label,
                                    width=15
                                )
                            ),
                            font = dict(size = 10)
                        ),
                        margin = {'r':0,'b':25}
                    )

                elif feature_number==2:
                    
                    feature_columns = self.feature_data.columns.tolist()

                    figure = go.Figure(data = px.scatter(
                        data_frame=self.feature_data,
                        x = feature_columns[0],
                        y = feature_columns[1],
                        color = 'label',
                        custom_data = 'Hidden',
                        title = '<br>'.join(
                            textwrap.wrap(
                                f'Scatter plot of {feature_columns[0]} and {feature_columns[1]} labeled by {label}',
                                width = 30
                                )
                            )
                    ))

                    figure.update_layout(
                        legend = dict(
                            orientation='h',
                            y = 0,
                            yanchor='top',
                            xanchor='left'
                        ),
                        margin = {'r':0,'b':25}
                    )

                elif feature_number>2:

                    self.umap_df.loc[:,'label']=[label_data[i] for i in list(self.feature_data.index)]
                    
                    figure = go.Figure(data = px.scatter(
                        data_frame = self.umap_df,
                        x = 'UMAP1',
                        y = 'UMAP2',
                        color = 'label',
                        custom_data = 'Hidden',
                        title = '<br>'.join(
                            textwrap.wrap(
                                f'UMAP of selected features labeled with {label}',
                                width=30
                            )
                        )
                    ))

                    figure.update_layout(
                        legend = dict(
                            orientation='h',
                            y = 0,
                            yanchor='top',
                            xanchor='left'
                        ),
                        margin = {'r':0,'b':25}
                    )
                
                return figure, label_info_children, filter_info_children, report_active_tab, download_plot_disable
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate

    def grab_image(self,sample_info):

        if len(sample_info)>100:
            sample_info = sample_info[0:100]

        img_list = []
        for idx,s in enumerate(sample_info):
            #print(f's: {s}')
            image_region = self.dataset_handler.get_annotation_image(s['Slide_Id'],s['Bounding_Box'])
            
            img_list.append(resize(np.array(image_region),output_shape=(512,512,3)))


        return img_list        

    def update_selected(self,click,selected):
        #print(click)
        if click is not None:
            if 'cluster-graph.selectedData' in list(ctx.triggered_prop_ids.keys()):
                # Custom data contains Slide_Id and BBox coordinates for pulling images
                if selected is None:
                    raise exceptions.PreventUpdate
                if 'points' not in selected:
                    raise exceptions.PreventUpdate
                try:
                    sample_info = [i['customdata'][0] for i in selected['points']]
                except KeyError:
                    sample_info = [i['customdata'] for i in selected['points']]

                # Index corresponds to the row in the feature dataframe used for pulling cell type/state information
                sample_index = [i['Index'] for i in sample_info]
            else:
                try:
                    sample_info = [click['points'][0]['customdata'][0]]
                except KeyError:
                    sample_info = [click['points'][0]['customdata']]

                sample_index = [sample_info[0]['Index']]

            self.current_selected_samples = sample_index

            # Creating selected_image_info children
            # This could have more functionality
            # Like adding markers or going to that region in the current slide (if the selected image is from the current slide)
            slide_names = [self.dataset_handler.gc.get(f'/item/{s_i["Slide_Id"]}')["name"] for s_i in sample_info]
            label_list = self.feature_data['label'].tolist()
            image_labels = [label_list[l] for l in sample_index]

            #TODO: Disable if the slide is changed
            if not self.wsi is None:
                if any([i==self.wsi.slide_name for i in slide_names]) and len(slide_names)>1:
                    selected_image_info = [
                        dbc.Row([
                            dbc.Col(
                                dbc.Button(
                                    f'Add All Markers ({slide_names.count(self.wsi.slide_name)})',
                                    id = {'type':'add-mark-cluster','index':0},
                                    className='d-grid col-12 mx-auto'
                                )
                            )
                        ])
                    ]
                else:
                    selected_image_info = []
            else:
                selected_image_info = []

            for i,j,k in zip(list(range(len(slide_names))),slide_names,image_labels):
                
                if not self.wsi is None:
                    if j == self.wsi.slide_name:
                        selected_image_info.append(
                            dbc.Row([
                                dbc.Col(
                                    html.P(f'Image: {i}, Slide: {j}, Label: {k}'),
                                    md = 8
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        'Add Marker',
                                        id = {'type':'add-mark-cluster','index':i+1},
                                        className='d-grid mx-auto'
                                    ),
                                    md = 4
                                )
                            ],style={'marginBottom':'5px'})
                        )
                    else:
                        selected_image_info.append(
                            dbc.Row(
                                dbc.Col(
                                    html.P(f'Image: {i}, Slide: {j}, Label: {k}')
                                )
                            )
                        )
                else:
                    selected_image_info.append(
                        dbc.Row(
                            dbc.Col(
                                html.P(f'Image: {i}, Slide: {j}, Label: {k}')
                            )
                        )
                    )
            
            selected_image_info = html.Div(
                selected_image_info,
                style = {'maxHeight':'100px','overflow':'scroll'}
            )

            current_image = self.grab_image(sample_info)
            if len(current_image)==1:
                selected_image = go.Figure(
                    data = px.imshow(current_image[0])['data'],
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                    )
            elif len(current_image)>1:
                selected_image = go.Figure(
                    data = px.imshow(np.stack(current_image,axis=0),animation_frame=0,binary_string=True),
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                    )
            else:
                selected_image = go.Figure()
                print(f'No images found')
                print(f'hover: {click}')
                print(f'selected:{selected}')
                print(f'self.current_selected_samples: {self.current_selected_samples}')

            if 'Main_Cell_Types' in self.feature_data.columns:
                # Preparing figure containing cell types + cell states info
                main_cell_types_list = self.feature_data['Main_Cell_Types'].tolist()
                #print(f'size of main_cell_types_list: {len(main_cell_types_list)}')
                #print(f'sample_index: {sample_index}')
                counts_data = pd.DataFrame([main_cell_types_list[i] for i in sample_index]).sum(axis=0).to_frame()
                counts_data.columns = ['Selected Data Points']
                counts_data = counts_data.reset_index()
                # Normalizing to sum to 1
                counts_data['Selected Data Points'] = counts_data['Selected Data Points']/(counts_data['Selected Data Points'].sum()+0.00000001)

                # Only getting the top-5
                counts_data = counts_data.sort_values(by='Selected Data Points',ascending=False)
                counts_data = counts_data[counts_data['Selected Data Points']>0]
                if counts_data.shape[0]>0:
                    f_pie = px.pie(counts_data,values='Selected Data Points',names='index')

                    # Getting initial cell state info
                    first_cell = counts_data['index'].tolist()[0]
                    cell_states_list = self.feature_data['Cell_States'].tolist()
                    state_data = pd.DataFrame([cell_states_list[i][first_cell] for i in sample_index]).sum(axis=0).to_frame()
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

                else:
                    selected_cell_types = go.Figure()
                    selected_cell_states = go.Figure()
            else:
                selected_cell_types = go.Figure()
                selected_cell_states = go.Figure()

            return selected_image, selected_cell_types, selected_cell_states, selected_image_info
        else:
            return go.Figure(), go.Figure(), go.Figure(),[]
    
    def update_selected_state_bar(self, selected_cell_click):
        print(f'Selected cell click: {selected_cell_click}')
        if not selected_cell_click is None:
            cell_type = selected_cell_click['points'][0]['label']

            cell_states_data = self.clustering_data['Cell_States'].tolist()
            state_data = pd.DataFrame([cell_states_data[i][cell_type] for i in self.current_selected_samples]).sum(axis=0).to_frame()
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
        
        if not self.wsi is None:
            try:
                # Used for pattern-matching callbacks
                triggered_id = ctx.triggered_id['type']
            except TypeError:
                # Used for normal callbacks
                triggered_id = ctx.triggered_id

            if triggered_id == 'edit_control':
                
                if triggered_id=='edit_control':
                    print(f'len of new_geojson: {len(new_geojson)}')
                    print(new_geojson)
                    if type(new_geojson)==list:
                        new_geojson = new_geojson[0]    

                if not new_geojson is None:
                    if len(new_geojson['features'])>0:
                        
                        # Adding each manual annotation iteratively (good for if annotations are edited or deleted as well as for new annotations)
                        self.wsi.manual_rois = []
                        self.wsi.marked_ftus = []
                        self.current_overlays = self.current_overlays[0:len(self.wsi.ftu_names)]
                        for geo in new_geojson['features']:

                            if not geo['properties']['type'] == 'marker':

                                new_roi = {'type':'FeatureCollection','features':[geo]}
                                
                                # New geojson has no properties which can be used for overlays or anything so we have to add those
                                # Step 1, find intersecting spots:
                                overlap_spot_props = self.wsi.find_intersecting_spots(shape(new_roi['features'][0]['geometry']))

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

                                new_roi['features'][0]['properties']['Main_Cell_Types'] = main_counts_dict
                                new_roi['features'][0]['properties']['Cell_States'] = agg_cell_states

                                new_manual_roi_dict = {
                                        'geojson':new_roi,
                                        'id':{'type':'ftu-bounds','index':len(self.current_overlays)},
                                        'popup_id':{'type':'ftu-popup','index':len(self.current_overlays)},
                                        'color':'white',
                                        'hover_color':'#32a852'
                                    }
                                self.wsi.manual_rois.append(new_manual_roi_dict)

                                new_child = dl.Overlay(
                                    dl.LayerGroup(
                                        dl.GeoJSON(data = new_roi, id = new_manual_roi_dict['id'], options = dict(style = self.ftu_style_handle),
                                                hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors, filter_vals = self.filter_vals),
                                                hoverStyle = arrow_function(dict(weight=5, color = new_manual_roi_dict['hover_color'], dashArray='')),
                                                children = [dl.Popup(id = new_manual_roi_dict['popup_id'])]
                                            )
                                    ), name = f'Manual ROI {len(self.wsi.manual_rois)}', checked = True, id = self.wsi.item_id+f'_manual_roi{len(self.wsi.manual_rois)}'
                                )

                                self.current_overlays.append(new_child)

                            elif geo['properties']['type']=='marker':
                                # Separate procedure for marking regions/FTUs with a marker
                                new_marked = {'type':'FeatureCollection','features':[geo]}

                                overlap_dict, overlap_poly = self.wsi.find_intersecting_ftu(shape(new_marked['features'][0]['geometry']),'all')
                                if not overlap_poly is None:
                                    # Getting the intersecting ROI geojson
                                    if len(self.wsi.marked_ftus)==0:
                                        if triggered_id=='edit_control':
                                            new_marked_roi = {
                                                'type':'FeatureCollection',
                                                'features':[
                                                    {
                                                        'type':'Feature',
                                                        'geometry':{
                                                            'type':'Polygon',
                                                            'coordinates':[list(overlap_poly.exterior.coords)],
                                                        },
                                                        'properties':overlap_dict
                                                    }
                                                ]
                                            }
                                        elif triggered_id=='add-marker-cluster':
                                            new_marked_roi = {
                                                'type':'FeatureCollection',
                                                'features':[
                                                    {
                                                        'type':'Feature',
                                                        'geometry':{
                                                            'type':'Polygon',
                                                            'coordinates':[list(overlap_poly.exterior.coords)]
                                                        },
                                                        'properties':overlap_dict
                                                    },
                                                    #new_marked['features'][0]
                                                ]
                                            }

                                        self.wsi.marked_ftus = [{
                                            'geojson':new_marked_roi,
                                        }]

                                    else:
                                        self.wsi.marked_ftus[0]['geojson']['features'].append(
                                            {
                                                'type':'Feature',
                                                'geometry':{
                                                    'type':'Polygon',
                                                    'coordinates':[list(overlap_poly.exterior.coords)],
                                                },
                                                'properties':overlap_dict
                                            }
                                        )

                        # Adding the marked ftus layer if any were added
                        if len(self.wsi.marked_ftus)>0:
                            print(f'Number of marked ftus: {len(self.wsi.marked_ftus[0]["geojson"]["features"])}')
                            
                            self.wsi.marked_ftus[0]['id'] = {'type':'ftu-bounds','index':len(self.current_overlays)}
                            self.wsi.marked_ftus[0]['hover_color'] = '#32a852'

                            new_marked_dict = {
                                'geojson':self.wsi.marked_ftus[0]['geojson'],
                                'id':{'type':'ftu-bounds','index':len(self.current_overlays)},
                                'color':'white',
                                'hover_color':'#32a852'
                            }
                            new_child = dl.Overlay(
                                dl.LayerGroup(
                                    dl.GeoJSON(data = new_marked_dict['geojson'], id = new_marked_dict['id'], options = dict(style = self.ftu_style_handle), pointToLayer = self.render_marker_handle,
                                            hideout = dict(color_key = self.hex_color_key, current_cell = self.current_cell, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors,filter_vals = self.filter_vals),
                                            hoverStyle = arrow_function(dict(weight=5, color = new_marked_dict['hover_color'],dashArray='')),
                                            children = []
                                            )
                                ), name = f'Marked FTUs', checked=True, id = self.wsi.item_id+f'_marked_ftus'
                            )
                            
                            self.current_overlays.append(new_child)

                        if len(self.wsi.manual_rois)>0:
                            data_select_options = self.layout_handler.data_options
                            data_select_options[4]['disabled'] = False
                        else:
                            data_select_options = self.layout_handler.data_options

                        if len(self.wsi.marked_ftus)>0:
                            data_select_options[3]['disabled'] = False

                        if not self.current_cell is None:
                            if self.current_cell in self.cell_names_key:
                                self.update_hex_color_key('cell_value')
                            elif '_' in self.current_cell:
                                self.update_hex_color_key('cell_sub_value')
                            elif self.current_cell=='max':
                                self.update_hex_color_key('max_cell')
                            elif self.current_cell == 'cluster':
                                self.update_hex_color_key('cluster')
                            else:
                                self.update_hex_color_key(self.current_cell)
                        else:
                            self.update_hex_color_key(self.current_cell)
                        
                        return self.current_overlays, data_select_options
                    else:

                        # Clearing manual ROIs and reverting overlays
                        self.wsi.manual_rois = []
                        self.wsi.marked_ftus = []
                        self.current_overlays = self.current_overlays[0:len(self.wsi.ftu_names)+1]
                        data_select_options = self.layout_handler.data_options

                        if not self.current_cell is None:
                            if self.current_cell in self.cell_names_key:
                                self.update_hex_color_key('cell_value')
                            elif '_' in self.current_cell:
                                self.update_hex_color_key('cell_sub_value')
                            elif self.current_cell=='max':
                                self.update_hex_color_key('max_cell')
                            elif self.current_cell == 'cluster':
                                self.update_hex_color_key('cluster')
                            else:
                                self.update_hex_color_key(self.current_cell)
                        else:
                            self.update_hex_color_key(self.current_cell)
                        
                        return self.current_overlays, data_select_options
                else:
                    raise exceptions.PreventUpdate
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate

    def add_marker_from_cluster(self,mark_click):

        # Adding marker(s) from graph returning geojson
        # index = 0 == mark all the samples in the current slide
        # index != 0 == mark a specific sample in the current slide
        if ctx.triggered[0]['value']:
            
            if ctx.triggered_id['index']==0:
                # Add marker for all samples in the current slide
                mark_geojson = {'type':'FeatureCollection','features':[]}

                # Iterating through all current selected samples
                # current_selected_samples is an index from self.feature_data
                current_selected_hidden = [self.feature_data['Hidden'].tolist()[i] for i in self.current_selected_samples]
                marker_bboxes = [i['Bounding_Box'] for i in current_selected_hidden if i['Slide_Id']==self.wsi.item_id]
                marker_map_coords = [self.wsi.convert_slide_coords([[i[0],i[1]],[i[2],i[3]]]) for i in marker_bboxes]
                marker_center_coords = [[(i[0][0]+i[1][0])/2,(i[0][1]+i[1][1])/2] for i in marker_map_coords]
                
                mark_geojson['features'].extend([
                    {
                        'type':'Feature',
                        'properties':{'type':'marker'},
                        'geometry': {'type':'Point','coordinates': i}
                    }
                    for i in marker_center_coords
                ])

                map_with_markers = [
                    dl.Marker(position=[i[1],i[0]])
                    for i in marker_center_coords
                ]
            else:
                # Add marker for a specific sample
                mark_geojson = {'type':'FeatureCollection','features':[]}

                # Pulling out one specific sample
                selected_bbox = self.feature_data['Hidden'].tolist()[self.current_selected_samples[ctx.triggered_id['index']-1]]['Bounding_Box']
                marker_map_coords = self.wsi.convert_slide_coords([[selected_bbox[0],selected_bbox[1]],[selected_bbox[2],selected_bbox[3]]])
                marker_center_coords = [(marker_map_coords[0][0]+marker_map_coords[1][0])/2,(marker_map_coords[0][1]+marker_map_coords[1][1])/2]
                
                mark_geojson['features'].append(
                    {
                        'type':'Feature',
                        'properties':{'type':'marker'},
                        'geometry':{'type':'Point','coordinates':marker_center_coords}
                    }
                )

                map_with_markers = dl.Marker(position=[marker_center_coords[1],marker_center_coords[0]])
                

            return [mark_geojson], map_with_markers
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
        #print(ctx.triggered_id)
        #print(options)
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
                    download_list = self.download_handler.extract_annotations(self.wsi,self.current_slide_bounds, options)
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

        # Making a new folder for this upload in the user's Public (could also change to private?) folder
        current_datetime = str(datetime.now())
        current_datetime = current_datetime.replace('-','_').replace(' ','_').replace(':','_').replace('.','_')
        parentId = self.dataset_handler.get_user_folder_id(f'Public/FUSION_Upload_{current_datetime}')
        print(f'parentId: {parentId}')
        self.latest_upload_folder = {
            'id':parentId,
            'path':f'Public/FUSION_Upload_{current_datetime}'
        }    

        if upload_type == 'Visium':
            upload_reqs = html.Div([
                dbc.Row([
                    html.Div(
                        id = {'type':'wsi-upload-div','index':0},
                        children = [
                            dbc.Label('Upload Whole Slide Image Here!'),
                            UploadComponent(
                                id = {'type':'wsi-upload','index':0},
                                uploadComplete=False,
                                baseurl=self.dataset_handler.apiUrl,
                                girderToken=self.dataset_handler.user_token,
                                parentId=parentId,
                                filetypes=['svs','ndpi','scn','tiff','tif']                      
                            )
                        ],
                        style={'marginBottom':'10px','display':'inline-block'}
                    )
                ],align='center'),
                dbc.Row([
                    html.Div(
                        id = {'type':'omics-upload-div','index':0},
                        children = [
                            dbc.Label('Upload your RDS file here!'),
                            UploadComponent(
                                id = {'type':'omics-upload','index':0},
                                uploadComplete=False,
                                baseurl=self.dataset_handler.apiUrl,
                                girderToken=self.dataset_handler.user_token,
                                parentId=parentId,
                                filetypes=['rds','csv']                 
                            )
                        ],
                        style = {'marginTop':'10px','display':'inline-block'}
                    )
                ],align='center')
            ])
        
            self.upload_check = {'WSI':False,'Omics':False}
        
        elif upload_type =='Regular':
            # Regular slide with no --omics
            upload_reqs = html.Div([
                dbc.Row([
                    html.Div(
                        id = {'type':'wsi-upload-div','index':0},
                        children = [
                            dbc.Label('Upload Whole Slide Image Here!'),
                            UploadComponent(
                                id = {'type':'wsi-upload','index':0},
                                uploadComplete=False,
                                baseurl=self.dataset_handler.apiUrl,
                                girderToken=self.dataset_handler.user_token,
                                parentId=parentId,
                                filetypes=['svs','ndpi','scn','tiff','tif']                      
                            )
                        ],
                        style={'marginBottom':'10px','display':'inline-block'}
                    )
                ],align='center')
            ])
        
            self.upload_check = {'WSI':False}
        
        self.upload_wsi_id = None
        self.upload_omics_id = None

        return upload_reqs, input_disabled

    def girder_login(self,p_butt,create_butt,username,pword,email,firstname,lastname):

        create_user_children = []
        print(f'login ctx.triggered_id: {ctx.triggered_id}')
        usability_disabled = no_update
        if ctx.triggered_id=='login-submit':

            try:
                user_info = self.dataset_handler.authenticate(username,pword)

                button_color = 'success'
                button_text = 'Success!'
                logged_in_user = f'Welcome: {username}'
                upload_disabled = False

                if not user_info is None:
                    usability_disabled = False

            except girder_client.AuthenticationError:

                button_color = 'warning'
                button_text = 'Login Failed'
                logged_in_user = ''
                upload_disabled = True

            return button_color, button_text, logged_in_user, upload_disabled, create_user_children, [usability_disabled]
        
        elif ctx.triggered_id=='create-user-submit':
            print(f'email:{email}')
            print(f'firstname: {firstname}')
            print(f'lastname:{lastname}')
            if len(email)==0 or len(firstname)==0 or len(lastname)==0:
                create_user_children = [
                    dbc.Label('Email:',width='auto'),
                    dbc.Col(
                        dcc.Input(type='email',id={'type':'email-input','index':0})
                    ),
                    dbc.Label('First Name:',width='auto'),
                    dbc.Col(
                        dcc.Input(type='text',id={'type':'first-name-input','index':0})
                    ),
                    dbc.Label('Last Name',width='auto'),
                    dbc.Col(
                        dcc.Input(type='text',id={'type':'last-name-input','index':0})
                    )
                ]

                button_color = 'success'
                button_text = 'Create Account'
                logged_in_user = ''
                upload_disabled = True

            else:
                create_user_children = no_update
                try:
                    user_info = self.dataset_handler.create_user(username,pword,email,firstname,lastname)

                    button_color = 'success',
                    button_text = 'Success!',
                    logged_in_user = f'Welcome: {username}'
                    upload_disabled = False

                    if not user_info is None:
                        usability_disabled = False

                except girder_client.AuthenticationError:

                    button_color = 'warning'
                    button_text = 'Login Failed'
                    logged_in_user = ''
                    upload_disabled = True
            
            return button_color, button_text, logged_in_user, upload_disabled, create_user_children, [usability_disabled]

        else:
            raise exceptions.PreventUpdate

    def upload_data(self,wsi_file,omics_file,wsi_file_flag,omics_file_flag):

        print(f'Triggered id for upload_data: {ctx.triggered_id}')
        if ctx.triggered_id['type']=='wsi-upload':

            # Getting the uploaded item id
            self.upload_wsi_id = self.dataset_handler.get_new_upload_id(self.latest_upload_folder['id'])

            if not self.upload_wsi_id is None:
                if not wsi_file_flag[0]:
                    wsi_upload_children = [
                        dbc.Alert('WSI Upload Success!',color='success')
                    ]

                    self.upload_check['WSI'] = True

                else:
                    wsi_upload_children = [
                        dbc.Alert('WSI Upload Failure! Accepted file types include svs, ndpi, scn, tiff, and tif',color='danger'),
                        UploadComponent(
                            id = {'type':'wsi-upload','index':0},
                            uploadComplete=False,
                            baseurl=self.dataset_handler.apiUrl,
                            girderToken=self.dataset_handler.user_token,
                            parentId=self.latest_upload_folder['id'],
                            filetypes=['svs','ndpi','scn','tiff','tif']                      
                        )
                    ]

            else:
                wsi_upload_children = no_update
            
            if 'Omics' in self.upload_check:
                if not self.upload_check['Omics']:
                    omics_upload_children = no_update
                else:
                    omics_upload_children = [
                        dbc.Alert('Omics Upload Success!')
                    ]
            else:
                omics_upload_children = no_update

        elif ctx.triggered_id['type']=='omics-upload':
            
            self.upload_omics_id = self.dataset_handler.get_new_upload_id(self.latest_upload_folder['id'])
            if not self.upload_omics_id is None:
                if type(omics_file_flag)==list:
                    if len(omics_file_flag)>0:
                        if not omics_file_flag[0]:
                            omics_upload_children = [
                                dbc.Alert('Omics Upload Success!')
                            ]
                            self.upload_check['Omics'] = True
                        else:
                            omics_upload_children = [
                                dbc.Alert('Omics Upload Failure! Accepted file types are: rds',color = 'danger'),
                                UploadComponent(
                                    id = {'type':'omics-upload','index':0},
                                    uploadComplete=False,
                                    baseurl=self.dataset_handler.apiUrl,
                                    girderToken=self.dataset_handler.user_token,
                                    parentId=self.latest_upload_folder['id'],
                                    filetypes=['rds','csv']                 
                                    )
                            ]
                    else:
                        omics_upload_children = no_update
                else:
                    omics_upload_children = no_update
            else:
                omics_upload_children = no_update

            if not self.upload_check['WSI']:
                wsi_upload_children = no_update
            else:
                wsi_upload_children = [
                    dbc.Alert('WSI Upload Success!',color='success')
                ]

        else:
            print(f'ctx.triggered_id["type"]: {ctx.triggered_id["type"]}')

        # Checking the upload check
        if all([self.upload_check[i] for i in self.upload_check]):
            print('All set!')

            slide_thumbnail, slide_qc_table = self.slide_qc(self.upload_wsi_id)
            print(slide_qc_table)

            if 'Omics' in self.upload_check:
                omics_upload_children = [
                    dbc.Alert('Omics Upload Success!')
                ]
            else:
                omics_upload_children = no_update

            wsi_upload_children = [
                dbc.Alert('WSI Upload Success!',color='success')
            ]

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

            structure_type_disabled = False
            post_upload_style = {'display':'flex'}
            disable_upload_type = True
            
            if 'Omics' in self.upload_check:
                return slide_qc_results, thumb_fig, [wsi_upload_children], [omics_upload_children], structure_type_disabled, post_upload_style, disable_upload_type
            else:
                return slide_qc_results, thumb_fig, [wsi_upload_children], [], structure_type_disabled, post_upload_style, disable_upload_type
        else:

            disable_upload_type = True

            if 'Omics' in self.upload_check:
                return no_update, no_update,[wsi_upload_children], [omics_upload_children], True, no_update, disable_upload_type
            else:
                return no_update, no_update, [wsi_upload_children], [], True, no_update, disable_upload_type

    def slide_qc(self, upload_id):

        #try:
        #collection_contents = self.dataset_handler.get_collection_items(upload_id)
        thumbnail = self.dataset_handler.get_slide_thumbnail(upload_id)
        collection_contents = self.dataset_handler.gc.get(f'/item/{upload_id}')

        #histo_qc_run = self.dataset_handler.run_histo_qc(self.latest_upload_folder['id'])

        #TODO: Activate the HistoQC plugin from here and return some metrics
        histo_qc_output = pd.DataFrame(collection_contents)

        return thumbnail, histo_qc_output
    
    def start_segmentation(self,structure_selection,go_butt):

        # Starting segmentation job and initializing dcc.Interval object to check logs
        # output = div children, disable structure_type, disable segment_butt
        if ctx.triggered_id=='segment-butt':
            if structure_selection is not None:
                if len(structure_selection)>0:
                    disable_structure = True
                    disable_seg_butt = True

                    print(f'Running segmentation!')
                    self.segmentation_job_info = self.prep_handler.segment_image(self.upload_wsi_id,structure_selection)
                    print(f'Running spot annotation!')
                    if not self.upload_omics_id is None:
                        self.cell_deconv_job_info = self.prep_handler.run_cell_deconvolution(self.upload_wsi_id,self.upload_omics_id)

                    seg_woodshed = [
                        dcc.Interval(
                            id = {'type':'seg-log-interval','index':0},
                            interval = 3000,
                            max_intervals = -1,
                            n_intervals = 0
                        ),
                        html.Div(
                            id = {'type':'seg-logs','index':0},
                            children = []
                        )
                    ]

                    return seg_woodshed, disable_structure, disable_seg_butt
                else:
                    raise exceptions.PreventUpdate
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate
    
    def create_seg_upload(self,up_click):

        if up_click:
            
            upload_style = {
                'width': '100%',
                #'height': '60px',
                #'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            }
            # Creating the upload component for separate annotation files
            seg_up_children = [
                dbc.Row([
                    dbc.Col(
                        dcc.Upload(
                            id = {'type':'seg-file-upload','index':0},
                            children = [
                                'Drag and Drop or ',
                                html.A('Select Annotation File(s)')
                            ],
                            style = upload_style
                        ),
                        md = 6
                    ),
                    dbc.Col(
                        dbc.Accordion(
                            id = {'type':'seg-file-accordion','index':0},
                            children = []
                        ),
                        md = 6
                    )
                ])
            ]

            return seg_up_children
        
        else:
            raise exceptions.PreventUpdate

    def new_seg_upload(self,seg_upload,current_up_anns,seg_upload_filename):

        if seg_upload:
            # Processing new uploaded file
            current_len = len(current_up_anns)
            new_filename = seg_upload_filename[0]
            new_upload = seg_upload[0]
            
            # Processing newly uploaded annotations
            processed_anns = self.prep_handler.process_uploaded_anns(new_filename,new_upload,self.upload_wsi_id)

            return_items = current_up_anns[0]
            return_items.append(
                dbc.AccordionItem(
                    title = new_filename
                )
            )

            return [return_items]
        else:
            raise exceptions.PreventUpdate

    def update_logs(self,new_interval):

        # Callback to update the segmentation logs div with more data
        # Also populates the post-segment-row when the jobs are completed
        # output = seg-logs div, seg-interval disabled, post-segment-row style, 
        # structure-type disabled, ftu-select options, ftu-select value, sub-comp-method value,
        # ex-ftu-img figure
        # Getting most recent logs:
        seg_status = 0
        seg_log = []
        for seg_job in self.segmentation_job_info:
            s_stat, s_log = self.dataset_handler.get_job_status(seg_job['_id'])
            seg_log.append(s_log)
            seg_status+=s_stat
        seg_log = [html.P(s) for s in seg_log]

        if not self.upload_omics_id is None:
            cell_status, cell_log = self.dataset_handler.get_job_status(self.cell_deconv_job_info['_id'])
            cell_log = [html.P(cell_log)]
        else:
            cell_status = 3
            cell_log = ''
        
        # This would be at the end of the two jobs
        if seg_status+cell_status==3*(1+len(self.segmentation_job_info)):

            # Div containing the job logs:
            if not self.upload_omics_id is None:
                seg_logs_div = html.Div(
                    dbc.Row([
                        dbc.Col(html.Div(
                            dbc.Alert(
                                'Segmentation Complete!',
                                color = 'success'
                            )
                        ),md=6),
                        dbc.Col(html.Div(
                            dbc.Alert(
                                'Cell Deconvolution Complete!',
                                color = 'success'
                            )
                        ),md=6)
                    ],align='center'),style={'height':'200px','display':'inline-block'}
                )
            else:
                seg_logs_div = html.Div(
                    dbc.Row(
                        dbc.Col(html.Div(
                            dbc.Alert(
                                'Segmentation Complete!',
                                color = 'success'
                            )
                        ),md=12),align='center'
                    ),style={'height':'200px','display':'inline-block'}
                )

            # disabling interval object
            seg_log_disable = True

        else:
            # For during segmentation/cell-deconvolution
            seg_log_disable = False

            if not self.upload_omics_id is None:
                if seg_status==3:
                    seg_part = dbc.Alert(
                        'Segmentation Complete!',
                        color = 'success'
                    )
                else:
                    seg_part = seg_log
                
                if cell_status==3:
                    cell_part = dbc.Alert(
                        'Cell Deconvolution Complete!',
                        color = 'success'
                    )
                else:
                    cell_part = cell_log
                
                seg_logs_div = html.Div(
                    dbc.Row([
                        dbc.Col(html.Div(
                            children = seg_part
                        ),md=6),
                        dbc.Col(html.Div(
                            children = cell_part
                        ),md=6)
                    ],align='center'),style={'height':'200px','display':'inline-block'}
                )
            else:
                if seg_status==3:
                    seg_part = dbc.Alert(
                        'Segmentation Complete!',
                        color = 'success'
                    )
                else:
                    seg_part = seg_log
                
                seg_logs_div = html.Div(
                    dbc.Row(
                        dbc.Col(html.Div(
                            children = seg_part
                        ),md=12),align='center'
                    ),style={'height':'200px','display':'inline-block'}
                )

        print(f'disabling interval: {seg_log_disable}')
        return [seg_logs_div], [seg_log_disable]

    def post_segmentation(self, seg_log_disable):

        print(f'seg_log_disable: {seg_log_disable}')
        if seg_log_disable[0] is not None:
            if seg_log_disable[0]:
                # post-segment-row stuff
                sub_comp_style = {'display':'flex'}
                disable_organ = True

                if not self.upload_omics_id is None:
                    # Generating spot annotations based on cell types
                    spot_annotation_info = self.prep_handler.run_spot_annotation(self.upload_wsi_id,self.upload_omics_id)
                    # Aggregating spot-level cell composition information to intersecting FTUs
                    spot_aggregation_info = self.prep_handler.run_spot_aggregation(self.upload_wsi_id)

                # Extracting annotations and initial sub-compartment mask
                self.upload_annotations = self.dataset_handler.get_annotations(self.upload_wsi_id)

                # Populate with default sub-compartment parameters
                self.sub_compartment_params = self.prep_handler.initial_segmentation_parameters

                # Adding options to FTU options dropdown menu
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

                sub_comp_method = 'Manual'
                # Initializing layer and annotation idxes (starting with the first one that isn't disabled)
                self.layer_ann = {
                    'current_layer':[i['value'] for i in ftu_names if not i['disabled']][0],
                    'current_annotation':0,
                    'previous_annotation':0,
                    'max_layers':[len(i['annotation']['elements']) for i in self.upload_annotations if 'annotation' in i]
                }

                self.feature_extract_ftus = ftu_names
                ftu_value = ftu_names[self.layer_ann['current_layer']]
                image, mask = self.prep_handler.get_annotation_image_mask(self.upload_wsi_id,self.upload_annotations, self.layer_ann['current_layer'],self.layer_ann['current_annotation'])

                self.layer_ann['current_image'] = image
                self.layer_ann['current_mask'] = mask

                image_figure = go.Figure(
                    data = px.imshow(image)['data'],
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                )                

                return sub_comp_style, disable_organ, ftu_names, ftu_value, sub_comp_method, image_figure
            else:
                return no_update, no_update, no_update, no_update,no_update, no_update
        else:
            
            return no_update, no_update, no_update, no_update, no_update, no_update
    
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
            
            new_image, new_mask = self.prep_handler.get_annotation_image_mask(self.upload_wsi_id,self.upload_annotations,self.layer_ann['current_layer'],self.layer_ann['current_annotation'])
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

    def run_feature_extraction(self,feat_butt):

        self.feat_ext_job = self.prep_handler.run_feature_extraction(self.upload_wsi_id,self.sub_compartment_params)

        # Returning a dcc.Interval object to check logs for feature extraction
        feat_log_interval = [
            dcc.Interval(
                id = {'type':'feat-interval','index':0},
                interval = 1000,
                max_intervals=-1,
                n_intervals = 0
            ),
            html.Div(
                id = {'type':'feat-log-output','index':0},
                children = []
            )
        ]

        return [feat_log_interval]
    
    def update_feat_logs(self,new_interval):

        # Updating logs for feature extraction job, disabling when the job is done
        feat_ext_status, feat_ext_log = self.dataset_handler.get_job_status(self.feat_ext_job['_id'])

        if feat_ext_status==3:
            feat_logs_disable = True
            feat_logs_div = html.Div(
                children = [
                    dbc.Alert('Feature Extraction Complete!',color='success'),
                    dbc.Button('Go to Dataset-Builder',href='/dataset-builder')
                ],
                style = {'display':'inline-block','height':'200px'}
            )
        else:
            feat_logs_disable = False
            feat_logs_div = html.Div(
                children = [
                    feat_ext_log
                ],
                style = {'height':'200px','display':'inline-block'}
            )
        
        return [feat_logs_disable],[feat_logs_div]

    def ask_fusey(self,butt_click,current_style):

        # Generate some summary of the current view
        fusey_child = []
        fusey_style = {}
        if not current_style is None:
            if current_style['visibility']=='hidden':
                fusey_style = {
                    'visibility':'visible',
                    'background':'white',
                    'background':'rgba(255,255,255,0.8)',
                    'box-shadow':'0 0 15px rgba(0,0,0,0.2)',
                    'border-radius':'5px',
                    'display':'inline-block',
                    'position':'absolute',
                    'top':'75px',
                    'right':'10px',
                    'zIndex':'1000',
                    'padding':'8px 10px',
                    'width':'115px'
                }

                parent_style = {
                    'visibility': 'visible',
                    'display':'inline-block',
                    'position':'absolute',
                    'top':'75px',
                    'right':'10px',
                    'zIndex':'1000',
                    'padding':'8px 10px',
                    'width':'125px'
                }

                # Getting data for Fusey to use
                if 'Spots' in self.fusey_data:

                    # This will be the top cell type overall for a region
                    top_cell = self.fusey_data['Spots']['top_cell']
                    top_cell_states = self.fusey_data['Spots']['pct_states'].to_dict('records')
                    top_cell_pct = round(self.fusey_data['Spots']['normalized_counts'].loc[top_cell]*100,2)

                    table_data = self.table_df.dropna(subset=['CT/1/ABBR'])
                    table_data = table_data[table_data['CT/1/ABBR'].str.match(top_cell)]

                    try:
                        id = table_data['CT/1/ID'].tolist()[0]
                        # Modifying base url to make this link to UBERON
                        base_url = self.node_cols['Cell Types']['base_url']
                        new_url = base_url+id.replace('CL:','')
                    except:
                        new_url = ''

                    cell_state_text = []
                    for cs in top_cell_states:
                        cell_state_text.append(f'{cs["Cell State"]} ({round(cs["Proportion"]*100,2)}%)')
                    
                    cell_state_text = ' '.join(cell_state_text)

                    top_cell_text = f'The current region is mostly {self.cell_graphics_key[top_cell]["full"]} ({top_cell_pct}% in intersecting Spots).'
                    top_cell_text += f' {self.cell_graphics_key[top_cell]["full"]} cells in this region exhibit {cell_state_text} cell states. '

                    # Finding the FTU with most of this cell type:
                    top_ftu_text = ''
                    ftu_exp_list = []
                    for f in self.fusey_data:
                        if not f=='Spots':
                            ftu_exp_list.append(round(self.fusey_data[f]['normalized_counts'].loc[top_cell]*100,2))
                    
                    if len(ftu_exp_list)>0:
                        ftu_list = [i for i in list(self.fusey_data.keys()) if not i == 'Spots']
                        ftu_exp_list = [i if not np.isnan(i) else 0 for i in ftu_exp_list]
                        top_ftu = ftu_list[np.argmax(ftu_exp_list)]
                        top_ftu_pct = np.max(ftu_exp_list)

                        top_ftu_text = f'This cell type is mostly represented in {top_ftu} ({self.fusey_data[top_ftu]["structure_number"]} intersecting) with an average of {top_ftu_pct}%'
                    else:
                        top_ftu_text = ''

                    final_text = top_cell_text+top_ftu_text

                else:
                    
                    final_text = 'Uh oh! It looks like there are no Spots in this area!'

                fusey_child = [
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Img(src='./assets/fusey_clean.svg',height='75px',width='75px')),
                        ]),
                        dbc.Row([
                            dbc.Col(html.H4('Hi, my name is Fusey!',style={'fontSize':11}))

                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col(html.P(final_text,style={'fontSize':10}))
                        ]),
                        dbc.Row([
                            dbc.Col(html.A('Learn more about this cell!',href=new_url))
                        ])
                    ],style = fusey_style)
                ]
            elif current_style['visibility']=='visible': 
                parent_style = {
                    'visibility':'hidden'
                }
        
        return fusey_child, parent_style

    def add_label(self,notes_click,delete_click,pop_input):

        expected_outputs = callback_context.outputs_list
        triggered_list = [i['value'] for i in ctx.triggered]
        if not any(triggered_list):
            raise exceptions.PreventUpdate
        
        if len(expected_outputs)>1:
            output_list = [no_update]*(len(expected_outputs)-1)
            pop_input = pop_input[-1]
        else:
            pop_input = pop_input[0]
            output_list = []

        if ctx.triggered_id['type']=='add-popup-note':
            if not pop_input is None:
                # Adding provided label to ftu's user_label property
                self.wsi.add_label(self.clicked_ftu,pop_input,'add')

            # Getting current user-labels
            pop_label_children = self.layout_handler.get_user_ftu_labels(self.wsi,self.clicked_ftu)

        elif ctx.triggered_id['type']=='delete-user-label':
            
            # Removing label based on index
            remove_index = ctx.triggered_id['index']

            self.wsi.add_label(self.clicked_ftu,remove_index,'remove')

            # Getting current user-labels
            pop_label_children = self.layout_handler.get_user_ftu_labels(self.wsi,self.clicked_ftu)

        output_list.append(pop_label_children)

        return output_list

    def populate_cluster_tab(self,active_tab, cluster_butt):

        if active_tab == 'clustering-tab':
            if ctx.triggered_id=='tools-tabs':

                # Checking current clustering data (either a full or empty pandas dataframe)
                if not self.clustering_data.empty:
                    # Checking the current slides to see if they match self.current_slides:
                    current_slide_ids = [i['Slide_Id'] for i in self.clustering_data['Hidden'].tolist()]
                    unique_ids = np.unique(current_slide_ids).tolist()
                    current_ids = [i['_id'] for i in self.current_slides]
                    print(f'in self.clustering_data: {unique_ids}')
                    print(f'slide ids in current_slides: {current_ids}')
                    # Checking if these are the same (regardless of order)
                    if set(unique_ids)==set(current_ids):

                        # Populating feature-select-tree (tree), label-select (dropdown), filter-select-tree (tree)
                        get_data_div_children = dbc.Alert(
                            'Clustering data aligned!',
                            color='success',
                            dismissable=True,
                            is_open=True,
                            className='d-grid col-12 mx-auto'
                            )
                        
                        # Setting style of download plot data button
                        download_style = {'display':'inline-block'}
                        
                        # Generating options
                        self.dataset_handler.generate_feature_dict(self.current_slides)

                        feature_select_data = self.dataset_handler.plotting_feature_dict
                        label_select_options = self.dataset_handler.label_dict
                        label_select_value = self.dataset_handler.label_dict[0]['value']
                        label_select_disabled = False
                        filter_select_data = self.dataset_handler.label_filter_dict

                    else:

                        # If they are not aligned, have to grab aligning clustering data manually
                        get_data_div_children = dbc.Button(
                            'Get Data for Clustering!',
                            className = 'd-grid col-12 mx-auto',
                            id = {'type':'get-clustering-butt','index':0}
                        )

                        # Setting style of download plot data button
                        download_style = {'display':None}
                        self.feature_data = None
                        self.umap_df = None

                        feature_select_data = []
                        label_select_options = []
                        label_select_value = []
                        label_select_disabled = True
                        filter_select_data = []
                else:
                    # If its empty, have to grab aligning clustering data manually
                    get_data_div_children = dbc.Button(
                        'Get Data for Clustering!',
                        className = 'd-grid col-12 mx-auto',
                        id = {'type':'get-clustering-butt','index':0}
                    )

                    # Setting style of download plot data button
                    download_style = {'display':None}
                    self.feature_data = None
                    self.umap_df = None

                    feature_select_data = []
                    label_select_options = []
                    label_select_value = []
                    label_select_disabled = True
                    filter_select_data = []
            else:

                # Retrieving clustering data
                data_getting_response = self.dataset_handler.get_collection_annotation_meta([i['_id'] for i in self.current_slides if i['included']])
                
                self.dataset_handler.generate_feature_dict([i for i in self.current_slides if i['included']])
                # Monitoring getting the data:
                data_get_status = 0
                while data_get_status<3:
                    data_status, data_log = self.dataset_handler.get_job_status(data_getting_response['_id'])
                    data_get_status=data_status
                    print(f'data_get_status: {data_get_status}')
                    time.sleep(1)

                self.clustering_data = self.dataset_handler.load_clustering_data()

                get_data_div_children = dbc.Alert(
                    'Clustering data aligned!',
                    color = 'success',
                    dismissable=True,
                    is_open = True,
                    className='d-grid col-12 mx-auto'
                )

                # Setting style of download plot data button
                download_style = {'display':'inline-block'}
                self.feature_data = None
                self.umap_df = None

                feature_select_data = self.dataset_handler.plotting_feature_dict
                label_select_options = self.dataset_handler.label_dict
                label_select_value = self.dataset_handler.label_dict[0]['value']
                label_select_disabled = False
                filter_select_data = self.dataset_handler.label_filter_dict
    
        else:
            # If in another tab, just leave alone
            raise exceptions.PreventUpdate

        return get_data_div_children, feature_select_data, label_select_disabled, label_select_options, label_select_value, filter_select_data, download_style

    def update_plot_report(self,report_tab):

        # Return the contents of the plot report tab according to selection
        report_tab_children = dbc.Alert('Generate a plot first!',color = 'warning')
        if not self.feature_data is None:
            if report_tab in self.reports_generated:
                # Report already generated, return the report
                report_tab_children = self.reports_generated[report_tab]
            else:
                # Report hasn't been generated yet, generate it
                report_tab_children = self.layout_handler.gen_report_child(self.feature_data,report_tab)
                self.reports_generated[report_tab] = report_tab_children
            
        return report_tab_children

    def download_plot_data(self,download_button_clicked):

        if not download_button_clicked:
            raise exceptions.PreventUpdate
        
        if not self.feature_data is None:

            feature_columns = [i for i in self.feature_data if i not in ['label','Hidden','Main_Cell_Types','Cell_States']]

            # If this is umap data then save one sheet with the raw data and another with the umap embeddings
            if len(feature_columns)<=2:

                download_data_df = {
                    'FUSION_Plot_Features': self.feature_data.copy()
                }
            elif len(feature_columns)>2:
                download_data_df = {
                    'FUSION_Plot_Features': self.feature_data.copy(),
                    'UMAP_Embeddings': self.umap_df[self.umap_df.columns.intersection(['UMAP1','UMAP2'])].copy()
                }

            with pd.ExcelWriter('Plot_Data.xlsx') as writer:
                for sheet in download_data_df:
                    download_data_df[sheet].to_excel(writer,sheet_name=sheet,engine='openpyxl')

            return dcc.send_file('Plot_Data.xlsx')
        else:
            raise exceptions.PreventUpdate

    def start_cluster_markers(self,butt_click):

        # Clicked Get Cluster Markers
        print(ctx.triggered)
        if butt_click:

            disable_button = True
            
            # Saving current feature data to user public folder
            features_for_markering = self.dataset_handler.save_to_user_folder({
                'filename':f'marker_features.csv',
                'content': self.feature_data
            })
            print(features_for_markering)

            # Starting cluster marker job
            cluster_marker_job = self.dataset_handler.gc.post(
                    '/slicer_cli_web/samborder2256_clustermarkers_fusion_latest/ClusterMarkers/run',
                    parameters = {
                        'feature_address': f'{self.dataset_handler.apiUrl}/item/{features_for_markering["itemId"]}/download?token={self.dataset_handler.user_token}',
                        'girderApiUrl': self.dataset_handler.apiUrl,
                        'girderToken': self.dataset_handler.user_token,
                    }
                )
            print(cluster_marker_job)
            self.cluster_marker_job_id = cluster_marker_job['_id']

            # Returning a dcc.Interval object and another div to store the logs
            marker_logs_children = [
                dcc.Interval(
                    id = {'type':'markers-interval','index':0},
                    interval = 2000,
                    disabled = False,
                    max_intervals = -1,
                    n_intervals = 0
                ),
                html.Div(
                    id = {'type':'marker-logs-div','index':0},
                    children = []
                )
            ]

            return [marker_logs_children],[disable_button]
        else:
            raise exceptions.PreventUpdate

    def update_cluster_logs(self,new_interval):

        if not new_interval is None:
            # Checking the cluster marker job status and printing the log to the cluster_log_div
            marker_status, marker_log = self.dataset_handler.get_job_status(self.cluster_marker_job_id)

            if marker_status<3:
                
                marker_interval_disable = False
                cluster_log_children = [
                    html.P(i)
                    for i in marker_log.split('\n')
                ]

            else:

                marker_interval_disable = True

                # Load cluster markers from user folder
                cluster_marker_data = pd.DataFrame(self.dataset_handler.grab_from_user_folder('FUSION_Cluster_Markers.json'))

                cluster_log_children = [
                    dbc.Row(
                        dbc.Col(
                            html.Div(
                                dbc.Alert(
                                    'Markers Found!',
                                    color = 'success',
                                    dismissable = True,
                                    className = 'd-grid col-12 mx-auto'
                                )
                            ), md = 12
                        ), align = 'center'
                    ),
                    html.Br(),
                    html.Div([
                        html.A('Cluster Markers found using Seurat v4',href='https://satijalab.org/seurat/articles/pbmc3k_tutorial.html'),
                        html.P('Cluster identities determined by plot labels. Marker features determined using a Wilcoxon Rank Sum test.')
                    ]),
                    html.Div(
                        dash_table.DataTable(
                            id = 'cluster-marker-table',
                            columns = [{'name':i,'id':i} for i in cluster_marker_data.columns],
                            data = cluster_marker_data.to_dict('records'),
                            style_cell = {
                                'overflow':'hidden',
                                'textOverflow':'ellipsis',
                                'maxWidth':0
                            },
                            tooltip_data = [
                                {
                                    column: {'value':str(value),'type':'markdown'}
                                    for column, value in row.items()
                                } for row in cluster_marker_data.to_dict('records')
                            ],
                            tooltip_duration = None,
                            style_data_conditional = [
                                {
                                    'if': {
                                        'filter_query': '{p_val_adj} <0.05',
                                        'column_id':'p_val_adj'
                                    },
                                    'backgroundColor':'green',
                                    'color':'white'
                                },
                                {
                                    'if': {
                                        'filter_query': '{p_val_adj}>=0.05',
                                        'column_id':'p_val_adj'
                                    },
                                    'backgroundColor':'tomato',
                                    'color':'white'
                                }
                            ]
                        )
                    )
                ]
            
            self.reports_generated['feat-cluster-tab'] = cluster_log_children

            return [marker_interval_disable], [cluster_log_children]
        else:
            raise exceptions.PreventUpdate


def app(*args):
    
    # Using DSA as base directory for storage and accessing files
    #dsa_url = 'http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/'
    dsa_url = os.environ.get('DSA_URL')
    try:
        username = os.environ.get('DSA_USER')
        p_word = os.environ.get('DSA_PWORD')
    except:
        username = ''
        p_word = ''
        print(f'Be sure to set an initial user dummy!')

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
    dataset_handler.set_default_slides(initial_collection_contents)
    dataset_handler.initialize_folder_structure(initial_collection,path_type)
    dataset_handler.get_collection_annotation_meta([i['_id'] for i in initial_collection_contents])

    # Getting graphics_reference.json from the FUSION Assets folder
    print(f'Getting asset items')
    assets_path = '/collection/FUSION Assets/'
    dataset_handler.get_asset_items(assets_path)

    # Getting the slide data for DSASlide()
    slide_names = [i['name'] for i in initial_collection_contents]

    # Starting off with no WSI, don't load annotations until on the /vis page
    wsi = None

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
    #fusion_cli = ['Segment Anything Model (SAM)','Contrastive Language-Image Pre-training (CLIP)']

    external_stylesheets = [
        dbc.themes.LUX,
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
        ]

    print(f'Generating layouts')
    layout_handler = LayoutHandler()
    layout_handler.gen_initial_layout(slide_names,username)
    layout_handler.gen_vis_layout(
        wsi,
        cli_list
        )
    layout_handler.gen_builder_layout(dataset_handler)
    layout_handler.gen_uploader_layout(dataset_handler)

    download_handler = DownloadHandler(dataset_handler)

    prep_handler = PrepHandler(dataset_handler)
    
    print('Ready to rumble!')
    main_app = DashProxy(__name__,
                         external_stylesheets=external_stylesheets,
                         transforms = [MultiplexerTransform()],
                         )
    
    vis_app = FUSION(
        main_app,
        layout_handler,
        dataset_handler,
        download_handler,
        prep_handler
        )


# Comment this portion out for web running
if __name__=='__main__':
    app()
