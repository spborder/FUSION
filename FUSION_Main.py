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

import threading

from tqdm import tqdm

import shapely
from shapely.geometry import Point, shape, box
from skimage.transform import resize
import random

from umap.umap_ import UMAP

from uuid import uuid4
import textwrap

import girder_client

import plotly.express as px
import plotly.graph_objects as go
from matplotlib import colormaps

import dash
# Requirement for dash_mantine_components
dash._dash_renderer._set_react_version('18.2.0')
from dash import dcc, ctx, MATCH, ALL, Patch, dash_table, exceptions, callback_context, no_update

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign, arrow_function
from dash_extensions.enrich import DashProxy, html, Input, Output, MultiplexerTransform, State
import dash_mantine_components as dmc
import dash_treeview_antd as dta

from timeit import default_timer as timer
import time

from FUSION_WSI import DSASlide, VisiumSlide, CODEXSlide
from FUSION_Handlers import LayoutHandler, DownloadHandler, GirderHandler, GeneHandler
from FUSION_Prep import CODEXPrep, VisiumPrep, Prepper
from FUSION_Utils import (
    get_pattern_matching_value, extract_overlay_value, 
    gen_umap,
    gen_violin_plot, process_filters,
    path_to_mask,
    make_marker_geojson,
    gen_clusters)

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

        self.gene_handler = GeneHandler()

        # Setting some app-related things
        self.app = app
        self.app.title = "FUSION"
        self.app.layout = self.layout_handler.current_initial_layout
        self.app._favicon = './assets/favicon.ico'
        self.app.validation_layout = html.Div(self.layout_handler.validation_layout)
        
        # Setup GoogleTag for event tracking
        self.app.index_string = """
        <!DOCTYPE html>
        <html>
        <head>
            <!-- Google Tag Manager -->
                <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
                new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
                j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
                'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
                })(window,document,'script','dataLayer','GTM-WWS4Q54M');</script>
            <!-- End Google Tag Manager -->
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
        </head>
        <body>
            <!-- Google Tag Manager (noscript) -->
                <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WWS4Q54M"
                height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
                <!-- End Google Tag Manager (noscript) -->
            <!-- End Google Tag Manager (noscript) -->
            {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        </body>
        </html>"""

        # clustering related properties (and also cell types, cell states, image_ids, etc.)
        self.cell_graphics_key = self.dataset_handler.cell_graphics_key

        # Inverting the graphics key to get {'full_name':'abbreviation'}
        self.cell_names_key = {}
        for ct in self.cell_graphics_key:
            self.cell_names_key[self.cell_graphics_key[ct]['full']] = ct

        # Getting cell graphics color key sorted {'Red,Green,Blue': 'abbreviation'}
        self.cell_colors_key = {}
        for ct in self.cell_graphics_key:
            for c_c in self.cell_graphics_key[ct]['color_code']:
                self.cell_colors_key[c_c] = ct

        # Getting morphometrics reference from dataset_handler
        self.morphometrics_reference = self.dataset_handler.morphometrics_reference["Morphometrics"]
        self.morphometrics_names = self.dataset_handler.morpho_names

        # Load clustering data
        self.filter_labels = []
        self.reports_generated = {}
        
        # Initializing fusey_data as empty
        self.fusey_data = None

        # Number of main cell types to include in pie-charts (currently set to all cell types)
        self.plot_cell_types_n = len(list(self.cell_names_key.keys()))

        # ASCT+B table for cell hierarchy generation
        self.table_df = self.dataset_handler.asct_b_table    

        # FTU settings
        self.wsi = None
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
            'Cluster','Main_Cell_Types','Max Cell Type','Gene Counts','Morphometrics','Cell_Subtypes'
        ]

        # Initializing some parameters
        self.overlay_prop = None

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
        self.filter_vals = None
        self.overlap_prop = None
        self.hex_color_key = {}
        self.current_channels = {}

        # Initializing current annotation session
        self.current_ann_session = None

        # JavaScript functions for controlling annotation properties
        self.ftu_style_handle = assign("""function(feature,context){
            const {color_key,overlay_prop,fillOpacity,ftu_colors,filter_vals} = context.hideout;
                                       
            var overlay_value = Number.Nan;
            if (overlay_prop) {
                if (overlay_prop.name) {
                    if (overlay_prop.name in feature.properties.user) {
                        if (overlay_prop.value) {
                            if (overlay_prop.value in feature.properties.user[overlay_prop.name]) {
                                if (overlay_prop.sub_value) {
                                    if (overlay_prop.sub_value in feature.properties.user[overlay_prop.name][overlay_prop.value]) {
                                        var overlay_value = feature.properties.user[overlay_prop.name][overlay_prop.value][overlay_prop.sub_value];
                                    } else {
                                        var overlay_value = Number.Nan;
                                    }
                                } else {
                                    var overlay_value = feature.properties.user[overlay_prop.name][overlay_prop.value];
                                }
                            } else if (overlay_prop.value==="max") {
                                // Finding max represented sub-value
                                var overlay_value = Number.Nan;
                                var test_value = 0.0;
                                var overlay_idx = -1.0;
                                for (var key in feature.properties.user[overlay_prop.name]) {
                                    var tester = feature.properties.user[overlay_prop.name][key];
                                    overlay_idx += 1.0;
                                    if (tester > test_value) {
                                        test_value = tester;
                                        overlay_value = overlay_idx;
                                    }
                                } 
                            } else {
                                var overlay_value = Number.Nan;
                            }
                        } else {
                            var overlay_value = feature.properties.user[overlay_prop.name];
                        }
                    } else {
                        var overlay_value = Number.Nan;
                    }
                } else {
                    var overlay_value = Number.Nan;
                }
            } else {
                var overlay_value = Number.Nan;
            }
                                       
            var style = {};
            if (overlay_value == overlay_value) {
                if (overlay_value in color_key) {
                    const fillColor = color_key[overlay_value];
                    style.fillColor = fillColor;
                    style.fillOpacity = fillOpacity;        
                } else if (Number(overlay_value).toFixed(1) in color_key) {
                    const fillColor = color_key[Number(overlay_value).toFixed(1)];
                    style.fillColor = fillColor;
                    style.fillOpacity = fillOpacity;
                }
                                                                              
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
                style.fillColor = "f00";
            }           
                                                                              
            return style;
        }
            """
        )

        self.ftu_filter = assign("""function(feature,context){
                const {color_key,overlay_prop,fillOpacity,ftu_colors,filter_vals} = context.hideout;
                var return_feature = true;               
                if (filter_vals){
                    // If there are filters, use them
                    for (let i = 0; i < filter_vals.length; i++) {
                        // Iterating through filter_vals dict
                        var filter = filter_vals[i];         
                                                
                        if (filter.name) {
                            // Checking if the filter name is in the feature
                            if (filter.name in feature.properties.user) {
                                
                                if (filter.value) {
                                    if (filter.value in feature.properties.user[filter.name]) {
                                        if (filter.sub_value) {
                                            if (filter.sub_value in feature.properties.user[filter.name][filter.value]) {
                                                var test_val = feature.properties.user[filter.name][filter.value][filter.sub_value];
                                            } else {
                                                return_feature = return_feature & false;
                                            }
                                        } else {
                                            var test_val = feature.properties.user[filter.name][filter.value];
                                        }
                                    } else if (filter.value==="max") {
                                        return_feature = return_feature & true;
                                    } else {
                                        return_feature = return_feature & false;
                                    }
                                } else {
                                    var test_val = feature.properties.user[filter.name];
                                }
                            } else {
                                return_feature = return_feature & false;
                            }
                        }
                                 
                        if (filter.range) {
                            if (typeof filter.range[0]==='number') {
                                if (test_val < filter.range[0]) {
                                    return_feature = return_feature & false;
                                }
                                if (test_val > filter.range[1]) {
                                    return_feature = return_feature & false;
                                }   
                            } else {
                                if (filter.range.includes(return_feature)) {
                                    return_feature = return_feature & true;
                                } else {
                                    return_feature = return_feature & false;
                                }
                            }
                        }
                    }
                    
                    return return_feature;
                                 
                } else {
                    // If no filters are provided, return true for everything.
                    return return_feature;
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
        self.ga_callbacks()
        self.vis_callbacks()
        self.all_layout_callbacks()
        self.builder_callbacks()
        self.welcome_callbacks()
        self.upload_callbacks()

        # Running server
        #self.app.run_server(host = '0.0.0.0',debug=False,use_reloader=False,port=8000)
        serve(self.app.server,host='0.0.0.0',port=8000,threads = 10)

    def view_instructions(self,n,n2,is_open,user_data_store):
        print(user_data_store)
        user_data_store = json.loads(user_data_store)

        # Opening collapse and populating internal div 
        if ctx.triggered_id['type']=='collapse-descrip':
            collapse_children = self.layout_handler.description_dict[self.current_page]
            usability_color = ['primary']
        elif ctx.triggered_id['type']=='usability-butt':
            if n2:
                self.dataset_handler.update_usability()
                user_info = user_data_store['user_info']
                collapse_children = self.layout_handler.gen_usability_report(user_info,self.dataset_handler.usability_users)
                if not is_open[-1]:
                    usability_color = ['success']
                else:
                    usability_color = ['primary']
            else:
                usability_color = ['primary']

        if n or n2:
            return [not i for i in is_open], collapse_children, usability_color
        return [i for i in is_open], collapse_children, usability_color
    
    def view_sidebar(self,n,is_open):
        if n:
            return not is_open
        return is_open

    def update_page(self,pathname, user_data, vis_sess_store):
        
        vis_sess_store = json.loads(vis_sess_store)
        user_data = json.loads(user_data)
        pathname = pathname[1:]
        print(f'Navigating to {pathname}')

        slide_select_value = ''

        self.dataset_handler.update_usability()
        self.dataset_handler.clean_old_annotations()

        if pathname in self.layout_handler.layout_dict:
            self.current_page = pathname

            if not pathname=='vis':
                if not pathname=='dataset-builder':
                    if pathname=='dataset-uploader':
                        # Double-checking that a user is logged in before giving access to dataset-uploader
                        if user_data['login']=='fusionguest':
                            self.current_page = 'welcome'
                    container_content = self.layout_handler.layout_dict[self.current_page]
                else:
                    # Checking if there was any new slides added via uploader (or just added externally)
                    self.dataset_handler.update_folder_structure(user_data['login'])
                    self.layout_handler.gen_builder_layout(self.dataset_handler,user_data['login'])

                    container_content = self.layout_handler.layout_dict[self.current_page]

            else:

                # Generating visualization layout with empty clustering data, default filter vals, and no WSI
                self.wsi = None
                self.filter_vals = None
                self.layout_handler.gen_vis_layout(
                    self.wsi
                )

                # Checking vis_sess_store for any slides, if there aren't any 'included'==True then revert to default set
                included_slides = [i['included'] for i in vis_sess_store if 'included' in i]
                if len(included_slides)==0:
                    vis_sess_store = []
                    for s in self.dataset_handler.default_slides:
                        s['included'] = True
                        vis_sess_store.append(s)

                container_content = self.layout_handler.layout_dict[self.current_page]

        else:
            self.current_page = 'welcome'
                
            container_content = self.layout_handler.layout_dict[self.current_page]

        if self.current_page == 'vis':
            slide_style = {'marginBottom':'20px','display':'inline-block'}
        else:
            slide_style = {'display':'none'}

        return container_content, slide_style, slide_select_value, json.dumps(vis_sess_store)

    def open_nav_collapse(self,n,is_open):
        if n:
            return not is_open
        return is_open

    def ga_callbacks(self):

        # Callbacks with recorded user data for Google Analytics
        # GTM tracking setup for every page
        self.app.clientside_callback(
            "dash_clientside.clientside.trackPageView",
            Output('ga-invisible-div', 'children'),
            [Input('url', 'pathname')]
        )

        # Tracking user login
        self.app.clientside_callback(
            "window.dash_clientside.clientside.updateDataLayerWithUserId",
            Output('dummy-div-for-userId', 'children'),
            [Input('user-id-div', 'children')]
        )

        # Tracking manual_roi data
        self.app.clientside_callback(
            "window.dash_clientside.clientside.trackManualRoiData",
            Output('dummy-div-user-annotations', 'children'),
            [Input('user-annotations-div', 'children')]
        )

        # Tracking PlugIns
        self.app.clientside_callback(
            "window.dash_clientside.clientside.trackPlugInData",
            Output('dummy-div-plugin-track', 'children'),
            [Input('plugin-ga-track', 'children')]
        )

    def all_layout_callbacks(self):

        # Adding callbacks for items in every page

        # Updating items in page
        self.app.callback(
            [Output('container-content','children'),
             Output('slide-select-card','style'),
             Output('slide-select','value'),
             Output('visualization-session-store','data')],
             Input('url','pathname'),
             [State('user-store','data'),
              State('visualization-session-store','data')],
             prevent_initial_call = True
        )(self.update_page)

        # Opening the description/usability collapse content
        self.app.callback(
            [Output({'type':'collapse-content','index':ALL},'is_open'),
             Output('descrip','children'),
             Output({'type':'usability-butt','index':ALL},'color')],
            [Input({'type':'collapse-descrip','index':ALL},'n_clicks'),
             Input({'type':'usability-butt','index':ALL},'n_clicks')],
            [State({'type':'collapse-content','index':ALL},'is_open'),
             State('user-store','data')],
            prevent_initial_call=True
        )(self.view_instructions)

        # Open/close nav bar when screen/window is too small
        self.app.callback(
            Output('navbar-collapse','is_open'),
            Input('navbar-toggler','n_clicks'),
            State('navbar-collapse','is_open')
        )(self.open_nav_collapse)

        # Opening the sidebar to access other pages
        self.app.callback(
            Output({'type':'sidebar-offcanvas','index':MATCH},'is_open'),
            Input({'type':'sidebar-button','index':MATCH},'n_clicks'),
            [State({'type':'sidebar-offcanvas','index':MATCH},'is_open')],
            prevent_initial_call=True
        )(self.view_sidebar)

        # Logging in to DSA instance
        self.app.callback(
            [   
                Output('login-submit','color'),
                Output('login-submit','children'),
                Output('logged-in-user','children'),
                Output('upload-sidebar','disabled'),
                Output('create-user-extras','children'),
                Output('user-id-div', 'children'),
                Output({'type':'usability-sign-up','index':ALL},'style'),
                Output({'type':'usability-butt','index':ALL},'style'),
                Output('user-store','data')
            ],
            [
                Input('login-submit','n_clicks'),
                Input('create-user-submit','n_clicks')
            ],
            [
                State('username-input','value'),
                State('pword-input','value'),
                State({'type':'email-input','index':ALL},'value'),
                State({'type':'first-name-input','index':ALL},'value'),
                State({'type':'last-name-input','index':ALL},'value')
            ],
            prevent_initial_call=True
        )(self.girder_login)

        # Loading new tutorial slides
        self.app.callback(
            Input({'type':'tutorial-tabs','index':ALL},'active_tab'),
            Output({'type':'tutorial-content','index':ALL},'children'),
        )(self.update_tutorial_slide)
        
        # Updating questions in question tab
        self.app.callback(
            Input({'type':'questions-tabs','index':ALL},'active_tab'),
            State('logged-in-user','children'),
            Output({'type':'question-div','index':ALL},'children'),
        )(self.update_question_div)

        # Consenting to participate in the study, activating other question tabs
        self.app.callback(
            Input({'type':'study-consent-butt','index':ALL},'n_clicks'),
            [Output({'type':'question-tab','index':ALL},'disabled'),
             Output({'type':'study-consent-butt','index':ALL},'color')],
            prevent_initial_call = True
        )(self.consent_to_usability_study)

        # Posting question responses to usability info file
        self.app.callback(
            Input({'type':'questions-submit','index':ALL},'n_clicks'),
            Output({'type':'questions-submit-alert','index':ALL},'children'),
            State({'type':'question-input','index':ALL},'value'),
            prevent_initial_call = True
        )(self.post_usability_response)

        # Downloading usability data for admins
        self.app.callback(
            Output({'type':'usability-download','index':ALL},'data'),
            Input({'type':'download-usability-butt','index':ALL},'n_clicks'),
            prevent_initial_call = True
        )(self.download_usability_response)

    def vis_callbacks(self):

        # Updating GeoJSON fill/color/filter
        self.app.callback(
            [Output({'type':'ftu-bounds','index':ALL},'hideout'),Output('colorbar-div','children'),
             Output('filter-slider','min'),Output('filter-slider','max'),Output('filter-slider','disabled'),
             Output('cell-sub-select-div','children'),Output({'type':'gene-info-div','index':ALL},'style'),
             Output({'type':'gene-info-div','index':ALL},'children')],
            [Input('cell-drop','value'),Input('vis-slider','value'),
             Input('filter-slider','value'), Input({'type':'added-filter-slider','index':ALL},'value'),
             Input({'type':'added-filter-drop','index':ALL},'value'),
             Input({'type':'ftu-bound-color-butt','index':ALL},'n_clicks'),
             Input({'type':'cell-sub-drop','index':ALL},'value')],
            [State({'type':'ftu-bound-color','index':ALL},'value'),
             State({'type':'added-filter-slider-div','index':ALL},'style')],
            prevent_initial_call = True
        )(self.update_overlays)

        # Getting Anatomical Structures and Cell Types that contain a certain biomarker
        self.app.callback(
            Output({'type':'asct-gene-table','index':ALL},'children'),
            Input({'type':'get-asct-butt','index':ALL},'n_clicks'),
            State({'type':'hgnc-id','index':ALL},'children'),
            prevent_initial_call = True
        )(self.get_asct_table)
        
        # Adding filter to apply to structures in the image
        self.app.callback(
            [
                Output('parent-added-filter-div','children',allow_duplicate=True),
            ],
            [
                Input('add-filter-button','n_clicks'),
                Input({'type':'delete-filter','index':ALL},'n_clicks')
            ],
            prevent_initial_call = True
        )(self.add_filter)

        # Adding min/max values for an added slider
        self.app.callback(
            [
                Output({'type':'added-filter-slider','index':MATCH},'min'),
                Output({'type':'added-filter-slider','index':MATCH},'max'),
                Output({'type':'added-filter-slider-div','index':MATCH},'style')
            ],
            Input({'type':'added-filter-drop','index':MATCH},'value'),
            prevent_initial_call = True
        )(self.add_filter_slider)

        # Updating Cell Composition pie charts
        self.app.callback(
            [Output('roi-pie-holder','children'),
             Output('annotation-session-div','children'),
             Output({'type':'frame-label-drop','index':ALL},'disabled'),
             Output({'type':'viewport-store-data','index':ALL},'data'),
             ],
            [Input('slide-map','bounds'),
             Input({'type':'roi-view-data','index':ALL},'value'),
             Input({'type':'frame-data-butt','index':ALL},'n_clicks'),
             Input({'type':'cell-label-cluster','index':ALL},'n_clicks')
             ],
            [State('tools-tabs','active_tab'),
             State('viewport-data-update','checked'),
             State({'type':'frame-histogram-drop','index':ALL},'value'),
             State({'type':'frame-label-drop','index':ALL},'value'),
             State({'type':'cell-label-eps','index':ALL},'value'),
             State({'type':'cell-label-min-samples','index':ALL},'value'),
             State({'type':'viewport-store-data','index':ALL},'data')
             ],
            prevent_initial_call = True
        )(self.update_viewport_data)      

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
            [Output('slide-tile-holder','children'),
             Output('layer-control','children'),
             Output({'type':'edit_control','index':ALL},'editToolbar'),
             Output('slide-map','center'),
             #Output('slide-map','bounds'),
             Output('cell-drop','options'),
             Output('ftu-bound-opts','children'),
             Output('special-overlays','children'),
             Output('ftu-structure-hierarchy','children')],
            Input('slide-load-interval','disabled'),
            prevent_initial_call=True,
            suppress_callback_exceptions=True
        )(self.ingest_wsi)

        # Displaying slide loading progress
        self.app.callback(
            [
                Output('slide-load-modal','is_open'),
                Output('slide-load-modal','children'),
                Output('slide-info-store','data'),
                Output('slide-load-interval','disabled')
            ],
            [
                Input('slide-select','value'),
                Input('slide-load-interval','n_intervals')
            ],
            [
                State('slide-load-modal','is_open'),
                State('slide-info-store','data')
            ],
            prevent_initial_call = True
        )(self.load_new_slide)
        
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
             Output('label-select','options'),
             Output('label-info','children'),
             Output('filter-info','children'),
             Output('plot-report-tab','active_tab'),
             Output('download-plot-butt','disabled'),
             Output('cluster-store','data')],
            [State('feature-select-tree','checked'),
             State('filter-select-tree','checked'),
             State('cell-states-clustering','value'),
             State('cluster-store','data'),
             State('user-store','data')],
            prevent_initial_call=True
        )(self.update_graph)

        # Grabbing image(s) from morphometric cluster plot
        self.app.callback(
            [Input('cluster-graph','clickData'),
            Input('cluster-graph','selectedData')],
            State('cluster-store','data'),
            [Output('selected-image','figure'),
            Output('selected-cell-types','figure'),
            Output('selected-cell-states','figure'),
            Output('selected-image-info','children'),
            Output('cluster-store','data')],
            prevent_initial_call=True
        )(self.update_selected)

        # Updating cell states bar chart from within the selected point(s) in morphometrics cluster plot
        self.app.callback(
            Input('selected-cell-types','clickData'),
            State('cluster-store','data'),
            Output('selected-cell-states','figure'),
            prevent_initial_call=True
        )(self.update_selected_state_bar)

        # Adding manual ROIs using EditControl
        self.app.callback(
            Input({'type':'edit_control','index':ALL},'geojson'),
            [Output('layer-control','children'),
             Output('data-select','options'),
             Output('user-annotations-div', 'children')],
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
            State('visualization-session-store','data'),
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
        """
        # Callback for Ask Fusey
        self.app.callback(
            [Output('ask-fusey-box','children'),
             Output('ask-fusey-box','style')],
            Input('fusey-button','n_clicks'),
            [State('ask-fusey-box','style'),
             State('tools-tabs','active_tab')],
            prevent_initial_call = True
        )(self.ask_fusey)
        """

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
             Output('download-plot-data-div','style'),
             Output('plugin-ga-track', 'children'),
             Output('cluster-store','data')],
            [Input('tools-tabs','active_tab'),
             Input({'type':'get-clustering-butt','index':ALL},'n_clicks')],
            [State('cluster-store','data'),
             State('visualization-session-store','data'),
             State('user-store','data')],
            prevent_initial_call=True
        )(self.populate_cluster_tab)

        # Updating contents of plot-report tabs when switched
        self.app.callback(
            Output('plot-report-div','children'),
            Input('plot-report-tab','active_tab'),
            State('cluster-store','data'),
            prevent_initial_call = True
        )(self.update_plot_report)

        # Downloading data in the current plot
        self.app.callback(
            Output('download-plot-data','data'),
            Input('download-plot-butt','n_clicks'),
            State('cluster-store','data'),
            prevent_initial_call = True
        )(self.download_plot_data)

        # Find cluster markers button clicked and return dcc.Interval object
        self.app.callback(
            [Output({'type':'cluster-marker-div','index':ALL},'children'),
             Output({'type':'cluster-markers-butt','index':ALL},'disabled'),
             Output('plugin-ga-track', 'children'),
             Output('user-store','data')],
            Input({'type':'cluster-markers-butt','index':ALL},'n_clicks'),
            [State('cluster-store','data'),
             State('user-store','data')],
            prevent_initial_call = True
        )(self.start_cluster_markers)

        # Updating logs from cluster markers job
        self.app.callback(
            [Output({'type':'markers-interval','index':ALL},'disabled'),
             Output({'type':'marker-logs-div','index':ALL},'children')],
            Input({'type':'markers-interval','index':ALL},'n_intervals'),
            State('user-store','data'),
            prevent_initial_call = True
        )(self.update_cluster_logs)

        # Special overlay populating
        self.app.callback(
            [Output({'type':'channel-overlay-select-div','index': ALL},'children'),
             Output({'type':'channel-overlay-butt','index':ALL},'disabled')],
            Input({'type':'channel-overlay-drop','index': ALL},'value'),
            prevent_initial_call = True,
        )(self.add_channel_color_select)

        # Adding CODEX channel overlay
        self.app.callback(
            [Output({'type':'codex-tile-layer','index':ALL},'url'),
             Output({'type':'overlay-channel-tab','index':ALL},'label_style')],
            Input({'type':'channel-overlay-butt','index':ALL},'n_clicks'),
            [State({'type':'overlay-channel-color','index':ALL},'value'),
            State({'type':'overlay-channel-tab','index':ALL},'label')],
            prevent_initial_call = True
        )(self.add_channel_overlay)

        # Populating structure hierarchy tab
        self.app.callback(
            [
                Output({'type':'structure-tree-div','index':ALL},'children'),
                Output({'type':'reset-structure-hierarchy','index':ALL},'disabled')
            ],
            Input('ftu-structure-hierarchy','active_tab'),
            prevent_initial_call = True
        )(self.populate_structure_hierarchy)

        # Grabbing points from current viewport umap (CODEX) and plotting other properties
        self.app.callback(
            [
                Output({'type':'extracted-cell-labeling','index':ALL},'children'),
                Output({'type':'cell-labeling-criteria','index':ALL},'children'),
                Output('marker-add-div','children')
            ],
            [
                Input({'type':'ftu-cell-pie','index':ALL},'selectedData'),
            ],
            [
                State({'type':'cell-checkbox','index':ALL},'value'),
                State({'type':'ftu-cell-pie','index':ALL},'selectedData')
            ],
            prevent_initial_call = True
        )(self.cell_labeling_initialize)

        # Populating tabs for each annotation session in Annotation Station
        self.app.callback(
            [
                Output({'type':'annotation-session-content','index':ALL},'children'),
                Output({'type':'annotation-tab-group','index':ALL},'children')
            ],
            [
                Input({'type':'annotation-tab-group','index':ALL},'active_tab'),
                Input({'type':'create-annotation-session-button','index':ALL},'n_clicks')
            ],
            [
                State({'type':'annotation-session-name','index':ALL},'value'),
                State({'type':'annotation-session-description','index':ALL},'value'),
                State({'type':'annotation-session-add-users','index':ALL},'value'),
                State({'type':'ann-sess-tab','index':ALL},'label'),
                State({'type':'new-annotation-class','index':ALL},'value'),
                State({'type':'new-annotation-color','index':ALL},'value'),
                State({'type':'new-annotation-label','index':ALL},'value'),
                State({'type':'new-annotation-user','index':ALL},'value'),
                State({'type':'new-user-type','index':ALL},'value')
            ],
            prevent_initial_call = True
        )(self.update_annotation_session)

        # Add new image annotation class
        self.app.callback(
            [
                Output({'type':'annotation-classes-parent-div','index':ALL},'children',allow_duplicate=True)
            ],
            [
                Input({'type':'add-annotation-class','index':ALL},'n_clicks'),
                Input({'type':'delete-annotation-class','index':ALL},'n_clicks')
            ],
            prevent_initial_call = True
        )(self.add_annotation_class)

        # Add new annotation text label
        self.app.callback(
            [
                Output({'type':'annotation-labels-parent-div','index':ALL},'children')
            ],
            [
                Input({'type':'add-annotation-label','index':ALL},'n_clicks'),
                Input({'type':'delete-annotation-label','index':ALL},'n_clicks')
            ],
            prevent_initial_call = True
        )(self.add_annotation_label)

        # Add new annotation session user
        self.app.callback(
            [
                Output({'type':'annotation-session-users-parent','index':ALL},'children')
            ],
            [
                Input({'type':'add-annotation-user','index':ALL},'n_clicks'),
                Input({'type':'delete-annotation-user','index':ALL},'n_clicks')
            ],
            prevent_initial_call = True
        )(self.add_annotation_user)

        # Updating annotation data and annotation structure
        self.app.callback(
            [
                Output({'type':'annotation-current-structure','index':ALL},'figure'),
                Output({'type':'annotation-station-ftu','index':ALL},'style'),
                Output({'type':'annotation-class-label','index':ALL},'value'),
                Output({'type':'annotation-image-label','index':ALL},'value'),
                Output({'type':'annotation-station-ftu-idx','index':ALL},'children'),
                Output({'type':'annotation-save-button','index':ALL},'color'),
                Output({'type':'annotation-session-progress','index':ALL},'value'),
                Output({'type':'annotation-session-progress','index':ALL},'label')
            ],
            [
                Input({'type':'annotation-station-ftu','index':ALL},'n_clicks'),
                Input({'type':'annotation-previous-button','index':ALL},'n_clicks'),
                Input({'type':'annotation-next-button','index':ALL},'n_clicks'),
                Input({'type':'annotation-save-button','index':ALL},'n_clicks'),
                Input({'type':'annotation-set-label','index':ALL},'n_clicks'),
                Input({'type':'annotation-delete-label','index':ALL},'n_clicks'),
                Input({'type':'annotation-line-slider','index':ALL},'value'),
                Input({'type':'annotation-class-select','index':ALL},'value')
            ],
            [
                State({'type':'annotation-current-structure','index':ALL},'relayoutData'),
                State({'type':'annotation-class-label','index':ALL},'value'),
                State({'type':'annotation-image-label','index':ALL},'value'),
                State({'type':'annotation-station-ftu','index':ALL},'children'),
                State({'type':'annotation-station-ftu-idx','index':ALL},'children')
            ],
            prevent_initial_call = True
        )(self.update_current_annotation)

        # Adding cell subtypes to current annotations based on contents of molecular data file
        self.app.callback(
            [
                Output('cell-drop','options'),
                Output({'type':'cell-subtype-drop','index':ALL},'value')
            ],
            [
                Input({'type':'cell-subtype-butt','index':ALL},'n_clicks')
            ],
            [
                State({'type':'cell-subtype-drop','index':ALL},'value')
            ],
            prevent_initial_call = True
        )(self.add_cell_subtypes)

    def builder_callbacks(self):

        # Initializing plot after selecting dataset(s)
        self.app.callback(
            Input('dataset-table','selected_rows'),
            State('visualization-session-store','data'),
            [Output('selected-dataset-slides','children'),
             Output('slide-metadata-plots','children'),
             Output('slide-select','options'),
             Output('visualization-session-store','data')],
             prevent_initial_call=True
        )(self.initialize_metadata_plots)

        # Updating metadata plot based on selected parameters
        self.app.callback(
            [Input({'type':'meta-drop','index':ALL},'value'),
             Input({'type':'cell-meta-drop','index':ALL},'value'),
             Input({'type':'agg-meta-drop','index':ALL},'value'),
             Input({'type':'slide-dataset-table','index':ALL},'selected_rows')],
             State('visualization-session-store','data'),
             [Output('slide-select','options'),
             Output({'type':'meta-plot','index':ALL},'figure'),
             Output({'type':'cell-meta-drop','index':ALL},'options'),
             Output({'type':'cell-meta-drop','index':ALL},'disabled'),
             Output({'type':'current-slide-count','index':ALL},'children'),
             Output('visualization-session-store','data')],
             prevent_initial_call = True
        )(self.update_metadata_plot)

    def welcome_callbacks(self):
        
        # Updating tutorial slides shown
        self.app.callback(
            [Input({'type':'tutorial-name','index':ALL},'n_clicks'),
             Input({'type':'tutorial-sub-part','index':ALL},'n_clicks')],
            [Output('welcome-tutorial','children'),
             Output('tutorial-name','children'),
             Output({'type':'welcome-tutorial-slides','index':ALL},'active_index'),
             Output({'type':'tutorial-name','index':ALL},'style'),
             Output({'type':'tutorial-parts','index':ALL},'children'),
             Output({'type':'tutorial-sub-part','index':ALL},'style')],
            [State({'type':'tutorial-sub-part','index':ALL},'children')]
        )(self.get_tutorial)

    def upload_callbacks(self):

        # Creating upload components depending on omics type
        self.app.callback(
            Input('upload-type','value'),
            [Output('upload-requirements','children'),
             Output('upload-type','disabled'),
             Output('user-store','data')],
            State('user-store','data'),
            prevent_initial_call=True
        )(self.update_upload_requirements)

        # Uploading data to DSA collection
        self.app.callback(
            [Input({'type':'wsi-upload','index':ALL},'uploadComplete'),
             Input({'type':'omics-upload','index':ALL},'uploadComplete'),
             Input({'type':'wsi-upload','index':ALL},'fileTypeFlag'),
             Input({'type':'omics-upload','index':ALL},'fileTypeFlag')],
             State('user-store','data'),
            [Output('slide-qc-results','children'),
             Output('slide-thumbnail-holder','children'),
             Output({'type':'wsi-upload-div','index':ALL},'children'),
             Output({'type':'omics-upload-div','index':ALL},'children'),
             Output('structure-type','disabled'),
             Output('post-upload-row','style'),
             Output('upload-type','disabled'),
             Output('plugin-ga-track','children'),
             Output('user-store','data')],
            prevent_initial_call=True
        )(self.upload_data)

        # Adding slide metadata
        self.app.callback(
            Input({'type':'add-slide-metadata','index':ALL},'n_clicks'),
            [State({'type':'slide-qc-table','index':ALL},'data'),
             State('user-store','data')],
            Output({'type':'slide-qc-table','index':ALL},'data'),
            prevent_initial_call = True
        )(self.add_slide_metadata)

        # Starting segmentation for selected structures
        self.app.callback(
            [
                Output('seg-woodshed','children'),
                Output('structure-type','disabled'),
                Output('segment-butt','disabled'),
                Output({'type':'seg-continue-butt','index':ALL},'disabled'),
                Output('user-store','data')
            ],
            [
                Input('structure-type','value'),
                Input('segment-butt','n_clicks')
            ],
            State('user-store','data'),
            prevent_initial_call = True
        )(self.start_segmentation)

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
             State({'type':'seg-file-upload','index':ALL},'filename'),
             State('user-store','data')],
            prevent_initial_call = True
        )(self.new_seg_upload)

        # Updating log output for segmentation
        self.app.callback(
            [
                Output({'type':'seg-logs','index':ALL},'children'),
                Output({'type':'seg-log-interval','index':ALL},'disabled'),
                Output({'type':'seg-continue-butt','index':ALL},'disabled')
            ],
            [
                Input({'type':'seg-log-interval','index':ALL},'n_intervals')
            ],
            State('user-store','data'),
            prevent_initial_call = True
        )(self.update_logs)

        # Populating the post-segmentation sub-compartment segmentation and feature extraction row
        self.app.callback(
            [
                Output('post-segment-row','style'),
                Output('structure-type','disabled'),
                Output({'type':'prep-div','index':ALL},'children')
            ],
            [
                Input({'type':'seg-log-interval','index':ALL},'disabled'),
                Input({'type':'seg-continue-butt','index':ALL},'n_clicks')
            ],
            [
                State({'type':'organ-select','index':ALL},'value'),
                State({'type':'gene-selection-method','index':ALL},'value'),
                State({'type':'gene-selection-n','index':ALL},'value'),
                State({'type':'gene-selection-list','index':ALL},'value'),
                State('user-store','data')
            ],
            prevent_initial_call = True
        )(self.post_segmentation)

        # Updating sub-compartment segmentation
        self.app.callback(
            [Input({'type':'ftu-select','index':ALL},'value'),
             Input({'type':'prev-butt','index':ALL},'n_clicks'),
             Input({'type':'next-butt','index':ALL},'n_clicks'),
             Input({'type':'go-to-feat','index':ALL},'n_clicks'),
             Input({'type':'ex-ftu-view','index':ALL},'value'),
             Input({'type':'ex-ftu-slider','index':ALL},'value'),
             Input({'type':'sub-thresh-slider','index':ALL},'value'),
             Input({'type':'sub-comp-method','index':ALL},'value')],
             [State({'type':'go-to-feat','index':ALL},'disabled'),
              State('user-store','data')],
             [Output({'type':'ex-ftu-img','index':ALL},'figure'),
              Output({'type':'sub-thresh-slider','index':ALL},'marks'),
              Output({'type':'feature-items','index':ALL},'children'),
              Output({'type':'sub-thresh-slider','index':ALL},'disabled'),
              Output({'type':'sub-comp-method','index':ALL},'disabled'),
              Output({'type':'go-to-feat','index':ALL},'disabled')],
            prevent_initial_call = True
        )(self.update_sub_compartment)

        # Grabbing new frame slide thumbnail for nucleus segmentation
        self.app.callback(
            [
                Input({'type':'frame-thumbnail','index':ALL},'clickData'),
                Input({'type':'frame-select','index':ALL},'value')
            ],
            [
                Output({'type':'frame-thumbnail','index':ALL},'figure'),
                Output({'type':'ex-nuc-img','index':ALL},'figure')
            ],
            prevent_initial_call = True
        )(self.grab_nuc_region)

        # Updating nucleus segmentation parameters for CODEX prep
        self.app.callback(
            [
                Input({'type':'nuc-seg-method','index':ALL},'value'),
                Input({'type':'nuc-thresh-slider','index':ALL},'value'),
                Input({'type':'ex-nuc-view','index':ALL},'value'),
                Input({'type':'go-to-feat','index':ALL},'n_clicks')
            ],
            [
                Output({'type':'ex-nuc-img','index':ALL},'figure'),
                Output({'type':'nuc-thresh-slider','index':ALL},'marks')
            ],
            prevent_initial_call = True
        )(self.update_nuc_segmentation)

        # Running feature extraction plugin
        self.app.callback(
            [Input({'type':'start-feat','index':ALL},'n_clicks'),
             Input({'type':'skip-feat','index':ALL},'n_clicks')],
            [Output({'type':'feat-logs','index':ALL},'children'),
             Output('user-store','data')],
            [State({'type': 'include-ftu-drop','index':ALL},'value'),
             State({'type': 'include-feature-drop','index':ALL},'value'),
             State('user-store','data')],
            prevent_initial_call=True
        )(self.run_feature_extraction)

        # Updating logs for feature extraction
        self.app.callback(
            [
                Output({'type':'feat-interval','index':ALL},'disabled'),
                Output({'type':'feat-log-output','index':ALL},'children')
            ],
            [
                Input({'type':'feat-interval','index':ALL},'n_intervals')
            ],
            State('user-store','data'),
            prevent_initial_call = True
        )(self.update_feat_logs)
    
    def get_tutorial(self,a_click, sub_click,current_parts):
        """
        Getting next slide deck tutorial
        """

        if not ctx.triggered[0]['value']:
            raise exceptions.PreventUpdate

        click_key = [i['name'] for i in self.layout_handler.tutorial_content['categories']]
        slide_active_index = [0]*len(ctx.outputs_list[2])
        sub_parts_style = [no_update]*len(ctx.outputs_list[-1])
        new_slides = no_update

        # Returning style list for html.A components
        selected_style = {
            'background':'rgba(255,255,255,0.8)',
            'box-shadow':'0 0 10px rgba(0,0,0,0.2)',
            'border-radius':'5px',
        }

        sub_selected_style = {
            'background':'rgba(255,255,255,0.8)',
            'box-shadow':'0 0 5px rgba(0,0,0,0.2)',
            'border-radius':'3px',
            'fontSize':12,
            'marginLeft':'25px'
        }

        if ctx.triggered_id['type']=='tutorial-name':
            tutorial_name = html.H3(click_key[ctx.triggered_id["index"]])
            new_items_list = [{
                'key':f'{i+1}',
                'src':f'./static/tutorials/{click_key[ctx.triggered_id["index"]]}/slide_{i}.svg',
                'img_style':{'height':'60vh','width':'80%'}
                }
                for i in range(len(os.listdir(f'./static/tutorials/{click_key[ctx.triggered_id["index"]]}/')))
            ]

            new_slides = dbc.Carousel(
                id = {'type':'welcome-tutorial-slides','index':0},
                items = new_items_list,
                active_index = 1,
                controls = True,
                indicators = True,
                variant = 'dark'
            )

            style_list = [{} if not i==ctx.triggered_id['index'] else selected_style for i in range(len(click_key))]

            sub_parts = [[
                dbc.Row([
                    html.A(
                        dcc.Markdown(f'* {i["label"]}'),
                        id = {'type':'tutorial-sub-part','index':idx},
                        style ={'fontSize':10,'marginLeft':'25px'}
                    ),
                    html.Br()
                ])
                for idx,i in enumerate(self.layout_handler.tutorial_content['categories'][click_key.index(click_key[ctx.triggered_id["index"]])]['parts'])
            ]
            if i==ctx.triggered_id['index'] else []
            for i in range(0,len(ctx.outputs_list[4]))
            ]

        elif ctx.triggered_id['type']=='tutorial-sub-part':

            tutorial_name = no_update
            style_list = [no_update]*len(ctx.outputs_list[3])
            sub_parts = [no_update]*len(ctx.outputs_list[4])
            
            tutorial_names = [[f['label'] for f in i['parts']] for i in self.layout_handler.tutorial_content['categories']]
            part_names = [i['props']['children'].split('* ')[-1] for i in current_parts]
            tutorial_index = tutorial_names.index(part_names)

            # Updating style
            sub_parts_style = [{'fontSize':10,'marginLeft':'25px'} if not i==ctx.triggered_id['index'] else sub_selected_style for i in range(len(ctx.outputs_list[-1]))]
            slide_active_index = [self.layout_handler.tutorial_content['categories'][tutorial_index]['parts'][ctx.triggered_id['index']]['value']]*len(ctx.outputs_list[2])


        return new_slides, tutorial_name, slide_active_index, style_list, sub_parts, sub_parts_style

    def initialize_metadata_plots(self,selected_dataset_list,vis_sess_store):
        """
        Creating plot and slide select divs in dataset-builder
        """
        vis_sess_store = json.loads(vis_sess_store)

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
        vis_sess_store = []
        for i in full_slides_list:
            i['included'] = True
            vis_sess_store.append(i)

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
                            dbc.Button('Go to Visualization!',className='d-grid mx-auto',id={'type':'go-to-vis-butt','index':0},color='success',href='/vis')
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
        current_slide_count, slide_select_options, vis_sess_store = self.update_current_slides(slide_rows,vis_sess_store)

        return drop_div, plot_div, slide_select_options, json.dumps(vis_sess_store)

    def update_metadata_plot(self,new_meta,sub_meta,group_type,slide_rows,vis_sess_store):
        """
        Plotting slide/dataset-level metadata in dataset-builder
        """

        vis_sess_store = json.loads(vis_sess_store)

        if not ctx.triggered[0]['value']:
            raise exceptions.PreventUpdate

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

        current_slide_count, slide_select_options, vis_sess_store = self.update_current_slides(slide_rows,vis_sess_store)
        #TODO: get_pattern_matching_value test here
        """
        if type(new_meta)==list:
            if len(new_meta)>0:
                new_meta = new_meta[0]
        """
        new_meta = get_pattern_matching_value(new_meta)

        #TODO: get_pattern_matching_value test here
        """
        if type(group_type)==list:
            if len(group_type)>0:
                group_type = group_type[0]
        """
        group_type = get_pattern_matching_value(group_type)

        #TODO: get_pattern_matching_value test here
        if not new_meta is None:
            if not len(new_meta)==0:
                # Filtering out de-selected slides
                present_slides = [s['name'] for s in vis_sess_store]
                included_slides = [t for t in present_slides if vis_sess_store]                    

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
                                    d_dict['Dataset'].append(dataset_name)
                                    d_dict['Slide Name'].append(s['name'])
                                    d_dict[new_meta].append(np.nan)

                            # Finding nan fill value
                            try:
                                d_dict_meta_types = [type(i) for i in d_dict[new_meta] if not np.isnan(i)]
                            except TypeError:
                                # Most likely this is because the new_meta type is str which can't be tested for nan
                                d_dict_meta_types = [type(i) for i in d_dict[new_meta] if not i!=i]
                                
                            if all([i in [int, float, complex] for i in d_dict_meta_types]):
                                nan_fill_val = 0.0
                            elif all([i==str for i in d_dict_meta_types]):
                                nan_fill_val = 'Not Recorded'
                            else:
                                nan_fill_val = None
                                print('type of new_meta in dataset builder is something strange')
                                print(f'd_dict_meta_types: {d_dict_meta_types}')

                        else:
                            nan_fill_val = 0.0
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
                                                    d_dict[new_meta].append(0.0)
                                                    d_dict['FTU'].append(f.replace(' Expression Statistics',''))
                                                    d_dict['Dataset'].append(dataset_name)
                                                    d_dict['Slide Name'].append(slide_name)
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
                    #plot_data = plot_data.dropna(subset=[new_meta]).convert_dtypes()
                    plot_data = plot_data.fillna(nan_fill_val).convert_dtypes()
                    
                    # Assigning grouping variable
                    if group_type=='By Dataset':
                        group_bar = 'Dataset'
                    elif group_type=='By Slide':
                        group_bar = 'Slide Name'

                    # Checking if new_meta is a number or a string. If not then it's a nested metadata feature.
                    # This is dumb, c'mon pandas
                    if plot_data[new_meta].dtype.kind in 'biufc':
                        if group_bar == 'Dataset':
                            print('Generating violin plot')
                            if not new_meta == 'FTU Expression Statistics':
                                fig = go.Figure(px.violin(plot_data,points = 'all', x=group_bar,y=new_meta,hover_data=['Slide Name']))
                            else:
                                if sub_meta[0]=='Main Cell Types':
                                    fig = go.Figure(px.violin(plot_data,points = 'all', x=group_bar,y=new_meta,hover_data=['Slide Name'],color='FTU'))
                                else:
                                    fig = go.Figure(px.violin(plot_data,points = 'all', x=group_bar,y=new_meta,hover_data=['Slide Name','State'],color='FTU'))
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

                return [slide_select_options, [fig], cell_types_options, cell_types_turn_off,[current_slide_count], json.dumps(vis_sess_store)]
            else:
                return [slide_select_options, [], [], [],[], json.dumps(vis_sess_store)]
        else:
            return [slide_select_options, [no_update], cell_types_options, cell_types_turn_off,[current_slide_count], json.dumps(vis_sess_store)]
        
    def update_current_slides(self,slide_rows,current_session):
        """
        Adding slides to current slides from dataset-builder
        """
        # Updating the current slides
        slide_options = []
        if len(slide_rows)>0:
            #TODO: get_pattern_matching_value test here
            if type(slide_rows[0])==list:
                slide_rows = slide_rows[0]
            for s in range(0,len(current_session)):
                if s in slide_rows:
                    current_session[s]['included'] = True
                else:
                    current_session[s]['included'] = False
                    
            slide_options = []
            unique_folders = np.unique([s['folderId'] for s in current_session if s['included']])
            for f in unique_folders:

                # Adding all the slides in the same folder under the same disabled folder label option (not selectable)
                folder_name = self.dataset_handler.slide_datasets[f]['name']

                slide_options.append({
                    'label': html.Span([
                        html.Img(src='/assets/fusey_clean.svg',height=20),
                        html.Span(folder_name,style={'font-size':25,'padding-left':10})
                    ], style = {'align-items':'center','justify-content':'center'}),
                    'value':'folder',
                    'disabled':True
                })

                slide_options.extend([
                    {
                        'label':i['name'],
                        'value':i['_id'],
                        'disabled':False
                    }
                    for i in current_session if i['included'] and i['folderId']==f
                ])

        if slide_options == []:
            slide_options = []
            unique_folders = np.unique([s['folderId'] for s in self.dataset_handler.default_slides])
            for f in unique_folders:
                
                # Getting the folder names for the slides
                folder_name = self.dataset_handler.slide_datasets[f]['name']

                slide_options.append(
                    {
                        'label': html.Span([
                            html.Img(src='/assets/fusey_clean.svg',height=20),
                            html.Span(folder_name,style={'font-size':25,'padding-left':10})
                        ], style = {'align-items':'center','justify-content':'center'}),
                        'value':'folder',
                        'disabled':True
                    }
                )

                slide_options.extend([
                    {
                        'label':i['name'],
                        'value':i['_id'],
                        'disabled':False
                    }
                    for i in self.dataset_handler.default_slides if i['folderId']==f
                ])


        return [html.P(f'Included Slide Count: {len(slide_rows)}')], slide_options, current_session

    def update_viewport_data(self,bounds,cell_view_type,frame_plot_butt,cluster_click,current_tab,update_data,frame_list,frame_label, cluster_eps, cluster_min_samples,current_data):
        """
        Updating data used for current viewport visualizations
        """
        #TODO: Update this for CODEX and Regular vis

        frame_label_disable = [no_update]*len(ctx.outputs_list[-2])
        viewport_store_data = ['']
        cell_view_type = get_pattern_matching_value(cell_view_type)
        if cell_view_type is None:
            cell_view_type = 'channel_hist'
        
        if update_data or ctx.triggered_id=='roi-view-data':
            if not self.wsi is None:
                if not bounds is None:
                    if len(bounds)==2:
                        bounds_box = shapely.geometry.box(bounds[0][1],bounds[0][0],bounds[1][1],bounds[1][0])
                    else:
                        bounds_box = shapely.geometry.box(*bounds)

                    # Storing current slide boundaries
                    self.current_slide_bounds = bounds_box
                else:
                    self.current_slide_bounds = self.wsi.map_bounds
            else:
                self.current_slide_bounds = None

                raise exceptions.PreventUpdate
        
        # Making a box-poly from the bounds
        if current_tab=='cell-compositions-tab':
            if not self.wsi is None:
                # Getting a dictionary containing all the intersecting spots with this current ROI
                if self.wsi.spatial_omics_type=='CODEX':
                    frame_label = get_pattern_matching_value(frame_label)
                    if len(frame_list)<2:
                        frame_label = None
                    
                    frame_list = get_pattern_matching_value(frame_list)
                    print(f'frame_list: {frame_list}')
                    if frame_list is None:
                        frame_list = [self.wsi.channel_names[0]]

                if update_data:
                    if self.wsi.spatial_omics_type=='Visium':
                        # Getting intersecting FTU information
                        intersecting_ftus, intersecting_ftu_polys = self.wsi.update_viewport_data(self.current_slide_bounds)

                    elif self.wsi.spatial_omics_type=='CODEX':
                        # Returns dictionary of intersecting tissue frame intensities
                        print(cell_view_type)
                        intersecting_ftus, intersecting_ftu_polys = self.wsi.update_viewport_data(bounds_box,cell_view_type,frame_list)
                    
                    self.current_ftus = intersecting_ftus

                    viewport_store_data = {}

                    if len(list(intersecting_ftus.keys()))>0:
                        for f_idx,f in enumerate(list(intersecting_ftus.keys())):

                            viewport_store_data[f] = {}

                            if self.wsi.spatial_omics_type=='Visium':
                                counts_dict_list = [i['Main_Cell_Types'] for i in intersecting_ftus[f] if 'Main_Cell_Types' in i]

                                if len(counts_dict_list)>0:
                                    counts_data = pd.DataFrame.from_records(counts_dict_list).sum(axis=0).to_frame()
                                    counts_data.columns = [f]

                                    counts_data = (counts_data[f]/counts_data[f].sum()).to_frame()
                                    counts_data.columns = [f]
                                    counts_data = counts_data.sort_values(by=f,ascending=False).iloc[0:self.plot_cell_types_n,:]
                                    counts_data = counts_data.reset_index()

                                    viewport_store_data[f]['data'] = counts_data.to_dict('records')
                                    viewport_store_data[f]['count'] = len(counts_dict_list)

                                    # Getting cell state information
                                    viewport_store_data[f]['states'] = {}
                                    for m in counts_data['index'].tolist():
                                        cell_states_data = [i['Cell_States'][m] for i in intersecting_ftus[f] if 'Cell_States' in i]

                                        cell_states_data = pd.DataFrame.from_records(cell_states_data).sum(axis=0).to_frame()
                                        cell_states_data = cell_states_data.reset_index()
                                        cell_states_data.columns = ['Cell State','Proportion']
                                        cell_states_data['Proportion'] = cell_states_data['Proportion']/cell_states_data['Proportion'].sum()

                                        viewport_store_data[f]['states'][m] = cell_states_data.to_dict('records')

                                else:
                                    viewport_store_data[f]['data'] = []
                                    viewport_store_data[f]['count'] = 0
                                    viewport_store_data[f]['states'] = {}

                            elif self.wsi.spatial_omics_type == 'CODEX':

                                if cell_view_type == 'channel_hist':
                                    for f in list(intersecting_ftus.keys()):

                                        viewport_store_data[f] = {}
                                        counts_dict_list = [i for i in intersecting_ftus]
                                        counts_data = pd.DataFrame()
                                        if type(intersecting_ftus[f])==dict:

                                            for frame_name in intersecting_ftus[f]:
                                                frame_data = intersecting_ftus[f][frame_name]

                                                frame_data_df = pd.DataFrame({'Frequency':[0]+frame_data['hist'],'Intensity':frame_data['bin_edges'],'Channel':[frame_name]*len(frame_data['bin_edges'])})

                                                if counts_data.empty:
                                                    counts_data = frame_data_df
                                                else:
                                                    counts_data = pd.concat([counts_data,frame_data_df],axis=0,ignore_index=True)

                                        viewport_store_data[f]['data'] = counts_data.to_dict('records')
                                        viewport_store_data[f]['count'] = len(counts_dict_list)
                                        viewport_store_data[f]['view_type'] = cell_view_type
                                
                                elif cell_view_type == 'features':
                                    
                                    for f in list(intersecting_ftus.keys()):
                                        viewport_store_data[f] = {}
                                        counts_dict_list = [i for i in intersecting_ftus if len(intersecting_ftus[i])>0]

                                        counts_data_list = []
                                        counts_data = pd.DataFrame()
                                        if type(intersecting_ftus[f])==list:
                                            for ftu_idx,ind_ftu in enumerate(intersecting_ftus[f]):

                                                cell_features = {}
                                                
                                                if 'Channel Means' in ind_ftu:
                                                    for frame in frame_list:
                                                        cell_features[f'Channel {self.wsi.channel_names.index(frame)}'] = ind_ftu['Channel Means'][self.wsi.channel_names.index(frame)]
                                                
                                                if 'Cell Type' in ind_ftu:
                                                    cell_features['Cell Type'] = ind_ftu['Cell Type']
                                                else:
                                                    cell_features['Cell Type'] = 'Unlabeled'

                                                if 'Cluster' in ind_ftu:
                                                    cell_features['Cluster'] = ind_ftu['Cluster']
                                                else:
                                                    cell_features['Cluster'] = 'Unclustered'

                                                if not frame_label is None and frame_label in self.wsi.channel_names:
                                                    cell_features[frame_label] = ind_ftu['Channel Means'][self.wsi.channel_names.index(frame_label)]
                                                
                                                cell_features['Hidden'] = {'Bbox':list(intersecting_ftu_polys[f][ftu_idx].bounds)}

                                                counts_data_list.append(cell_features)

                                        if len(counts_data_list)>0:
                                            counts_data = pd.DataFrame.from_records(counts_data_list)

                                            viewport_store_data[f]['data'] = counts_data.to_dict('records')
                                        else:
                                            viewport_store_data[f]['data'] = []

                                        viewport_store_data[f]['count'] = len(counts_data_list)
                                        viewport_store_data[f]['view_type'] = cell_view_type

                                elif cell_view_type == 'cell':
                                    counts_dict_list = [i for i in intersecting_ftus if len(intersecting_ftus[i])>0]
                                    
                                    viewport_store_data[f] = {}

                                    counts_data_list = []
                                    counts_data = pd.DataFrame()
                                    if type(intersecting_ftus[f])==list:
                                        for ind_ftu in intersecting_ftus[f]:

                                            if 'Cell' in ind_ftu:
                                                counts_data_list.append({
                                                    'Cell': ind_ftu['Cell'],
                                                    'Cluster': ind_ftu['Cluster']
                                                })
                                            elif 'Cluster' in ind_ftu:
                                                counts_data_list.append({
                                                    'Cluster':ind_ftu['Cluster'],
                                                    'Cell': 'Unlabeled'
                                                })
                                            else:
                                                counts_data_list.append({
                                                    'Cluster': 'No Cluster',
                                                    'Cell': 'Unlabeled'
                                                })

                                    if len(counts_data_list)>0:
                                        counts_data = pd.DataFrame.from_records(counts_data_list)

                                        viewport_store_data[f]['data'] = counts_data.to_dict('records')
                                    else:
                                        viewport_store_data[f]['data'] = []

                                    viewport_store_data[f]['count'] = len(counts_data_list)
                                    viewport_store_data[f]['view_type'] = cell_view_type

                else:
                    current_data = get_pattern_matching_value(current_data)
                    viewport_store_data = json.loads(current_data)

                included_ftus = list(viewport_store_data.keys())
                if len(included_ftus)>0:

                    tab_list = []
                    self.fusey_data = {}
                    for f_idx,f in enumerate(included_ftus):
                        counts_data = viewport_store_data[f]['data']

                        if self.wsi.spatial_omics_type=='Visium':
                            
                            counts_data = pd.DataFrame.from_records(counts_data)
                            first_chart_label = f'{f} Cell Type Proportions'

                        elif self.wsi.spatial_omics_type=='CODEX':
                            
                            counts_data = pd.DataFrame.from_records(counts_data)

                            if cell_view_type == 'channel_hist':
                                first_chart_label = 'Channel Intensity Histogram'

                            elif cell_view_type == 'features':
                                first_chart_label = 'Nucleus Features'

                            elif cell_view_type == 'cell':
                                first_chart_label = 'Cell Compositions'

                        if not counts_data.empty:
                            
                            if self.wsi.spatial_omics_type=='Visium':
                                f_pie = px.pie(counts_data,values=f,names='index')
                                f_pie.update_traces(textposition='inside')
                                f_pie.update_layout(uniformtext_minsize=12,uniformtext_mode='hide')

                            elif self.wsi.spatial_omics_type=='CODEX':
                                # For CODEX images, have the tissue tab be one histogram                                    
                                # Getting one of the channels. Maybe just the first one
                                #print(f'counts_data shape: {counts_data.shape}')
                                cluster_disable = True
                                eps = 0.3
                                min_samples = 10

                                if cell_view_type=='channel_hist':
                                    cluster_disable = True
                                    f_pie = px.bar(
                                        data_frame = counts_data,
                                        x = 'Intensity',
                                        y = 'Frequency',
                                        color = 'Channel'
                                    )
                                elif cell_view_type=='features':

                                    if len(frame_list)==1:
                                        f_pie = gen_violin_plot(
                                            feature_data = counts_data,
                                            label_col = 'Cell Type' if 'Cell Type' in counts_data.columns else 'Cluster',
                                            label_name = 'Cell Type' if 'Cell Type' in counts_data.columns else 'Cluster',
                                            feature_col = f'Channel {self.wsi.channel_names.index(frame_list[0])}',
                                            custom_col = 'Hidden'
                                        )

                                    elif len(frame_list)==2:
                                        f_names = [i for i in counts_data.columns.tolist() if 'Channel' in i]
                                        f_pie = px.scatter(
                                            data_frame = counts_data,
                                            x = f_names[0],
                                            y = f_names[1],
                                            color = 'Cell Type' if frame_label is None else frame_label,
                                            custom_data = 'Hidden'
                                        )
                                        f_pie.update_layout({
                                            'xaxis': {
                                                'title': {
                                                    'text': f'{self.wsi.channel_names[int(float(f_names[0].split(" ")[-1]))]}'
                                                }
                                            },
                                            'yaxis': {
                                                'title': {
                                                    'text': f'{self.wsi.channel_names[int(float(f_names[1].split(" ")[-1]))]}'
                                                }
                                            }
                                        })
                                        
                                    elif len(frame_list)>2:
                                        # Generate UMAP dimensional reduction for channel mean data
                                        if update_data:
                                            f_umap_data = gen_umap(
                                                feature_data = counts_data,
                                                feature_cols = [i for i in counts_data.columns.tolist() if 'Channel' in i],
                                                label_and_custom_cols=[i for i in counts_data.columns.tolist() if not 'Channel' in i]
                                            )

                                            viewport_store_data[f]['UMAP'] = f_umap_data.to_dict('records')
                                        else:
                                            f_umap_data = pd.DataFrame.from_records(viewport_store_data[f]['UMAP'])

                                        cluster_disable = False

                                        if not ctx.triggered_id=='slide-map':
                                            if ctx.triggered_id['type']=='cell-label-cluster':
                                                # Get clustering labels:
                                                eps = get_pattern_matching_value(cluster_eps)
                                                min_samples = get_pattern_matching_value(cluster_min_samples)
                                                cluster_labels = gen_clusters(f_umap_data,['UMAP1','UMAP2'],eps=eps, min_samples=min_samples)

                                                f_umap_data['Cluster'] = cluster_labels

                                                frame_label = 'Cluster'


                                        f_pie = px.scatter(
                                            data_frame = f_umap_data,
                                            x = 'UMAP1',
                                            y = 'UMAP2',
                                            color = 'Cell Type' if frame_label is None else frame_label,
                                            custom_data = 'Hidden'
                                        )

                                elif cell_view_type=='cell':

                                    cluster_disable = True
                                    counts_data = counts_data['Cluster'].value_counts().to_dict()
                                    pie_chart_data = []
                                    for key,val in counts_data.items():
                                        pie_chart_data.append(
                                            {'Cluster': key, 'Count': val}
                                        )
                                    pie_chart_df = pd.DataFrame.from_records(pie_chart_data)
                                    f_pie = px.pie(
                                        data_frame = pie_chart_df,
                                        values = 'Count',
                                        names = 'Cluster'
                                    )
                                    f_pie.update_traces(textposition='inside')
                                    f_pie.update_layout(uniformtext_minsize=12,uniformtext_mode='hide')
                                    
                            if self.wsi.spatial_omics_type=='Visium':
                                
                                # Finding cell state proportions per main cell type in the current region
                                second_plot_label = f'{f} Cell State Proportions'

                                top_cell = counts_data['index'].tolist()[0]

                                pct_states = pd.DataFrame.from_records(viewport_store_data[f]['states'][top_cell])
                                # Fusey data
                                self.fusey_data[f] = {
                                    'normalized_counts':counts_data,
                                    'pct_states':pct_states,
                                    'top_cell':top_cell
                                }
                                state_bar = px.bar(pct_states,x='Cell State',y = 'Proportion', title = f'Cell State Proportions for:<br><sup>{self.cell_graphics_key[top_cell]["full"]} in:</sup><br><sup>{f}</sup>')

                                f_tab = dbc.Tab(
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label(first_chart_label),
                                            dcc.Graph(
                                                id = {'type':'ftu-cell-pie','index':f_idx},
                                                figure = go.Figure(f_pie)
                                            )
                                        ],md=6),
                                        dbc.Col([
                                            dbc.Label(second_plot_label),
                                            dcc.Graph(
                                                id = {'type':'ftu-state-bar','index':f_idx},
                                                figure = go.Figure(state_bar)
                                            )
                                        ],md=6)
                                    ]),label = f+f' ({viewport_store_data[f]["count"]})',tab_id = f'tab_{f_idx}'
                                )

                            elif self.wsi.spatial_omics_type=='CODEX':

                                # Returning blank for now
                                state_bar = []
                                second_plot_label = 'This is CODEX'

                                self.fusey_data['Tissue'] = {}

                                f_tab = dbc.Tab([
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Label('Select a Channel Name:',html_for={'type':'frame-histogram-drop','index':0}),
                                            md = 3, align = 'center',
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                options = [i for i in self.wsi.channel_names if not i=='Histology (H&E)'],
                                                value = frame_list,
                                                multi = True,
                                                id = {'type':'frame-histogram-drop','index':0}
                                            ),
                                            md = 6, align = 'center'
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                'Plot it!',
                                                className = 'd-grid col-12 mx-auto',
                                                n_clicks = 0,
                                                style = {'width':'100%'},
                                                id = {'type':'frame-data-butt','index':0}
                                            ),
                                            md = 3
                                        )
                                    ], style = {'display':'none'} if not cell_view_type in ['channel_hist','features'] else {}),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Label('Select a label: ',html_for = {'type':'frame-label-drop','index':0}),
                                            md = 3
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                options = [
                                                    {
                                                        'label': i,
                                                        'value': i
                                                    }
                                                    if not i=='Cluster' else {'label':i, 'value': i, 'disabled': True if 'Cluster' in counts_data.columns else False}
                                                    for i in self.wsi.channel_names+['Cluster']
                                                ],
                                                value = frame_label,
                                                id = {'type':'frame-label-drop','index':0},
                                                disabled = True if len(frame_list)<2 else False
                                            )
                                        )
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label(first_chart_label),
                                            dcc.Graph(
                                                id = {'type':'ftu-cell-pie','index':f_idx},
                                                figure = go.Figure(f_pie)
                                            )
                                        ],md=12)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Label('Clustering parameters',html_for={'type':'cell-label-cluster','index':len(tab_list)}),
                                            md = 3
                                        ),
                                        dbc.Col(
                                            dcc.Input(
                                                id = {'type':'cell-label-eps','index':len(tab_list)},
                                                placeholder = 'Epsilon (0.01-1.0)',
                                                type = 'number',
                                                value = eps,
                                                min = 0.0,
                                                step = 0.001,
                                                max = 10000000
                                            ),
                                            md = 2,
                                            style = {'marginLeft':'1px'}
                                        ),
                                        dbc.Col(
                                            dcc.Input(
                                                id = {'type':'cell-label-min-samples','index':len(tab_list)},
                                                placeholder = 'Minimum number per cluster',
                                                type = 'number',
                                                value = min_samples,
                                                step = 1,
                                                min = 1,
                                                max = counts_data.shape[0]
                                            ),
                                            md = 2,
                                            style = {'marginLeft':'1px'}
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                'Cluster it!',
                                                id = {'type':'cell-label-cluster','index':len(tab_list)},
                                                className = 'd-grid col-12 mx-auto',
                                                n_clicks = 0,
                                                disabled = cluster_disable
                                            ),
                                            style = {'marginLeft':'1px'}
                                        )
                                    ],
                                    style = {'marginTop':'10px'})
                                    ],
                                    label = f+f' ({viewport_store_data[f]["count"]})',tab_id = f'tab_{len(tab_list)}'
                                )

                            tab_list.append(f_tab)

                    if self.wsi.spatial_omics_type=='Visium' or self.wsi.spatial_omics_type=='Regular':
                        return dbc.Tabs(tab_list,active_tab = 'tab_0'), no_update, frame_label_disable, [json.dumps(viewport_store_data)]
                    elif self.wsi.spatial_omics_type=='CODEX':

                        return_div = html.Div([
                            dbc.Row([
                                dbc.Col(
                                    children = [
                                        dbc.Label('Select View Type: '),
                                        self.layout_handler.gen_info_button('Select different options here to vary how to view information in the current viewport')
                                    ],
                                    md = 4
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options = [
                                            {'label':'Channel Intensity Histograms','value':'channel_hist'},
                                            {'label':'Nucleus Features','value':'features'},
                                            {'label':'Cell Type Composition','value':'cell'}
                                        ],
                                        value = cell_view_type,
                                        multi = False,
                                        id = {'type':'roi-view-data','index':0}
                                    )
                                )
                            ]),
                            html.Hr(),
                            dbc.Row(
                                dbc.Tabs(tab_list,active_tab=f'tab_0')
                            ),
                            html.Hr(),
                            dbc.Row([
                                html.Div(
                                    id = {'type':'extracted-cell-labeling','index':0},
                                    children = ['Select points to start labeling!']
                                )
                            ]),
                            dbc.Row([
                                html.Div(
                                    id = {'type':'cell-labeling-criteria','index':0},
                                    children = []
                                )
                            ])
                        ])

                        return return_div, no_update, frame_label_disable, [json.dumps(viewport_store_data)]

                else:
                    return html.P('No FTUs in current view'), [no_update], frame_label_disable, [json.dumps(viewport_store_data)]
            else:
                return html.P('Select a slide to get started!'), [no_update], frame_label_disable, [json.dumps(viewport_store_data)]
    
        elif current_tab=='annotation-tab':
            if self.dataset_handler.username=='fusionguest':
                return no_update, html.P('Sign in or create an account to start annotating!'), frame_label_disable, [json.dumps(viewport_store_data)]
            else:
                
                # Getting intersecting FTU information
                intersecting_ftus = {}
                for ftu in self.current_ftu_layers:
                    if not ftu=='Spots':
                        intersecting_ftus[ftu], _ = self.wsi.find_intersecting_ftu(bounds_box,ftu)

                for m_idx,m_ftu in enumerate(self.wsi.manual_rois):
                    intersecting_ftus[f'Manual ROI: {m_idx+1}'] = [m_ftu['geojson']['features'][0]['properties']['user']]

                for marked_idx, marked_ftu in enumerate(self.wsi.marked_ftus):
                    intersecting_ftus[f'Marked FTUs: {marked_idx+1}'] = [i['properties']['user'] for i in marked_ftu['geojson']['features']]
                
                self.current_ftus = intersecting_ftus

                #TODO: Check for current annotation session in user's public folder
                annotation_tabs, first_tab, first_session = self.layout_handler.gen_annotation_card(self.dataset_handler,self.current_ftus)
                self.current_ann_session = first_session

                annotation_tab_group = html.Div([
                        dbc.Tabs(
                            id = {'type':'annotation-tab-group','index':0},
                            active_tab = 'ann-sess-0',
                            children = annotation_tabs
                        ),
                        html.Div(
                            id = {'type':'annotation-session-content','index':0},
                            children = [
                                first_tab
                            ]
                        )
                ])

                return html.Div(), annotation_tab_group, frame_label_disable, [json.dumps(viewport_store_data)]
        
        else:
            raise exceptions.PreventUpdate

    def update_state_bar(self,cell_click):
        """
        View cell state proportions for clicked main cell type
        
        """

        if not cell_click is None:
            self.pie_cell = cell_click['points'][0]['label']

            self.pie_ftu = list(self.current_ftus.keys())[ctx.triggered_id['index']]

            pct_states = pd.DataFrame.from_records([i['Cell_States'][self.pie_cell] for i in self.current_ftus[self.pie_ftu] if 'Cell_States' in i]).sum(axis=0).to_frame()
            pct_states = pct_states.reset_index()
            pct_states.columns = ['Cell State', 'Proportion']
            pct_states['Proportion'] = pct_states['Proportion']/pct_states['Proportion'].sum()

            state_bar = go.Figure(px.bar(pct_states,x='Cell State', y = 'Proportion', title = f'Cell State Proportions for:<br><sup>{self.cell_graphics_key[self.pie_cell]["full"]} in:</sup><br><sup>{self.pie_ftu}</sup>'))

            return state_bar
        else:
            return go.Figure()
    
    def update_hex_color_key(self):
        
        # Iterate through all structures (and spots) in current wsi,
        # concatenate all of their proportions of specific cell types together
        # scale it with self.color_map (make sure to multiply by 255 after)
        # convert uint8 RGB colors to hex
        # create look-up table for original value --> hex color
        # add that as get_color() function in style dict (fillColor) along with fillOpacity

        #TODO: Add a scaling/range thing for values used as overlays
        raw_values_list = []
        if not self.overlay_prop is None:
            if not self.overlay_prop['name'] is None:
                raw_values_list.extend(self.wsi.get_overlay_value_list(self.overlay_prop))

        # Converting to RGB
        if len(raw_values_list)>0:
            if max(raw_values_list)>0:
                scaled_values = [(i-min(raw_values_list))/max(raw_values_list) for i in raw_values_list]
            else:
                scaled_values = raw_values_list
            rgb_values = np.uint8(255*self.color_map(scaled_values))[:,0:3]

            hex_list = []
            for row in range(rgb_values.shape[0]):
                hex_list.append('#'+"%02x%02x%02x" % (rgb_values[row,0],rgb_values[row,1],rgb_values[row,2]))

            self.hex_color_key = {i:j for i,j in zip(raw_values_list,hex_list)}
        else:
            self.hex_color_key = {}

    def update_overlays(self,cell_val,vis_val,filter_vals,added_filter_slider,added_filter_keys,ftu_color_butt,cell_sub_val,ftu_bound_color, added_slide_style):
        """
        Updating overlay hideout property in GeoJSON layer used in self.ftu_style_handle        
        """

        print(f'Updating overlays for current slide: {self.wsi.slide_name}, {cell_val}')

        m_prop = None
        cell_sub_select_children = no_update

        if self.wsi.spatial_omics_type=='Visium':
            gene_info_style = [{'display':'none'}]
            gene_info_components = [[]]
        else:
            gene_info_style = []
            gene_info_components = []

        color_bar_style = {
            'visibility':'visible',
            'background':'white',
            'background':'rgba(255,255,255,0.8)',
            'box-shadow':'0 0 15px rgba(0,0,0,0.2)',
            'border-radius':'10px',
            'width':'',
            'padding':'0px 0px 0px 25px',

        }

        cell_sub_val = get_pattern_matching_value(cell_sub_val)

        if type(ctx.triggered_id)==list:
            triggered_id = ctx.triggered_id[0]
        else:
            triggered_id = ctx.triggered_id

        #TODO: Add a button to update FTU/boundary structure colors instead of making it responsive to the color selector-=
        try:
            if triggered_id['type']=='ftu-bound-color-butt':
                if not ftu_bound_color is None:
                    self.ftu_colors[self.wsi.ftu_names[triggered_id['index']]] = ftu_bound_color[triggered_id['index']]
        
        except TypeError:
            # This is for non-pattern matching components so the ctx.triggered_id is just a str
            pass
        
        # Extracting cell val if there are sub-properties
        if not cell_val is None:
            if '-->' in cell_val:
                cell_val_parts = cell_val.split(' --> ')
                m_prop = cell_val_parts[0]
                cell_val = cell_val_parts[1]
            else:
                m_prop = cell_val
            
            # Updating overlay property
            if m_prop == 'Main_Cell_Types':
                
                #TODO: Find a better way to define cell names, integrate HRA but clean up long nasty names
                if cell_val in self.cell_names_key:
                    cell_name = self.cell_names_key[cell_val]
                else:
                    cell_name = cell_val
                
                # Picking just a Main_Cell_Type
                if ctx.triggered_id == 'cell-drop':
                    
                    cell_sub_val= None
                    self.overlay_prop = {
                        'name': m_prop,
                        'value': cell_name,
                        'sub_value': cell_sub_val
                    }

                    if cell_name in self.cell_graphics_key:
                        # Getting all possible cell states for this cell type:
                        possible_cell_states = np.unique(self.cell_graphics_key[cell_name]['states'])
                        # Creating dropdown for cell states
                        cell_sub_select_children = [
                            dcc.Dropdown(
                                options = [{'label':p,'value':p,'disabled':False} for p in possible_cell_states]+[{'label':'All','value':'All','disabled':False}],
                                placeholder = 'Select A Cell State Value',
                                id = {'type':'cell-sub-drop','index':0}
                            )
                        ]
                    else:
                        cell_sub_select_children = []

                else:
                    cell_sub_select_children = no_update

                if cell_sub_val is None:

                    self.overlap_prop = {
                        'name': m_prop,
                        'value': cell_name,
                        'sub_value': cell_sub_val
                    }
                
                elif cell_sub_val=='All':
                    self.overlay_prop = {
                        'name': m_prop,
                        'value': cell_name,
                        'sub_value': None
                    }
                
                else:
                    # Visualizing a sub-property of a main cell type
                    self.overlay_prop = {
                        'name': 'Cell_States',
                        'value': cell_name,
                        'sub_value': cell_sub_val
                    }
                self.update_hex_color_key()
                color_bar_style['width'] = '350px'

                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width=300,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                filter_min_val = np.min(list(self.hex_color_key.keys()))
                filter_max_val = np.max(list(self.hex_color_key.keys()))
                filter_disable = False
            
            elif m_prop == 'Cell_Subtypes':

                if cell_val in self.cell_names_key:
                    cell_name = self.cell_names_key[cell_val]
                else:
                    cell_name = cell_val

                if cell_name in self.cell_graphics_key:

                    if ctx.triggered_id=='cell-drop':
                        cell_sub_val = None
                        # Getting all possible cell sub-types:
                        possible_cell_subtypes = np.unique(self.cell_graphics_key[cell_name]['subtypes'])

                        cell_sub_select_children = [
                            dcc.Dropdown(
                                options = [{'label':p,'value':p,'disabled':False} for p in possible_cell_subtypes] + [{'label':'All','value':'All','disabled':False}],
                                placeholder = 'Select A Cell Subtype Value',
                                id = {'type':'cell-sub-drop','index':0}
                            )
                        ]
                    else:
                        cell_sub_select_children = no_update
                else:
                    cell_sub_select_children = []

                if cell_sub_val == 'All':
                    self.overlay_prop = {
                        'name': m_prop,
                        'value':cell_name,
                        'sub_value':None
                    }

                elif not cell_sub_val is None:
                    self.overlay_prop = {
                        'name': m_prop,
                        'value':cell_name,
                        'sub_value': cell_sub_val
                    }

                else:
                    self.overlay_prop = {
                        'name': 'Main_Cell_Types',
                        'value': cell_name,
                        'sub_value': None
                    }
                
                self.update_hex_color_key()
                color_bar_style['width'] = '350px'

                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width = 300,height = 10,position = 'bottomleft',
                    id = f'colorbar{random.randint(0,100)}',
                    style = color_bar_style
                )

                if len(list(self.hex_color_key.keys()))==0:
                    filter_min_val = 0.0
                    filter_max_val = 1.0
                    filter_disable = True
                else:
                    filter_min_val = np.min(list(self.hex_color_key.keys()))
                    filter_max_val = np.max(list(self.hex_color_key.keys()))
                    filter_disable = False

            elif m_prop == 'Cell_States':
                # Selecting a specific cell state value for overlays
                self.overlay_prop = {
                    'name': m_prop,
                    'value': cell_val,
                    'sub_value': None
                }

                self.update_hex_color_key()
                color_bar_style['width'] = '350px'

                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width=300,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)
                
                filter_min_val = np.min(list(self.hex_color_key.keys()))
                filter_max_val = np.max(list(self.hex_color_key.keys()))
                filter_disable = False

            elif m_prop == 'Max Cell Type':
                # Getting the maximum cell type present for each structure
                self.overlay_prop = {
                    'name': 'Main_Cell_Types',
                    'value':'max',
                    'sub_value': None
                }
                self.update_hex_color_key()
                color_bar_style['width'] = '650px'

                cell_sub_select_children = []

                cell_types = sorted(list(self.cell_graphics_key.keys()))
                color_bar = dlx.categorical_colorbar(
                    categories = cell_types,
                    colorscale = list(self.hex_color_key.values()),
                    width=600,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                filter_min_val = 0.0
                filter_max_val = 1.0
                filter_disable = True

            elif m_prop == 'Cluster':
                # Getting cluster value associated with each structure
                self.overlay_prop = {
                    'name': 'Cluster',
                    'value': None,
                    'sub_value': None
                }
                self.update_hex_color_key()

                cluster_labels = sorted(list(self.hex_color_key.keys()))
                color_bar = dlx.categorical_colorbar(
                    categories = cluster_labels,
                    colorscale = list(self.hex_color_key.values()),
                    width = 600, height = 10, position = 'bottomleft',
                    id = f'colorbar{random.randint(0,100)}',
                    style = color_bar_style
                )
                cell_sub_select_children = []

                filter_min_val = 0.0
                filter_max_val = 1.0
                filter_disable = True
                
            elif m_prop == 'FTU Label':
                self.overlay_prop = {
                    'name': 'Structure',
                    'value': None,
                    'sub_value': None
                }
                self.update_hex_color_key()

                cell_sub_select_children = []

                ftu_types = list(self.hex_color_key.keys())
                color_bar = dlx.categorical_colorbar(
                    categories = ftu_types,
                    colorscale = list(self.hex_color_key.values()),
                    width = 600, height = 10, position = 'bottom left',
                    id = f'colorbar{random.randint(0,100)}',
                    style = color_bar_style
                )

                filter_min_val = 0.0
                filter_max_val = 1.0
                filter_disable = True
            
            elif m_prop == 'Gene Counts':

                self.overlay_prop = {
                    'name': 'Gene Counts',
                    'value': cell_val,
                    'sub_value': None
                }
                self.update_hex_color_key()

                # Now displaying gene info
                if triggered_id=='cell-drop':
                    gene_info_style = [{'display':'inline-block'}]
                    gene_info_components = [self.gene_handler.get_layout(gene_id=cell_val)]
                else:
                    gene_info_style = [no_update]
                    gene_info_components = [no_update]
                

                cell_sub_select_children = []

                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width=300,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                if len(list(self.hex_color_key.keys()))>0:
                    filter_min_val = np.min(list(self.hex_color_key.keys()))
                    filter_max_val = np.max(list(self.hex_color_key.keys()))
                else:
                    filter_min_val = 0.0
                    filter_max_val = 1.0
                filter_disable = False
            
            elif m_prop == 'Morphometrics':

                self.overlay_prop = {
                    'name': m_prop,
                    'value': cell_val,
                    'sub_value': None
                }
                self.update_hex_color_key()

                cell_sub_select_children = []

                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width=300,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                filter_min_val = np.min(list(self.hex_color_key.keys()))
                filter_max_val = np.max(list(self.hex_color_key.keys()))
                filter_disable = False
            
            else:
                if m_prop==cell_val:
                    self.overlay_prop = {
                        'name': m_prop,
                        'value': None,
                        'sub_value': None
                    }
                else:
                    self.overlay_prop = {
                        'name': m_prop,
                        'value': cell_val,
                        'sub_value': None
                    }
                self.update_hex_color_key()

                cell_sub_select_children = []

                color_bar = dl.Colorbar(
                    colorscale = list(self.hex_color_key.values()),
                    width=300,height=10,position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style)

                if not len(list(self.hex_color_key.keys()))==0:
                    filter_min_val = np.min(list(self.hex_color_key.keys()))
                    filter_max_val = np.max(list(self.hex_color_key.keys()))
                    filter_disable = False
                else:
                    filter_min_val = 0
                    filter_max_val = 1.0
                    filter_disable = True

        else:
            self.overlay_prop = {
                'name': None,
                'value': None,
                'sub_value': None
            }
            self.update_hex_color_key()

            cell_sub_select_children = []

            color_bar = no_update
            filter_disable = True
            filter_min_val = 0
            filter_max_val = 1

        if not filter_disable:
            self.filter_vals = [
                {
                    'name': self.overlay_prop['name'],
                    'value': self.overlay_prop['value'],
                    'sub_value': self.overlay_prop['sub_value'],
                    'range': filter_vals
                }
            ]

            # Processing the add-on filters:
            processed_filters = process_filters(added_filter_keys, added_filter_slider,added_slide_style, self.cell_names_key)

            self.filter_vals.extend(processed_filters)

        else:
            self.filter_vals = None

        self.cell_vis_val = vis_val/100
        n_layers = len(callback_context.outputs_list[0])
        geojson_hideout = [
            {
                'color_key':self.hex_color_key,
                'overlay_prop':self.overlay_prop,
                'fillOpacity': self.cell_vis_val,
                'ftu_colors':self.ftu_colors,
                'filter_vals':self.filter_vals
            }
            for i in range(0,n_layers)
        ]

        return geojson_hideout, color_bar, filter_min_val, filter_max_val, filter_disable, cell_sub_select_children, gene_info_style, gene_info_components

    def update_cell_hierarchy(self,cell_clickData):
        """
        Updating cell cytoscape visualization        
        """

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
            color = cell_clickData['points'][0]['color']

            color_code = [str(color[i]) for i in color]

            if not color_code[-1]=='0':
                color_code = ','.join(color_code[0:-1])

                if color_code in list(self.cell_colors_key.keys()):

                    cell_val = self.cell_graphics_key[self.cell_colors_key[color_code]]['full']                    
                    cell_name = html.H3(cell_val)
                    if self.cell_names_key[cell_val] in self.cell_graphics_key:
                        cell_graphic = self.cell_graphics_key[self.cell_names_key[cell_val]]['graphic']
                        cell_hierarchy = self.gen_cyto(self.cell_names_key[cell_val])
                        cell_state_droptions = np.unique(self.cell_graphics_key[self.cell_names_key[cell_val]]['states'])

        return cell_graphic, cell_hierarchy, cell_state_droptions, cell_name

    def get_neph_hover(self,neph_hover):
        """
        Getting color on nephron diagram and returning associated cell type
        """
        if neph_hover is None:
            return False, no_update, no_update
        
        pt = neph_hover['points'][0]
        color = pt['color']
        tool_bbox = pt['bbox']

        # Color code is assigned in graphics_reference.json
        color_code = [str(color[i]) for i in color]

        if not color_code[-1]=='0':
            color_code = ','.join(color_code[0:-1])

            if color_code in list(self.cell_colors_key.keys()):

                cell_name = self.cell_graphics_key[self.cell_colors_key[color_code]]['full']
                tool_children = [
                    html.Div([
                        cell_name
                    ])
                ]
            else:
                tool_children = []
        else:
            tool_children = []


        return True, tool_bbox, tool_children

    def get_click_popup(self,ftu_click):
        """
        Controlling popup components when clicking on FTU in GeoJSON layer        
        """
        if not ftu_click is None:
            self.clicked_ftu = ftu_click
            if 'unique_index' in ftu_click['properties']:
                ftu_idx = ftu_click['properties']['unique_index']

            if self.wsi.spatial_omics_type=='Visium':
                if 'Main_Cell_Types' in ftu_click['properties']['user']:

                    # Getting the main cell type data (only using top-n)
                    main_cell_types = ftu_click['properties']['user']['Main_Cell_Types']
                    chart_data = [main_cell_types[i] for i in main_cell_types]

                    if not len(chart_data)==0:

                        # Only keeping the first self.plot_cell_n
                        top_idx = np.argsort(chart_data)[::-1][0:self.plot_cell_types_n]
                        chart_data = [chart_data[i] for i in top_idx]
                        chart_labels = [list(main_cell_types.keys())[i] for i in top_idx]
                        chart_full_labels = [self.cell_graphics_key[i]['full'] for i in chart_labels]

                        # Getting the cell state info for one of the cells and getting the names of the cells for a dropdown menu
                        cell_states = ftu_click['properties']['user']['Cell_States']
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
                        all_properties = list(ftu_click['properties']['user'].keys())
                        all_properties = [i for i in all_properties if not type(ftu_click['properties']['user'][i])==dict]
                        all_props_dict = {'Property':all_properties,'Value':[ftu_click['properties']['user'][i] for i in all_properties]}
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
                        if 'unique_index' in ftu_click['properties']['user']:
                            add_labels_children = self.layout_handler.get_user_ftu_labels(self.wsi,ftu_click)
                            accordion_children = [
                                dbc.AccordionItem([
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col(-
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
                    # Getting other FTU/Spot properties
                    all_properties = list(ftu_click['properties']['user'].keys())
                    all_properties = [i for i in all_properties if not type(ftu_click['properties']['user'][i])==dict]
                    all_props_dict = {'Property':all_properties,'Value':[ftu_click['properties']['user'][i] for i in all_properties]}
                    all_properties_df = pd.DataFrame(all_props_dict)

                    # popup divs
                    if 'unique_index' in ftu_click['properties']['user']:
                        add_labels_children = self.layout_handler.get_user_ftu_labels(self.wsi,ftu_click)
                        accordion_children = [
                            dbc.AccordionItem([
                                html.Div([
                                    dash_table.DataTable(
                                        id = 'popup-table',
                                        columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in all_properties_df],
                                        data = all_properties_df.to_dict('records'),
                                        editable=False,                                        
                                        sort_mode='multi',
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
                                    dash_table.DataTable(
                                        id = 'popup-table',
                                        columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in all_properties_df],
                                        data = all_properties_df.to_dict('records'),
                                        editable=False,                                        
                                        sort_mode='multi',
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

                # Getting other FTU/Spot properties
                all_properties = list(ftu_click['properties']['user'].keys())
                all_properties = [i for i in all_properties if not type(ftu_click['properties']['user'][i])==dict]
                all_props_dict = {'Property':all_properties,'Value':[ftu_click['properties']['user'][i] for i in all_properties]}
                all_properties_df = pd.DataFrame(all_props_dict)

                # popup divs
                if 'unique_index' in ftu_click['properties']['user']:
                    add_labels_children = self.layout_handler.get_user_ftu_labels(self.wsi,ftu_click)
                    accordion_children = [
                        dbc.AccordionItem([
                            html.Div([
                                dash_table.DataTable(
                                    id = 'popup-table',
                                    columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in all_properties_df],
                                    data = all_properties_df.to_dict('records'),
                                    editable=False,                                        
                                    sort_mode='multi',
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
                                dash_table.DataTable(
                                    id = 'popup-table',
                                    columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in all_properties_df],
                                    data = all_properties_df.to_dict('records'),
                                    editable=False,                                        
                                    sort_mode='multi',
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
            raise exceptions.PreventUpdate
        
    def gen_cyto(self,cell_val):
        """
        Generating cytoscape for selected cell type (referenced in self.update_cell_hierarchy)
        """
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
        """
        Getting information for clicked node in cell hierarchy cytoscape vis.
        """
        if not clicked is None:
            if 'ST' in clicked['id']:
                table_data = self.table_df.dropna(subset=['CT/1/ABBR'])
                table_data = table_data[table_data['CT/1/ABBR'].str.match(clicked['label'])]

                label = clicked['label']
                try:
                    try:
                        id = table_data['CT/1/ID'].tolist()[0]
                        # Modifying base url to make this link to UBERON
                        base_url = self.node_cols['Cell Types']['base_url']
                        new_url = base_url+id.replace('CL:','')
                    except AttributeError:
                        # for float objects
                        print(id)
                        base_url = self.node_cols['Cell Types']['base_url']
                        new_url = base_url+str(id).replace('CL:','')

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
    
    def load_new_slide(self,slide_id,new_interval,modal_is_open,slide_info_store):
        """
        Progress indicator for loading a new WSI and annotations
        """
        modal_open = True
        modal_children = []
        slide_info_store_data = [no_update]
        disable_slide_load = False

        if ctx.triggered_id=='slide-select':

            if not slide_id:
                raise exceptions.PreventUpdate

            # "slide_name" here is the itemId of a slide
            print(f'Getting info for slide: {slide_id}')
            slide_info = self.dataset_handler.get_item_info(slide_id)
            annotation_info = self.dataset_handler.get_available_annotation_ids(slide_id)

            if 'Spatial Omics Type' in slide_info['meta']:
                slide_type = slide_info['meta']['Spatial Omics Type']
                if slide_type=='Visium':
                    self.wsi = VisiumSlide(
                        item_id = slide_info['_id'],
                        girder_handler=self.dataset_handler,
                        ftu_colors = self.ftu_colors,
                        manual_rois = [],
                        marked_ftus = []
                    )
                elif slide_type=='CODEX':
                    self.wsi = CODEXSlide(
                        item_id = slide_info['_id'],
                        girder_handler = self.dataset_handler,
                        ftu_colors = self.ftu_colors,
                        manual_rois = [],
                        marked_ftus = []
                    )
                elif slide_type=='Regular':
                    self.wsi = DSASlide(
                        item_id = slide_info['_id'],
                        girder_handler = self.dataset_handler,
                        ftu_colors = self.ftu_colors,
                        manual_rois = [],
                        marked_ftus = []
                    )
                else:
                    self.wsi = DSASlide(
                        item_id = slide_info['_id'],
                        girder_handler = self.dataset_handler,
                        ftu_colors = self.ftu_colors,
                        manual_rois = [],
                        marked_ftus = []
                    )
            else:
                self.wsi = DSASlide(
                    item_id = slide_info['_id'],
                    girder_handler = self.dataset_handler,
                    ftu_colors = self.ftu_colors,
                    manual_rois = [],
                    marked_ftus = []
                )

            # Storing some info in a data store component (erases upon refresh)
            slide_annotation_info = {
                'slide_info': slide_info,
                'annotations': annotation_info
            }

            n_annotations = len(annotation_info)
            if n_annotations>0:
                first_annotation = annotation_info[0]['annotation']['name']
                slide_annotation_info['loading_annotation'] = annotation_info[0]['_id']

                slide_info_store_data = [
                    json.dumps(slide_annotation_info)
                ]

                # Starting the get_annotation_geojson function on a new thread
                new_thread = threading.Thread(target = self.wsi.get_annotation_geojson, name = first_annotation, args = [0])
                new_thread.daemon = True
                new_thread.start()

                modal_children = [
                    html.Div([
                        dbc.ModalHeader(html.H4(f'Loading Annotations for: {slide_info["name"]}')),
                        dbc.ModalBody([
                            html.Div(
                                html.H6(f'Working on: {first_annotation}')
                            ),
                            dbc.Progress(
                            id = {'type':'slide-load-progress','index':0},
                            value = 0,
                            label = f'0%'
                            )
                        ])
                    ])
                ]
            else:

                slide_info_store_data = [
                    json.dumps(slide_annotation_info)
                ]         
                modal_children = [
                    html.Div(
                        dbc.ModalHeader(html.H4(f'Loading Slide: {slide_info["name"]}'))
                    )
                ]
        
        elif ctx.triggered_id=='slide-load-interval':
            
            if not new_interval:
                raise exceptions.PreventUpdate
            
            slide_info_store = json.loads(slide_info_store[0])

            if len(slide_info_store['annotations'])>0:

                # Checking if previous annotation is complete
                all_annotation_ids = [i['_id'] for i in slide_info_store['annotations']]
                previous_annotation_id = slide_info_store['loading_annotation']
                
                if not previous_annotation_id == 'done':
                    previous_annotation_name = slide_info_store['annotations'][all_annotation_ids.index(previous_annotation_id)]['annotation']['name']

                    # Getting names of current threads to see if that one is still active
                    thread_names = [i.name for i in threading.enumerate()]

                    if not previous_annotation_name in thread_names:

                        next_ann_idx = all_annotation_ids.index(previous_annotation_id) + 1

                        n_annotations = len(all_annotation_ids)
                        slide_name = slide_info_store['slide_info']['name']

                        if next_ann_idx<len(all_annotation_ids):

                            next_annotation = slide_info_store['annotations'][next_ann_idx]['annotation']['name']
                            slide_info_store['loading_annotation'] = slide_info_store['annotations'][next_ann_idx]['_id']
                            
                            # Starting the get_annotation_geojson function on a new thread
                            new_thread = threading.Thread(target = self.wsi.get_annotation_geojson, name = next_annotation, args = [next_ann_idx])
                            new_thread.daemon = True
                            new_thread.start()

                            modal_children = [
                                html.Div([
                                    dbc.ModalHeader(html.H4(f'Loading Annotations for: {slide_name}')),
                                    dbc.ModalBody([
                                        html.Div(
                                            html.H6(f'Working on: {next_annotation}')
                                        ),
                                        dbc.Progress(
                                            id = {'type':'slide-load-progress','index':0},
                                            value = int(100*((next_ann_idx)/n_annotations)),
                                            label = f'{int(100*((next_ann_idx)/n_annotations))}%'
                                        )
                                    ])
                                ])
                            ]
                        else:
                            slide_info_store['loading_annotation'] = 'done'

                            modal_children = [
                                html.Div([
                                    dbc.ModalHeader(html.H4(f'All done!')),
                                    dbc.ModalBody([
                                        dbc.Progress(
                                            id = {'type':'slide-load-progress','index':0},
                                            value = 100,
                                            label = f'100%'
                                        )
                                    ])
                                ])
                            ]

                    else:
                        old_ann_idx = all_annotation_ids.index(previous_annotation_id)
                        old_ann_name = slide_info_store['annotations'][old_ann_idx]['annotation']['name']
                        n_annotations = len(all_annotation_ids)
                        slide_name = slide_info_store['slide_info']['name']

                        modal_children = [
                            html.Div([
                                dbc.ModalHeader(html.H4(f'Loading Annotations for: {slide_name}')),
                                dbc.ModalBody([
                                    html.Div(
                                        html.H6(f'Working on: {old_ann_name}')
                                    ),
                                    dbc.Progress(
                                        id = {'type':'slide-load-progress','index':0},
                                        value = int(100*((old_ann_idx)/n_annotations)),
                                        label = f'{int(100*((old_ann_idx)/n_annotations))}%'
                                    )
                                ])
                            ])
                        ]

                else:
                    modal_children = []
                    disable_slide_load = True
                    modal_open = False

                slide_info_store_data = [
                    json.dumps(slide_info_store)
                ]

            else:
                disable_slide_load = True
                modal_open = False

        return modal_open, modal_children, slide_info_store_data, disable_slide_load

    def ingest_wsi(self,load_slide_done):
        """
        Populating slide visualization components after loading annotation and tile information
        """
        load_slide_done = get_pattern_matching_value(load_slide_done)
        if load_slide_done:
            # Updating in the case that an FTU isn't in the previous set of ftu_colors
            self.ftu_colors = self.wsi.ftu_colors

            # Updating overlays colors according to the current cell
            self.update_hex_color_key()

            special_overlays_opts = self.layout_handler.gen_special_overlay_opts(self.wsi)

            new_children = []
            if self.wsi.spatial_omics_type=='CODEX':
                # Adding the different frames to the layers control object
                new_children+=[
                    dl.BaseLayer(
                        dl.TileLayer(
                            url = self.wsi.channel_tile_url[c_idx],
                            tileSize = 240,
                            maxNativeZoom = self.wsi.zoom_levels-1,
                            id = {'type':'codex-tile-layer','index':c_idx}
                        ),
                        name = c_name,
                        checked = c_name== self.wsi.channel_names[0]
                    )
                    for c_idx,c_name in enumerate(self.wsi.channel_names)
                ]

                slide_tile_layer = []

            else:
                
                slide_tile_layer = [
                    dl.TileLayer(
                        id = 'slide-tile',
                        url = self.wsi.tile_url,
                        tileSize = self.wsi.tile_dims[0],
                        maxNativeZoom = self.wsi.zoom_levels-1
                    )
                ]

            new_children += [
                dl.Overlay(
                    dl.LayerGroup(
                        dl.GeoJSON(url = self.wsi.map_dict['FTUs'][struct]['url'], id = self.wsi.map_dict['FTUs'][struct]['id'], options = dict(style = self.ftu_style_handle, filter = self.ftu_filter),
                                    hideout = dict(color_key = self.hex_color_key, overlay_prop = self.overlay_prop, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors, filter_vals = self.filter_vals),
                                    hoverStyle = arrow_function(dict(weight=5, color = self.wsi.map_dict['FTUs'][struct]['hover_color'], dashArray = '')),
                                    zoomToBounds=False,children = [dl.Popup(id=self.wsi.map_dict['FTUs'][struct]['popup_id'])])
                    ), name = struct, checked = True, id = self.wsi.item_id+'_'+struct
                )
                for struct in self.wsi.map_dict['FTUs']
            ]

            # Now iterating through manual ROIs
            for m_idx, man in enumerate(self.wsi.manual_rois):
                new_children.append(
                    dl.Overlay(
                        dl.LayerGroup(
                            dl.GeoJSON(data = man['geojson'], id = man['id'], options = dict(style = self.ftu_style_handle,filter = self.ftu_filter),
                                        hideout = dict(color_key = self.hex_color_key, overlay_prop = self.overlay_prop, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors,filter_vals = self.filter_vals),
                                        hoverStyle = arrow_function(dict(weight=5,color=man['hover_color'], dashArray='')),
                                        children = [dl.Popup(id=man['popup_id'])])
                        ),
                        name = f'Manual ROI {m_idx}', checked = True, id = self.wsi.item_id+f'_manual_roi{m_idx}'
                    )
                )

            center_point = [2.0*(self.wsi.map_bounds[0][0]+self.wsi.map_bounds[1][0]),1.0*(self.wsi.map_bounds[0][1]+self.wsi.map_bounds[1][1])]

            map_center = {
                'center': center_point,
                'zoom': 3,
                'transition':'flyTo'
            }

            self.current_ftus = self.wsi.ftu_props
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
                        dmc.ColorPicker(
                            id = {'type':'ftu-bound-color','index':idx},
                            format = 'hex',
                            value = combined_colors_dict[struct]['color'],
                            fullWidth=True
                        ),
                        dbc.Button(
                            'Update Structure Boundaries',
                            className = 'd-grid col-12 md-auto',
                            id = {'type':'ftu-bound-color-butt','index':idx},
                            style = {'marginTop':'5px'}
                        )
                    ], label = struct
                )
                for idx, struct in enumerate(list(combined_colors_dict.keys()))
            ]

            # Structure hierarchy tab creation
            structure_hierarchy_tabs = [
                dbc.Tab(
                    children = [
                        html.Div(
                            id = {'type':'structure-tree-div','index':idx},
                            children = []
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

            return slide_tile_layer, new_children, remove_old_edits, map_center, self.wsi.properties_list, boundary_options_children, special_overlays_opts, structure_hierarchy_tabs
        else:
            raise exceptions.PreventUpdate

    def update_graph_label_children(self,leftover_labels):
        """
        Generating distribution/count for each label in current plot in morphological clustering tab
        """

        if len(leftover_labels)>0:
            unique_labels = np.unique(leftover_labels).tolist()
            u_l_data = []

            if type(unique_labels[0]) in [int,float]:
                quantile_labels = np.quantile(leftover_labels,[0,0.25,0.5,0.75,1.0])
                quantile_labels = [
                    [quantile_labels[0],quantile_labels[1]],
                    [quantile_labels[1],quantile_labels[2]],
                    [quantile_labels[2],quantile_labels[3]],
                    [quantile_labels[3],quantile_labels[4]]
                ]

                for q in quantile_labels:
                    q_label = '-->'.join([str(round(i,2)) for i in q])
                    q_count = ((leftover_labels > q[0]) & (leftover_labels < q[1])).sum()
                    u_l_data.append({
                        'label': q_label,
                        'count': q_count
                    })

            else:
                for u in unique_labels:
                    one_u_l = {'label':u, 'count': leftover_labels.count(u)}
                    u_l_data.append(one_u_l)

            label_pie_data = pd.DataFrame.from_records(u_l_data)

            if type(unique_labels[0]) == str:
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

            elif type(unique_labels[0]) in [int,float]:
                label_pie = px.bar(
                    data_frame = label_pie_data,
                    x = 'label',
                    y = 'count',
                    title = '<br>'.join(
                        textwrap.wrap('Count of samples in each label category',width = 20)
                    )
                )
                label_pie.update_layout(
                    showlegend = False,
                    autosize = True,
                    margin = {'b':0,'l':0,'r':0}
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
                    trigger='legacy',
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

    def update_graph(self,gen_plot_butt,label,checked_feature,filter_labels,states_option,cluster_data_store,user_data_store):
        """
        Updating plot displayed in morphological clustering tab according to checked features, filters, and labels
        """
        
        # Grabbing current metadata from user private folder        
        # Finding features checked:

        # Placeholder, replacing samples with missing cell types with 0 if cell features are included
        replace_missing = False
        cluster_data_store = json.loads(cluster_data_store)
        user_data_store = json.loads(user_data_store)

        if not cluster_data_store['feature_data'] is None:
            feature_data = pd.DataFrame.from_records(cluster_data_store['feature_data'])
        else:
            feature_data = None
        
        if not cluster_data_store['umap_df'] is None:
            umap_df = pd.DataFrame.from_records(cluster_data_store['umap_df'])
        else:
            umap_df = None

        if not cluster_data_store['clustering_data'] is None:   
            clustering_data = pd.DataFrame.from_records(cluster_data_store['clustering_data'])
        else:
            clustering_data = None

        if ctx.triggered_id=='gen-plot-butt':

            self.reports_generated = {}
            report_active_tab = 'feat-summ-tab'

            # Enabling download plot data button
            download_plot_disable = False

            if clustering_data.empty:
                print(f'Getting new clustering data')
                clustering_data = self.dataset_handler.load_clustering_data(user_data_store)
                cluster_data_store['clustering_data'] = clustering_data.to_dict('records')

            feature_names = [i['title'] for i in self.dataset_handler.feature_keys if i['key'] in checked_feature]
            cell_features = [i for i in feature_names if i in self.cell_names_key]
            feature_data = pd.DataFrame()

            if len(cell_features)>0:
                feature_data = clustering_data.loc[:,[i for i in feature_names if i in clustering_data.columns]]
                feature_data = feature_data.reset_index(drop=True)

                if 'Main_Cell_Types' in clustering_data.columns:
                    cell_values = clustering_data['Main_Cell_Types'].tolist()
                    state_values = clustering_data['Cell_States'].tolist()
                    if states_option=='main':
                        for c in cell_features:
                            cell_abbrev = self.cell_names_key[c]

                            specific_cell_values = []
                            for i in cell_values:
                                if not i is None:
                                    if cell_abbrev in i:
                                        specific_cell_values.append(i[cell_abbrev])
                                    else:
                                        # This is adding 0 for all samples without this specific cell type measured
                                        if replace_missing:
                                            specific_cell_values.append(0)
                                        else:
                                            specific_cell_values.append(np.nan)
                                else:
                                    # This is adding 0 for all samples without any cell types measured
                                    if replace_missing:
                                        specific_cell_values.append(0)
                                    else:
                                        specific_cell_values.append(np.nan)

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
                                        # This is for if this specific cell type is not measured
                                        if replace_missing:
                                            main_val = 0.0
                                            states = {c_state:0.0 for c_state in np.unique(self.cell_graphics_key[cell_abbrev]['states'])}
                                        else:
                                            main_val = np.nan
                                            states = {c_state: np.nan for c_state in np.unique(self.cell_graphics_key[cell_abbrev]['states'])}
                                else:
                                    # This is for if no cell types are measured
                                    if replace_missing:
                                        main_val = 0.0
                                        states = {c_state:0.0 for c_state in np.unique(self.cell_graphics_key[cell_abbrev]['states'])}
                                    else:
                                        main_val = np.nan
                                        states = {c_state: np.nan for c_state in np.unique(self.cell_graphics_key[cell_abbrev]['states'])}

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
                feature_data = clustering_data.loc[:,[i for i in feature_names if i in clustering_data.columns]]
                feature_data = feature_data.reset_index(drop=True)

            # Removing duplicate columns
            feature_data = feature_data.loc[:,~feature_data.columns.duplicated()].copy()

            # Coercing dtypes of columns in feature_data
            for f in feature_data.columns.tolist():
                feature_data[f] = pd.to_numeric(feature_data[f],errors='coerce')

            label_options = ['FTU','Slide Name','Folder Name'] 

            # Getting the label data
            label_data = []
            if label in clustering_data.columns or label in feature_data.columns:
                if label in clustering_data.columns:
                    label_data = clustering_data[label].tolist()
                else:
                    # This has to have an index portion just so they line up consistently
                    label_data = [0]*(max(list(feature_data.index))+1)
                    for idx in list(feature_data.index):
                        label_data[idx] = float(feature_data.loc[idx,label])
            else:
                sample_ids = [i['Slide_Id'] for i in clustering_data['Hidden'].tolist()]
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

                elif label in self.cell_names_key:
                    # For labeling with cell type
                    label_data = [float(i[self.cell_names_key[label]]) if self.cell_names_key[label] in i else 0 for i in clustering_data['Main_Cell_Types'].tolist()]

                else:
                    # Default labels just to FTU
                    label = 'FTU'
                    label_data = clustering_data[label].tolist()


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
                    ftu_labels = clustering_data['FTU'].tolist()
                    filter_idx.extend([i for i in range(len(ftu_labels)) if ftu_labels[i] in filter_label_names])
                    unique_parent_filters = [i for i in unique_parent_filters if not i=='FTUs']
                    
                if len(unique_parent_filters)>0:
                    # Grab slide metadata
                    sample_ids = [i['Slide_Id'] for i in clustering_data['Hidden'].tolist()]
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
                feature_data['Hidden'] = clustering_data['Hidden'].tolist()       
                feature_data['label'] = [label_data[i] for i in list(feature_data.index)]
                if 'Main_Cell_Types' in clustering_data.columns:
                    feature_data['Main_Cell_Types'] = clustering_data['Main_Cell_Types'].tolist()
                    feature_data['Cell_States'] = clustering_data['Cell_States'].tolist()
                
                # Dropping filtered out rows
                if len(filter_idx)>0:
                    feature_data = feature_data.iloc[[int(i) for i in range(feature_data.shape[0]) if int(i) not in filter_idx],:]

                feature_data = feature_data.dropna(subset=[i for i in feature_names if i in feature_data.columns]+['label','Hidden'])

                # Generating labels_info_children
                labels_left = feature_data['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)

                if feature_data.shape[0]>0:

                    # Adding unique point index to hidden data
                    hidden_data = feature_data['Hidden'].tolist()
                    for h_i,h in enumerate(hidden_data):
                        h['Index'] = h_i

                    feature_data.loc[:,'Hidden'] = hidden_data

                    figure = gen_violin_plot(feature_data, 'label', label, feature_names[0], 'Hidden')

                else:
                    figure = go.Figure()
                
                cluster_data_store['feature_data'] = feature_data.to_dict('records')

            elif len(feature_names)==2:
                print(f'Generating a scatter plot')
                feature_columns = feature_names

                label_options += [i for i in feature_data.columns.tolist() if not i == 'Hidden']

                # Adding "Hidden" column with image grabbing info
                feature_data['Hidden'] = clustering_data['Hidden'].tolist()     
                feature_data['label'] = label_data
                if 'Main_Cell_Types' in clustering_data.columns:
                    feature_data['Main_Cell_Types'] = clustering_data['Main_Cell_Types'].tolist()
                    feature_data['Cell_States'] = clustering_data['Cell_States'].tolist()
                
                # Dropping filtered out rows
                if len(filter_idx)>0:
                    feature_data = feature_data.iloc[[int(i) for i in range(feature_data.shape[0]) if int(i) not in filter_idx],:]
                
                feature_data = feature_data.dropna(subset=[i for i in feature_names if i in feature_data.columns]+['label','Hidden'])
                # Adding point index to hidden data
                hidden_data = feature_data['Hidden'].tolist()
                for h_i,h in enumerate(hidden_data):
                    h['Index'] = h_i

                feature_data.loc[:,'Hidden'] = hidden_data
                # Generating labels_info_children and filter_info_children
                labels_left = feature_data['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)

                cluster_data_store['feature_data'] = feature_data.to_dict('records')

                figure = go.Figure(data = px.scatter(
                    data_frame=feature_data,
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

                if not feature_data['label'].dtype == np.number:
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

                label_options += [i for i in feature_data.columns.tolist() if not i == 'Hidden']

                # Scaling and reducing feature data using UMAP
                feature_data['Hidden'] = clustering_data['Hidden'].tolist()
                feature_data['label'] = label_data
                if 'Main_Cell_Types' in clustering_data.columns:
                    feature_data['Main_Cell_Types'] = clustering_data['Main_Cell_Types'].tolist()
                    feature_data['Cell_States'] = clustering_data['Cell_States'].tolist()

                # Dropping filtered out rows
                if len(filter_idx)>0:
                    feature_data = feature_data.iloc[[int(i) for i in range(feature_data.shape[0]) if int(i) not in filter_idx],:]

                feature_data = feature_data.dropna(subset=[i for i in feature_names if i in feature_data.columns]+['label','Hidden'])
                hidden_col = feature_data['Hidden'].tolist()

                # Adding point index to hidden_col
                for h_i,h in enumerate(hidden_col):
                    h['Index'] = h_i

                feature_data['Hidden'] = hidden_col

                cluster_data_store['feature_data'] = feature_data.to_dict('records')

                umap_df = gen_umap(feature_data, feature_names, ['Hidden','label','Main_Cell_Types','Cell_States'])
                # Saving this so we can update the label separately without re-running scaling or reduction

                cluster_data_store['umap_df'] = umap_df.to_dict('records')

                # Generating labels_info_children and filter_info_children
                labels_left = umap_df['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)


                if not type(label_data[0]) in [int,float]:

                    figure = go.Figure(px.scatter(
                        data_frame = umap_df,
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
                        )
                    )
                else:

                    figure = go.Figure(
                        go.Scatter(
                            x = umap_df['UMAP1'].values,
                            y = umap_df['UMAP2'].values,
                            customdata = umap_df['Hidden'].tolist(),
                            mode = 'markers',
                            marker = {
                                'color': umap_df['label'].values,
                                'colorbar':{
                                    'title': 'label'
                                },
                                'colorscale':'jet'
                            },
                            text = umap_df['label'].values,
                            hovertemplate = "label: %{text}"

                        )
                    )

                    figure.update_layout(
                        showlegend = False,
                        title = {
                            'text': '<br>'.join(
                                textwrap.wrap(
                                    f'UMAP of selected features labeled with {label}',
                                    width=30
                                )
                            )
                        }
                    )

                figure.update_layout(
                    margin = {'r':0,'b':25,'l':0},
                    yaxis = dict(
                        title = {'text':''},
                        ticks = '',
                        tickfont = {'size':1}
                    ),
                    xaxis = dict(
                        title = {'text':''},
                        ticks = '',
                        tickfont = {'size':1}
                    )
                )

            else:
                # No features selected
                figure = go.Figure()
                label_info_children = []
                filter_info_children = no_update

            return figure, label_options, label_info_children, filter_info_children, report_active_tab, download_plot_disable, json.dumps(cluster_data_store)
        
        elif ctx.triggered_id=='label-select':
            self.reports_generated = {}
            report_active_tab = 'feat-summ-tab'

            # Enabling download plot data button
            download_plot_disable = False
            label_options = no_update

            label_data = []
            if not feature_data is None:
                # Getting the label data
                if label in clustering_data.columns or label in feature_data.columns:
                    if label in clustering_data.columns:
                        label_data = clustering_data[label].tolist()
                    else:
                        # This has to have an index portion just so they line up consistently
                        label_data = [0]*(max(list(feature_data.index))+1)
                        for idx in list(feature_data.index):
                            label_data[idx] = float(feature_data.loc[idx,label])
                else:
                    sample_ids = [i['Slide_Id'] for i in clustering_data['Hidden'].tolist()]
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

                    elif label in self.cell_names_key:
                        # For labeling with cell type
                        label_data = [float(i[self.cell_names_key[label]]) if self.cell_names_key[label] in i else 0 for i in clustering_data['Main_Cell_Types'].tolist()]

                    else:
                        label = 'FTU'
                        label_data = clustering_data[label].tolist()


                label_data = [label_data[i] for i in list(feature_data.index)]
                # Needs an alignment step going from the full clustering_data to the na-dropped and filtered feature_data
                # feature_data contains the features included in the current plot, label, Hidden, Main_Cell_Types, and Cell_States
                # So for a violin plot the shape should be nX5
                feature_data = feature_data.drop(columns = ['label'])
                feature_data['label'] = label_data

                if type(label_data[0]) in [int,float]:
                    feature_data['label'] = feature_data['label'].astype(float)

                cluster_data_store['feature_data'] = feature_data.to_dict('records')

                # Generating labels_info_children
                labels_left = feature_data['label'].tolist()
                label_info_children, filter_info_children = self.update_graph_label_children(labels_left)

                feature_number = len([i for i in feature_data.columns.tolist() if i not in ['label','Hidden','Main_Cell_Types','Cell_States']])

                if feature_number==1:
                    feature_names = feature_data.columns.tolist()

                    figure = gen_violin_plot(feature_data, 'label', label, feature_names[0],'Hidden')
                    
                elif feature_number==2:
                    
                    feature_columns = feature_data.columns.tolist()

                    figure = go.Figure(px.scatter(
                        data_frame=feature_data,
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

                    if not type(label_data[0]) in [int,float]:
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

                    umap_df.loc[:,'label'] = label_data

                    cluster_data_store['umap_df'] = umap_df.to_dict('records')
                    
                    if not type(label_data[0]) in [int,float]:

                        figure = go.Figure(px.scatter(
                            data_frame = umap_df,
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
                            )
                        )
                    else:

                        figure = go.Figure(
                            go.Scatter(
                                x = umap_df['UMAP1'].values,
                                y = umap_df['UMAP2'].values,
                                customdata = umap_df['Hidden'].tolist(),
                                mode = 'markers',
                                marker = {
                                    'color': umap_df['label'].values,
                                    'colorbar':{
                                        'title': 'label'
                                    },
                                    'colorscale':'jet'
                                },
                                text = umap_df['label'].tolist(),
                                hovertemplate = "label: %{text}"
                            )
                        )

                        figure.update_layout(
                            showlegend = False,
                            title = {
                                'text': '<br>'.join(
                                    textwrap.wrap(
                                        f'UMAP of selected features labeled with {label}',
                                        width=30
                                    )
                                )
                            }
                        )

                    figure.update_layout(
                        margin = {'r':0,'b':25,'l':0},
                        yaxis = {
                            'title':{'text':''},
                            'ticks':"",
                            'tickfont':{'size':1}
                        },
                        xaxis = {
                            'title':{'text':''},
                            'ticks':"",
                            'tickfont':{'size':1}
                        }
                    )
                
                return figure, label_options, label_info_children, filter_info_children, report_active_tab, download_plot_disable, json.dumps(cluster_data_store)
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate

    def grab_image(self,sample_info):
        """
        Getting image for clicked point in morphological clustering tab. (referenced in self.update_selected)
        """
        if len(sample_info)>100:
            sample_info = sample_info[0:100]

        img_list = []
        img_dims = np.zeros((len(sample_info),2))
        for idx,s in enumerate(sample_info):
            image_region = np.array(self.dataset_handler.get_annotation_image(s['Slide_Id'],s['Bounding_Box']))
            
            # Resizing images so that each one is the same size
            img_list.append(resize(np.array(image_region),output_shape=(512,512,3),anti_aliasing=True))
            #TODO: Find some efficient way to pad images equally up to the max size and then resize to 512x512x3
            #img_list.append(image_region)
            #img_dims[idx,:] += np.shape(image_region).tolist()[0:2]
        
        return img_list        

    def update_selected(self,click,selected,cluster_data_store):
        """
        Getting cell/state information and image for selected point in plot.
        """

        cluster_data_store = json.loads(cluster_data_store)
        feature_data = pd.DataFrame.from_records(cluster_data_store['feature_data'])

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

            cluster_data_store['current_selected_samples'] = sample_index

            # Creating selected_image_info children
            # This could have more functionality
            # Like adding markers or going to that region in the current slide (if the selected image is from the current slide)
            slide_names = [self.dataset_handler.gc.get(f'/item/{s_i["Slide_Id"]}')["name"] for s_i in sample_info]
            label_list = feature_data['label'].tolist()
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
                                    className='d-grid col-12 mx-auto',
                                    style = {'marginBottom':'5px'}
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
                style = {'maxHeight':'500px','overflow':'scroll'}
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
                print(f'self.current_selected_samples: {sample_index}')

            if 'Main_Cell_Types' in feature_data.columns:
                # Preparing figure containing cell types + cell states info
                main_cell_types_list = feature_data['Main_Cell_Types'].tolist()
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
                    cell_states_list = feature_data['Cell_States'].tolist()
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

            return selected_image, selected_cell_types, selected_cell_states, selected_image_info, json.dumps(cluster_data_store)
        else:
            return go.Figure(), go.Figure(), go.Figure(),[], json.dumps(cluster_data_store)
    
    def update_selected_state_bar(self, selected_cell_click,cluster_data_store):
        """
        Getting cell state distribution plot for clicked main cell type
        """
        cluster_data_store = json.loads(cluster_data_store)
        feature_data = pd.DataFrame.from_records(cluster_data_store['feature_data'])
        current_selected_samples = cluster_data_store['current_selected_samples']

        if not selected_cell_click is None:
            cell_type = selected_cell_click['points'][0]['label']

            cell_states_data = feature_data['Cell_States'].tolist()
            state_data = pd.DataFrame([cell_states_data[i][cell_type] for i in current_selected_samples]).sum(axis=0).to_frame()
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
        """
        Adding a rectangle, polygon, or marker annotation to the current image
        """
        
        if not self.wsi is None:
            try:
                # Used for pattern-matching callbacks
                triggered_id = ctx.triggered_id['type']
            except TypeError:
                # Used for normal callbacks
                triggered_id = ctx.triggered_id

            if triggered_id == 'edit_control':
                
                if triggered_id=='edit_control':
                    #TODO: get_pattern_matching_value test here
                    if type(new_geojson)==list:
                        new_geojson = new_geojson[0]    

                if not new_geojson is None:
                    new_roi = None
                    if len(new_geojson['features'])>0:
                        
                        # Adding each manual annotation iteratively (good for if annotations are edited or deleted as well as for new annotations)
                        self.wsi.manual_rois = []
                        self.wsi.marked_ftus = []

                        if not self.wsi.spatial_omics_type=='CODEX':
                            # This is starting off only with the FTU annotations for Visium and Regular slides
                            self.current_overlays = self.current_overlays[0:len(self.wsi.ftu_names)]
                        else:
                            # This is for CODEX images where each frame is added as a BaseLayer
                            self.current_overlays = self.current_overlays[0:self.wsi.n_frames+len(self.wsi.ftu_names)]

                        for geo in new_geojson['features']:

                            if not geo['properties']['type'] == 'marker':

                                new_roi = {'type':'FeatureCollection','features':[geo]}
                                
                                # New geojson has no properties which can be used for overlays or anything so we have to add those
                                if self.wsi.spatial_omics_type=='Visium':
                                    # Step 1, find intersecting spots:
                                    overlap_props,_ = self.wsi.find_intersecting_ftu(shape(new_roi['features'][0]['geometry']),'Spots')
                                    # Adding Main_Cell_Types from intersecting spots data
                                    main_counts_data = pd.DataFrame.from_records([i['Main_Cell_Types'] for i in overlap_props if 'Main_Cell_Types' in i]).sum(axis=0).to_frame()
                                    main_counts_data = (main_counts_data/main_counts_data.sum()).fillna(0.000).round(decimals=18)
                                    main_counts_data[0] = main_counts_data[0].map('{:.19f}'.format)
                                    main_counts_dict = main_counts_data.astype(float).to_dict()[0]

                                    # Repeating for Gene Counts
                                    gene_counts_data = pd.DataFrame.from_records([i['Gene Counts'] for i in overlap_props if 'Gene Counts' in i]).sum(axis=0).to_frame()
                                    main_counts_data = (gene_counts_data/gene_counts_data.sum()).fillna(0.000).round(decimals=18)
                                    gene_counts_data[0] = gene_counts_data[0].map('{:.19f}'.format)
                                    gene_counts_dict = gene_counts_data.astype(float).to_dict()[0]

                                    # Aggregating cell state information from intersecting spots
                                    agg_cell_states = {}
                                    for m_c in list(main_counts_dict.keys()):

                                        cell_states = pd.DataFrame.from_records([i['Cell_States'][m_c] for i in overlap_props]).sum(axis=0).to_frame()
                                        cell_states = (cell_states/cell_states.sum()).fillna(0.000).round(decimals=18)
                                        cell_states[0] = cell_states[0].map('{:.19f}'.format)

                                        agg_cell_states[m_c] = cell_states.astype(float).to_dict()[0]

                                    new_roi['features'][0]['properties']['user'] = {
                                        'Main_Cell_Types': main_counts_dict,
                                        'Cell_States': agg_cell_states,
                                        'Gene Counts': gene_counts_dict
                                    }


                                elif self.wsi.spatial_omics_type=='CODEX':
                                    overlap_props = self.wsi.intersecting_frame_intensity(shape(new_roi['features'][0]['geometry']),'all')

                                    # Adding channel intensity histogram to properties.
                                    new_roi['features'][0]['properties']['user'] = {}
                                    for frame in overlap_props:
                                        new_roi['features'][0]['properties']['user'][frame] = overlap_props[frame]

                                #TODO: include overlapping FTU properties in visualization within Manual ROIs
                                # Now including properties of intersecting FTUs
                                #overlap_ftu_props, _ = self.wsi.find_intersecting_ftu(shape(new_roi['features'][0]['geometry']),'all')

                                # Have to normalize and make this visible in the "Cell Compositions" or something tab (since it might be more than just cells)

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
                                                hideout = dict(color_key = self.hex_color_key, overlay_prop = self.overlay_prop, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors, filter_vals = self.filter_vals),
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
                                                        'properties': {
                                                            'user': overlap_dict
                                                        }
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
                                                        'properties':{
                                                            'user': overlap_dict
                                                        }
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
                                                'properties':{
                                                    'user':overlap_dict
                                                }
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
                                            hideout = dict(color_key = self.hex_color_key, overlay_prop = self.overlay_prop, fillOpacity = self.cell_vis_val, ftu_colors = self.ftu_colors,filter_vals = self.filter_vals),
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

                        self.update_hex_color_key()
                        
                        if new_roi:
                            user_ann_tracking = json.dumps({ 'slide_name': self.wsi.slide_name, 'item_id': self.wsi.item_id })
                            return self.current_overlays, data_select_options, user_ann_tracking
                        
                        return self.current_overlays, data_select_options, no_update
                    else:

                        # Clearing manual ROIs and reverting overlays
                        self.wsi.manual_rois = []
                        self.wsi.marked_ftus = []

                        if not self.wsi.spatial_omics_type=='CODEX':
                            self.current_overlays = self.current_overlays[0:len(self.wsi.ftu_names)]
                        else:
                            self.current_overlays = self.current_overlays[0:self.wsi.n_frames+len(self.wsi.ftu_names)]

                        data_select_options = self.layout_handler.data_options

                        self.update_hex_color_key()
                        
                        return self.current_overlays, data_select_options, no_update
                else:
                    raise exceptions.PreventUpdate
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate

    def add_marker_from_cluster(self,mark_click,cluster_data_store):
        """
        Marking structure in GeoJSON layer from selected points in plot (in the current slide)
        """

        cluster_data_store = json.loads(cluster_data_store)
        feature_data = pd.DataFrame.from_records(cluster_data_store['feature_data'])
        current_selected_samples = cluster_data_store['current_selected_samples']

        # Adding marker(s) from graph returning geojson
        # index = 0 == mark all the samples in the current slide
        # index != 0 == mark a specific sample in the current slide
        if ctx.triggered[0]['value']:
            
            if ctx.triggered_id['index']==0:
                # Add marker for all samples in the current slide
                mark_geojson = {'type':'FeatureCollection','features':[]}

                # Iterating through all current selected samples
                # current_selected_samples is an index from feature_data
                current_selected_hidden = [feature_data['Hidden'].tolist()[i] for i in current_selected_samples]
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
                selected_bbox = feature_data['Hidden'].tolist()[current_selected_samples[ctx.triggered_id['index']-1]]['Bounding_Box']
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

    def update_download_options(self,selected_data,vis_sess_store):
        
        print(f'selected_data: {selected_data}')
        new_children = []
        tab_labels = []

        vis_sess_store = json.loads(vis_sess_store)

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
                            [i['name'] for i in vis_sess_store],
                            [i['name'] for i in vis_sess_store],
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

        if not self.wsi is None:
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
                    elif download_type == 'manual_rois':
                        #TODO: Find bounding box of all manually annotated regions for current slide, create new image,
                        # get those files as well as annotations for manual ROIs
                        download_list = self.download_handler.extract_manual_rois(self.wsi,self.dataset_handler,options)

                    elif download_type == 'select_ftus':
                        #TODO: Find all manually selected FTUs, extract image bounding box of those regions,
                        # create binary mask for each one, save along with cell data if needed
                        download_list = []

                    else:
                        print('Working on it!')
                        download_list = []

                    self.download_handler.zip_data(download_list)
                    
                    return dcc.send_file('./assets/FUSION_Download.zip')

                else:
                    raise exceptions.PreventUpdate
            else:
                raise exceptions.PreventUpdate
        else:
            #TODO: Add some error handling here for if there isn't anything to save
            # Or just make this tab not enabled
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

    def update_upload_requirements(self,upload_type,user_data_store):
        """
        Creating upload components for different file types        
        """

        print(f'update_upload_requirements: {ctx.triggered_id}')
        print(ctx.triggered)
        if not ctx.triggered[0]['value']:
            raise exceptions.PreventUpdate

        input_disabled = True
        # Creating an upload div specifying which files are needed for a given upload type
        # Getting the collection id
        user_data_store = json.loads(user_data_store)

        # Making a new folder for this upload in the user's Public (could also change to private?) folder
        current_datetime = str(datetime.now())
        current_datetime = current_datetime.replace('-','_').replace(' ','_').replace(':','_').replace('.','_')
        parentId = self.dataset_handler.get_user_folder_id(f'Public/FUSION_Upload_{current_datetime}',user_data_store['login'])

        user_data_store['latest_upload_folder'] = {
            'id': parentId,
            'path':f'Public/FUSION_Upload_{current_datetime}'
        }

        if upload_type == 'Visium':

            self.prep_handler = VisiumPrep(self.dataset_handler)

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
                                girderToken= user_data_store['token'],
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
                            dbc.Label('Upload your counts file here!'),
                            self.layout_handler.gen_info_button('Upload either an RDS file or h5ad file containing gene counts per-spot.'),
                            UploadComponent(
                                id = {'type':'omics-upload','index':0},
                                uploadComplete=False,
                                baseurl=self.dataset_handler.apiUrl,
                                girderToken = user_data_store['token'],
                                parentId=parentId,
                                filetypes=['rds','csv','h5ad']                 
                            )
                        ],
                        style = {'marginTop':'10px','display':'inline-block'}
                    )
                ],align='center'),
                dbc.Row([
                    dbc.Col(dbc.Label('Select Organ: ',html_for = {'type':'organ-select','index':0}),md=4),
                    dbc.Col(
                        dcc.Dropdown(
                            options = [
                                'Kidney', 'Other Organs'
                            ],
                            placeholder = 'Organ',
                            id = {'type':'organ-select','index':0}
                        )
                    )
                ])
            ])

            user_data_store['upload_check'] = {
                "WSI": False,
                "Omics": False
            }

            user_data_store['upload_type'] = 'Visium'
        
        elif upload_type =='Regular':
            # Regular slide with no --omics

            self.prep_handler = Prepper(self.dataset_handler)

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
                                girderToken=user_data_store['token'],
                                parentId=parentId,
                                filetypes=['svs','ndpi','scn','tiff','tif']                      
                            )
                        ],
                        style={'marginBottom':'10px','display':'inline-block'}
                    )
                ],align='center'),
                dbc.Row([
                    dbc.Col(dbc.Label('Select Organ: ',html_for = {'type':'organ-select','index':0}),md=4),
                    dbc.Col(
                        dcc.Dropdown(
                            options = [
                                'Kidney', 'Other Organs'
                            ],
                            placeholder = 'Organ',
                            id = {'type':'organ-select','index':0}
                        )
                    )
                ])
            ])

            user_data_store['upload_check'] = {
                'WSI': False
            }
            user_data_store['upload_type'] = 'Regular'

        elif upload_type == 'CODEX':
            # CODEX uploads include histology and multi-frame CODEX image (or just CODEX?)

            self.prep_handler = CODEXPrep(self.dataset_handler)

            upload_reqs = html.Div([
                dbc.Row([
                    html.Div(
                        id = {'type': 'wsi-upload-div','index':0},
                        children = [
                            dbc.Label('Upload Whole Slide Image (CODEX) Here!'),
                            UploadComponent(
                                id = {'type':'wsi-upload','index':0},
                                uploadComplete = False,
                                baseurl = self.dataset_handler.apiUrl,
                                girderToken = user_data_store['token'],
                                parentId = parentId,
                                filetypes = ['svs','ndpi','scn','tiff','tif']
                            )
                        ],
                        style = {'marginBottom':'10px','display':'inline-block'}
                    )
                ],align='center'),
                dbc.Row([
                    dbc.Col(dbc.Label('Select Organ: ',html_for = {'type':'organ-select','index':0}),md=4),
                    dbc.Col(
                        dcc.Dropdown(
                            options = [
                                'Kidney', 'Other Organs'
                            ],
                            placeholder = 'Organ',
                            id = {'type':'organ-select','index':0}
                        )
                    )
                ])
            ])

            self.upload_check = {'WSI':False}
            self.current_upload_type = 'CODEX'

            user_data_store['upload_check'] = {
                'WSI': False
            }

            user_data_store['upload_type'] = 'CODEX'
        
        user_data_store['upload_wsi_id'] = None
        user_data_store['upload_omics_id'] = None

        return upload_reqs, input_disabled, json.dumps(user_data_store)

    def girder_login(self,p_butt,create_butt,username,pword,email,firstname,lastname):

        create_user_children = []
        print(f'login ctx.triggered_id: {ctx.triggered_id}')
        usability_signup_style = no_update
        usability_butt_style = no_update

        user_data_output = no_update

        if ctx.triggered_id=='login-submit':

            try:
                user_info, user_details = self.dataset_handler.authenticate(username,pword)

                user_id = user_details['_id']

                button_color = 'success'
                button_text = 'Success!'
                logged_in_user = f'Welcome: {username}'
                upload_disabled = False

                user_data_output = {
                    'user_info': user_info,
                    'user_details': user_details,
                    'token': self.dataset_handler.user_token
                }

                if not user_info is None:
                    usability_signup_style = {'display':'none'}
                    usability_butt_style = {'marginLeft':'5px','display':'inline-block'}


            except girder_client.AuthenticationError:

                button_color = 'warning'
                button_text = 'Login Failed'
                logged_in_user = 'Welcome: fusionguest'
                upload_disabled = True

            return button_color, button_text, logged_in_user, upload_disabled, create_user_children, json.dumps({'user_id': username}), [usability_signup_style],[usability_butt_style], json.dumps(user_data_output)
        
        elif ctx.triggered_id=='create-user-submit':
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
                        dcc.Input(type='text',id={'type':'last-name-input','index':0}),
                        style = {'marginBottom':'5px'}
                    )
                ]

                button_color = 'success'
                button_text = 'Login'
                logged_in_user = 'Welcome: fusionguest'
                upload_disabled = True

                return button_color, button_text, logged_in_user, upload_disabled, create_user_children, no_update, [usability_signup_style],[usability_butt_style], json.dumps(user_data_output)


            else:
                create_user_children = no_update
                try:
                    user_info, user_details = self.dataset_handler.create_user(username,pword,email,firstname,lastname)

                    button_color = 'success',
                    button_text = 'Success!',
                    logged_in_user = f'Welcome: {username}'
                    upload_disabled = False

                    user_data_output = {
                        'user_info': user_info,
                        'user_details': user_details,
                        'token': self.dataset_handler.user_token
                    }

                    if not user_info is None:
                        usability_signup_style = {'display':'none'}
                        usability_butt_style = {'marginLeft':'5px','display':'inline-block'}


                except girder_client.AuthenticationError:

                    button_color = 'warning'
                    button_text = 'Login Failed'
                    logged_in_user = f'Welcome: fusionguest'
                    upload_disabled = True
            
                return button_color, button_text, logged_in_user, upload_disabled, create_user_children, json.dumps({'user_id': username}), [usability_signup_style],[usability_butt_style], json.dumps(user_data_output)

        else:
            raise exceptions.PreventUpdate

    def upload_data(self,wsi_file,omics_file,wsi_file_flag,omics_file_flag, user_data_store):
        
        user_data_store = json.loads(user_data_store)

        print(f'Triggered id for upload_data: {ctx.triggered_id}')
        if ctx.triggered_id['type']=='wsi-upload':

            # Getting the uploaded item id
            upload_wsi_id = self.dataset_handler.get_new_upload_id(user_data_store['latest_upload_folder']['id'])

            user_data_store['upload_wsi_id'] = upload_wsi_id

            if not upload_wsi_id is None:
                if not wsi_file_flag[0]:
                    wsi_upload_children = [
                        dbc.Alert('WSI Upload Success!',color='success')
                    ]

                    user_data_store['upload_check']['WSI'] = True

                    # Adding metadata to the uploaded slide
                    self.dataset_handler.add_slide_metadata(
                        item_id = upload_wsi_id,
                        metadata_dict = {
                            'Spatial Omics Type': user_data_store['upload_type']
                        }
                    )

                else:
                    wsi_upload_children = [
                        dbc.Alert('WSI Upload Failure! Accepted file types include svs, ndpi, scn, tiff, and tif',color='danger'),
                        UploadComponent(
                            id = {'type':'wsi-upload','index':0},
                            uploadComplete=False,
                            baseurl=self.dataset_handler.apiUrl,
                            girderToken=user_data_store['token'],
                            parentId=user_data_store['latest_upload_folder']['id'],
                            filetypes=['svs','ndpi','scn','tiff','tif']                      
                        )
                    ]

            else:
                wsi_upload_children = no_update
            
            if 'Omics' in user_data_store['upload_check']:
                if not user_data_store['upload_check']['Omics']:
                    omics_upload_children = no_update
                else:
                    omics_upload_children = [
                        dbc.Alert('Omics Upload Success!')
                    ]
            else:
                omics_upload_children = no_update

        elif ctx.triggered_id['type']=='omics-upload':
            
            upload_omics_id = self.dataset_handler.get_new_upload_id(user_data_store['latest_upload_folder']['id'])
            
            user_data_store['upload_omics_id'] = upload_omics_id

            if not self.upload_omics_id is None:
                #TODO: get_pattern_matching_value test here
                if type(omics_file_flag)==list:
                    if len(omics_file_flag)>0:
                        if not omics_file_flag[0]:
                            omics_upload_children = [
                                dbc.Alert('Omics Upload Success!')
                            ]

                            user_data_store['upload_check']['Omics'] = True
                        else:
                            omics_upload_children = [
                                dbc.Alert('Omics Upload Failure! Accepted file types are: rds',color = 'danger'),
                                UploadComponent(
                                    id = {'type':'omics-upload','index':0},
                                    uploadComplete=False,
                                    baseurl=self.dataset_handler.apiUrl,
                                    girderToken=user_data_store['token'],
                                    parentId=user_data_store['latest_upload_folder']['id'],
                                    filetypes=['rds','csv','h5ad']                 
                                    )
                            ]
                    else:
                        omics_upload_children = no_update
                else:
                    omics_upload_children = no_update
            else:
                omics_upload_children = no_update

            if not user_data_store['upload_check']['WSI']:
                wsi_upload_children = no_update
            else:
                wsi_upload_children = [
                    dbc.Alert('WSI Upload Success!',color='success')
                ]

        else:
            print(f'ctx.triggered_id["type"]: {ctx.triggered_id["type"]}')

        # Checking the upload check
        if all([user_data_store['upload_check'][i] for i in user_data_store['upload_check']]):
            print('All set!')

            slide_thumbnail = self.dataset_handler.get_slide_thumbnail(user_data_store['upload_wsi_id'])

            if 'Omics' in user_data_store['upload_check']:
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

            # This slide_meta is a dictionary which should just have the "Spatial Omics Type"
            slide_meta = self.dataset_handler.gc.get(f'/item/{user_data_store["upload_wsi_id"]}')['meta']
            # Now create the dataframe with columns like "metadata name" "value"
            slide_meta_df = pd.DataFrame({'Metadata Name':list(slide_meta.keys())+[''],'Value':list(slide_meta.values())+['']})

            slide_metadata_table = dash_table.DataTable(
                id = {'type':'slide-qc-table','index':0},
                columns = [{'name':i, 'id': i, 'deletable':False, 'selectable':True} for i in slide_meta_df],
                data = slide_meta_df.to_dict('records'),
                editable=True,
                row_deletable = True,
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
                    } for row in slide_meta_df.to_dict('records')
                ],
                tooltip_duration = None
            )

            structure_type_disabled = False
            post_upload_style = {'display':'flex'}
            disable_upload_type = True
            
            if 'Omics' in user_data_store['upload_check']:
                return slide_metadata_table, thumb_fig, [wsi_upload_children], [omics_upload_children], structure_type_disabled, post_upload_style, disable_upload_type, json.dumps({'plugin_used': 'upload', 'type': 'Visium' }), json.dumps(user_data_store)
            else:
                return slide_metadata_table, thumb_fig, [wsi_upload_children], [], structure_type_disabled, post_upload_style, disable_upload_type, json.dumps({'plugin_used': 'upload', 'type': 'non-Omnics' }), json.dumps(user_data_store)
        else:

            disable_upload_type = True

            if 'Omics' in user_data_store['upload_check']:
                return no_update, no_update,[wsi_upload_children], [omics_upload_children], True, no_update, disable_upload_type, no_update, json.dumps(user_data_store)
            else:
                return no_update, no_update, [wsi_upload_children], [], True, no_update, disable_upload_type, no_update, json.dumps(user_data_store)
    
    def add_slide_metadata(self, button_click, table_data, user_data_store):

        user_data_store = json.loads(user_data_store)
        # Adding slide metadata according to user inputs
        if not button_click:
            raise exceptions.PreventUpdate
        
        # Formatting the data so instead of a list of dicts it's one dictionary
        table_data = table_data[0]
        # Making it so you can't delete this metadata
        slide_metadata = {
            'Spatial Omics Type': user_data_store['upload_type']
        }
        add_row = True
        for m in table_data:
            if not m['Value']=='':
                slide_metadata[m['Metadata Name']] = m['Value']
            else:
                add_row = False

        # Checking current slide metadata and seeing if it differs from "slide_metadata"
        current_slide_metadata = self.dataset_handler.gc.get(f'/item/{user_data_store["upload_wsi_id"]}')['meta']

        if not list(slide_metadata.keys())==list(current_slide_metadata.keys()):
            # Finding the ones that are different
            add_keys = [i for i in list(slide_metadata.keys()) if i not in list(current_slide_metadata.keys())]
            rm_keys = [i for i in list(current_slide_metadata.keys()) if i not in list(slide_metadata.keys())]

            if len(rm_keys)>0:
                for m in rm_keys:
                    self.dataset_handler.gc.delete(
                        f'/item/{user_data_store["upload_wsi_id"]}/metadata',
                        parameters = {
                            'fields':f'["{m}"]'
                        }
                    )

            # Adding metadata through GirderHandler
            if add_row:
                self.dataset_handler.add_slide_metadata(user_data_store["upload_wsi_id"],slide_metadata)
                # Adding new empty row
                table_data.append({'Metadata Name':'', 'Value': ''})

        return [table_data]

    def start_segmentation(self,structure_selection,go_butt,user_data_store):

        # Starting segmentation job and initializing dcc.Interval object to check logs
        # output = div children, disable structure_type, disable segment_butt
        user_data_store = json.loads(user_data_store)

        if ctx.triggered_id=='segment-butt':
            if structure_selection is not None:
                if len(structure_selection)>0:
                    disable_structure = True
                    disable_seg_butt = True
                    disable_continue_butt = True

                    print(f'Running segmentation!')
                    user_data_store['segmentation_job'] = self.prep_handler.segment_image(user_data_store['upload_wsi_id'],structure_selection)
                    #TODO: Make it so you don't have to run segmentation to access cell deconvolution/spot stuff
                    if not user_data_store['upload_omics_id'] is None:
                        user_data_store['cell_deconv_job'] = self.prep_handler.run_cell_deconvolution(user_data_store['upload_wsi_id'],user_data_store['upload_omics_id'])

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

                    return seg_woodshed, disable_structure, disable_seg_butt, [disable_continue_butt], json.dumps(user_data_store)
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

    def new_seg_upload(self,seg_upload,current_up_anns,seg_upload_filename,user_data_store):

        user_data_store = json.loads(user_data_store)

        if seg_upload:
            # Processing new uploaded file
            current_len = len(current_up_anns)
            new_filename = seg_upload_filename[0]
            new_upload = seg_upload[0]
            
            # Processing newly uploaded annotations
            processed_anns = self.prep_handler.process_uploaded_anns(new_filename,new_upload,user_data_store['upload_wsi_id'])

            if not processed_anns is None:
                return_items = current_up_anns[0]
                return_items.append(
                    dbc.AccordionItem(
                        title = new_filename,
                        children = [
                            html.P(f'{a}: {processed_anns[a]}')
                            for a in processed_anns
                        ]
                    )
                )
            else:
                return_items = current_up_anns[0]
                return_items.append(
                    dbc.AccordionItem(
                        title = new_filename,
                        children = [
                            dbc.Alert('Invalid file type! Vali',color='danger'),
                            'Valid file types: Aperio XML (.xml), JSON (.json), GeoJSON (.geojson)'
                        ]
                    )
                )

            return [return_items]
        else:
            raise exceptions.PreventUpdate

    def update_logs(self,new_interval,user_data_store):

        # Callback to update the segmentation logs div with more data
        # Also populates the post-segment-row when the jobs are completed
        # output = seg-logs div, seg-interval disabled, post-segment-row style, 
        # structure-type disabled, ftu-select options, ftu-select value, sub-comp-method value,
        # ex-ftu-img figure
        # Getting most recent logs:

        user_data_store = json.loads(user_data_store)
        seg_status = 0
        seg_log = []
        for seg_job in user_data_store['segmentation_job']:
            s_stat, s_log = self.dataset_handler.get_job_status(seg_job['_id'])
            seg_log.append(s_log)
            seg_status+=s_stat
        seg_log = [html.P(s) for s in seg_log]

        if not user_data_store['upload_omics_id'] is None:
            cell_status, cell_log = self.dataset_handler.get_job_status(user_data_store['cell_deconv_job']['_id'])
            cell_log = [html.P(cell_log)]
        else:
            cell_status = 3
            cell_log = ''
        
        # This would be at the end of the two jobs
        if seg_status+cell_status==3*(1+len(user_data_store['segmentation_job'])):

            # Div containing the job logs:
            if not user_data_store['upload_omics_id'] is None:
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
            continue_disable = False

        else:
            # For during segmentation/cell-deconvolution
            seg_log_disable = False
            continue_disable = True

            if not user_data_store['upload_omics_id'] is None:
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
        return [seg_logs_div], [seg_log_disable], [continue_disable]

    def post_segmentation(self, seg_log_disable, continue_butt, organ, gene_method, gene_n, gene_list, user_data_store):

        user_data_store = json.loads(user_data_store)

        if ctx.triggered[0]['value']:
            # post-segment-row stuff
            sub_comp_style = {'display':'flex'}
            disable_organ = True

            # Getting values for organ, gene_method, gene_n, and gene_list
            organ = get_pattern_matching_value(organ)
            if organ is None:
                organ = ""
            
            gene_method = get_pattern_matching_value(gene_method)
            if gene_method is None:
                gene_method = ""
            
            gene_n = get_pattern_matching_value(gene_n)
            if gene_n is None:
                gene_n = 0

            gene_list = get_pattern_matching_value(gene_list)
            if gene_list is None:
                gene_list = ""

            if user_data_store['upload_type'] == 'Regular':
                # Extracting annotations and initilaiz sub-compartment mask
                self.upload_annotations = self.dataset_handler.get_annotations(user_data_store['upload_wsi_id'])

                # Running post-segmentation worfklow from Prepper
                self.feature_extract_ftus, self.layer_ann = self.prep_handler.post_segmentation(user_data_store['upload_wsi_id'],self.upload_annotations)

            if user_data_store['upload_type'] == 'Visium':
                # Extracting annotations and initial sub-compartment mask
                self.upload_annotations = self.dataset_handler.get_annotations(user_data_store['upload_wsi_id'])
                
                # Running post-segmentation workflowfrom VisiumPrep
                self.feature_extract_ftus, self.layer_ann = self.prep_handler.post_segmentation(
                    user_data_store['upload_wsi_id'], 
                    user_data_store['upload_omics_id'], 
                    self.upload_annotations,
                    organ,
                    gene_method,
                    gene_n,
                    gene_list
                )

            elif user_data_store['upload_type'] == 'CODEX':

                # Running post-segmentation workflow from CODEX Prep
                frame_names, current_frame = self.prep_handler.post_segmentation(user_data_store['upload_wsi_id']) 

            # Populate with default sub-compartment parameters
            self.sub_compartment_params = self.prep_handler.initial_segmentation_parameters

            sub_comp_method = 'Manual'

            if user_data_store['upload_type'] in ['Visium','Regular']:

                if not self.layer_ann is None:

                    ftu_value = self.feature_extract_ftus[self.layer_ann['current_layer']]
                    image, mask = self.prep_handler.get_annotation_image_mask(user_data_store['upload_wsi_id'],self.upload_annotations, self.layer_ann['current_layer'],self.layer_ann['current_annotation'])

                    self.layer_ann['current_image'] = image
                    self.layer_ann['current_mask'] = mask
 
                else:
                    ftu_value = ''
                    image = np.ones((10,10))

                prep_values = {
                    'ftu_names': self.feature_extract_ftus,
                    'image': image
                }

            elif user_data_store['upload_type'] == 'CODEX':
                self.feature_extract_ftus = None

                prep_values = {
                    'frames': frame_names
                }

            # Generating upload preprocessing row
            prep_row = self.layout_handler.gen_uploader_prep_type(user_data_store['upload_type'],prep_values)
                
            return sub_comp_style, disable_organ, [prep_row]
        else:
            return no_update, no_update, [no_update]
    
    def update_sub_compartment(self,select_ftu,prev,next,go_to_feat,ex_ftu_view,ftu_slider,thresh_slider,sub_method,go_to_feat_state,user_data_store):

        user_data_store = json.loads(user_data_store)

        new_ex_ftu = [go.Figure()]
        feature_extract_children = [[]]
        go_to_feat_disabled = [go_to_feat_state[0]]
        disable_slider = [go_to_feat_state[0]]
        disable_method = [go_to_feat_state[0]]
        select_ftu = select_ftu[0]
        prev = prev[0]
        next = next[0]
        go_to_feat = go_to_feat[0]
        ex_ftu_view = ex_ftu_view[0]
        thresh_slider = thresh_slider[0]
        sub_method = sub_method[0]
        ftu_slider = ftu_slider[0]


        try:
            ctx_triggered_id = ctx.triggered_id['type']
        except KeyError:
            ctx_triggered_id = ctx.triggered_id

        if not self.layer_ann is None:

            slider_marks = [{
                val:{'label':f'{sub_comp["name"]}: {val}','style':{'color':sub_comp["marks_color"]}}
                for val,sub_comp in zip(thresh_slider[::-1],self.sub_compartment_params)
            }]

            for idx,ftu,thresh in zip(list(range(len(self.sub_compartment_params))),self.sub_compartment_params,thresh_slider[::-1]):
                ftu['threshold'] = thresh
                self.sub_compartment_params[idx] = ftu

            if ctx_triggered_id=='next-butt':
                # Moving to next annotation in current layer
                self.layer_ann['previous_annotation'] = self.layer_ann['current_annotation']

                if self.layer_ann['current_annotation']+1>=self.layer_ann['max_layers'][self.layer_ann['current_layer']]:
                    self.layer_ann['current_annotation'] = 0
                else:
                    self.layer_ann['current_annotation'] += 1

            elif ctx_triggered_id=='prev-butt':
                # Moving back to previous annotation in current layer
                self.layer_ann['previous_annotation'] = self.layer_ann['current_annotation']

                if self.layer_ann['current_annotation']==0:
                    self.layer_ann['current_annotation'] = self.layer_ann['max_layers'][self.layer_ann['current_layer']]-1
                else:
                    self.layer_ann['current_annotation'] -= 1
            
            elif ctx_triggered_id=='ftu-select':
                # Moving to next annotation layer, restarting annotation count
                if type(select_ftu)==dict:
                    self.layer_ann['current_layer']=select_ftu['value']
                elif type(select_ftu)==int:
                    self.layer_ann['current_layer'] = select_ftu

                self.layer_ann['current_annotation'] = 0
                self.layer_ann['previous_annotation'] = self.layer_ann['max_layers'][self.layer_ann['current_layer']]

            if ctx_triggered_id not in ['go-to-feat','ex-ftu-slider','sub-comp-method']:
                
                new_image, new_mask = self.prep_handler.get_annotation_image_mask(user_data_store['upload_wsi_id'],self.upload_annotations,self.layer_ann['current_layer'],self.layer_ann['current_annotation'])
                self.layer_ann['current_image'] = new_image
                self.layer_ann['current_mask'] = new_mask

            if ctx_triggered_id not in ['go-to-feat']:
                
                sub_compartment_image = self.prep_handler.sub_segment_image(self.layer_ann['current_image'],self.layer_ann['current_mask'],self.sub_compartment_params,ex_ftu_view,ftu_slider)

                new_ex_ftu = [go.Figure(
                    data = px.imshow(sub_compartment_image)['data'],
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                )]
            else:
                go_to_feat_disabled = [True]
                disable_slider = [True]
                disable_method = [True]

                new_ex_ftu = [go.Figure(
                    data = px.imshow(self.prep_handler.current_sub_comp_image)['data'],
                    layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
                )]

                feature_extract_children = [self.prep_handler.gen_feat_extract_card(self.feature_extract_ftus)]

            if go_to_feat_state[0]:
                return new_ex_ftu, slider_marks, [no_update], disable_slider, disable_method, go_to_feat_disabled
            else:
                return new_ex_ftu, slider_marks, feature_extract_children, disable_slider, disable_method, go_to_feat_disabled
        else:
            slider_marks = [{
                val:{'label':f'{sub_comp["name"]}: {val}','style':{'color':sub_comp["marks_color"]}}
                for val,sub_comp in zip(thresh_slider[::-1],self.sub_compartment_params)
            }]
            
            go_to_feat_disabled = [True]
            disable_slider = [True]
            disable_method = [True]
            new_ex_ftu = [go.Figure()]
            feature_extract_children = [self.prep_handler.gen_feat_extract_card(self.feature_extract_ftus)]

            return new_ex_ftu, slider_marks, feature_extract_children, disable_slider, disable_method, go_to_feat_disabled

    def grab_nuc_region(self, thumb_fig_click, frame):

        # Get a new region of the full image, or update the frame thumbnail.
        print(thumb_fig_click)
        print(frame)
        print(ctx.triggered)

        print(ctx.outputs_list)

        raise exceptions.PreventUpdate

        pass

    def update_nuc_segmentation(self,nuc_method,nuc_thresh,view_type,go_to_feat):

        # Generating nucleus segmentation image from user inputs
        print(ctx.triggered)
        print(nuc_method)
        print(nuc_thresh)
        print(view_type)
        print(go_to_feat)

        print(ctx.outputs_list)

        raise exceptions.PreventUpdate

        pass

    def run_feature_extraction(self,feat_butt,skip_feat,include_ftus,include_features,user_data_store):

        user_data_store = json.loads(user_data_store)

        if ctx.triggered_id['type']=='start-feat':
            # Prepping input arguments to feature extraction
            include_ftus = get_pattern_matching_value(include_ftus)
            include_ftus = [self.feature_extract_ftus[i]['label'].split(' (')[0] for i in include_ftus]
            include_features = get_pattern_matching_value(include_features)
            include_features = ','.join(include_features)
            ignore_ftus = ','.join([i['label'].split(' (')[0] for i in self.feature_extract_ftus if not i['label'].split(' (')[0] in include_ftus])

            user_data_store['feat_ext_job'] = self.prep_handler.run_feature_extraction(user_data_store['upload_wsi_id'],self.sub_compartment_params,include_features,ignore_ftus)

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
        elif ctx.triggered_id['type']=='skip-feat':
            
            user_data_store['feat_ext_job'] = None
            # Returning a shorty dcc.Interval that should automatically trigger the next step
            feat_log_interval = [
                dcc.Interval(
                    id = {'type':'feat-interval','index':0},
                    interval = 500,
                    max_intervals = 1,
                    n_intervals = 0
                ),
                html.Div(
                    id = {'type': 'feat-log-output','index':0},
                    children = []
                )
            ]

        return [feat_log_interval], json.dumps(user_data_store)
    
    def update_feat_logs(self,new_interval,user_data_store):

        user_data_store = json.loads(user_data_store)

        # Updating logs for feature extraction job, disabling when the job is done
        feat_ext_status, feat_ext_log = self.dataset_handler.get_job_status(user_data_store['feat_ext_job']['_id'])

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

    """
    def ask_fusey(self,butt_click,current_style,current_tab):
        
        # Don't do anything if there isn't a WSI in view
        if self.wsi is None:
            raise exceptions.PreventUpdate

        # Generate some summary of the current view
        if current_tab == 'cell-compositions-tab':
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
    """

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

    def populate_cluster_tab(self,active_tab, cluster_butt,cluster_data_store,vis_sess_store,user_data_store):

        cluster_data_store = json.loads(cluster_data_store)
        vis_sess_store = json.loads(vis_sess_store)
        user_data_store = json.loads(user_data_store)

        used_get_cluster_data_plugin = None
        clustering_data = pd.DataFrame.from_records(cluster_data_store['clustering_data'])

        if active_tab == 'clustering-tab':
             # Checking the current slides to see if they match vis_sess_store:
            if not clustering_data.empty:
                current_slide_ids = [i['Slide_Id'] for i in clustering_data['Hidden'].tolist() if 'Hidden' in clustering_data.columns]
                unique_ids = np.unique(current_slide_ids).tolist()
                current_ids = [i['_id'] for i in vis_sess_store]
            else:
                current_ids = [i['_id'] for i in vis_sess_store]
        
            if ctx.triggered_id=='tools-tabs':

                # Checking current clustering data (either a full or empty pandas dataframe)
                if not clustering_data.empty:
                    print(f'in clustering_data: {unique_ids}')
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
                        self.dataset_handler.generate_feature_dict(vis_sess_store)

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

                        used_get_cluster_data_plugin = True

                        # Setting style of download plot data button
                        download_style = {'display':None}

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
                    cluster_data_store['feature_data'] = None
                    cluster_data_store['umap_df'] = None

                    feature_select_data = []
                    label_select_options = []
                    label_select_value = []
                    label_select_disabled = True
                    filter_select_data = []
            else:

                # Retrieving clustering data
                data_getting_response = self.dataset_handler.get_collection_annotation_meta([i['_id'] for i in vis_sess_store if i['included']])

                used_get_cluster_data_plugin = True
            
                self.dataset_handler.generate_feature_dict([i for i in vis_sess_store if i['included']])
                # Monitoring getting the data:
                data_get_status = 0
                while data_get_status<3:
                    data_status, data_log = self.dataset_handler.get_job_status(data_getting_response['_id'])
                    data_get_status=data_status
                    print(f'data_get_status: {data_get_status}')
                    time.sleep(1)

                cluster_data_store['clustering_data'] = self.dataset_handler.load_clustering_data(user_data_store).to_dict('records')

                get_data_div_children = dbc.Alert(
                    'Clustering data aligned!',
                    color = 'success',
                    dismissable=True,
                    is_open = True,
                    className='d-grid col-12 mx-auto'
                )

                # Setting style of download plot data button
                download_style = {'display':'inline-block'}
                cluster_data_store['feature_data'] = None
                cluster_data_store['umap_df'] = None

                feature_select_data = self.dataset_handler.plotting_feature_dict
                label_select_options = self.dataset_handler.label_dict
                label_select_value = self.dataset_handler.label_dict[0]['value']
                label_select_disabled = False
                filter_select_data = self.dataset_handler.label_filter_dict
    
        else:
            # If in another tab, just leave alone
            raise exceptions.PreventUpdate
        
        if used_get_cluster_data_plugin:
            return get_data_div_children, feature_select_data, label_select_disabled, label_select_options, label_select_value, filter_select_data, download_style, json.dumps({'plugin_used': 'get_cluster_data', 'slide_ids': current_ids }), json.dumps(cluster_data_store)

        return get_data_div_children, feature_select_data, label_select_disabled, label_select_options, label_select_value, filter_select_data, download_style, no_update, json.dumps(cluster_data_store)

    def update_plot_report(self,report_tab,cluster_data_store):

        cluster_data_store = json.loads(cluster_data_store)

        # Return the contents of the plot report tab according to selection
        report_tab_children = dbc.Alert('Generate a plot first!',color = 'warning')
        if not cluster_data_store['feature_data'] is None:
            if report_tab in self.reports_generated:
                # Report already generated, return the report
                report_tab_children = self.reports_generated[report_tab]
            else:
                # Report hasn't been generated yet, generate it
                report_tab_children = self.layout_handler.gen_report_child(cluster_data_store['feature_data'],report_tab)
                self.reports_generated[report_tab] = report_tab_children
            
        return report_tab_children

    def download_plot_data(self,download_button_clicked,cluster_data_store):

        cluster_data_store = json.loads(cluster_data_store)

        if not download_button_clicked:
            raise exceptions.PreventUpdate
        
        if not cluster_data_store['feature_data'] is None:
            feature_data = pd.DataFrame.from_records(cluster_data_store['feature_data'])
            feature_columns = [i for i in feature_data if i not in ['label','Hidden','Main_Cell_Types','Cell_States']]

            # If this is umap data then save one sheet with the raw data and another with the umap embeddings
            if len(feature_columns)<=2:

                download_data_df = {
                    'FUSION_Plot_Features': feature_data.copy()
                }
            elif len(feature_columns)>2:

                umap_df = pd.DataFrame.from_records(cluster_data_store['umap_df'])
                download_data_df = {
                    'FUSION_Plot_Features': feature_data.copy(),
                    'UMAP_Embeddings': umap_df[umap_df.columns.intersection(['UMAP1','UMAP2'])].copy()
                }

            with pd.ExcelWriter('Plot_Data.xlsx') as writer:
                for sheet in download_data_df:
                    download_data_df[sheet].to_excel(writer,sheet_name=sheet,engine='openpyxl')

            return dcc.send_file('Plot_Data.xlsx')
        else:
            raise exceptions.PreventUpdate

    def start_cluster_markers(self,butt_click,cluster_data_store,user_data_store):

        cluster_data_store = json.loads(cluster_data_store)
        user_data_store = json.loads(user_data_store)

        # Clicked Get Cluster Markers
        if ctx.triggered[0]['value']:

            disable_button = True
            feature_data = pd.DataFrame.from_records(cluster_data_store['feature_data'])
            
            # Saving current feature data to user public folder
            features_for_markering = self.dataset_handler.save_to_user_folder(
                save_object = {
                    'filename':f'marker_features.csv',
                    'content': feature_data
                },
                username = user_data_store['login']
            )

            # Starting cluster marker job
            cluster_marker_job = self.dataset_handler.gc.post(
                    '/slicer_cli_web/samborder2256_clustermarkers_fusion_latest/ClusterMarkers/run',
                    parameters = {
                        'feature_address': f'{self.dataset_handler.apiUrl}/item/{features_for_markering["itemId"]}/download?token={user_data_store["token"]}',
                        'girderApiUrl': self.dataset_handler.apiUrl,
                        'girderToken': user_data_store['token'],
                    }
                )
            user_data_store['cluster_marker_job_id'] = cluster_marker_job['_id']

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

            labels = np.unique(feature_data['label'].tolist()).tolist()
            slide_ids = np.unique([i['Slide_Id'] for i in feature_data['Hidden'].tolist()]).tolist()

            return [marker_logs_children],[disable_button], json.dumps({'plugin_used': 'clustermarkers_fusion', 'features': feature_data.columns.tolist(), 'label': labels, 'slide_ids': slide_ids }), json.dumps(user_data_store)
        else:
            raise exceptions.PreventUpdate

    def update_cluster_logs(self,new_interval,user_data_store):

        user_data_store = json.loads(user_data_store)

        if not new_interval is None:
            # Checking the cluster marker job status and printing the log to the cluster_log_div
            marker_status, marker_log = self.dataset_handler.get_job_status(user_data_store['cluster_marker_job_id'])

            if marker_status<3:
                
                marker_interval_disable = False
                cluster_log_children = [
                    html.P(i)
                    for i in marker_log.split('\n')
                ]

            else:

                marker_interval_disable = True

                # Load cluster markers from user folder
                cluster_marker_data = pd.DataFrame(self.dataset_handler.grab_from_user_folder('FUSION_Cluster_Markers.json',user_data_store['login'])).round(decimals=4)

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
                                'overflowX':'auto'
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

    def update_tutorial_slide(self,tutorial_tab):

        tab_key = {
            'background-tab':'Background',
            'start-tab':'Start',
            'histo-tab':'Histology',
            'omics-tab':'Spatial -Omics',
            'answer-tab':'Answer Hypothesis',
            'generate-tab':'Generate Hypothesis'
        }

        if tutorial_tab:
            # Getting tutorial content from FUSION Assets folder in DSA instance
            tutorial_slides = os.listdir(f'./static/{tab_key[tutorial_tab[0]]}/')
            tutorial_children = [
                dbc.Carousel(
                    id = 'tutorial-carousel',
                    items = [
                        {'key':f'{i+1}','src':f'./static/{tab_key[tutorial_tab[0]]}/slide_{i}.svg'}
                        for i in range(len(tutorial_slides))
                    ],
                    controls = True,
                    indicators = True
                )
            ]

            return [tutorial_children]
        else:
            raise exceptions.PreventUpdate

    def update_question_div(self,question_tab,logged_in_user):

        logged_in_username = logged_in_user.split(' ')[-1]
        # Updating the questions that the user sees in the usability questions tab
        usability_info = self.dataset_handler.update_usability()
        # Username is also present in the welcome div thing
        user_info = self.dataset_handler.check_usability(logged_in_username)
        user_type = user_info['type']

        # Getting questions for that type
        usability_questions = usability_info['usability_study_questions'][user_type]
        
        question_list = []
        # Narrowing down level by the index that the tab is on.
        if 'level' in question_tab[0]:
            level_index = int(question_tab[0].split('-')[1])
            level_questions = usability_questions[f'Level {level_index}']["questions"]

            for q_idx,l_q in enumerate(level_questions):

                # Checking if the user has already responded to this question
                if f'Level {level_index}' in list(user_info['responses'].keys()):
                    if len(user_info['responses'][f'Level {level_index}'])>q_idx:
                        q_val = user_info['responses'][f'Level {level_index}'][q_idx]
                    else:
                        q_val = []
                else:
                    q_val = []
                
                if not l_q['input_type']=='bool':
                    question_list.append(
                        html.Div([
                            dbc.Label(l_q['text'],size='lg'),
                            dbc.Input(
                                placeholder="Input response",
                                type=l_q['input_type'],
                                id={'type':'question-input','index':q_idx},
                                value = q_val
                            ),
                            html.Hr()
                        ])
                    )
                else:
                    question_list.append(
                        html.Div([
                            dbc.Label(l_q['text'],size='lg'),
                            dbc.RadioItems(
                                options = [
                                    {'label':dbc.Label('No',size='lg',style={'marginBottom':'15px','marginRight':'10px'}),'value':'No'},
                                    {'label':dbc.Label('Yes',size='lg',style={'marginBottom':'15px'}),'value':'Yes'}
                                ],
                                value = q_val,
                                id = {'type':'question-input','index':q_idx},
                                inline=True,
                                labelCheckedClassName="text-success",
                                inputCheckedClassName='border border-success bg-success'
                            )
                        ])
                    )
        
        elif question_tab[0]=='consent-tab':
            # Consent tab, holds disclaimers and official study information required by UF
            question_list.append(
                html.Div([
                    dcc.Markdown(
                        '''
                        #### Title of Project: _FUSION_ Usability Study
                        #### Principle Investigator: Pinaki Sarder, Ph.D.

                        Please read the information below carefully before you decide to participate in this research study.
                        **Your participation is voluntary. You can decide not to participate or later decide to stop participating at any time without penalty or lose any benefits that would normally be expected.**

                        1. **Purpose of the Study:** The purpose of this research study is to assess the usability of _FUSION_.
                        2. **What will you be asked to do:** You will be asked to answer a number of questions using tools within _FUSION_ assessing histology and spatial -omics data.
                        3. **Time Required:** It will take about 1 hour to participate in the research.
                        4. **Research Benefits:** Participants in this study will be listed as second-tier authors in the _FUSION_ manuscript submission.
                        5. **Research Risks:** There are no risks or discomforts anticipated.
                        6. **Statement of Confidentiality:** Your participation in this research is confidential.
                        Information collected about you will be stored in computers with security passwords or 
                        in locked filing cabinets. Only certain people have the legal right to review these 
                        research records, and they will protect the secrecy (confidentiality) of these records 
                        as much as the law allows. These people include the researchers for this study, certain 
                        University of Florida officials, and the Institutional Review Board (IRB; an IRB is a 
                        group of people who are responsible for looking after the rights and welfare of people 
                        taking part in research). Otherwise your research records will not be released without 
                        your permission unless required by law or a court order. The researchers will not share 
                        your name or other identifiable information about you if they publish, present, or share 
                        the results of this research. 
                        7. **Who to contact if you have questions or injured:** Please contact Annika Holmstrom at
                        annika.holmstrom@medicine.ufl.edu with questions or concerns about this study.
                        8. **Voluntary Participation:** Your decision to be in this research is voluntary. You do 
                        not have to do any study activities that you do not want to take part in. You can stop at 
                        any time. If you decide you want to stop participating in the research, you can let the 
                        research team know or call the Principle Investigator any time at (352) 273-6018. If 
                        you choose not to take part, this will have no effect on you or your relationships with 
                        the University of Florida. If you have any questions about your rights as a research subject, 
                        you can phone the institutional Review Board at (352) 273-9600.

                        Participation in the research implies that you have read the information in this form and 
                        consent to take part in the research. Please save a copy of this form for your records or 
                        for future reference.
                        
                        If you want to participate in this research study, click the "I agree to participate" button below.
                        If you do not want to participate, you may simply close this window.
                        ''',
                        style = {'font-size':'0.8rem'}
                    ),
                    dbc.Button(
                        'I agree to participate',
                        className = 'd-grid col-12 mx-auto',
                        id = {'type':'study-consent-butt','index':0},
                        style = {'marginTop':'10px','marginBottom':'10px'}
                    )
                ],style={'maxHeight':'55vh','overflow':'scroll'})
            )

        else:
            # Comments tab
            level_index = len(list(usability_questions.keys()))

            if 'Comments' in list(user_info['responses'].keys()):

                question_list.append(
                    html.Div([
                        dbc.Row(dbc.Label('Add any comments here!',size='lg')),
                        dbc.Row(
                            dcc.Textarea(
                                id = {'type':'question-input','index':0},
                                placeholder = 'Comments',
                                style = {'width':'100%','marginBottom':'10px'},
                                maxLength = 10000,
                                value = user_info['responses']['Comments']
                            )
                        )
                    ])
                )
            else:
                question_list.append(
                    html.Div([
                        dbc.Row(dbc.Label('Add any comments here!',size='lg')),
                        dbc.Row(
                            dcc.Textarea(
                                id = {'type':'question-input','index':0},
                                placeholder = 'Comments',
                                style = {'width':'100%','marginBottom':'10px'},
                                maxLength = 10000
                            )
                        )
                    ])
                )

        if question_tab[0] != 'consent-tab':
            question_list.append(html.Div([
                dbc.Button(
                    'Save Responses',
                    className = 'd-grid col-12 mx-auto',
                    id = {'type':'questions-submit','index':level_index},
                    style = {'marginBottom':'15px'}
                ),
                dbc.Button(
                    'Submit Recording',
                    className = 'd-grid col-12 mx-auto',
                    id = {'type':'recording-upload','index':0},
                    target='_blank',
                    href = 'https://trailblazer.app.box.com/f/f843d7b1da204b538dd3173c81ce66cf',
                    disabled = False
                ),
                html.Div(id = {'type':'questions-submit-alert','index':0})
                ])
            )

        question_return = dbc.Form(question_list,style = {'maxHeight':'55vh','overflow':'scroll'})

        return [question_return]

    def consent_to_usability_study(self,butt_click):
        
        # Enabling all the question tabs after user clicks the I agree to participate button
        if not ctx.triggered[0]['value']:
            raise exceptions.PreventUpdate
        
        n_outputs = len(ctx.outputs_list[0])

        return [False]*n_outputs,['success']

    def post_usability_response(self,butt_click,questions_inputs):

        # Updating usability info file in DSA after user clicks "Submit" button
        if butt_click:
            # Checking if all of the responses are not empty
            responses_check = [True if not i==[] and not i is None else False for i in questions_inputs]
            if all(responses_check):
                submit_alert = dbc.Alert('Submitted!',color='success')

                # Getting the most recent usability info to update
                usability_info = self.dataset_handler.update_usability()

                # Updating responses for the current user
                level_idx = ctx.triggered_id['index']
                if level_idx<=4:
                    level_name = f'Level {level_idx}'
                else:
                    level_name = 'Comments'
                usability_info['usability_study_users'][self.dataset_handler.username]['responses'][f'{level_name}'] = questions_inputs

                # Posting to DSA
                self.dataset_handler.update_usability(usability_info)
            else:
                submit_alert = dbc.Alert(f'Uh oh! {len([i for i in responses_check if not i])} responses are missing',color='warning')
                
            return [submit_alert]
        else:
            raise exceptions.PreventUpdate

    def download_usability_response(self,butt_click):

        if not butt_click:
            raise exceptions.PreventUpdate

        # Getting most recent usability data
        usability_info = self.dataset_handler.usability_users

        all_users = usability_info['usability_study_users']
        user_data = []
        for u in all_users:
            user_data.append({
                'Username':u,
                'User Type':all_users[u]['type'],
                'Responded?':'Yes' if len(list(all_users[u]['responses'].keys()))>0 else 'No',
                'Task Responses':all_users[u]['responses'] if len(list(all_users[u]['responses'].keys()))>0 else 'No Responses'
            })

        user_df = pd.DataFrame.from_records(user_data)

        # Breaking up users into separate user types
        user_types = np.unique(user_df['User Type'].tolist()).tolist()
        user_type_dict = {}
        for u_t in user_types:
            user_type_data = user_df[user_df['User Type'].str.match(u_t)]

            final_user_type_list = []
            users = np.unique(user_type_data['Username'].tolist()).tolist()
            for u in users:
                u_list = []
                u_responses = user_type_data[user_type_data['Username'].str.match(u)]['Task Responses'].tolist()[0]
                if type(u_responses)==dict:
                    for lvl in list(u_responses.keys()):
                        for q_idx,q in enumerate(u_responses[lvl]):
                            u_list.append(
                                {
                                    'Username':u,
                                    'Level': lvl,
                                    'Question':f'Question {q_idx+1}',
                                    'Response': q
                                }
                            )
                    final_user_type_list.extend(u_list)
                else:
                    final_user_type_list.append(
                        {
                            'Username':u,
                            'Level':'No Responses',
                            'Question':'No Responses',
                            'Response':'No Responses'
                        }
                    )

            if len(final_user_type_list)>0:
                user_type_lvl_df = pd.DataFrame.from_records(final_user_type_list)
                user_type_dict[u_t] = user_type_lvl_df

        # Creating excel file writer with different sheets for each user type
        if len(list(user_type_dict.keys()))>0:
            with pd.ExcelWriter('Usability_Response_Data.xlsx') as writer:
                for sheet in user_type_dict:
                    user_type_dict[sheet].to_excel(writer,sheet_name = sheet, engine = 'openpyxl')

            return [dcc.send_file('Usability_Response_Data.xlsx')]
        else:
            raise exceptions.PreventUpdate

    def add_channel_color_select(self,channel_opts):

        # Creating a new color selector thing for overlaid channels?
        #TODO: get_pattern_matching_value test here
        if not channel_opts is None:
            if type(channel_opts)==list:
                if len(channel_opts[0])>0:
                    channel_opts = channel_opts[0]
                    active_tab = channel_opts[0]
                    disable_butt = False
                else:
                    active_tab = None
                    disable_butt = True
                    channel_opts = channel_opts[0]
                    self.current_channels = {}
            
            # Removing any channels which aren't included from self.current_channels
            intermediate_dict = self.current_channels.copy()
            current_channels = list(self.current_channels.keys())

            self.current_channels = {}
            channel_tab_list = []
            for c_idx,channel in enumerate(channel_opts):
                
                if channel in intermediate_dict:
                    channel_color = intermediate_dict[channel]['color']
                else:
                    channel_color = 'rgba(255,255,255,255)'

                self.current_channels[channel] = {
                    'index': self.wsi.channel_names.index(channel),
                    'color': channel_color
                }

                channel_tab_list.append(
                    dbc.Tab(
                        id = {'type':'overlay-channel-tab','index':c_idx},
                        tab_id = channel,
                        label = channel,
                        activeTabClassName='fw-bold fst-italic',
                        children = [
                            dmc.ColorPicker(
                                id =  {'type':'overlay-channel-color','index':c_idx},
                                format = 'rgba',
                                value = channel_color,
                                fullWidth=True,
                            ),
                        ]
                    )
                )

            channel_tabs = dbc.Tabs(
                id = {'type':'channel-color-tabs','index':0},
                children = channel_tab_list,
                active_tab = active_tab
            )

            return [channel_tabs],[disable_butt]
        else:
            self.current_channels = {}

            raise exceptions.PreventUpdate
    
    def add_channel_overlay(self,butt_click,channel_colors,channels):

        # Adding an overlay for channel with a TileLayer containing stylized grayscale info
        if not ctx.triggered[0]['value']:
            raise exceptions.PreventUpdate
        
        # self.current_channels contains a list of all the currently overlaid channels
        # Updating the color for this channel
        for ch, co in zip(channels,channel_colors):
            self.current_channels[ch]['color'] = co

        updated_channel_idxes = [self.current_channels[i]['index'] for i in self.current_channels]
        updated_channel_colors = [self.current_channels[i]['color'] for i in self.current_channels]

        update_style = {
            c_idx: color
            for c_idx,color in zip(updated_channel_idxes,updated_channel_colors)
        }

        updated_urls = []
        for c_idx, old_style in enumerate(self.wsi.channel_tile_url):
            if c_idx not in updated_channel_idxes:
                base_style = {
                    c_idx: "rgba(255,255,255,255)"
                }

                new_url = self.wsi.update_url_style(base_style | update_style)
            else:

                new_url = self.wsi.update_url_style(update_style)

            updated_urls.append(new_url)

        # Adding label style to tabs, just adding the overlay color to the text for that tab:
        tab_label_style = [
            {'color': co}
            for co in channel_colors
        ]

        return updated_urls, tab_label_style

    def get_asct_table(self,butt_click,hgnc_id):
        """
        Get AS and CT associated with a given biomarker (hgnc id)
        """

        if not butt_click:
            raise exceptions.PreventUpdate
        
        hgnc_id = get_pattern_matching_value(hgnc_id)
        # Getting only the number part of the id
        hgnc_id = hgnc_id.split(' ')[-1]

        # Getting ASCT part of table from gene_handler
        asct_table = self.gene_handler.get_asct(hgnc_id)
        if not asct_table.empty:
            asct_table = asct_table.loc[:,asct_table.columns.str.contains('_label.value')]

            # Formatting to dash_table
            return_table = dash_table.DataTable(
                id = {'type':'asct-dash-table','index':0},
                columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in asct_table],
                data = asct_table.to_dict('records'),
                filter_action='native',
                page_size = 5,
                style_cell = {
                    'overflow':'hidden',
                    'textOverflow':'ellipsis',
                    'maxWidth':0
                },
                tooltip_data = [
                    {
                        column: {'value':str(value),'type':'markdown'}
                        for column, value in row.items()
                    } for row in asct_table.to_dict('records')
                ],
                tooltip_duration = None
            )
        else:
            return_table = "Not in HRA!"

        return [return_table]

    def populate_structure_hierarchy(self, active_tab):
        """
        Showing which structures are contained within which structure in a tree/dropdown structure for set operations
        """
        if not active_tab:
            raise exceptions.PreventUpdate
        
        print(f'active_tab: {active_tab}')
        tree_data = []
        button_disable = True

        # Getting structure hierarchy for current structure
        if not active_tab in self.wsi.ftu_hierarchy:
            tree_data = self.wsi.gen_structure_hierarchy(active_tab)

        else:
            tree_data = self.wsi.ftu_hierarchy[active_tab]

        if len(tree_data['children'])==0:
            button_disable = False

        if active_tab in self.wsi.ftu_names:
            use_idx = self.wsi.ftu_names.index(active_tab)
        elif active_tab=='Spots':
            use_idx = len(self.wsi.ftu_names)
        
        tree_select = dta.TreeView(
            id = {'type':'structure-tree','index': use_idx},
            multiple = True,
            checkable = True,
            selected = [],
            checked = [],
            data = tree_data
        )
        
        return_list = [tree_select]*len(ctx.outputs_list[0])

        return return_list, [button_disable]*len(ctx.outputs_list[1])

    def update_structure_hierarchy(self,butt_click):
        """
        Updating structure boundaries according to selected hierarchy        
        """
        pass
    
    def add_filter(self,butt_click, delete_click):
        """
        Adding a new filter to apply to current FTUs
        """
        if not self.wsi is None and not self.overlay_prop is None and not self.filter_vals is None:
            if ctx.triggered_id=='add-filter-button':
                # Returning a new dropdown for selecting properties to use for a filter

                patched_list = Patch()
                # Find min and max vals
                filter_val = [i for i in self.wsi.properties_list if not 'Cell_Subtypes' in i and not i=='Max Cell Type'][0]
                if '-->' in filter_val:
                    filter_val_parts = filter_val.split(' --> ')
                    m_prop = filter_val_parts[0]
                    val = filter_val_parts[1]
                    # For full cell names
                    if val in self.cell_names_key:
                        val = self.cell_names_key[val]
                else:
                    m_prop = filter_val
                    val = None
                
                unique_values = self.wsi.get_overlay_value_list({'name':m_prop,'value':val,'sub_value': None})
                value_types = [type(i) for i in unique_values][0]
                if value_types in [int,float]:
                    value_disable = False
                    value_display = {'display':'inline-block','margin':'auto','width':'100%'}
                else:
                    value_disable = True
                    value_display = {'display':'none','margin':'auto','width':'100%'}

                value_slider = html.Div(
                        dcc.RangeSlider(
                            id = {'type':'added-filter-slider','index':butt_click},
                            min = np.min(unique_values),
                            max = np.max(unique_values),
                            step = 0.01,
                            marks = None,
                            tooltip = {'placement':'bottom','always_visible':True},
                            allowCross = False,
                            disabled = value_disable
                        ),
                        style = value_display,
                        id = {'type':'added-filter-slider-div','index':butt_click}  
                    )
                
                def new_filter_item():
                    return html.Div([
                        dbc.Row([
                            dbc.Col(
                                dcc.Dropdown(
                                    options = [i for i in self.wsi.properties_list if not i=='Max Cell Type' and not 'Cell_Subtypes' in i],
                                    value = [i for i in self.wsi.properties_list if not i=='Max Cell Type' and not 'Cell_Subtypes' in i][0],
                                    placeholder = 'Select new property to filter FTUs',
                                    id = {'type':'added-filter-drop','index':butt_click}
                                ),
                                md = 10
                            ),
                            dbc.Col(
                                html.I(
                                    id = {'type':'delete-filter','index':butt_click},
                                    n_clicks= 0,
                                    className = 'bi bi-x-circle-fill fa-2x',
                                    style = {'color':'rgb(255,0,0)'}
                                ),
                                md = 2
                            )
                        ],align='center'),
                        value_slider,
                    ])

                patched_list.append(new_filter_item())

            elif ctx.triggered_id['type']=='delete-filter':

                patched_list = Patch()
                values_to_remove = []
                for i,val in enumerate(delete_click):
                    if val:
                        values_to_remove.insert(0,i)
                
                for v in values_to_remove:
                    del patched_list[v]
                
            return patched_list
        else:
            raise exceptions.PreventUpdate
        
    def add_filter_slider(self, filter_drop):

        # Find min and max vals
        filter_val = filter_drop
        if '-->' in filter_val:
            filter_val_parts = filter_val.split(' --> ')
            m_prop = filter_val_parts[0]
            val = filter_val_parts[1]
            # For full cell names
            if val in self.cell_names_key:
                val = self.cell_names_key[val]
        else:
            m_prop = filter_val
            val = None

        unique_values = self.wsi.get_overlay_value_list({'name':m_prop,'value':val,'sub_value': None})
        value_types = [type(i) for i in unique_values][0]
        if value_types in [int,float]:
            slider_style = {'display':'inline-block','margin':'auto','width':'100%'}
            return np.min(unique_values), np.max(unique_values), slider_style

        else:
            raise exceptions.PreventUpdate

    def cell_labeling_initialize(self, selectedData, checked_cells, current_selectedData):
        """
        Takes as input some selected cells, a frame to plot their distribution of intensities, and then a Set button that labels those cells
        """
        selectedData = get_pattern_matching_value(selectedData)
        return_div = html.Div()
        cell_markers = no_update

        if ctx.triggered_id is None:
            raise exceptions.PreventUpdate
        
        if ctx.triggered_id['type'] in ['ftu-cell-pie', 'cell-labeling-channel-drop']:

            if not selectedData is None:
                print('Pulling new cells from clusters')
                # Pulling customdata from each point object
                labeling_cells = [i['customdata'] for i in selectedData['points']]

                if not ctx.triggered_id['type']=='cell-labeling-channel-drop':
                    channel_val = []

                updated_cell_geojson, cell_markers = make_marker_geojson([i[0]['Bbox'] if type(i)==list else i['Bbox'] for i in labeling_cells],convert_coords = True, wsi = self.wsi)

                return_div = html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Row(html.H6('Selected Cells')),
                            dbc.Card(
                                dbc.CardBody(
                                    children = [
                                        dcc.Checklist(
                                            options = [
                                                {'label':f'Cell {idx}','value':'_'.join([str(j) for j in i[0]['Bbox']]) if type(i)==list else '_'.join([str(j) for j in i['Bbox']])}
                                                for idx,i in enumerate(labeling_cells)
                                            ],
                                            value = ['_'.join([str(j) for j in i[0]['Bbox']]) if type(i)==list else '_'.join([str(j) for j in i['Bbox']]) for i in labeling_cells]
                                        )
                                        ],
                                    style = {'maxHeight':'60vh','overflow':'scroll'}
                                    )
                                )
                            ],
                            md = 4
                        ),
                        dbc.Col([
                            dbc.Row(html.H6('Selected Cell Labeling')),
                            dbc.Row([
                                dbc.Col(
                                    'Label for Checked Cells: ',
                                    md = 3
                                ),
                                dbc.Col(
                                    dcc.Input(
                                        id = {'type':'checked-cell-label','index':0},
                                        placeholder = 'Type cell label here'
                                    ),
                                    md = 6
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        'Submit',
                                        className = 'd-grid col-12 mx-auto',
                                        id = {'type':'checked-cell-label-butt','index':0}
                                    ),
                                    md = 3
                                )
                            ]),
                            dbc.Row([
                                html.Div(
                                    id = 'cell-labeling-rationale',
                                    children = [
                                        'This is where the cell labeling rationale goes'
                                    ]
                                )
                            ])
                        ],md = 8)
                    ])
                ])
            else:
                raise exceptions.PreventUpdate

        return [return_div], [f'{len(selectedData["points"])} Cells selected'], cell_markers

    def update_annotation_session(self, session_tab, new_session, new_session_name, new_session_description, new_session_users, current_session_names, annotation_classes, annotation_colors, annotation_labels, annotation_users, annotation_user_types):
        """
        Populating annotation station tab with necessary information
        """
        return_div = [no_update]
        new_annotation_tab_group = [no_update]

        session_tab = get_pattern_matching_value(session_tab)

        if ctx.triggered_id['type']=='create-annotation-session-button':
            new_session_name = get_pattern_matching_value(new_session_name)
            new_session_description = get_pattern_matching_value(new_session_description)
            new_session_users = get_pattern_matching_value(new_session_users)
            
            # Creating folder in user's public folder called FUSION Annotation Sessions, with sub-folder named the new_session_name
            ann_sessions_folder = self.dataset_handler.check_user_folder("FUSION Annotation Sessions")
            if ann_sessions_folder is None:
                # Creating parent folder if it doesn't exist
                parent_folder = self.dataset_handler.create_user_folder(
                    parent_path = f'/user/{self.dataset_handler.username}/Public',
                    folder_name = 'FUSION Annotation Sessions',
                    metadata = {"Folder Description":"Data related to annotation sessions completed in FUSION"}
                )

            new_session_folder = self.dataset_handler.create_user_folder(
                parent_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions',
                folder_name = new_session_name,
                metadata = {
                    "Session Description":new_session_description,
                    "Annotations": [
                        {'label': i, 'value': j}
                        for i,j in zip(annotation_classes, annotation_colors)
                    ],
                    "Labels": annotation_labels,
                    "Users": [
                        {i: {'type': j}}
                        for i,j in zip(annotation_users,annotation_user_types)
                    ]
                }
            )

            #TODO: Find some way to share this with indicated users

            updated_tabs,first_tab, first_session = self.layout_handler.gen_annotation_card(self.dataset_handler, self.current_ftus)
            session_progress, self.current_ann_session = self.dataset_handler.get_annotation_session_progress(first_session['name'])

            new_annotation_tab_group = [html.Div([
                dbc.Tabs(
                    id = {'type':'annotation-tab-group','index':0},
                    active_tab = 'ann-sess-0',
                    children = updated_tabs
                )
            ])]

            return_div = [first_tab]

        if ctx.triggered_id['type']=='annotation-tab-group':
            session_name = current_session_names[int(session_tab.split('-')[-1])]

            if session_name=='Create New Session':
                self.current_ann_session = {
                    'name':'Create New Session',
                    'Annotations':[],
                    'Labels': []
                }

                first_tab = self.layout_handler.gen_annotation_content(new = True, current_ftus = self.current_ftus)
            else:
                session_progress, self.current_ann_session = self.dataset_handler.get_annotation_session_progress(session_name)
                first_tab = self.layout_handler.gen_annotation_content(new = False, current_ftus = self.current_ftus, classes = self.current_ann_session['Annotations'], labels = self.current_ann_session['Labels'], ann_progress = session_progress, user_type = self.current_ann_session['user_type'])

            return_div = [first_tab]

        
        return return_div, new_annotation_tab_group
    
    def update_current_annotation(self, ftus, prev_click, next_click, save_click, set_click, delete_click,  line_slide, ann_class, all_annotations, class_label, image_label, ftu_names, ftu_idx):
        """
        Getting current annotation data (text or image) and saving to annotation session folder on the server
        """
        current_structure_fig = [no_update]
        ftu_styles = [no_update]*len(ctx.outputs_list[1])
        class_labels = [no_update]*len(ctx.outputs_list[2])
        image_labels = [no_update]*len(ctx.outputs_list[3])
        save_button_style = ['primary']
        session_ftu_progress = [no_update]*len(ctx.outputs_list[6])
        session_ftu_progress_label = [no_update]*len(ctx.outputs_list[7])

        line_slide = get_pattern_matching_value(line_slide)
        if line_slide is None:
            line_slide = 5
        ann_class = get_pattern_matching_value(ann_class)
        ann_class_color = ann_class

        all_annotations = get_pattern_matching_value(all_annotations)
        if not all_annotations is None:
            annotations = []
            if 'shapes' in all_annotations.keys():
                annotations += all_annotations['shapes']
                if 'line' in all_annotations.keys():
                    annotations += all_annotations['line']
            else:
                annotations = []
        else:
            annotations = []

        ftu_idx = get_pattern_matching_value(ftu_idx)
        if type(ftu_idx)==list:
            ftu_idx=ftu_idx[0]

        selected_style = {
            'background':'rgba(255,255,255,0.8)',
            'box-shadow':'0 0 10px rgba(0,0,0,0.2)',
            'border-radius':'5px',
            'marginBottom':'15px'
        }

        if ctx.triggered_id['type']=='annotation-station-ftu':
            ftu_styles = [{'display':'inline-block','marginBottom':'15px'} if not i==ctx.triggered_id['index'] else selected_style for i in range(len(ctx.outputs_list[1]))]

            clicked_ftu_name = ftu_names[ctx.triggered_id['index']][0]
            if 'Manual' in clicked_ftu_name:
                clicked_ftu_name = 'Manual:'+clicked_ftu_name.split(':')[1].strip()
            elif 'Marked' in clicked_ftu_name:
                clicked_ftu_name = 'Marked:'+clicked_ftu_name.split(':')[1].strip()
            else:
                clicked_ftu_name = clicked_ftu_name.split(':')[0]
            ftu_idx = 0

            # Clearing image labels
            image_labels = ['']*len(ctx.outputs_list[3])

        else:
            clicked_ftu_name = ftu_idx.split(':')[0]
            ftu_idx = int(ftu_idx.split(':')[-1])

        if ctx.triggered_id['type'] in ['annotation-station-ftu','annotation-previous-button','annotation-next-button','annotation-line-slider','annotation-class-select']:
            
            # Grab the first member of the clicked ftu
            if not 'Manual' in clicked_ftu_name and not 'Marked' in clicked_ftu_name:
                if clicked_ftu_name=='FTU':
                    clicked_ftu_name = ftu_names[0][0].split(':')[0]
                intersecting_ftu_props, intersecting_ftu_polys = self.wsi.find_intersecting_ftu(self.current_slide_bounds,clicked_ftu_name)
            elif 'Manual' in clicked_ftu_name:
                manual_idx = int(clicked_ftu_name.split(':')[-1])-1
                intersecting_ftu_polys = [shape(self.wsi.manual_rois[manual_idx]['geojson']['features'][0]['geometry'])]
            elif 'Marked' in clicked_ftu_name:
                intersecting_ftu_polys = [shape(i['geojson']['features'][0]['geometry']) for i in self.wsi.marked_ftus]

            # Getting bounding box of this ftu
            if ctx.triggered_id['type']=='annotation-station-ftu':
                ftu_idx = 0
            else:
                if ctx.triggered_id['type']=='annotation-previous-button':
                    if ftu_idx-1<0:
                        ftu_idx = len(intersecting_ftu_polys)-1
                    else:
                        ftu_idx -= 1
                elif ctx.triggered_id['type']=='annotation-next-button':
                    if ftu_idx+1>=len(intersecting_ftu_polys):
                        ftu_idx = 0
                    else:
                        ftu_idx +=1 

            if ctx.triggered_id['type'] in ['annotation-previous-button','annotation-next-button']:
                annotations = []

            ftu_bbox_coords = list(intersecting_ftu_polys[ftu_idx].exterior.coords)
            
            ftu_bbox = np.array(self.wsi.convert_map_coords(ftu_bbox_coords))
            ftu_bbox = [np.min(ftu_bbox[:,0])-50,np.min(ftu_bbox[:,1])-50,np.max(ftu_bbox[:,0])+50,np.max(ftu_bbox[:,1])+50]
            #TODO: This method only grabs histology images that are single frame or multi-frame with RGB at 0,1,2
            ftu_image = self.dataset_handler.get_image_region(self.wsi.item_id,[int(i) for i in ftu_bbox])

            if not ann_class_color is None:
                color_parts = ann_class_color.replace('rgb(','').replace(')','').split(',')
                fill_color = f'rgba({color_parts[0]},{color_parts[1]},{color_parts[2]},0.2)'
            else:
                fill_color = 'rgba(255,255,255,0.2)'

            current_structure_fig = go.Figure(px.imshow(np.array(ftu_image)))
            current_structure_fig.update_layout(
                {
                    'margin': {'l':0,'r':0,'t':0,'b':0},
                    'xaxis':{'showticklabels':False,'showgrid':False},
                    'yaxis':{'showticklabels':False,'showgrid':False},
                    'dragmode':'drawclosedpath',
                    "shapes":annotations,
                    "newshape.line.width": line_slide,
                    "newshape.line.color": ann_class_color,
                    "newshape.fillcolor": fill_color
                }
            )
            current_structure_fig = [current_structure_fig]

            # Updating image labels
            image_labels = ['']*len(ctx.outputs_list[3])

        elif ctx.triggered_id['type'] in ['annotation-set-label','annotation-delete-label']:
            
            #TODO: Add more labels after clicking the check-mark
            class_label = get_pattern_matching_value(class_label)
            image_label = get_pattern_matching_value(image_label)

            if type(class_label)==str:
                label_dict = {class_label: image_label}
            elif type(class_label)==list:
                label_dict = {i:j for i,j in zip(class_label,image_label)}
            
            label_dict['Annotator'] = self.dataset_handler.username
            
            if ctx.triggered_id['type']=='annotation-set-label':
                # Grab the first member of the clicked ftu
                if not 'Manual' in clicked_ftu_name or 'Marked' in clicked_ftu_name:
                    intersecting_ftu_props, intersecting_ftu_polys = self.wsi.find_intersecting_ftu(self.current_slide_bounds,clicked_ftu_name)
                elif 'Manual' in clicked_ftu_name:
                    manual_idx = int(clicked_ftu_name.split(':')[-1])
                    intersecting_ftu_polys = [shape(self.wsi.manual_rois[manual_idx]['geojson'])]
                elif 'Marked' in clicked_ftu_name:
                    intersecting_ftu_polys = [shape(i['geojson']) for i in self.wsi.marked_ftus]

                ftu_coords = list(intersecting_ftu_polys[ftu_idx].exterior.coords)
                ftu_coords = np.array(self.wsi.convert_map_coords(ftu_coords))
                ftu_bbox = [np.min(ftu_coords[:,0])-50,np.min(ftu_coords[:,1])-50,np.max(ftu_coords[:,0])+50,np.max(ftu_coords[:,1])+50]

                # Now saving image with label in metadata
                ftu_image = np.array(self.dataset_handler.get_image_region(self.wsi.item_id,[int(i) for i in ftu_bbox]))

                ftu_content = {
                    'content': ftu_image,
                    'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.png',
                    'metadata': label_dict
                }

                # Checking if slide folder is created
                slide_folder = self.dataset_handler.check_user_folder(
                    folder_name = 'FUSION Annotation Sessions',
                    subfolder = self.current_ann_session['name'],
                    sub_sub_folder = self.wsi.slide_name
                )
                if slide_folder is None:
                    # Creating slide folder in current_ann_session
                    new_folder = self.dataset_handler.create_user_folder(
                        parent_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}',
                        folder_name = self.wsi.slide_name
                    )
                    new_folder = self.dataset_handler.create_user_folder(
                        parent_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}/{self.wsi.slide_name}',
                        folder_name = 'Images'
                    )

                # Saving data
                self.dataset_handler.save_to_user_folder(ftu_content,output_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}/{self.wsi.slide_name}/Images')
                
                # Updating image labels
                image_labels = ['']*len(ctx.outputs_list[3])

            else:
                #TODO: delete label from image metadata if that metadata is already attached to an image in the annotation session

                image_labels = [no_update if not i==ctx.triggered_id['index'] else '' for i in range(len(ctx.outputs_list[3]))]

        elif ctx.triggered_id['type'] == 'annotation-save-button':

            # Grab the first member of the clicked ftu
            if not 'Manual' in clicked_ftu_name and not 'Marked' in clicked_ftu_name:
                intersecting_ftu_props, intersecting_ftu_polys = self.wsi.find_intersecting_ftu(self.current_slide_bounds,clicked_ftu_name)
            elif 'Manual' in clicked_ftu_name:
                manual_idx = int(clicked_ftu_name.split(':')[-1])
                intersecting_ftu_polys = [shape(self.wsi.manual_rois[manual_idx]['geojson'])]
            elif 'Marked' in clicked_ftu_name:
                intersecting_ftu_polys = [shape(i['geojson']) for i in self.wsi.marked_ftus]

            ftu_coords = list(intersecting_ftu_polys[ftu_idx].exterior.coords)
            ftu_coords = np.array(self.wsi.convert_map_coords(ftu_coords))
            ftu_bbox = [np.min(ftu_coords[:,0])-50,np.min(ftu_coords[:,1])-50,np.max(ftu_coords[:,0])+50,np.max(ftu_coords[:,1])+50]
            height = int(ftu_bbox[3]-ftu_bbox[1])
            width = int(ftu_bbox[2]-ftu_bbox[0])
            
            # Convert annotations relayoutData to a mask and save both image and mask to current annotation session
            #TODO: combined mask here should have a channel dimension equal to the number of classes (e.g. if two classes overlap)
            combined_mask = np.zeros((height,width,len(self.current_ann_session["Annotations"])))
            annotation_colors_list = [i['value'] for i in self.current_ann_session['Annotations']]
            for key in annotations:
                for i in range(len(annotations)):
                    #TODO: Add color property to this function for multiple classes
                    #TODO: Add lines to this mask
                    if 'path' in annotations[i]:
                        mask = path_to_mask(annotations[i]['path'],(combined_mask.shape[0],combined_mask.shape[1]))
                        mask_class = annotations[i]['line']['color']

                        combined_mask[:,:,annotation_colors_list.index(mask_class)] += 255*mask
            
            # Now saving both image and mask to the annotation session folder
            ftu_image = np.array(self.dataset_handler.get_image_region(self.wsi.item_id,[int(i) for i in ftu_bbox]))
            
            mask_image = np.uint8(combined_mask)

            ftu_content = {
                'content': ftu_image,
                'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.png',
                'metadata': {
                    "Annotator": self.dataset_handler.username
                }
            }

            if mask_image.shape[-1] in [1,3]:
                mask_content = {
                    'content': mask_image,
                    'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.png',
                    'metadata': {
                        "Annotator": self.dataset_handler.username
                    }
                }
            else:
                mask_content = {
                    'content': mask_image,
                    'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.tiff',
                    'metadata': {
                        "Annotator": self.dataset_handler.username
                    }
                }

            # Checking if slide folder is created
            slide_folder = self.dataset_handler.check_user_folder(
                folder_name = 'FUSION Annotation Sessions',
                subfolder = self.current_ann_session['name'],
                sub_sub_folder = self.wsi.slide_name
            )
            if slide_folder is None:
                # Creating slide folder in current_ann_session
                new_folder = self.dataset_handler.create_user_folder(
                    parent_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}',
                    folder_name = self.wsi.slide_name
                )
                new_folder = self.dataset_handler.create_user_folder(
                    parent_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}/{self.wsi.slide_name}',
                    folder_name = 'Images'
                )
                new_folder = self.dataset_handler.create_user_folder(
                    parent_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}/{self.wsi.slide_name}',
                    folder_name = 'Masks'
                )

            # Saving data
            self.dataset_handler.save_to_user_folder(ftu_content,output_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}/{self.wsi.slide_name}/Images')
            self.dataset_handler.save_to_user_folder(mask_content,output_path = f'/user/{self.dataset_handler.username}/Public/FUSION Annotation Sessions/{self.current_ann_session["name"]}/{self.wsi.slide_name}/Masks')

            save_button_style = ['success']

        session_ftu_progress = [int(100*(ftu_idx+1)/len(intersecting_ftu_polys))]
        session_ftu_progress_label = [f'{clicked_ftu_name}: {session_ftu_progress[0]}%']
        
        return current_structure_fig, ftu_styles, class_labels, image_labels, [f'{clicked_ftu_name}:{ftu_idx}'], save_button_style, session_ftu_progress, session_ftu_progress_label

    def add_annotation_class(self,add_click,delete_click):
        """
        Adding a new annotation class when pre-setting an annotation session
        """
        add_click
        add_click = get_pattern_matching_value(add_click)
        if ctx.triggered_id['type']=='add-annotation-class':
            patched_list = Patch()

            def new_class_input():
                return html.Div([
                    dbc.Row([
                        dbc.Col(
                            dcc.Input(
                                placeholder = 'New Class Name',
                                type = 'text',
                                maxLength=1000,
                                id = {'type':'new-annotation-class','index':add_click},
                                style = {'width':'100%'}
                            ),
                            md = 8
                        ),
                        dbc.Col(
                            html.Div(
                                dmc.ColorInput(
                                    id = {'type':'new-annotation-color','index':add_click},
                                    label = 'Color',
                                    format = 'rgb',
                                    value = f'rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})'
                                ),
                                style = {'width':'100%'}
                            ),
                            md = 2
                        ),
                        dbc.Col(
                            html.I(
                                id = {'type':'delete-annotation-class','index':add_click},
                                n_clicks = 0,
                                className = 'bi bi-x-circle-fill fa-2x',
                                style = {'color':'rgb(255,0,0)'}
                            ),
                            md = 2
                        )
                    ],align = 'center')
                ])
        
            patched_list.append(new_class_input())

        elif ctx.triggered_id['type']=='delete-annotation-class':
    
            patched_list = Patch()
            values_to_remove = []
            for i,val in enumerate(delete_click):
                if val:
                    values_to_remove.insert(0,i)
            
            for v in values_to_remove:
                del patched_list[v]

        else:
            raise exceptions.PreventUpdate

        return [patched_list]

    def add_annotation_label(self,add_click,delete_click):
        """
        Adding a new annotation label type when pre-setting an annotation session
        """
        add_click = get_pattern_matching_value(add_click)

        if ctx.triggered_id['type']=='add-annotation-label':

            patched_list = Patch()

            def new_label_item():
                return html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                placeholder = 'New Class Label',
                                type = 'text',
                                maxLength=1000,
                                id = {'type':'new-annotation-label','index':add_click},
                                style = {'width':'100%'}
                            )
                        ], md = 10),
                        dbc.Col(
                            html.I(
                                id = {'type':'delete-annotation-label','index':add_click},
                                n_clicks = 0,
                                className = 'bi bi-x-circle-fill fa-2x',
                                style = {'color':'rgb(255,0,0)'}
                            ),
                            md = 2
                        )
                    ])
                ])
            
            patched_list.append(new_label_item())
        
        elif ctx.triggered_id['type']=='delete-annotation-label':

            patched_list = Patch()
            values_to_remove = []
            for i,val in enumerate(delete_click):
                if val:
                    values_to_remove.insert(0,i)
            
            for v in values_to_remove:
                del patched_list[v]

        else:
            raise exceptions.PreventUpdate
        
        return [patched_list]

    def add_annotation_user(self,add_click,delete_click):
        """
        Adding a new user to the annotation session preset
        """
        add_click = get_pattern_matching_value(add_click)

        if ctx.triggered_id['type']=='add-annotation-user':

            patched_list = Patch()

            def new_user_input():
                return html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                placeholder = 'Add username here',
                                type = 'text',
                                maxLength = 1000,
                                id = {'type':'new-annotation-user','index':add_click},
                                style = {'width':'100%'}
                            )
                        ], md = 8),
                        dbc.Col(
                            dcc.Dropdown(
                                options = ['annotator','admin'],
                                value = 'annotator',
                                id = {'type':'new-user-type','index':add_click}
                            )
                        ),
                        dbc.Col(
                            html.I(
                                id = {'type':'delete-annotation-user','index':add_click},
                                n_clicks = 0,
                                className = 'bi bi-x-circle-fill fa-2x',
                                style = {'color':'rgb(255,0,0)'}
                            ),
                            md = 2
                        )
                    ])
                ])

            patched_list.append(new_user_input())
        
        elif ctx.triggered_id['type']=='delete-annotation-user':
            patched_list = Patch()
            values_to_remove = []
            for i,val in enumerate(delete_click):
                if val:
                    values_to_remove.insert(0,i)
            for v in values_to_remove:
                del patched_list[v]

        else:
            raise exceptions.PreventUpdate

        return [patched_list]



        return [patched_list]

    def add_cell_subtypes(self,butt_click,main_cell_types):
        """
        Extracting subtypes for a given "Main Cell Type" and adding those as a visualizable property for overlaid heatmaps
        """

        main_cell_types = get_pattern_matching_value(main_cell_types)

        if len(main_cell_types)==0:
            raise exceptions.PreventUpdate
        
        if not ctx.triggered:
            raise exceptions.PreventUpdate

        new_cell_droptions = no_update
        new_cell_subtype_dropue = [no_update]

        if not main_cell_types is None:
            # Running change-level plugin with these options
            job_id = self.wsi.run_change_level(main_cell_types)        
            job_status,most_recent_log = self.dataset_handler.get_job_status(job_id)
            # Checking progress of the job
            while not job_status==3 and not job_status==4:
                time.sleep(1)
                job_status,most_recent_log = self.dataset_handler.get_job_status(job_id)
                print(f'change_level job status: {job_status}')

            if job_status==4:
                raise exceptions.PreventUpdate
            
            # Grabbing from user "Supplemental_Annotations" folder
            updated_annotations_file = f'{self.wsi.item_id}.json'
            updated_annotations = self.dataset_handler.grab_from_user_folder(updated_annotations_file,'Supplemental_Annotations')

            self.wsi.update_ftu_props(updated_annotations)
            self.wsi.update_annotation_geojson(updated_annotations)

            new_cell_droptions = self.wsi.properties_list
            new_cell_subtype_dropue= ['']


        return new_cell_droptions, new_cell_subtype_dropue




def app(*args):
    
    # Using DSA as base directory for storage and accessing files
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
    initial_user_info = dataset_handler.user_details

    # Initial collection
    #TODO: initial collection can be added as an environment variable
    initial_collection = ['/collection/10X_Visium']
    path_type = ['collection','collection']
    print(f'initial collection: {initial_collection}')

    if isinstance(initial_collection,str):
        initial_collection_id = [dataset_handler.gc.get('resource/lookup',parameters={'path':initial_collection})]
    elif isinstance(initial_collection,list):
        initial_collection_id = [dataset_handler.gc.get('resource/lookup',parameters={'path':i}) for i in initial_collection]

    print(f'loading initial slide(s)')
    # Contents of folder (used for testing to initialize with one slide)
    initial_collection_contents = []
    for p_type,i in zip(path_type,initial_collection_id):
        initial_collection_contents.extend(dataset_handler.gc.get(f'resource/{i["_id"]}/items',parameters={'type':p_type}))
    initial_collection_contents = [i for i in initial_collection_contents if 'largeImage' in i]

    # For testing, setting initial slide
    # Avoiding image with elbow (permeabilization artifact present)
    default_images = ['XY04_IU-21-020F.svs','XY03_IU-21-019F.svs']
    initial_collection_contents = [i for i in initial_collection_contents if i['name'] in default_images]
    
    # Saving & organizing relevant id's in GirderHandler
    print('Getting initial items metadata')
    dataset_handler.set_default_slides(initial_collection_contents)
    dataset_handler.initialize_folder_structure(username,initial_collection,path_type)

    # Getting graphics_reference.json from the FUSION Assets folder
    print(f'Getting asset items')
    assets_path = '/collection/FUSION Assets/'
    dataset_handler.get_asset_items(assets_path)

    # Getting the slide data for DSASlide()
    slide_names = [
        {'label': i['name'],'value':i['_id']}
        for i in initial_collection_contents
    ]

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
    layout_handler.gen_initial_layout(slide_names,initial_user_info,dataset_handler.default_slides)
    layout_handler.gen_vis_layout(
        wsi,
        cli_list
        )
    layout_handler.gen_builder_layout(dataset_handler,initial_user_info)
    layout_handler.gen_uploader_layout()

    download_handler = DownloadHandler(dataset_handler)

    prep_handler = Prepper(dataset_handler)
    
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
