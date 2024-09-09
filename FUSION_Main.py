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

#from FUSION_WSI import DSASlide, VisiumSlide, CODEXSlide, XeniumSlide
from FUSION_WSI import SlideHandler
from FUSION_Handlers import LayoutHandler, DownloadHandler, GirderHandler, GeneHandler
from FUSION_Prep import XeniumPrep, CODEXPrep, VisiumPrep, Prepper
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

        self.download_handler = download_handler
        self.prep_handler = prep_handler

        self.gene_handler = GeneHandler()
        self.slide_handler = SlideHandler(
            girder_handler = self.dataset_handler
        )

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
        #TODO: Get rid of self.filter_labels, potential for user overlap
        self.filter_labels = []
        #TODO: get rid of self.reports_generated, potential for user overlap
        self.reports_generated = {}

        # Number of main cell types to include in pie-charts (currently set to all cell types)
        self.plot_cell_types_n = len(list(self.cell_names_key.keys()))

        # ASCT+B table for cell hierarchy generation
        self.table_df = self.dataset_handler.asct_b_table    

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

        # JavaScript functions for controlling annotation properties
        self.ftu_style_handle = assign("""function(feature,context){
            const {color_key,overlay_prop,fillOpacity,ftu_colors,filter_vals} = context.hideout;
                                       
            var overlay_value = Number.Nan;
            if (overlay_prop && "user" in feature.properties) {
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
                                    // TODO: Might have to do something here for non-aggregated String props
                                    var overlay_value = feature.properties.user[overlay_prop.name][overlay_prop.value];
                                }
                            } else if (overlay_prop.value==="max") {
                                // Finding max represented sub-value
                                var overlay_value = Number.Nan;
                                var test_value = 0.0;
                                for (var key in feature.properties.user[overlay_prop.name]) {
                                    var tester = feature.properties.user[overlay_prop.name][key];
                                    if (tester > test_value) {
                                        test_value = tester;
                                        overlay_value = key;
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
                                            // TODO: Might have to do something here for non-aggregated String props
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
                                if (filter.range.includes(test_val)) {
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

        # Running server (waitress)
        serve(self.app.server,host='0.0.0.0',port=8000,threads = 10)

    def view_instructions(self,n,n2,is_open,user_data_store):
        """
        Opens collapsible component underneath buttons containing usability questions        
        """

        user_data_store = json.loads(user_data_store)

        # Opening collapse and populating internal div 
        if ctx.triggered_id['type']=='collapse-descrip':
            collapse_children = self.layout_handler.description_dict[self.current_page]
            usability_color = ['primary']
        elif ctx.triggered_id['type']=='usability-butt':
            if n2:
                self.dataset_handler.update_usability()
                usability_info = self.dataset_handler.check_usability(user_data_store['login'])
                usability_info['login'] = user_data_store['login']
                collapse_children = self.layout_handler.gen_usability_report(usability_info,self.dataset_handler.usability_users)
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

    def update_page(self,pathname, user_data, vis_sess_store, available_datasets_store):
        
        vis_sess_store = json.loads(vis_sess_store)
        if len(available_datasets_store)>0:
            available_datasets_store = json.loads(get_pattern_matching_value(available_datasets_store))
        else:
            available_datasets_store = None
        user_data = json.loads(user_data)
        pathname = pathname[1:]
        #print(f'Navigating to {pathname}')

        slide_select_value = ''

        #TODO: Make this one more generalizable like grab extra data for any studies, etc.
        self.dataset_handler.update_usability()
        #TODO: Currently this method does not remove old annotations as expected
        self.dataset_handler.clean_old_annotations()

        if pathname in self.layout_handler.layout_dict:
            if not pathname=='vis':
                if not pathname=='dataset-builder':
                    if pathname=='dataset-uploader':
                        # Double-checking that a user is logged in before giving access to dataset-uploader
                        if user_data['login']=='fusionguest':
                            #self.current_page = 'welcome'
                            pathname = 'welcome'
                    container_content = self.layout_handler.layout_dict[pathname]
                else:
                    # Checking if there was any new slides added via uploader (or just added externally)
                    #self.dataset_handler.update_folder_structure(user_data['login'])
                    available_datasets_store = self.layout_handler.gen_builder_layout(self.dataset_handler,user_data,available_datasets_store)
                    container_content = self.layout_handler.layout_dict[pathname]

            else:

                # Generating visualization layout with empty clustering data, default filter vals, and no WSI
                self.layout_handler.gen_vis_layout(
                    self.gene_handler
                )

                # Checking vis_sess_store for any slides, if there aren't any 'included'==True then revert to default set
                included_slides = [i['included'] for i in vis_sess_store if 'included' in i]
                if len(included_slides)==0:
                    vis_sess_store = []
                    for s in self.dataset_handler.default_slides:
                        s['included'] = True
                        vis_sess_store.append(s)

                container_content = self.layout_handler.layout_dict[pathname]

        else:
            #self.current_page = 'welcome'
            pathname = 'welcome'
            container_content = self.layout_handler.layout_dict[pathname]

        if pathname == 'vis':
            slide_style = {'marginBottom':'20px','display':'inline-block'}
        else:
            slide_style = {'display':'none'}

        # Clearing slide_info_store and updating vis_sess_store
        slide_info_store = json.dumps({})
        vis_sess_store = json.dumps(vis_sess_store)
        available_datasets_store = [json.dumps(available_datasets_store)]*len(ctx.outputs_list[5])

        return container_content, slide_style, slide_select_value, vis_sess_store, slide_info_store, available_datasets_store

    def open_nav_collapse(self,n,is_open):
        if n:
            return not is_open
        return is_open

    def open_plugin_collapse(self,n,is_open):
        """
        Opening the running plugins collapse to see all the user's current jobs
        """
        if n:
            return not is_open
        return is_open

    def open_job_collapse(self,n,is_open,user_store_data):
        """
        Opening a specific job's collapse and updating with logs
        """
        if n:
            user_store_data = json.loads(user_store_data)

            # Getting the clicked job index:
            user_jobs = self.dataset_handler.get_user_jobs(user_store_data['_id'])

            job_id = user_jobs[ctx.triggered_id['index']]['_id']

            job_status, job_logs = self.dataset_handler.get_job_status(job_id)
            job_logs = job_logs.split('\n')

            if job_status==3:
                badge_children = html.A('Complete')
                badge_color = 'success'
            elif job_status==2:
                badge_children = html.A('Running')
                badge_color = 'warning'
            elif job_status==4:
                badge_children = html.A('Failed')
                badge_color = 'danger'
            else:
                badge_children = html.A(f'Unknown: ({job_status})')
                badge_color = 'info'

            return not is_open, badge_children, badge_color, [html.P(i) for i in job_logs]

        return is_open, no_update, no_update, []

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
             Output('visualization-session-store','data'),
             Output('slide-info-store','data'),
             Output({'type': 'available-datasets-store','index': ALL},'data')],
             Input('url','pathname'),
             [State('user-store','data'),
              State('visualization-session-store','data'),
              State({'type':'available-datasets-store','index': ALL},'data')],
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
                Output('user-store','data'),
                Output('long-plugin-div','children')
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

        # Open up job status collapse
        self.app.callback(
            Output('plugin-collapse','is_open'),
            Input('long-plugin-butt','n_clicks'),
            State('plugin-collapse','is_open'),
            prevent_initial_call = True
        )(self.open_plugin_collapse)

        # Update job status and view logs
        self.app.callback(
            [
                Output({'type':'checked-job-collapse','index': MATCH},'is_open'),
                Output({'type': 'job-status-badge','index': MATCH},'children'),
                Output({'type': 'job-status-badge','index':MATCH},'color'),
                Output({'type': 'checked-job-logs','index': MATCH},'children')
            ],
            Input({'type':'job-status-button','index': MATCH},'n_clicks'),
            [
                State({'type':'checked-job-collapse','index': MATCH},'is_open'),
                State('user-store','data')
            ],
        prevent_initial_call = True
        )(self.open_job_collapse)

        # Loading new tutorial slides
        self.app.callback(
            Input({'type':'tutorial-tabs','index':ALL},'active_tab'),
            Output({'type':'tutorial-content','index':ALL},'children'),
        )(self.update_tutorial_slide)
        
        # Updating questions in question tab
        self.app.callback(
            Input({'type':'questions-tabs','index':ALL},'active_tab'),
            State('user-store','data'),
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
            [State({'type':'question-input','index':ALL},'value'),
             State('user-store','data')],
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
             Output({'type':'gene-info-div','index':ALL},'children'),
             Output('slide-info-store','data')],
            [Input('cell-drop','value'),Input('vis-slider','value'),
             Input('filter-slider','value'), Input({'type':'added-filter-slider','index':ALL},'value'),
             Input({'type':'added-filter-drop','index':ALL},'value'),
             Input({'type':'ftu-bound-color-butt','index':ALL},'n_clicks'),
             Input({'type':'cell-sub-drop','index':ALL},'value')],
            [State({'type':'ftu-bound-color','index':ALL},'value'),
             State({'type':'added-filter-slider-div','index':ALL},'style'),
             State('slide-info-store','data'),State('cell-drop','options'),
             State('window-size','width')],
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
            [
                State('slide-info-store','data'),
                State('cell-drop','options')
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
            State('slide-info-store','data'),
            prevent_initial_call = True
        )(self.add_filter_slider)

        # Updating viewport data components
        self.app.callback(
            [
                Output('roi-pie-holder','children'),
                Output('annotation-session-div','children'),
                Output('cell-annotation-div','children'),
                Output({'type':'viewport-store-data','index':ALL},'data'),
                Output('user-store','data')
             ],
            [
                Input('slide-map','bounds'),
                Input({'type':'roi-view-data','index':ALL},'value'),
                Input({'type':'viewport-plot-butt','index':ALL},'n_clicks'),
             ],
            [
                State('tools-tabs','active_tab'),
                State('viewport-data-update','checked'),
                State({'type':'viewport-values-drop','index':ALL},'value'),
                State({'type':'viewport-label-drop','index':ALL},'value'),
                State({'type':'viewport-store-data','index':ALL},'data'),
                State('user-store','data'),
                State('slide-info-store','data')
             ],
            prevent_initial_call = True
        )(self.update_viewport_data)      

        # Updating cell hierarchy data
        self.app.callback(
            [Output('cell-graphic','src'),
             Output('cell-hierarchy','elements'),
             Output('cell-vis-drop','options'),
             Output('cell-graphic-name','children'),
             Output('organ-hierarchy-store','data'),
             Output('organ-hierarchy-cell-select','options'),
             Output('organ-hierarchy-cell-select','value'),
             Output('organ-hierarchy-cell-select-div','style'),
             Output('nephron-diagram','style')],
            [Input('neph-img','clickData'),
             Input('organ-hierarchy-select','value'),
             Input('organ-hierarchy-cell-select','value')],
            State('organ-hierarchy-store','data')
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
            State({'type': 'viewport-store-data','index': ALL},'data'),
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
            [
                Output('slide-tile-holder','children'),
                Output('layer-control-holder','children'),
                Output({'type':'edit-control','index':ALL},'editToolbar'),
                Output('slide-map','center'),
                Output('cell-drop','options'),
                Output('ftu-bound-opts','children'),
                Output('special-overlays','children'),
                Output('cell-annotation-tab','disabled'),
                Output('marker-add-div','children'),
                Output('marker-add-geojson','data')
            ],
            Input('slide-load-interval','disabled'),
            State('slide-info-store','data'),
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
                State('slide-info-store','data'),
                State('user-store','data'),
            ],
            prevent_initial_call = True
        )(self.load_new_slide)
        
        # Updating cytoscapes plot for cell hierarchy
        self.app.callback(
            [Output('label-p','children'),
            Output('id-p','children'),
            Output('notes-p','children')],
            Input('cell-hierarchy','tapNodeData'),
            State('organ-hierarchy-store','data'),
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
            [State('cluster-store','data'),
             State('user-store','data'),
             State('slide-info-store','data')],
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
            Input({'type':'edit-control','index':ALL},'geojson'),
            [Output({'type': 'layer-control','index': ALL},'children'),
             Output('user-annotations-div', 'children')],
            State('slide-info-store','data'),
            prevent_initial_call=True
        )(self.add_manual_roi)

        # Add histology marker from clustering/plot
        self.app.callback(
            Input({'type':'add-mark-cluster','index':ALL},'n_clicks'),
            [State('cluster-store','data'),
             State('slide-info-store','data')],
            [Output({'type':'edit_control','index':ALL},'geojson'),
             Output('marker-add-div','children')],
            prevent_initial_call=True
        )(self.add_marker_from_cluster)

        # Downloading data
        self.app.callback(
            [
                Input({'type':'download-annotations-butt','index':ALL},'n_clicks'),
                Input({'type':'download-cell-butt','index':ALL},'n_clicks'),
                Input({'type':'download-meta-butt','index': ALL},'n_clicks'),
                Input({'type':'download-manual-butt','index': ALL},'n_clicks')
            ],
            Output({'type':'download-data','index':ALL},'data'),
            [
                State('user-store','data'),
                State('slide-map','bounds'),
                State('slide-info-store','data'),
                State({'type': 'download-ann-format','index': ALL},'value'),
                State({'type': 'download-cell-format','index': ALL},'value')
            ],
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
            State('slide-info-store','data'),
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
             Output({'type':'channel-overlay-butt','index':ALL},'disabled'),
             Output('slide-info-store','data')],
            Input({'type':'channel-overlay-drop','index': ALL},'value'),
            State('slide-info-store','data'),
            prevent_initial_call = True,
        )(self.add_channel_color_select)

        # Adding CODEX channel overlay
        self.app.callback(
            [Output({'type':'codex-tile-layer','index':ALL},'url'),
             Output({'type':'overlay-channel-tab','index':ALL},'label_style'),
             Output('slide-info-store','data')],
            Input({'type':'channel-overlay-butt','index':ALL},'n_clicks'),
            [State({'type':'overlay-channel-color','index':ALL},'value'),
            State({'type':'overlay-channel-tab','index':ALL},'label'),
            State('slide-info-store','data'), State('user-store','data')],
            prevent_initial_call = True
        )(self.add_channel_overlay)

        # Grabbing points from current viewport umap (CODEX) and plotting other properties
        self.app.callback(
            [
                Output('marker-add-div','children'),
                Output('marker-add-geojson','data'),
                Output({'type': 'cell-marker-count','index': ALL},'children'),
                Output({'type':'cell-marker-source','index': ALL},'children'),
                Output({'type':'cell-marker-summary','index':ALL},'children'),
                Output({'type': 'cell-marker-label','index': ALL},'value'),
                Output({'type': 'cell-marker-rationale','index': ALL},'value')
            ],
            [
                Input({'type':'ftu-cell-pie','index':ALL},'selectedData'),
                Input({'type':'cell-marker-apply','index': ALL},'n_clicks')
            ],
            [
                State({'type':'cell-marker-label','index':ALL},'value'),
                State({'type': 'cell-marker-rationale','index': ALL},'value'),
                State({'type':'ftu-cell-pie','index':ALL},'selectedData'),
                State({'type':'viewport-values-drop','index':ALL},'value'),
                State('slide-info-store','data'),
                State('marker-add-geojson','data')
            ],
            prevent_initial_call = True
        )(self.cell_labeling_initialize)

        # Removing cell labeling marker
        self.app.callback(
            [
                Output('marker-add-div','children'),
                Output({'type':'cell-marker-count','index': ALL},'children'),
                Output('marker-add-geojson','data')
            ],
            [
                Input({'type':'cell-marker-butt','index': ALL},'n_clicks')
            ],
            State('marker-add-geojson','data'),
            prevent_initial_call = True
        )(self.remove_cell_label)

        # Populating tabs for each annotation session in Annotation Station
        self.app.callback(
            [
                Output({'type':'annotation-session-content','index':ALL},'children'),
                Output({'type':'annotation-tab-group','index':ALL},'children'),
                Output('user-store','data')
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
                State({'type':'new-user-type','index':ALL},'value'),
                State('user-store','data'),
                State('slide-info-store','data')
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
                State({'type':'annotation-station-ftu-idx','index':ALL},'children'),
                State('user-store','data'),
                State({'type':'viewport-store-data','index': ALL},'data'),
                State('slide-info-store','data')
            ],
            prevent_initial_call = True
        )(self.update_current_annotation)

        # Downloading annotation session data with modal and progress indicator
        self.app.callback(
            [
                Output({'type':'download-ann-session-data','index':ALL},'data'),
                Output({'type':'ann-session-interval','index':ALL},'disabled'),
                Output({'type':'download-ann-session-modal','index':ALL},'is_open'),
                Output({'type':'download-ann-session-modal','index':ALL},'children')
            ],
            [
                Input({'type':'download-ann-session','index':ALL},'n_clicks'),
                Input({'type':'ann-session-interval','index':ALL},'n_intervals')
            ],
            [
                State('user-store','data'),
                State({'type':'download-ann-session-modal','index':ALL},'is_open')
            ],
            prevent_initial_call = True
        )(self.download_annotation_session_log)

    def builder_callbacks(self):

        # Initializing plot after selecting dataset(s)
        self.app.callback(
            Input('dataset-table','selected_rows'),
            [State('visualization-session-store','data'),
             State({'type': 'available-datasets-store','index': ALL},'data')],
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
            [State('visualization-session-store','data'),
             State({'type': 'available-datasets-store','index': ALL},'data')],
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
             Input({'type':'wsi-files-upload','index':ALL},'uploadComplete'),
             Input({'type':'wsi-upload','index':ALL},'fileTypeFlag'),
             Input({'type':'wsi-files-upload','index':ALL},'fileTypeFlag')],
             State('user-store','data'),
            [Output('slide-qc-results','children'),
             Output('slide-thumbnail-holder','children'),
             Output({'type':'wsi-upload-div','index':ALL},'children'),
             Output({'type':'wsi-files-upload-div','index':ALL},'children'),
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

    def initialize_metadata_plots(self,selected_dataset_list,vis_sess_store,dataset_store):
        """
        Creating plot and slide select divs in dataset-builder
        """
        vis_sess_store = json.loads(vis_sess_store)
        dataset_store = json.loads(get_pattern_matching_value(dataset_store))

        # Extracting metadata from selected datasets and plotting
        all_metadata_labels = []
        slide_dataset_dict = []
        full_slides_list = []

        for d in selected_dataset_list:

            # Pulling dataset metadata
            # Dataset name in this case is associated with a folder or collection
            metadata_available = list(dataset_store['slide_dataset'][d]['Aggregated_Metadata'].keys())
            # This will store metadata, id, name, etc. for every slide in that dataset
            slides_list = self.dataset_handler.get_folder_slides(dataset_store['slide_dataset'][d]["_id"])
            full_slides_list.extend(slides_list)

            slide_dataset_dict.extend([{'Slide Names':s['name'],'Dataset':dataset_store['slide_dataset'][d]['name']} for s in slides_list])

            # Grabbing dataset-level metadata
            metadata_available += ['FTU Expression Statistics', 'FTU Morphometrics']

            all_metadata_labels.extend(metadata_available)

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

        if not len(all_metadata_labels)==0:
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

            plot_div = html.Div([
                dcc.Graph(id = {'type':'meta-plot','index':0},figure = go.Figure())
            ])

        else:
            drop_div = html.Div()
            plot_div = html.Div()

        slide_rows = [list(range(len(slide_dataset_dict)))]
        current_slide_count, slide_select_options, vis_sess_store = self.update_current_slides(slide_rows,vis_sess_store)

        return drop_div, plot_div, slide_select_options, json.dumps(vis_sess_store)

    def update_metadata_plot(self,new_meta,sub_meta,group_type,slide_rows,vis_sess_store,dataset_store):
        """
        Plotting slide/dataset-level metadata in dataset-builder
        """

        vis_sess_store = json.loads(vis_sess_store)
        if len(dataset_store)>0:
            dataset_store = json.loads(get_pattern_matching_value(dataset_store))
        else:
            dataset_store = {}
        
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

        new_meta = get_pattern_matching_value(new_meta)
        group_type = get_pattern_matching_value(group_type)

        if not new_meta is None:
            if not len(new_meta)==0:
                # Filtering out de-selected slides
                included_datasets = [f for f in dataset_store['slide_dataset'] if any([f["_id"]==j['folderId'] and j['included'] for j in vis_sess_store])]
                dataset_metadata = []
                for dataset in included_datasets:
                    dataset_name = dataset['name']
                    slide_data = [s for s in vis_sess_store if s['included'] and s['folderId']==dataset['_id']]

                    if len(slide_data)>0:
                        
                        # Adding new_meta value to dataset dictionary
                        if not new_meta=='FTU Expression Statistics' and not new_meta=='FTU Morphometrics':
                            d_dict = {'Dataset':[],'Slide Name':[],new_meta:[]}

                            for s in slide_data:
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

                                    for d_i in slide_data:
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
                                    
                                    for d_i in slide_data:
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
                            if not new_meta == 'FTU Expression Statistics':
                                fig = go.Figure(px.violin(plot_data,points = 'all', x=group_bar,y=new_meta,hover_data=['Slide Name']))
                            else:
                                if sub_meta[0]=='Main Cell Types':
                                    fig = go.Figure(px.violin(plot_data,points = 'all', x=group_bar,y=new_meta,hover_data=['Slide Name'],color='FTU'))
                                else:
                                    fig = go.Figure(px.violin(plot_data,points = 'all', x=group_bar,y=new_meta,hover_data=['Slide Name','State'],color='FTU'))
                        else:
                            if not new_meta=='FTU Expression Statistics' and not new_meta=='FTU Morphometrics':
                                fig = go.Figure(px.bar(plot_data,x=group_bar,y=new_meta,hover_data = ['Dataset']))
                            else:
                                if sub_meta[0]=='Main Cell Types':
                                    fig = go.Figure(px.bar(plot_data,x=group_bar,y=new_meta,hover_data=['Dataset'],color='FTU'))
                                else:
                                    fig = go.Figure(px.bar(plot_data,x=group_bar,y=new_meta,hover_data=['Dataset','State'],color='FTU'))
                    else:
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
                folder_name = self.dataset_handler.get_folder_name(f)

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
                folder_name = self.dataset_handler.get_folder_name(f)

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

    def update_viewport_data(self,bounds,cell_view_type,frame_plot_butt,current_tab,update_data,frame_list,frame_label,current_data, user_data_store, slide_info_store):
        """
        Updating data used for current viewport visualizations
        """
        user_data_store = json.loads(user_data_store)
        slide_info_store = json.loads(slide_info_store)
        
        if not len(get_pattern_matching_value(current_data))==0:
            viewport_store_data = json.loads(get_pattern_matching_value(current_data))
        else:
            viewport_store_data = {}
        cell_view_type = get_pattern_matching_value(cell_view_type)

        viewport_data_components = no_update
        annotation_session_components = no_update
        cell_annotation_components = no_update
        if update_data or ctx.triggered_id=='roi-view-data':
            if not len(list(slide_info_store.keys()))==0:
                if not bounds is None:
                    if len(bounds)==2:
                        bounds_box = shapely.geometry.box(bounds[0][1],bounds[0][0],bounds[1][1],bounds[1][0])
                    else:
                        bounds_box = shapely.geometry.box(*bounds)

                    # Storing current slide boundaries
                    viewport_store_data['current_slide_bounds'] = list(bounds_box.bounds)
                else:
                    viewport_store_data['current_slide_bounds'] = slide_info_store['map_bounds']
            else:
                # If no slide is currently in view
                viewport_store_data['current_slide_bounds'] = None
                raise exceptions.PreventUpdate
        
        # Making a box-poly from the bounds
        if current_tab=='cell-compositions-tab':
            if not len(list(slide_info_store.keys()))==0:
                # Getting a dictionary containing all the intersecting spots with this current ROI
                frame_list = get_pattern_matching_value(frame_list)
                frame_label = get_pattern_matching_value(frame_label)
                if slide_info_store['slide_type']=='CODEX':
                    if frame_list is None or len(frame_list)==0:
                        frame_list = [0]

                    view_type_dict = {
                        'name': cell_view_type,
                        'values': frame_list if type(frame_list)==list else [frame_list],
                        'label': frame_label
                    }
                else:
                    view_type_dict = {
                        'name': cell_view_type
                    }

                viewport_store_data['view_type_dict'] = view_type_dict

                if update_data or not update_data and not ctx.triggered_id=='slide-map':
                    #TODO: Generalized get_viewport_data method for current wsi which should return a list of all the stuff needed for that tab
                    viewport_data_components, viewport_data = self.slide_handler.update_viewport_data(
                        bounds_box = viewport_store_data['current_slide_bounds'],
                        view_type = view_type_dict,
                        slide_info = slide_info_store
                    )

                    viewport_store_data['data'] = viewport_data

                    viewport_data_return = json.dumps(viewport_store_data)
                    user_data_return = json.dumps(user_data_store)

                    return viewport_data_components, annotation_session_components, cell_annotation_components, [viewport_data_return], user_data_return
                else:
                    raise exceptions.PreventUpdate

            else:
                return [html.P('Select a slide to get started!')], [no_update], [no_update], [viewport_data_return], json.dumps(user_data_store)
    
        elif current_tab=='annotation-tab':
            if user_data_store["login"]=='fusionguest':
                return no_update, html.P('Sign in or create an account to start annotating!'), [no_update], [json.dumps(viewport_store_data)], json.dumps(user_data_store)
            else:
                
                # Getting intersecting FTU information
                intersecting_ftus = {}
                current_ftu_names = [i['annotation']['name'] for i in slide_info_store['annotations']]
                for ftu in current_ftu_names:
                    if not ftu=='Spots' and not ftu=='Cells':
                        intersecting_ftus[ftu], _ = self.slide_handler.find_intersecting_ftu(
                            viewport_store_data['current_slide_bounds'],
                            ftu,
                            slide_info_store
                        )

                for m_idx,m_ftu in enumerate(slide_info_store['manual_ROIs']):
                    intersecting_ftus[f'Manual ROI: {m_idx+1}'] = [m_ftu['geojson']['features'][0]['properties']['user']]

                for marked_idx, marked_ftu in enumerate(slide_info_store['marked_FTUs']):
                    intersecting_ftus[f'Marked FTUs: {marked_idx+1}'] = [i['properties']['user'] for i in marked_ftu['geojson']['features']]
                
                #TODO: Check for current annotation session in user's public folder
                annotation_tabs, first_tab, first_session = self.layout_handler.gen_annotation_card(self.dataset_handler,intersecting_ftus, user_data_store)
                user_data_store["current_ann_session"] = first_session

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

                return viewport_data_components, annotation_tab_group, cell_annotation_components, [json.dumps(viewport_store_data)], json.dumps(user_data_store)
        
        elif current_tab == 'cell-annotation-tab':

            # Getting cell annotation components from CODEXSlide
            current_cell_annotation_data = {
                'current_ontology': None,
                'n_labeled': 0,
                'current_rules': []
            }

            cell_annotation_components = self.slide_handler.populate_cell_annotation(
                viewport_store_data['current_slide_bounds'], 
                current_cell_annotation_data
            )

            return viewport_data_components, annotation_session_components,cell_annotation_components, [json.dumps(viewport_store_data)], json.dumps(user_data_store)
        else:
            raise exceptions.PreventUpdate

    def update_state_bar(self,cell_click, viewport_data_store):
        """
        View cell state proportions for clicked main cell type
        
        """
        if not cell_click is None:
            pie_cell = cell_click['points'][0]['label']
            viewport_data_store = json.loads(get_pattern_matching_value(viewport_data_store))

            current_ftu_data = viewport_data_store['data']
            pie_ftu = list(current_ftu_data.keys())[ctx.triggered_id['index']]

            assembled_state_data = []
            for i in current_ftu_data[pie_ftu]['data']:
                if 'states' in i:
                    if pie_cell in i['states']:
                        assembled_state_data.append(i['states'][pie_cell])
            pct_states = pd.DataFrame.from_records(assembled_state_data).sum(axis=0).to_frame()
            pct_states = pct_states.reset_index()
            pct_states.columns = ['Cell State','Proportion']
            pct_states['Proportion'] = pct_states['Proportion']/pct_states['Proportion'].sum()
            if pie_cell in self.cell_graphics_key:
                cell_title = self.cell_graphics_key[pie_cell]["full"]
            else:
                cell_title = pie_cell

            state_bar = go.Figure(px.bar(pct_states,x='Cell State', y = 'Proportion', title = f'Cell State Proportions for:<br><sup>{cell_title} in:</sup><br><sup>{pie_ftu}</sup>'))

            return state_bar
        else:
            return go.Figure()
    
    def update_hex_color_key(self, overlay_prop, slide_info_store):
        
        # Iterate through all structures (and spots) in current wsi,
        # concatenate all of their proportions of specific cell types together
        # scale it with self.color_map (make sure to multiply by 255 after)
        # convert uint8 RGB colors to hex
        # create look-up table for original value --> hex color
        # add that as get_color() function in style dict (fillColor) along with fillOpacity

        #TODO: Add a scaling/range thing for values used as overlays
        raw_values_list = []
        if not overlay_prop is None:
            if not overlay_prop['name'] is None:
                #TODO: Attach this method to something else
                raw_values_list.extend(self.slide_handler.get_overlay_value_list(overlay_prop,slide_info_store))

        # Converting to RGB
        if len(raw_values_list)>0:
            raw_values_list = np.unique(raw_values_list).tolist()
            if all([not type(i)==str for i in raw_values_list]):
                if max(raw_values_list)>0:
                    scaled_values = [(i-min(raw_values_list))/max(raw_values_list) for i in raw_values_list]
                else:
                    scaled_values = raw_values_list
                
                rgb_values = np.uint8(255*self.color_map(scaled_values))[:,0:3]
            else:
                scaled_values = [i/len(raw_values_list) for i in range(len(raw_values_list))]
                rgb_values = np.uint8(255*self.color_map(scaled_values))[:,0:3]

            hex_list = []
            for row in range(rgb_values.shape[0]):
                hex_list.append('#'+"%02x%02x%02x" % (rgb_values[row,0],rgb_values[row,1],rgb_values[row,2]))

            hex_color_key = {i:j for i,j in zip(raw_values_list,hex_list)}
        else:
            hex_color_key = {}

        return hex_color_key

    def update_overlays(self,cell_val,vis_val,filter_vals,added_filter_slider,added_filter_keys,ftu_color_butt,cell_sub_val,ftu_bound_color, added_slide_style, slide_info_store, all_overlay_options,window_width):
        """
        Updating overlay hideout property in GeoJSON layer used in self.ftu_style_handle        
        """

        m_prop = None
        cell_sub_select_children = no_update
        slide_info_store = json.loads(slide_info_store)

        if len(list(slide_info_store.keys()))==0 or all([i['value'] is None for i in ctx.triggered]):
            raise exceptions.PreventUpdate
        
        if 'overlay_prop' in slide_info_store:
            overlay_prop = slide_info_store['overlay_prop']
        else:
            overlay_prop = None

        if 'filter_vals' in slide_info_store:
            ftu_filter_vals = slide_info_store['filter_vals']
        else:
            ftu_filter_vals = None

        if slide_info_store['slide_type']=='Visium':
            gene_info_style = [{'display':'none'}]*len(ctx.outputs_list[6])
            gene_info_components = [[]]*len(ctx.outputs_list[7])
        else:
            gene_info_style = []
            gene_info_components = []

        color_bar_style = {
            'visibility':'visible',
            'background':'white',
            'background':'rgba(255,255,255,0.8)',
            'box-shadow':'0 0 15px rgba(0,0,0,0.2)',
            'border-radius':'10px',
            'width': f'{round(0.3*window_width)}px',
            'padding':'0px 0px 0px 25px'
        }

        cell_sub_val = get_pattern_matching_value(cell_sub_val)

        if type(ctx.triggered_id)==list:
            triggered_id = ctx.triggered_id[0]
        else:
            triggered_id = ctx.triggered_id

        try:
            if triggered_id['type']=='ftu-bound-color-butt':
                if not ftu_bound_color is None:
                    ftu_colors = slide_info_store['ftu_colors']
                    ftu_names = [i['annotation']['name'] for i in slide_info_store['annotations']]
                    ftu_colors[ftu_names[triggered_id['index']]] = ftu_bound_color[triggered_id['index']]
        
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
                    overlay_prop = {
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
                    overlay_prop = {
                        'name': m_prop,
                        'value': cell_name,
                        'sub_value': cell_sub_val
                    }
                
                elif cell_sub_val=='All':
                    overlay_prop = {
                        'name': m_prop,
                        'value': cell_name,
                        'sub_value': None
                    }
                
                else:
                    # Visualizing a sub-property of a main cell type
                    overlay_prop = {
                        'name': 'Cell_States',
                        'value': cell_name,
                        'sub_value': cell_sub_val
                    }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                if len(list(hex_color_key.keys()))>0:
                    filter_min_val = np.min(list(hex_color_key.keys()))
                    filter_max_val = np.max(list(hex_color_key.keys()))
                    filter_disable = False
                else:
                    filter_min_val = 0
                    filter_max_val = 1
                    filter_disable = True
            
            elif m_prop == 'Cell_Subtypes':

                cell_sub_select_children = []

                overlay_prop = {
                    'name': m_prop,
                    'value':cell_val,
                    'sub_value': None
                }

                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                if len(list(hex_color_key.keys()))==0:
                    filter_min_val = 0.0
                    filter_max_val = 1.0
                    filter_disable = True
                else:
                    filter_min_val = np.min(list(hex_color_key.keys()))
                    filter_max_val = np.max(list(hex_color_key.keys()))
                    filter_disable = False

            elif m_prop == 'Cell_States':
                # Selecting a specific cell state value for overlays
                overlay_prop = {
                    'name': m_prop,
                    'value': cell_val,
                    'sub_value': None
                }

                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                filter_min_val = np.min(list(hex_color_key.keys()))
                filter_max_val = np.max(list(hex_color_key.keys()))
                filter_disable = False

            elif m_prop == 'Max Main Cell Type':
                # Getting the maximum cell type present for each structure
                overlay_prop = {
                    'name': 'Main_Cell_Types',
                    'value':'max',
                    'sub_value': None
                }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                cell_sub_select_children = []

                filter_min_val = 0.0
                filter_max_val = 1.0
                filter_disable = True

            elif m_prop == 'Max Cell Subtype':
                # Getting the maximum cell type present for each structure
                overlay_prop = {
                    'name': 'Cell_Subtypes',
                    'value':'max',
                    'sub_value': None
                }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                cell_sub_select_children = []

                filter_min_val = 0.0
                filter_max_val = 1.0
                filter_disable = True

            elif m_prop == 'Cluster':
                # Getting cluster value associated with each structure
                overlay_prop = {
                    'name': 'Cluster',
                    'value': None,
                    'sub_value': None
                }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                cell_sub_select_children = []

                filter_min_val = 0.0
                filter_max_val = 1.0
                filter_disable = True
                
            elif m_prop == 'FTU Label':
                overlay_prop = {
                    'name': 'Structure',
                    'value': None,
                    'sub_value': None
                }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                cell_sub_select_children = []

                filter_min_val = 0.0
                filter_max_val = 1.0
                filter_disable = True
            
            elif m_prop == 'Gene Counts':

                overlay_prop = {
                    'name': 'Gene Counts',
                    'value': cell_val,
                    'sub_value': None
                }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                # Now displaying gene info
                if triggered_id=='cell-drop':
                    gene_info_style = [{'display':'inline-block'}]
                    gene_info_components = [self.gene_handler.get_layout(gene_id=cell_val)]
                else:
                    gene_info_style = [no_update]
                    gene_info_components = [no_update]
                

                cell_sub_select_children = []

                if len(list(hex_color_key.keys()))>0:
                    filter_min_val = np.min(list(hex_color_key.keys()))
                    filter_max_val = np.max(list(hex_color_key.keys()))
                else:
                    filter_min_val = 0.0
                    filter_max_val = 1.0
                filter_disable = False
            
            elif m_prop == 'Morphometrics':

                overlay_prop = {
                    'name': m_prop,
                    'value': cell_val,
                    'sub_value': None
                }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                cell_sub_select_children = []

                filter_min_val = np.min(list(hex_color_key.keys()))
                filter_max_val = np.max(list(hex_color_key.keys()))
                filter_disable = False
            
            else:
                if m_prop==cell_val:
                    overlay_prop = {
                        'name': m_prop,
                        'value': None,
                        'sub_value': None
                    }
                else:
                    overlay_prop = {
                        'name': m_prop,
                        'value': cell_val,
                        'sub_value': None
                    }
                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                cell_sub_select_children = []

                if not len(list(hex_color_key.keys()))==0:
                    if all([not type(i)==str for i in list(hex_color_key.keys())]):
                        filter_min_val = np.min(list(hex_color_key.keys()))
                        filter_max_val = np.max(list(hex_color_key.keys()))
                        filter_disable = False
                    else:
                        filter_min_val = 0.0
                        filter_max_val = 1.0
                        filter_disable = True
                else:
                    filter_min_val = 0
                    filter_max_val = 1.0
                    filter_disable = True

        else:
            overlay_prop = {
                'name': None,
                'value': None,
                'sub_value': None
            }
            hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

            cell_sub_select_children = []

            color_bar = no_update
            filter_disable = True
            filter_min_val = 0
            filter_max_val = 1

        if not filter_disable:
            ftu_filter_vals = [
                {
                    'name': overlay_prop['name'],
                    'value': overlay_prop['value'],
                    'sub_value': overlay_prop['sub_value'],
                    'range': filter_vals
                }
            ]

            # Processing the add-on filters:
            processed_filters = process_filters(added_filter_keys, added_filter_slider,added_slide_style, self.cell_names_key)

            ftu_filter_vals.extend(processed_filters)

        else:
            ftu_filter_vals = None

        cell_vis_val = vis_val/100
        n_layers = len(callback_context.outputs_list[0])

        color_bar_width = round(0.25*window_width)
        if all([not type(i)==str for i in list(hex_color_key.keys())]) and not overlay_prop['value']=='max':
            if len(list(hex_color_key.values()))>0:
                color_bar = dl.Colorbar(
                    colorscale = list(hex_color_key.values()),
                    width = color_bar_width,
                    height = 15,
                    position='bottomleft',
                    id=f'colorbar{random.randint(0,100)}',
                    style = color_bar_style,
                    tooltip=True
                )
                
            else:
                # This is actually unreachable because the value of cell-drop is None and is the only trigger, leading to exceptions.PreventUpdate in line 2034
                color_bar = dlx.categorical_colorbar(
                    categories = list(slide_info_store['ftu_colors'].keys()),
                    colorscale=list(slide_info_store['ftu_colors'].values()),
                    width = color_bar_width,
                    height = 15,
                    position = 'bottomleft',
                    id = f'colorbar{random.randint(0,100)}',
                    style = color_bar_style
                )
        else:
            """
            if overlay_prop['value']=='max':
                categories = [i.split(' --> ')[-1] for i in all_overlay_options if overlay_prop['name'] in i]
            else:
            """
            categories = list(hex_color_key.keys())

            color_bar = dlx.categorical_colorbar(
                categories = sorted(categories),
                colorscale = list(hex_color_key.values()),
                width = color_bar_width,
                position = 'bottomleft',
                id = f'colorbar{random.randint(0,100)}',
                style = color_bar_style
            )
    
        geojson_hideout = [
            {
                'color_key':hex_color_key,
                'overlay_prop':overlay_prop,
                'fillOpacity': cell_vis_val,
                'ftu_colors': slide_info_store['ftu_colors'],
                'filter_vals': ftu_filter_vals
            }
            for i in range(0,n_layers)
        ]

        slide_info_store['overlay_prop'] = overlay_prop
        slide_info_store['filter_vals'] = ftu_filter_vals
        slide_info_store['cell_vis_val'] = cell_vis_val
        slide_info_store = json.dumps(slide_info_store)

        return geojson_hideout, color_bar, filter_min_val, filter_max_val, filter_disable, cell_sub_select_children, gene_info_style, gene_info_components, slide_info_store

    def update_cell_hierarchy(self,cell_clickData,organ_select,organ_cell_select,organ_store):
        """
        Updating cell cytoscape visualization        
        """
        organ_store = json.loads(organ_store)

        if not organ_store['organ'] is None:
            if not organ_store['organ']==organ_select:
                new_table, new_table_info = self.gene_handler.get_table(organ_select.lower())

                organ_store['organ'] = organ_select
                organ_store['table'] = new_table.to_dict('records')
                organ_store['info'] = new_table_info.to_dict('records')

                if 'CT/1/LABEL' in new_table.columns.tolist():
                    organ_cell_options = new_table['CT/1/LABEL'].dropna().unique().tolist()
                else:
                    organ_cell_options = new_table['CT/1'].dropna().unique().tolist()

            else:
                organ_cell_options = no_update
        else:
            new_table, new_table_info = self.gene_handler.get_table(organ_select.lower())

            organ_store['organ'] = organ_select
            organ_store['table'] = new_table.to_dict('records')
            organ_store['info'] = new_table_info.to_dict('records')
        
            if 'CT/1/LABEL' in new_table.columns.tolist():
                organ_cell_options = new_table['CT/1/LABEL'].dropna().unique().tolist()
            else:
                organ_cell_options = new_table['CT/1'].dropna().unique().tolist()


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
        if organ_select == 'kidney':

            nephron_diagram_style = {'display':'inline-block'}
            organ_cell_style = {'display': 'none'}
            organ_cell_value = no_update
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
                            cell_hierarchy = self.gen_cyto(cell_val,organ_store['table'])
                            cell_state_droptions = np.unique(self.cell_graphics_key[self.cell_names_key[cell_val]]['states'])

        else:
            # This is just because no graphics are available for other organs
            if not ctx.triggered_id=='organ-hierarchy-cell-select':
                organ_df = pd.DataFrame.from_records(organ_store['table'])
                if 'CT/1/LABEL' in organ_df.columns.tolist():
                    organ_cell_value = organ_df['CT/1/LABEL'].dropna().tolist()[0]
                else:
                    organ_cell_value = organ_df['CT/1'].dropna().tolist()[0]
                organ_cell_select = organ_cell_value
            else:
                organ_cell_value = no_update
                if organ_cell_select=='':
                    organ_df = pd.DataFrame.from_records(organ_store['table'])
                    if 'CT/1/LABEL' in organ_df.columns.tolist():
                        organ_cell_select = organ_df['CT/1/LABEL'].dropna().tolist()[0]
                    else:
                        organ_cell_select = organ_df['CT/1'].dropna().tolist()[0]

            cell_hierarchy = self.gen_cyto(organ_cell_select,organ_store['table'])
            nephron_diagram_style = {'display':'none'}
            organ_cell_style = {'display': 'inline-block'}


        organ_store = json.dumps(organ_store)

        return cell_graphic, cell_hierarchy, cell_state_droptions, cell_name, organ_store, organ_cell_options, organ_cell_value, organ_cell_style, nephron_diagram_style

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

        pie_chart_features = [
            'Main_Cell_Types','Cell_States', 'Cell Type', 'Channel Means', 'Channel Stds','Cell_Subtypes', 'All_Subtypes'
        ]

        def make_dash_table(df:pd.DataFrame):
            """
            Populate dash_table.DataTable
            """
            return_table = dash_table.DataTable(
                columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in df],
                data = df.to_dict('records'),
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
                    } for row in df.to_dict('records')
                ],
                tooltip_duration = None
            )

            return return_table
    
        def make_pie_chart(df:pd.DataFrame, top_n = 10):
            """
            Simple pie chart with provided dataframe
            change top_n as needed to only show top_n slices
            """
            df = df.sort_values(by = 'Value',ascending = False)
            df = df.iloc[0:top_n+1,:]

            simple_pie = dcc.Graph(
                figure = go.Figure(
                    data = [
                        go.Pie(
                            name = '',
                            values = df['Value'].tolist(),
                            labels = df['Property'].tolist()
                        )
                    ],
                    layout = {
                        'margin': {'t': 0, 'b': 0, 'l': 0,'r': 0},
                        'width': 300,
                        'uniformtext_minsize': 12,
                        'uniformtext_mode': 'hide',
                        'showlegend': False
                    }
                )
            )

            return simple_pie


        if not ctx.triggered[0]['value']:
            print('raising preventupdate')
            return no_update

        if not ftu_click is None:
            self.clicked_ftu = ftu_click
            if 'unique_index' in ftu_click['properties']:
                ftu_idx = ftu_click['properties']['unique_index']

            accordion_children = []
            # Getting other FTU/Spot properties
            all_properties = list(ftu_click['properties']['user'].keys())
            all_properties = [i for i in all_properties if not type(ftu_click['properties']['user'][i])==dict]
            all_props_dict = {'Property':all_properties,'Value':[ftu_click['properties']['user'][i] for i in all_properties]}
            all_properties_df = pd.DataFrame(all_props_dict)

            # Getting nested properties
            nested_properties = list(ftu_click['properties']['user'].keys())
            nested_properties = [i for i in nested_properties if type(ftu_click['properties']['user'][i])==dict]

            # Nested properties are limited to 3 levels (arbitrarily)
            # Manual ROIs will have properties like: { ftu_name: { main_column: { sub_column: [] } } }
            nested_prop_list = []
            for n_idx, n in enumerate(nested_properties):
                n_prop_data = ftu_click['properties']['user'][n]
                sub_n_props = list(n_prop_data.keys())
                
                if len(sub_n_props)>0:
                    nested_sub_props = [i for i in n_prop_data if type(n_prop_data[i])==dict]
                    for s_n_idx, s_n in enumerate(nested_sub_props):
                        sub_n_data = n_prop_data[s_n]

                        if type(sub_n_data)==dict:
                            sub_sub_n_props = list(sub_n_data.keys())
                            if len(sub_sub_n_props)>0:
                                if type(sub_n_data[sub_sub_n_props[0]])==dict:
                                    for s_s_n_idx, s_s_n in enumerate(sub_sub_n_props):
                                        sub_sub_n_data = sub_n_data[s_s_n]

                                        if type(sub_sub_n_data)==dict:
                                            # Cell_States and Cell_Subtypes for Manual ROIs
                                            nested_prop_list.append({
                                                'name': n,
                                                'sub_name': s_n,
                                                'sub_sub_name': s_s_n,
                                                'table': pd.DataFrame({'Property': list(sub_sub_n_data.keys()),'Value': list(sub_sub_n_data.values())})
                                            })

                                else:
                                    nested_prop_list.append({
                                        'name': n,
                                        'sub_name': s_n,
                                        'sub_sub_name': None,
                                        'table': pd.DataFrame({'Property': list(sub_n_data.keys()),'Value': list(sub_n_data.values())})
                                    })

                            else:
                                # Cell_States and Cell_Subtypes or Main_Cell_Types for Manual ROIs
                                nested_prop_list.append({
                                    'name': n,
                                    'sub_name': s_n,
                                    'sub_sub_name': None,
                                    'table': pd.DataFrame({'Property': list(sub_n_data.keys()), 'Value': list(sub_n_data.values())})
                                })

                    non_nested_sub_props = [i for i in n_prop_data if not type(n_prop_data[i])==dict]
                    if len(non_nested_sub_props)>0:
                        # Main_Cell_Types 
                        nested_prop_list.append({
                            'name': n,
                            'sub_name': None,
                            'sub_sub_name': None,
                            'table': pd.DataFrame({'Property': non_nested_sub_props,'Value': [n_prop_data[i] for i in non_nested_sub_props]})
                        })

            # popup divs
            accordion_children.append(
                dbc.AccordionItem([
                    html.Div([
                        make_dash_table(all_properties_df)
                    ])
                ],title = 'Other Properties')
            )

            if len(nested_prop_list)>0:

                # Start from the bottom and go up as opposed to top-down
                trunk_props = [i for i in nested_prop_list if not i['sub_sub_name'] is None]
                if len(trunk_props)>0:
                    unique_names = np.unique([i['name'] for  i in trunk_props]).tolist()
                    for u_n in unique_names:
                        shared_name_trunk = [i for i in trunk_props if i['name']==u_n]

                        shared_name_sub = np.unique([i['sub_name'] for i in shared_name_trunk]).tolist()
                        sub_accordion_list = []
                        for u_sub in shared_name_sub:
                            sub_prop_data = [i for i in shared_name_trunk if i['sub_name']==u_sub]
                            sub_tab_list = []
                            for sub_tab in sub_prop_data:
                                sub_tab_data = sub_tab['table']
                                sub_tab_data = sub_tab_data[sub_tab_data['Value']!=0]

                                if not sub_tab_data.empty:
                                    sub_tab_list.append(
                                        dbc.Tab(
                                            html.Div(
                                                make_pie_chart(sub_tab_data) if u_sub in pie_chart_features else make_dash_table(sub_tab_data)
                                            ),
                                            label = sub_tab['sub_sub_name']
                                        )
                                    )

                            # Checking if any other features are in this sub_name
                            other_sub_name_props = [i for i in nested_prop_list if i['sub_name']==u_sub and i['sub_sub_name'] is None]
                            for o_sub in other_sub_name_props:
                                o_sub_data = o_sub['table']
                                o_sub_data = o_sub_data[o_sub_data['Value']!=0]
                                if not o_sub_data.empty:
                                    sub_accordion_list.append(
                                        dbc.AccordionItem(
                                            html.Div(
                                                make_pie_chart(o_sub_data) if u_sub in pie_chart_features else make_dash_table(o_sub_data)
                                            ),
                                            title = u_sub
                                        )
                                    )

                            if len(sub_tab_list)>0:
                                sub_accordion_list.append(
                                    dbc.AccordionItem(
                                        dbc.Tabs(
                                            sub_tab_list
                                        ),
                                        title = u_sub
                                    )
                                )
                        
                        # Getting shared name props the no sub_sub_name
                        other_name_props = [i for i in nested_prop_list if i['name']==u_n and i['sub_sub_name'] is None and not i['sub_name'] is None]
                        for o_n_prop in other_name_props:
                            o_prop_data = o_n_prop['table']
                            o_prop_data = o_prop_data[o_prop_data['Value']!=0]
                            if not o_prop_data.empty:
                                sub_accordion_list.append(
                                    dbc.AccordionItem(
                                        html.Div(
                                            make_pie_chart(o_prop_data) if o_n_prop['sub_name'] in pie_chart_features else make_dash_table(o_prop_data)
                                        ),
                                        title = o_n_prop['sub_name']
                                    )
                                )

                        # Adding remainder props and adding the AccordionItem to the main list
                        l_props = [i for i in nested_prop_list if i['name']==u_n and i['sub_name'] is None]
                        if len(l_props)>0:
                            other_data_tab_list = []
                            for l_idx,l in enumerate(l_props):
                                # Not sure which properties will actually fall under here but need to make sure everything is gotten prior to adding to the accordion_children list
                                l_data = l['table']
                                l_data = l_data[l_data['Value']!=0]
                                if not l_data.empty:
                                    other_data_tab_list.append(
                                        dbc.Tab(
                                            html.Div(
                                                make_pie_chart(l_data) if l['name'] in pie_chart_features else make_dash_table(l_data)
                                            ),
                                            label = f'{u_n} Data {l_idx}'
                                        )
                                    )
                            
                            accordion_children.append(
                                dbc.AccordionItem([
                                    html.Div(
                                        dbc.Tabs(other_data_tab_list) if len(other_data_tab_list)>0 else html.Div()
                                    ),
                                    html.Div(
                                        dbc.Accordion(sub_accordion_list) if len(sub_accordion_list)>0 else html.Div()
                                    )
                                ], title = u_n)
                            )
                        else:

                            accordion_children.append(
                                dbc.AccordionItem([
                                    html.Div(
                                        dbc.Accordion(sub_accordion_list) if len(sub_accordion_list)>0 else html.Div()
                                    )
                                ], title = u_n)
                            )

                    # Getting the leftover properties
                    remainder_props = [i for i in nested_prop_list if i['name'] not in unique_names]
                    if len(remainder_props)>0:
                        for r in remainder_props:
                            r_data = r['table']
                            r_data = r_data[r_data['Value']!=0]
                            if not r_data.empty:
                                accordion_children.append(
                                    dbc.AccordionItem([
                                        html.Div(
                                            make_pie_chart(r_data) if r['name'] in pie_chart_features else make_dash_table(r_data)
                                        )
                                    ], title = r['name'])
                                )

                else:
                    # Now getting the sub_properties
                    branch_props = [i for i in nested_prop_list if not i['sub_name'] is None]
                    if len(branch_props)>0:
                        unique_names = np.unique([i['name'] for i in branch_props]).tolist()

                        for u_n in unique_names:
                            u_n_list = [i for i in branch_props if i['name']==u_n]
                            sub_tab_list = []
                            for s_u_n in u_n_list:
                                s_u_n_data = s_u_n['table']
                                s_u_n_data = s_u_n_data[s_u_n_data['Value']!=0]
                                if not s_u_n_data.empty:
                                    sub_tab_list.append(
                                        dbc.Tab(
                                            html.Div(
                                                make_pie_chart(s_u_n_data) if u_n in pie_chart_features or s_u_n['sub_name'] in pie_chart_features else make_dash_table(s_u_n_data),
                                            ),
                                            label = s_u_n['sub_name']
                                        )
                                    )

                            # Adding main props
                            l_props = [i for i in nested_prop_list if i['name']==u_n and i['sub_name'] is None]
                            if len(l_props)>0:
                                other_data_tab_list = []
                                for l_idx,l in enumerate(l_props):
                                    l_data = l['table']
                                    l_data = l_data[l_data['Value']!=0]
                                    if not l_data.empty:
                                        other_data_tab_list.append(
                                            dbc.Tab(
                                                html.Div(
                                                    make_pie_chart(l_data) if l['name'] in pie_chart_features else make_dash_table(l_data)
                                                ),
                                                label = f'{u_n} Data {l_idx}'
                                            )
                                        )

                                accordion_children.append(
                                    dbc.AccordionItem([
                                        html.Div(
                                            dbc.Tabs(other_data_tab_list) if len(other_data_tab_list)>0 else html.Div()
                                        ),
                                        html.Div(
                                            dbc.Tabs(sub_tab_list) if len(sub_tab_list)>0 else html.Div()
                                        )
                                    ], title = u_n)
                                )

                            else:
                                accordion_children.append(
                                    dbc.AccordionItem([
                                        html.Div(
                                            dbc.Tabs(
                                                sub_tab_list
                                            )
                                            if len(sub_tab_list)>0 else []
                                        )
                                    ], title = u_n)
                                )

                        # All other "name" props
                        remainder_props = [i for i in nested_prop_list if i['name'] not in unique_names]
                        if len(remainder_props)>0:
                            for r in remainder_props:
                                r_data = r['table']
                                r_data = r_data[r_data['Value']!=0]
                                if not r_data.empty:
                                    accordion_children.append(
                                        dbc.AccordionItem([
                                            html.Div(
                                                make_pie_chart(r_data) if r['name'] in pie_chart_features else make_dash_table(r_data)
                                            )
                                        ], title = r['name'])
                                    )
                        

                    else:
                        # Only has main props
                        for l in nested_prop_list:
                            l_data = l['table']
                            l_data = l_data[l_data['Value']!=0]
                            if not l_data.empty:
                                accordion_children.append(
                                    dbc.AccordionItem(
                                        html.Div(
                                            make_pie_chart(l['table']) if l['name'] in pie_chart_features else make_dash_table(l['table'])
                                        ),
                                        title = l['name']
                                    )
                                )

            if 'unique_index' in ftu_click['properties']['user']:
                #TODO: Store these user ftu labels somewhere else
                add_labels_children = self.layout_handler.get_user_ftu_labels(self.wsi,ftu_click)

                accordion_children.append(
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
                )
            
            popup_div = html.Div([
                dbc.Accordion(
                    children = accordion_children
                )
            ],style={'display':'inline-block'})

            return popup_div

        else:
            return no_update
        
    def gen_cyto(self,cell_val,table):
        """
        Generating cytoscape for selected cell type (referenced in self.update_cell_hierarchy)
        """
        cyto_elements = []
        
        table = pd.DataFrame.from_records(table)
        # Getting all the rows that contain these sub-types
        if not cell_val in self.cell_names_key:
            if 'CT/1/LABEL' in table.columns.tolist():
                table_data = table.dropna(subset=['CT/1/LABEL'])
                cell_data = table_data[table_data['CT/1/LABEL'].isin([cell_val])]
            else:
                table_data = table.dropna(subset=['CT/1'])
                cell_data = table_data[table_data['CT/1'].isin([cell_val])]
            cell_types_table = cell_data.filter(regex=self.node_cols['Cell Types']['abbrev']).dropna(axis=1)
            cell_subtypes = []
            unique_subtypes = []
            if any(['LABEL' in i for i in cell_types_table.columns.tolist()]):
                for c in [i for i in cell_types_table.columns.tolist() if 'LABEL' in i]:
                    unique_subtype_col = cell_types_table[c].unique().tolist()
                    for u_s in unique_subtype_col:
                        if not u_s in unique_subtypes:
                            cell_subtypes.append(
                                {
                                    'col': c.replace('/LABEL',''),
                                    'val': u_s
                                }
                            )
                            unique_subtypes.append(u_s)
                
            else:
                for c in [i for i in cell_types_table.columns.tolist() if len(i.split('/'))==2]:
                    unique_subtypes_col = cell_types_table[c].unique().tolist()
                    for u_s in unique_subtypes_col:
                        if not u_s in unique_subtypes:
                            cell_subtypes.append(
                                {
                                    'col': c,
                                    'val': u_s
                                }
                            )
                            unique_subtypes.append(u_s)
            
            main_cell_col = cell_subtypes[unique_subtypes.index(cell_val)]['col']

        else:
            table_data = table.dropna(subset=['CT/1/ABBR'])
            main_cell_col = 'CT/1'

            cell_subtypes = []
            for u_s in self.cell_graphics_key[self.cell_names_key[cell_val]]['subtypes']:
                cell_subtypes.append(
                    {
                        'col': 'CT/1',
                        'val': u_s
                    }
                )
            cell_data = table_data[table_data['CT/1/ABBR'].isin([i['val'] for i in cell_subtypes])]
        
        # Getting the anatomical structures for this cell type
        as_data = cell_data.filter(regex=self.node_cols['Anatomical Structure']['abbrev']).dropna(axis=1)
        an_start_y = self.node_cols['Anatomical Structure']['y_start']
        as_col_vals = as_data.columns.values.tolist()

        if any(['LABEL' in i for i in as_col_vals]):
            as_col_vals = [i for i in as_col_vals if 'LABEL' in i]
        else:
            as_col_vals = [i for i in as_col_vals if len(i.split('/'))==2]

        # Adding main cell node to connect edges to
        cyto_elements.append(
            {
                'data':{
                    'id':'Main_Cell',
                    'label':cell_val,
                    'url':'./assets/cell.png',
                    'col': main_cell_col
                },
            'classes': 'CT',
            'position':{
                'x':self.node_cols['Cell Types']['x_start'],
                'y':self.node_cols['Cell Types']['y_start']
            },
        }
        )

        for idx,col in enumerate(as_col_vals):
            as_value = as_data[col].tolist()[0]
            cyto_elements.append(
                {
                'data':{
                        'id':col,
                        'label':as_value,
                        'url':'./assets/kidney.png',
                        'col': col.replace('/LABEL','')
                    },
                'classes':'AS',
                'position':{
                    'x':self.node_cols['Anatomical Structure']['x_start'],
                    'y':an_start_y
                    }
                }
            )
            
            if idx>0:
                cyto_elements.append(
                    {'data':{'source':as_col_vals[idx-1],'target':col}}
                )
            an_start_y+=75
        
        last_struct = col
        cyto_elements.append(
            {'data':{'source':last_struct,'target':'Main_Cell'}}
        )
        
        cell_start_y = self.node_cols['Cell Types']['y_start']
        gene_start_y = self.node_cols['Genes']['y_start']

        # Iterating through cell subtypes
        unique_genes = []
        for idx_1,c in enumerate(cell_subtypes):

            # Finding row which matches this cell subtype
            if cell_val in self.cell_names_key:
                matching_rows = table_data.dropna(subset=['CT/1/ABBR'])
                matching_rows = matching_rows[matching_rows['CT/1/ABBR'].astype(str).str.match(c['val'])]
            else:
                matching_rows = table_data.dropna(subset=[c["col"]])
                matching_rows = matching_rows[matching_rows[c['col']].astype(str).str.match(c['val'])]

            if not matching_rows.empty:
                cell_start_y += 75
                # Adding subtype node to cytoscape
                cyto_elements.append(
                    {
                        'data': {
                            'id': f'ST_{idx_1}',
                            'label': c['val'],
                            'url': './assets/cell.png',
                            'col': c['col'].replace('/LABEL','')
                        },
                        'classes': 'CT',
                        'position': {
                            'x': self.node_cols['Cell Types']['x_start'],
                            'y': cell_start_y
                        }
                    }
                )

                # Adding edge from main cell to subtype node
                cyto_elements.append(
                    {
                        'data': {
                            'source': 'Main_Cell',
                            'target': f'ST_{idx_1}'
                        }
                    }
                )


                genes = matching_rows.filter(regex=self.node_cols['Genes']['abbrev']).dropna(axis=1)
                gene_col_vals = genes.columns.values.tolist()
                if any(['LABEL' in i for i in gene_col_vals]):
                    gene_col_vals = [i for i in gene_col_vals if 'LABEL' in i]
                else:
                    gene_col_vals = [i for i in gene_col_vals if len(i.split('/'))==2]

                gene_col_vals = np.unique(gene_col_vals).tolist()
                if len(gene_col_vals)>0:
                    for g_idx,g_col in enumerate(gene_col_vals):
                        gene_label = genes[g_col].tolist()[0]
                        if not gene_label in unique_genes:
                            unique_genes.append(gene_label)

                            # Adding new gene node to cytoscape
                            cyto_elements.append(
                                {
                                    'data':{
                                        'id':f'G_{unique_genes.index(gene_label)}',
                                        'label':gene_label,
                                        'url':'./assets/gene.png',
                                        'col': g_col.replace('/LABEL','')
                                    },
                                    'classes':'G',
                                    'position':{'x':self.node_cols['Genes']['x_start'],'y':gene_start_y}
                                }
                            )
                            gene_start_y+=75

                        # Adding edge between gene node and cell subtype node
                        cyto_elements.append(
                            {
                                'data': {
                                    'source': f'ST_{idx_1}',
                                    'target': f'G_{unique_genes.index(gene_label)}'
                                }
                            }
                        )


        return cyto_elements

    def get_cyto_data(self,clicked, organ_store):
        """
        Getting information for clicked node in cell hierarchy cytoscape vis.
        """
        if not clicked is None:
            
            organ_store = json.loads(organ_store)
            table = pd.DataFrame.from_records(organ_store['table'])
            if organ_store['organ']=='kidney':
                if 'CT' in clicked['col']:
                    if not clicked['id']=='Main_Cell':
                        table_data = table.dropna(subset=['CT/1/ABBR'])
                        table_data = table_data[table_data['CT/1/ABBR'].astype(str).str.match(clicked['label'])]
                    else:
                        table_data = pd.DataFrame()
                else:
                    if clicked['col']+'/LABEL' in table.columns.tolist():
                        table_data = table.dropna(subset=[clicked['col']+'/LABEL'])
                        table_data = table_data[table_data[clicked['col']+'/LABEL'].astype(str).str.match(clicked['label'])]
                    else:
                        table_data = table.dropna(subset=[clicked['col']])
                        table_data = table_data[table_data[clicked['col']].astype(str).str.match(clicked['label'])]
            else:
                if clicked['col']+'/LABEL' in table.columns.tolist():
                    table_data = table.dropna(subset=[clicked['col']+'/LABEL'])
                    table_data = table_data[table_data[clicked['col']+'/LABEL'].astype(str).str.match(clicked['label'])]
                else:
                    table_data = table.dropna(subset=[clicked['col']])
                    table_data = table_data[table_data[clicked['col']].astype(str).str.match(clicked['label'])]

            label = clicked['label']

            if not table_data.empty:
                id = table_data[f'{clicked["col"]}/ID'].tolist()[0]

                if 'CT' in clicked['col']:
                    base_url = self.node_cols['Cell Types']['base_url']
                    new_url = base_url+str(id).replace('CL:','')
                elif 'AS' in clicked['col']:
                    base_url = self.node_cols['Anatomical Structure']['base_url']
                    new_url = base_url+str(id).replace('UBERON:','')
                elif 'BGene' in clicked['col']:
                    base_url = self.node_cols['Genes']['base_url']
                    new_url = base_url+str(id).replace('HGNC:','')
                else:
                    new_url = ''

                if f'{clicked["col"]}/NOTES' in table_data.columns.tolist():
                    notes = table_data[f'{clicked["col"]}/NOTES'].tolist()[0]
                    if not notes:
                        notes = 'No notes.'
                else:
                    notes = 'No notes.'
            else:
                raise exceptions.PreventUpdate

        else:
            label = ''
            id = ''
            notes = ''
            new_url = ''

        return f'Label: {label}', dcc.Link(f'ID: {id}', href = new_url), f'Notes: {notes}'
    
    def load_new_slide(self,slide_id,new_interval,modal_is_open,slide_info_store_data, user_data_store):
        """
        Progress indicator for loading a new WSI and annotations
        """
        modal_open = True
        modal_children = []
        disable_slide_load = False

        slide_info_store_data = json.loads(slide_info_store_data)
        user_data_store = json.loads(user_data_store)

        #TODO: Change the active_tab to overlays when loading a new slide just to reset other tabs' content
        if ctx.triggered_id=='slide-select':

            if not slide_id:
                raise exceptions.PreventUpdate

            print(f'Getting info for slide: {slide_id}')
            slide_info = self.dataset_handler.get_item_info(slide_id)

            if 'Spatial Omics Type' in slide_info['meta']:
                slide_type = slide_info['meta']['Spatial Omics Type']
            else:
                slide_type = 'Regular'

            # Storing some info in a data store component (erases upon refresh)
            slide_annotation_info = {
                'slide_info': slide_info,
                'slide_type': slide_type,
                'overlay_prop': None,
                'cell_vis_val': 0.5,
                'filter_vals': None,
                'current_channels': {}
            }

            slide_annotation_info = slide_annotation_info | self.slide_handler.get_slide_map_data(slide_id,user_data_store)

            annotation_info = slide_annotation_info['annotations']
            slide_annotation_info['ftu_colors'] = {
                i['annotation']['name']: '#%02x%02x%02x' % (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                for i in annotation_info
            }

            n_annotations = len(annotation_info)
            if n_annotations>0:
                first_annotation = annotation_info[0]['annotation']['name']
                slide_annotation_info['loading_annotation'] = annotation_info[0]['_id']

                slide_info_store_data = json.dumps(slide_annotation_info)
                
                # Starting the get_annotation_geojson function on a new thread
                #TODO: Change name of the thread to the annotation id
                new_thread = threading.Thread(target = self.slide_handler.get_annotation_geojson, name = first_annotation, args = [slide_annotation_info, user_data_store, 0])
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

                slide_info_store_data = json.dumps(slide_annotation_info)
                         
                modal_children = [
                    html.Div(
                        dbc.ModalHeader(html.H4(f'Loading Slide: {slide_info["name"]}'))
                    )
                ]
        
        elif ctx.triggered_id=='slide-load-interval':
            
            if not new_interval or len(list(slide_info_store_data.keys()))==0:
                raise exceptions.PreventUpdate
            
            if len(slide_info_store_data['annotations'])>0:

                # Checking if previous annotation is complete
                all_annotation_ids = [i['_id'] for i in slide_info_store_data['annotations']]
                previous_annotation_id = slide_info_store_data['loading_annotation']
                
                if not previous_annotation_id == 'done':
                    previous_annotation_name = slide_info_store_data['annotations'][all_annotation_ids.index(previous_annotation_id)]['annotation']['name']

                    # Getting names of current threads to see if that one is still active
                    thread_names = [i.name for i in threading.enumerate()]

                    if not previous_annotation_name in thread_names:

                        next_ann_idx = all_annotation_ids.index(previous_annotation_id) + 1

                        n_annotations = len(all_annotation_ids)
                        slide_name = slide_info_store_data['slide_info']['name']

                        if next_ann_idx<len(all_annotation_ids):

                            next_annotation = slide_info_store_data['annotations'][next_ann_idx]['annotation']['name']
                            slide_info_store_data['loading_annotation'] = slide_info_store_data['annotations'][next_ann_idx]['_id']
                            
                            # Starting the get_annotation_geojson function on a new thread
                            #TODO: Change name of the thread to the annotation id
                            new_thread = threading.Thread(target = self.slide_handler.get_annotation_geojson, name = next_annotation, args = [slide_info_store_data, user_data_store,next_ann_idx])
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
                            slide_info_store_data['loading_annotation'] = 'done'

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
                        old_ann_name = slide_info_store_data['annotations'][old_ann_idx]['annotation']['name']
                        n_annotations = len(all_annotation_ids)
                        slide_name = slide_info_store_data['slide_info']['name']

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

                slide_info_store_data = json.dumps(slide_info_store_data)

            else:
                slide_info_store_data = json.dumps(slide_info_store_data)
                disable_slide_load = True
                modal_open = False

        return modal_open, modal_children, slide_info_store_data, disable_slide_load

    def ingest_wsi(self,load_slide_done,slide_info_store):
        """
        Populating slide visualization components after loading annotation and tile information
        """
        slide_info_store = json.loads(slide_info_store)
        load_slide_done = get_pattern_matching_value(load_slide_done)
        if load_slide_done:

            # Updating overlays colors according to the current cell
            hex_color_key = self.update_hex_color_key(None,slide_info_store)

            special_overlays_opts = self.layout_handler.gen_special_overlay_opts(slide_info_store)
            visualizable_properties_list = self.slide_handler.get_properties_list(slide_info_store)

            if slide_info_store['slide_type']=='CODEX':

                # Enabling cell annotation tab
                cell_annotation_tab_disable = False
                slide_tile_layer = []

            else:
                cell_annotation_tab_disable = True
                slide_tile_layer = [
                    dl.TileLayer(
                        id = f'slide-tile{np.random.randint(0,100)}',
                        url = slide_info_store['tile_url'],
                        tileSize = slide_info_store['tile_dims'][0],
                        maxNativeZoom = slide_info_store['zoom_levels']-1,
                        bounds = [[0,0],slide_info_store['map_bounds']]
                    )
                ]

            map_center = {
                'center': slide_info_store['map_bounds'],
                'zoom': 3,
                'transition':'flyTo'
            }

            # Adding the layers to be a property for the edit_control callback
            current_overlays = self.slide_handler.generate_annotation_overlays(
                slide_info = slide_info_store,
                style_handler = self.ftu_style_handle,
                filter_handler = self.ftu_filter,
                color_key = hex_color_key
            )

            # Removes manual ROIs added via dl.EditControl
            remove_old_edits = [{
                'mode':'remove',
                'n_clicks':0,
                'action':'clear all'
            }]

            # Populating FTU boundary options:
            combined_colors_dict = {}
            for f in slide_info_store['ftu_colors']:
                combined_colors_dict[f] = {'color':slide_info_store['ftu_colors'][f]}
            
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

            new_layer_control = dl.LayersControl(
                id = {'type': 'layer-control','index': random.randint(0,100)},
                children = current_overlays
            )

            # Clearing marker-add-div
            new_marker_div_children = []
            # Clearing marker-add-geojson
            new_marker_add_geojson = json.dumps({'type': 'FeatureCollection','features': []})

            return slide_tile_layer, new_layer_control, remove_old_edits, map_center, visualizable_properties_list, boundary_options_children, special_overlays_opts, cell_annotation_tab_disable, new_marker_div_children, new_marker_add_geojson
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
            feature_data = pd.DataFrame.from_dict(cluster_data_store['feature_data'],orient='index')
        else:
            feature_data = None
        
        if not cluster_data_store['umap_df'] is None:
            umap_df = pd.DataFrame.from_dict(cluster_data_store['umap_df'],orient='index')
        else:
            umap_df = None

        if not cluster_data_store['clustering_data'] is None:   
            clustering_data = pd.DataFrame.from_dict(cluster_data_store['clustering_data'],orient='index')
        else:
            clustering_data = None

        if ctx.triggered_id=='gen-plot-butt':
            #TODO: get rid of self.reports_generated, potential for user overlap
            self.reports_generated = {}
            report_active_tab = 'feat-summ-tab'

            # Enabling download plot data button
            download_plot_disable = False

            if clustering_data.empty:
                print(f'Getting new clustering data')
                clustering_data = self.dataset_handler.load_clustering_data(user_data_store)
                cluster_data_store['clustering_data'] = clustering_data.to_dict('index')

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

                        folder_names.append(self.dataset_handler.get_folder_name(folderId))
                    
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
                        item_folder = self.dataset_handler.get_folder_name(item_data['folderId'])
                        
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
                
                cluster_data_store['feature_data'] = feature_data.to_dict('index')

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

                cluster_data_store['feature_data'] = feature_data.to_dict('index')

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

                cluster_data_store['feature_data'] = feature_data.to_dict('index')

                umap_df = gen_umap(feature_data, feature_names, ['Hidden','label','Main_Cell_Types','Cell_States'])
                # Saving this so we can update the label separately without re-running scaling or reduction

                cluster_data_store['umap_df'] = umap_df.to_dict('index')

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
            #TODO: get rid of self.reports_generated, potential for user overlap
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
                        label_data = [0]*(max([int(i) for i in list(feature_data.index)])+1)
                        for idx in list(feature_data.index):
                            label_data[int(idx)] = float(feature_data.loc[idx,label])
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

                            folder_names.append(self.dataset_handler.get_folder_name(folderId))
                        
                        label_data = [folder_names[unique_ids.index(i)] for i in sample_ids]

                    elif label in self.cell_names_key:
                        # For labeling with cell type
                        label_data = [float(i[self.cell_names_key[label]]) if self.cell_names_key[label] in i else 0 for i in clustering_data['Main_Cell_Types'].tolist()]

                    else:
                        label = 'FTU'
                        label_data = clustering_data[label].tolist()

                #TODO: original index is not preserved after reading it from "records"
                label_data = [label_data[i] for i in list(feature_data.index)]
                # Needs an alignment step going from the full clustering_data to the na-dropped and filtered feature_data
                # feature_data contains the features included in the current plot, label, Hidden, Main_Cell_Types, and Cell_States
                # So for a violin plot the shape should be nX5
                feature_data = feature_data.drop(columns = ['label'])
                feature_data['label'] = label_data

                if type(label_data[0]) in [int,float]:
                    feature_data['label'] = feature_data['label'].astype(float)

                cluster_data_store['feature_data'] = feature_data.to_dict('index')

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

                    cluster_data_store['umap_df'] = umap_df.to_dict('index')
                    
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

    def grab_image(self,sample_info, user_info):
        """
        Getting image for clicked point in morphological clustering tab. (referenced in self.update_selected)
        """
        if len(sample_info)>100:
            sample_info = sample_info[0:100]

        img_list = []
        img_dims = np.zeros((len(sample_info),2))
        for idx,s in enumerate(sample_info):
            image_region = np.array(self.dataset_handler.get_annotation_image(s['Slide_Id'],user_info,s['Bounding_Box']))
            
            # Resizing images so that each one is the same size
            img_list.append(resize(np.array(image_region),output_shape=(512,512,3),anti_aliasing=True))
            #TODO: Find some efficient way to pad images equally up to the max size and then resize to 512x512x3
            #img_list.append(image_region)
            #img_dims[idx,:] += np.shape(image_region).tolist()[0:2]
        
        return img_list        

    def update_selected(self,click,selected,cluster_data_store,user_store_data,slide_info_store):
        """
        Getting cell/state information and image for selected point in plot.
        """

        cluster_data_store = json.loads(cluster_data_store)
        user_store_data = json.loads(user_store_data)
        slide_info_store = json.loads(slide_info_store)
        feature_data = pd.DataFrame.from_dict(cluster_data_store['feature_data'],orient = 'index')

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
            if not len(list(slide_info_store.keys()))==0:
                if any([i==slide_info_store['slide_info']['name'] for i in slide_names]) and len(slide_names)>1:
                    selected_image_info = [
                        dbc.Row([
                            dbc.Col(
                                dbc.Button(
                                    f'Add All Markers ({slide_names.count(slide_info_store["slide_info"]["name"])})',
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
                
                if not len(list(slide_info_store.keys()))==0:
                    if j == slide_info_store['slide_info']['name']:
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

            current_image = self.grab_image(sample_info,user_store_data)
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
        feature_data = pd.DataFrame.from_dict(cluster_data_store['feature_data'],orient='index')
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

    def add_manual_roi(self,new_geojson_list, slide_info_store):
        """
        Adding a rectangle, polygon, or marker annotation to the current image
        """
        slide_info_store = json.loads(slide_info_store)

        if not len(list(slide_info_store.keys()))==0:
            if 'overlay_prop' in slide_info_store:
                overlay_prop = slide_info_store['overlay_prop']
            else:
                overlay_prop = None
            
            if 'hex_color_key' in slide_info_store:
                hex_color_key = slide_info_store['hex_color_key']
            else:
                hex_color_key = {}

            try:
                # Used for pattern-matching callbacks
                triggered_id = ctx.triggered_id['type']
            except TypeError:
                # Used for normal callbacks
                triggered_id = ctx.triggered_id

            if not ctx.triggered[0]['value']:
                return [no_update], no_update, no_update

            if triggered_id == 'edit-control':
                
                if not new_geojson_list is None:
                    new_roi = None
                    for new_geo in new_geojson_list:
                        if not new_geo is None:
                            if len(new_geo['features'])>0:
                                
                                # Adding each manual annotation iteratively (good for if annotations are edited or deleted as well as for new annotations)
                                for geo in new_geo['features']:

                                    if not geo['properties']['type'] == 'marker':

                                        new_roi = {'type':'FeatureCollection','features':[geo]}

                                        agg_properties, polygon_properties = self.slide_handler.spatial_aggregation(shape(geo['geometry']),slide_info_store)
                                        new_roi['features'][0]['properties'] = {'user':agg_properties | {'name': 'Manual_ROI'} | polygon_properties}
                                        
                                        new_manual_roi_dict = {
                                                'geojson':new_roi,
                                            }
                                        
                                        slide_info_store['manual_ROIs'].append(new_manual_roi_dict)

                                    elif geo['properties']['type']=='marker':
                                        # Separate procedure for marking regions/FTUs with a marker
                                        new_marked = {'type':'FeatureCollection','features':[geo]}
                                        
                                        intersect_dict, intersect_poly = self.slide_handler.find_intersecting_ftu(
                                            shape(new_marked['features'][0]['geometry']),
                                            [i['annotation']['name'] for i in slide_info_store['annotations']],
                                            slide_info_store
                                        )

                                        # Checking which one is non-empty:
                                        intersect_count = [len(intersect_dict[i]) for i in intersect_dict]
                                        print(f'intersect_count: {intersect_count}')
                                        
                                        if not all([i==0 for i in intersect_count]):
                                            non_zeros = [list(intersect_dict.keys())[i] for i in range(len(intersect_count)) if not intersect_count[i]==0]
                                            for structure in non_zeros:
                                                overlap_polys = intersect_poly[structure]
                                                overlap_props = intersect_dict[structure]
                                                for o_idx in range(len(overlap_polys)):
                                                    # Getting the intersecting ROI geojson
                                                    if len(slide_info_store["marked_FTUs"])==0:
                                                        if triggered_id=='edit_control':
                                                            new_marked_roi = {
                                                                'type':'FeatureCollection',
                                                                'features':[
                                                                    {
                                                                        'type':'Feature',
                                                                        'geometry':{
                                                                            'type':'Polygon',
                                                                            'coordinates':[list(overlap_polys[o_idx].exterior.coords)],
                                                                        },
                                                                        'properties': {
                                                                            'user': overlap_props[o_idx]
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
                                                                            'coordinates':[list(overlap_polys[o_idx].exterior.coords)]
                                                                        },
                                                                        'properties':{
                                                                            'user': overlap_props[o_idx]
                                                                        }
                                                                    },
                                                                ]
                                                            }

                                                        slide_info_store['marked_FTUs'] = [{
                                                            'geojson':new_marked_roi,
                                                        }]

                                                    else:
                                                        slide_info_store['marked_FTUs'][0]['geojson']['features'].append(
                                                            {
                                                                'type':'Feature',
                                                                'geometry':{
                                                                    'type':'Polygon',
                                                                    'coordinates':[list(overlap_polys[o_idx].exterior.coords)],
                                                                },
                                                                'properties':{
                                                                    'user':overlap_props[o_idx]
                                                                }
                                                            }
                                                        )

                                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)

                                current_overlays = self.slide_handler.generate_annotation_overlays(
                                    slide_info = slide_info_store,
                                    style_handler=self.ftu_style_handle,
                                    filter_handler=self.ftu_filter,
                                    color_key = hex_color_key
                                )
                                
                                if new_roi:
                                    user_ann_tracking = json.dumps({ 'slide_name': slide_info_store['slide_info']["name"], 'item_id': slide_info_store['slide_info']['_id'] })
                                else:
                                    user_ann_tracking = no_update
                                
                            else:

                                # Clearing manual ROIs and reverting overlays
                                slide_info_store['manual_ROIs'] = []
                                slide_info_store['marked_FTUs'] = []

                                hex_color_key = self.update_hex_color_key(overlay_prop,slide_info_store)
                                user_ann_tracking = no_update

                                current_overlays = self.slide_handler.generate_annotation_overlays(
                                    slide_info = slide_info_store,
                                    style_handler = self.ftu_style_handle,
                                    filter_handler = self.ftu_filter,
                                    color_key = hex_color_key
                                )
                                                                                    
                        else:
                            
                            user_ann_tracking = no_update

                    return [current_overlays], user_ann_tracking
                else:
                    raise exceptions.PreventUpdate
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate

    def add_marker_from_cluster(self,mark_click,cluster_data_store,slide_info_store):
        """
        Marking structure in GeoJSON layer from selected points in plot (in the current slide)
        """

        cluster_data_store = json.loads(cluster_data_store)
        slide_info_store = json.loads(slide_info_store)
        if not cluster_data_store['feature_data'] is None and not len(list(slide_info_store.keys()))==0:
            feature_data = pd.DataFrame.from_dict(cluster_data_store['feature_data'],orient='index')
        else:
            feature_data = pd.DataFrame()
        if 'current_selected_samples' in cluster_data_store:
            current_selected_samples = cluster_data_store['current_selected_samples']

        # Adding marker(s) from graph returning geojson
        # index = 0 == mark all the samples in the current slide
        # index != 0 == mark a specific sample in the current slide

        if ctx.triggered[0]['value']:
            print(ctx.triggered_id)
            if ctx.triggered_id['index']==0:
                # Add marker for all samples in the current slide
                mark_geojson = {'type':'FeatureCollection','features':[]}

                # Iterating through all current selected samples
                # current_selected_samples is an index from feature_data
                current_selected_hidden = [feature_data['Hidden'].tolist()[i] for i in current_selected_samples]
                marker_bboxes = [i['Bounding_Box'] for i in current_selected_hidden if i['Slide_Id']==slide_info_store['slide_info']['_id']]
                
                slide_x_scale = slide_info_store['scale'][0]
                slide_y_scale = slide_info_store['scale'][1]
                marker_map_coords = [
                    [
                        [i[0]*slide_x_scale,i[1]*slide_y_scale],
                        [i[2]*slide_x_scale,i[3]*slide_y_scale]
                    ]
                    for i in marker_bboxes
                ]
                marker_center_coords = [
                    [
                        (i[0][0]+i[1][0])/2,
                        (i[0][1]+i[1][1])/2
                    ] 
                    for i in marker_map_coords
                ]
                
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

                slide_x_scale = slide_info_store['scale'][0]
                slide_y_scale = slide_info_store['scale'][1]

                # Pulling out one specific sample
                selected_bbox = feature_data['Hidden'].tolist()[current_selected_samples[ctx.triggered_id['index']-1]]['Bounding_Box']
                marker_map_coords = [
                    [selected_bbox[0]*slide_x_scale,selected_bbox[1]*slide_y_scale],
                    [selected_bbox[2]*slide_x_scale,selected_bbox[3]*slide_y_scale]
                ]
                marker_center_coords = [
                    (marker_map_coords[0][0]+marker_map_coords[1][0])/2,
                    (marker_map_coords[0][1]+marker_map_coords[1][1])/2
                ]
                
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

    def download_data(self,ann_click, cell_click, meta_click, manual_click, user_store_data,slide_bounds,slide_info_store_data, ann_format, cell_format):
        """
        Download data for external secondary analysis
        """
        user_store_data = json.loads(user_store_data)
        slide_info_store = json.loads(slide_info_store_data)

        ann_format = get_pattern_matching_value(ann_format)
        cell_format = get_pattern_matching_value(cell_format)
        
        if not any([i['value'] for i in ctx.triggered]) or len(list(slide_info_store.keys()))==0:
            raise exceptions.PreventUpdate
        
        if ctx.triggered_id['type']=='download-annotations-butt':
            if ctx.triggered_id['index']==0:
                return_dict = self.download_handler.prep_annotations(ann_format,slide_info_store,'all')
            else:
                current_region = slide_bounds

                return_dict = self.download_handler.prep_annotations(ann_format,slide_info_store,current_region)

            return [return_dict]

        elif ctx.triggered_id['type']=='download-cell-butt':
            if ctx.triggered_id['index']==0:
                # Grabbing cell info for everywhere
                return_dict = self.download_handler.prep_cell(cell_format,slide_info_store,'all')
            else:
                # Grabbing current region
                current_region = slide_bounds
                return_dict = self.download_handler.prep_cell(cell_format, slide_info_store, current_region)

            return [dcc.send_file(path = './assets/download/FUSION_Download.zip')]

        elif ctx.triggered_id['type']=='download-meta-butt':
            print(f'Downloading metadata')

            slide_metadata = json.dumps(slide_info_store['slide_info']['meta'])

            return [dict(content = slide_metadata, filename = f"{slide_info_store['slide_info']['name']}_Metadata.json")]
        
        elif ctx.triggered_id['type']=='download-manual-butt':
            print(f'Downloading manual rois')
            print(f'found: {len(slide_info_store["manual_ROIs"])} manual rois')
            if len(slide_info_store['manual_ROIs'])==0:
                raise exceptions.PreventUpdate
            else:

                manual_roi_geojson = json.dumps(slide_info_store['manual_ROIs']['geojson'])
                return [dict(content = manual_roi_geojson,filename = f'{slide_info_store["slide_info"]["name"]}_manual_rois.geojson')]

    def run_analysis(self,cli_name,cli_butt,slide_info_store):

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
            cli_results = 'Click "Run Job!" to do the thing!'

            # Getting current image region:
            wsi_coords = np.array(self.slide_handler.convert_map_coords(list(self.current_slide_bounds.exterior.coords)),slide_info_store)
            min_x = np.min(wsi_coords[:,0])
            min_y = np.min(wsi_coords[:,1])
            max_x = np.max(wsi_coords[:,0])
            max_y = np.max(wsi_coords[:,1])
            #TODO: Set maximum image size or pull from certain magnification level
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
            user_data_store['upload_check'] = {
                "WSI": False,
                "Omics": False
            }

            user_data_store['upload_type'] = 'Visium'

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
                        id = {'type':'wsi-files-upload-div','index':0},
                        children = [
                            dbc.Label('Upload your counts file here!'),
                            self.layout_handler.gen_info_button('Upload either an RDS file or h5ad file containing gene counts per-spot.'),
                            UploadComponent(
                                id = {'type':'wsi-files-upload','index':0},
                                uploadComplete=False,
                                baseurl=self.dataset_handler.apiUrl,
                                girderToken = user_data_store['token'],
                                parentId=parentId,
                                filetypes=['rds','h5ad']                 
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
        
        elif upload_type =='Regular':
            # Regular slide with no --omics

            self.prep_handler = Prepper(self.dataset_handler)
            user_data_store['upload_check'] = {
                'WSI': False
            }
            user_data_store['upload_type'] = 'Regular'

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

        elif upload_type == 'CODEX':
            # CODEX uploads include histology and multi-frame CODEX image (or just CODEX?)

            self.prep_handler = CODEXPrep(self.dataset_handler)
            user_data_store['upload_check'] = {
                'WSI': False,
                'Histology': False
            }
            user_data_store['upload_type'] = 'CODEX'

            upload_reqs = html.Div([
                dbc.Row([
                    html.Div(
                        id = {'type': 'wsi-upload-div','index':0},
                        children = [
                            dbc.Label('Upload CODEX Image Here!'),
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
                    ),
                    html.Div(
                        id = {'type':'wsi-files-upload-div','index':0},
                        children = [
                            dbc.Label('Upload Histology (H&E) Image Here!'),
                            UploadComponent(
                                id = {'type':'wsi-files-upload','index':0},
                                uploadComplete=False,
                                baseurl=self.dataset_handler.apiUrl,
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
                                'Kidney', 'Skin','Intestine','Lung','Other Organs'
                            ],
                            placeholder = 'Organ',
                            id = {'type':'organ-select','index':0}
                        )
                    )
                ])
            ])

        elif upload_type == 'Xenium':
            # Xenium uploads include an H&E image, a DAPI image, an alignment file (csv), and cell id/coordinates/group(anchors)(csv)

            self.prep_handler = XeniumPrep(self.dataset_handler)
            user_data_store['upload_check'] = {
                'WSI': False,
                'Morphology': False,
                'Alignment': False
            }
            user_data_store['upload_type'] = 'Xenium'

            upload_reqs = html.Div([
                dbc.Row([
                    html.Div(
                        id = {'type': 'wsi-upload-div','index':0},
                        children = [
                            dbc.Label('Upload H&E Histology Image Here!'),
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
                    ),
                    html.Div(
                        id = {'type':'wsi-files-upload-div','index':0},
                        children = [
                            dbc.Label('Upload DAPI (morphology) Image Here!'),
                            UploadComponent(
                                id = {'type':'wsi-files-upload','index':0},
                                uploadComplete = False,
                                baseurl = self.dataset_handler.apiUrl,
                                girderToken = user_data_store['token'],
                                parentId = parentId,
                                filetypes = ['svs','ndpi','scn','tiff','tif']
                            )
                        ]
                    ),
                    html.Div(
                        id = {'type':'wsi-files-upload-div','index':1},
                        children = [
                            dbc.Label('Upload Alignment File Here!'),
                            UploadComponent(
                                id = {'type':'wsi-files-upload','index':1},
                                uploadComplete = False,
                                baseurl = self.dataset_handler.apiUrl,
                                girderToken = user_data_store['token'],
                                parentId = parentId,
                                filetypes = ['csv','zip']
                            )
                        ]
                    )
                ],align='center'),
                dbc.Row([
                    dbc.Col(dbc.Label('Select Organ: ',html_for = {'type':'organ-select','index':0}),md=4),
                    dbc.Col(
                        dcc.Dropdown(
                            options = [
                                'Kidney', 'Skin', 'Intestine', 'Lung', 'Other Organs'
                            ],
                            placeholder = 'Organ',
                            id = {'type':'organ-select','index':0}
                        )
                    )
                ])
            ])
        
        user_data_store['upload_wsi_id'] = None
        user_data_store['upload_omics_id'] = None

        return upload_reqs, input_disabled, json.dumps(user_data_store)

    def girder_login(self,p_butt,create_butt,username,pword,email,firstname,lastname):

        create_user_children = []
        usability_signup_style = no_update
        usability_butt_style = no_update
        long_plugin_components = [no_update]

        if ctx.triggered_id=='login-submit':

            try:
                user_info, user_details = self.dataset_handler.authenticate(username,pword)
                user_id = user_details['_id']

                user_jobs = self.dataset_handler.get_user_jobs(user_id)
                if len(user_jobs)>0:
                    plugin_progress_div = []
                    for job_idx, job in enumerate(user_jobs):
                        plugin_name = job['title']
                        plugin_status = job['status']
                        
                        if plugin_status==3:
                            current_status_badge = dbc.Badge(
                                html.A('Complete'),
                                color = 'success',
                                id = {'type': 'job-status-badge','index': job_idx}
                            )
                        elif plugin_status==2:
                            current_status_badge = dbc.Badge(
                                html.A('Running'),
                                color = 'warning',
                                id = {'type': 'job-status-badge','index': job_idx}
                            )
                        elif plugin_status==4:
                            current_status_badge = dbc.Badge(
                                html.A('Failed'),
                                color = 'danger',
                                id = {'type': 'job-status-badge','index': job_idx}
                            )
                        else:
                            current_status_badge = dbc.Badge(
                                html.A(f'Unknown: ({plugin_status})'),
                                color = 'info',
                                id = {'type': 'job-status-badge','index': job_idx}
                            )

                        plugin_progress_div.append(
                            dbc.Card([
                                dbc.CardHeader([
                                    plugin_name,
                                    current_status_badge,
                                    html.I(
                                        className = 'bi bi-info-circle-fill me-5',
                                        id = {'type': 'job-status-button','index': job_idx}
                                    ),
                                    dbc.Popover(
                                        children = [
                                            html.Img(src = './assets/fusey_clean.svg',height=20,width=20),
                                            'Update job status and view logs'
                                        ],
                                        target = {'type': 'job-status-button','index': job_idx},
                                        body = True,
                                        trigger = 'hover'
                                    )
                                ]),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            html.Div(
                                                id = {'type': 'checked-job-logs','index': job_idx},
                                                children = [],
                                                style = {'maxHeight': '20vh','overflow':'scroll'}
                                            )
                                        )
                                    ),
                                    id = {'type': 'checked-job-collapse','index':job_idx},
                                    is_open = False
                                )
                            ])
                        )

                else:
                    plugin_progress_div = ['No jobs run yet!']
                
                long_plugin_components = [html.Div(
                    children = plugin_progress_div,
                    style = {'maxHeight': '30vh','overflow':'scroll'}
                )]


                button_color = 'success'
                button_text = 'Success!'
                logged_in_user = [
                    f'Welcome: {username}',
                    dbc.Badge(
                        html.A('Jobs'),
                        color = 'success' if len(user_jobs)>0 else 'secondary',
                        id = 'long-plugin-butt'
                    )
                ]
                upload_disabled = False

                user_data_output = user_details
                user_data_output['token'] = self.dataset_handler.user_token

                if not user_info is None:
                    usability_signup_style = {'display':'none'}
                    usability_butt_style = {'marginLeft':'5px','display':'inline-block'}

                user_data_output = json.dumps(user_data_output)

            except girder_client.AuthenticationError:

                button_color = 'warning'
                button_text = 'Login Failed'
                logged_in_user = [
                    'Welcome: fusionguest',
                    dbc.Badge(
                        html.A('Jobs'),
                        color = 'secondary',
                        id = 'long-plugin-butt'
                    )
                ]
                upload_disabled = True

                user_data_output = no_update

            return button_color, button_text, logged_in_user, upload_disabled, create_user_children, json.dumps({'user_id': username}), [usability_signup_style],[usability_butt_style], user_data_output, long_plugin_components
        
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
                logged_in_user = [
                    'Welcome: fusionguest',
                    dbc.Badge(
                        html.A('Jobs'),
                        color = 'secondary',
                        id = 'long-plugin-butt'
                    )
                ]
                upload_disabled = True

                return button_color, button_text, logged_in_user, upload_disabled, create_user_children, no_update, [usability_signup_style],[usability_butt_style], no_update, long_plugin_components

                return button_color, button_text, logged_in_user, upload_disabled, create_user_children, no_update, [usability_signup_style],[usability_butt_style], user_data_output

            else:
                create_user_children = no_update
                try:
                    user_info, user_details = self.dataset_handler.create_user(username,pword,email,firstname,lastname)

                    button_color = 'success',
                    button_text = 'Success!',
                    logged_in_user = [
                        f'Welcome: {username}',
                        dbc.Badge(
                            html.A('Jobs'),
                            color = 'secondary',
                            id = 'long-plugin-butt'
                        )
                    ]
                    upload_disabled = False

                    user_data_output = {
                        'user_info': user_info,
                        'user_details': user_details,
                        'token': self.dataset_handler.user_token
                    }

                    user_data_output = json.dumps(user_data_output)

                    if not user_info is None:
                        usability_signup_style = {'display':'none'}
                        usability_butt_style = {'marginLeft':'5px','display':'inline-block'}

                    user_data_output = json.dumps(user_data_output)

                except girder_client.AuthenticationError:

                    button_color = 'warning'
                    button_text = 'Login Failed'
                    logged_in_user = [
                        f'Welcome: fusionguest',
                        dbc.Badge(
                            html.A('Jobs'),
                            color = 'secondary',
                            id = 'long-plugin-butt'
                        )
                    ]
                    upload_disabled = True

                    user_data_output = no_update
            
                return button_color, button_text, logged_in_user, upload_disabled, create_user_children, json.dumps({'user_id': username}), [usability_signup_style],[usability_butt_style], user_data_output, long_plugin_components

        else:
            raise exceptions.PreventUpdate

    def upload_data(self,wsi_file,omics_file,wsi_file_flag,omics_file_flag, user_data_store):
        """
        User uploads either a WSI or an associated file
        """
        user_data_store = json.loads(user_data_store)

        if not ctx.triggered[0]['value']:
            raise exceptions.PreventUpdate

        if ctx.triggered_id['type']=='wsi-upload':
            # This ctx.triggered component has a "fileTypeFlag" and "uploadComplete" trigger
            wsi_file_flag = get_pattern_matching_value(wsi_file_flag)
            if not user_data_store['upload_wsi_id']:
                # If a wsi has not yet been uploaded
                if not wsi_file_flag:
                    # This is a valid file type for wsi uploads (will be None if untriggered or True for unaccepted file type)
                    prop_id = ctx.triggered[0]['prop_id'].split('.')[-1]
                    if prop_id=='uploadComplete':
                        # Getting the uploaded item id
                        upload_wsi_id = self.dataset_handler.get_new_upload_id(user_data_store['latest_upload_folder']['id'])
                        user_data_store['upload_wsi_id'] = upload_wsi_id

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
                        # upload incomplete
                        wsi_upload_children = [no_update]*len(ctx.outputs_list[2])
                else:
                    # This is for an invalid file type
                    wsi_upload_children = [
                        html.Div([
                            dbc.Alert('WSI Upload Failure! Accepted file types include svs, ndpi, scn, tiff, and tif',color='danger'),
                            UploadComponent(
                                id = {'type':'wsi-upload','index':0},
                                uploadComplete=False,
                                baseurl=self.dataset_handler.apiUrl,
                                girderToken=user_data_store['token'],
                                parentId=user_data_store['latest_upload_folder']['id'],
                                filetypes=['svs','ndpi','scn','tiff','tif']                      
                            )
                        ])
                    ]
            else:
                # This is for if one has already been uploaded
                wsi_upload_children = [no_update]*len(ctx.outputs_list[2])
           
            if not user_data_store['upload_type']=='Regular':
                omics_upload_children = [no_update]*len(ctx.outputs_list[3])

        elif ctx.triggered_id['type']=='wsi-files-upload':
            
            if user_data_store['upload_type']=='Visium':
                # Uploading omics file (h5ad, rds)
                # There will only be one additional file uploaded here
                omics_file_flag = get_pattern_matching_value(omics_file_flag)
                if not user_data_store['upload_omics_id'] is None:
                    # If an omics file hasn't been uploaded yet
                    if not omics_file_flag:
                        # If this is a valid file type
                        prop_id = ctx.triggered[0]['prop_id'].split('.')[-1]

                        if prop_id=='uploadComplete':
                            upload_omics_id = self.dataset_handler.get_new_upload_id(user_data_store['latest_upload_folder']['id'])
                            user_data_store['upload_omics_id'] = upload_omics_id
                            
                            omics_upload_children = [
                                dbc.Alert('Omics Upload Success!')
                            ]

                            user_data_store['upload_check']['Omics'] = True

                            #TODO: Add this to the files of the WSI
                        else:
                            omics_upload_children = [no_update]*len(ctx.outputs_list[3])
                    else:
                        # This is an invalid file tye:
                        omics_upload_children = [
                            html.Div([
                                dbc.Alert('Omics Upload Failure! Accepted file types are: h5ad, rds',color = 'danger'),
                                UploadComponent(
                                    id = {'type':'wsi-files-upload','index':0},
                                    uploadComplete=False,
                                    baseurl=self.dataset_handler.apiUrl,
                                    girderToken=user_data_store['token'],
                                    parentId=user_data_store['latest_upload_folder']['id'],
                                    filetypes=['rds','h5ad']                 
                                    )
                            ])
                        ]
                else:
                    # This has already been uploaded
                    omics_upload_children = [no_update]*len(ctx.outputs_list[3])

            elif user_data_store['upload_type']=='CODEX':
                # Check index of triggered id and add to main item files
                # Additional file is the histology image
                omics_file_flag = get_pattern_matching_value(omics_file_flag)

                if not user_data_store['upload_histology_id'] is None:
                    # If the histology image hasn't been uploaded yet
                    if not omics_file_flag:
                        # If this is a valid file type
                        prop_id = ctx.triggered[0]['prop_id'].split('.')[-1]

                        if prop_id=='uploadComplete':
                            upload_histology_id = self.dataset_handler.get_new_upload_id(user_data_store['latest_upload_folder']['id'])
                            user_data_store['upload_histology_id'] = upload_histology_id

                            omics_upload_children = [
                                dbc.Alert('Histology Image Upload Success!')
                            ]

                            user_data_store['upload_check']['Histology'] = True
                        else:
                            omics_upload_children = [no_update]*len(ctx.outputs_list[3])

                    else:
                        # This is an invalid file type
                        omics_upload_children = [
                            html.Div([
                                dbc.Alert('Histology Upload Failure! Accepted file types are: svs, ndpi, scn, tiff, tif',color = 'danger'),
                                UploadComponent(
                                    id = {'type':'wsi-files-upload','index':0},
                                    uploadComplete = False,
                                    baseurl = self.dataset_handler.apiUrl,
                                    girderToken = user_data_store['token'],
                                    parentId = user_data_store['latest_upload_folder']['id'],
                                    filetypes = ['svs','ndpi','scn','tiff','tif']
                                )
                            ])
                        ]
                else:
                    # This has already been uploaded
                    omics_upload_children = [no_update]*len(ctx.outputs_list[3])

            elif user_data_store['upload_type'] == 'Xenium':
                # Check index of triggered id and add to main item files
                # inputs go [morphology, alignment.csv.zip]
                omics_upload_children = [no_update]*len(ctx.outputs_list[3])

                triggered_prop = ctx.triggered[0]['prop_id']
                prop_id = triggered_prop.split('.')[-1]
                triggered_index = ctx.triggered_id['index']

                if len(omics_file_flag)>triggered_index:
                    bad_file_type = omics_file_flag[triggered_index]
                else:
                    bad_file_type = omics_file_flag[0]

                if not bad_file_type:
                    if prop_id=='uploadComplete':
                        omics_upload_children[ctx.triggered_id['index']] = dbc.Alert(
                            'Upload Success!',
                            color = 'success'
                        )

                        user_data_store['upload_check'][list(user_data_store['upload_check'].keys())[ctx.triggered_id['index']+1]] = True

                        #TODO: Add this file to the main item's files      
                else:
                    omics_upload_children[ctx.triggered_id['index']] = [
                        html.Div([
                            dbc.Alert('Upload Failure!',color='danger'),
                            UploadComponent(
                                id = {'type':'wsi-files-upload','index': triggered_index},
                                baseurl=self.dataset_handler.apiUrl,
                                girderToken = user_data_store['token'],
                                parentId = user_data_store['latest_upload_folder']['id'],
                                filetypes = ['svs','ndpi','scn','tiff','tif','csv','zip']
                            )
                        ])
                    ]
                
                
            wsi_upload_children = [no_update]*len(ctx.outputs_list[2])

        else:
            print(f'ctx.triggered_id["type"]: {ctx.triggered_id["type"]}')

        # Checking the upload check
        print(f'upload check: {user_data_store["upload_check"]}')
        if all([user_data_store['upload_check'][i] for i in user_data_store['upload_check']]):
            print('All set!')

            slide_thumbnail = self.dataset_handler.get_slide_thumbnail(user_data_store['upload_wsi_id'])

            if 'Omics' in user_data_store['upload_check']:
                omics_upload_children = [
                    dbc.Alert('Omics Upload Success!')
                ]
            else:
                omics_upload_children = [no_update]*len(ctx.outputs_list[3])

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
            
            if not user_data_store['upload_type']=='Regular':
                return slide_metadata_table, thumb_fig, wsi_upload_children, omics_upload_children, structure_type_disabled, post_upload_style, disable_upload_type, json.dumps({'plugin_used': 'upload', 'type': 'Visium' }), json.dumps(user_data_store)
            else:
                return slide_metadata_table, thumb_fig, wsi_upload_children, [], structure_type_disabled, post_upload_style, disable_upload_type, json.dumps({'plugin_used': 'upload', 'type': 'non-Omnics' }), json.dumps(user_data_store)
        else:

            disable_upload_type = True

            if not user_data_store['upload_type'] == 'Regular':
                return no_update, no_update,wsi_upload_children, omics_upload_children, True, no_update, disable_upload_type, no_update, json.dumps(user_data_store)
            else:
                return no_update, no_update, wsi_upload_children, [], True, no_update, disable_upload_type, no_update, json.dumps(user_data_store)
    
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
            if user_data_store['upload_type']=='Xenium':
                alignment_matrix = self.dataset_handler.grab_from_user_folder(
                    filename = 'matrix.csv',
                    username = user_data_store['login'],
                    folder = user_data_store['latest_upload_folder']['path'].replace('Public/','')
                ).T.reset_index().values
                processed_anns = self.prep_handler.process_uploaded_anns(new_filename,new_upload,user_data_store['upload_wsi_id'],np.float64(alignment_matrix))
            else:
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
                            'Valid file types: Aperio XML (.xml), JSON (.json), GeoJSON (.geojson), or CSV (.csv)'
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

            elif user_data_store['upload_type'] == 'Visium':
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

            elif user_data_store['upload_type'] == 'Xenium':

                # Running post-segmentation workflow from XeniumPrep
                self.feature_extract_ftus, self.layer_ann = self.prep_handler.post_segmentation(
                    user_data_store['upload_wsi_id']
                )

            # Populate with default sub-compartment parameters
            self.sub_compartment_params = self.prep_handler.initial_segmentation_parameters

            sub_comp_method = 'Manual'

            if user_data_store['upload_type'] in ['Visium','Regular','Xenium']:

                if not self.layer_ann is None:

                    ftu_value = self.feature_extract_ftus[self.layer_ann['current_layer']]
                    image, mask = self.prep_handler.get_annotation_image_mask(user_data_store['upload_wsi_id'],user_data_store,self.upload_annotations, self.layer_ann['current_layer'],self.layer_ann['current_annotation'])

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
                
                new_image, new_mask = self.prep_handler.get_annotation_image_mask(user_data_store['upload_wsi_id'],user_data_store,self.upload_annotations,self.layer_ann['current_layer'],self.layer_ann['current_annotation'])
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

    def run_feature_extraction(self,feat_butt,skip_feat,include_ftus,include_features,user_data_store):

        user_data_store = json.loads(user_data_store)

        if ctx.triggered_id is None:
            raise exceptions.PreventUpdate
        
        if not ctx.triggered['value']:
            raise exceptions.PreventUpdate

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

        if not ctx.triggered_id['value']:
            raise exceptions.PreventUpdate
        
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
                #TODO: Update labeling individual structures 
                self.wsi.add_label(self.clicked_ftu,pop_input,'add')

            # Getting current user-labels
            #TODO: Update labeling individual structures
            pop_label_children = self.layout_handler.get_user_ftu_labels(self.wsi,self.clicked_ftu)

        elif ctx.triggered_id['type']=='delete-user-label':
            
            # Removing label based on index
            remove_index = ctx.triggered_id['index']
            #TODO: Update labeling individual structures
            self.wsi.add_label(self.clicked_ftu,remove_index,'remove')

            # Getting current user-labels
            #TODO: Update labeling individual structures
            pop_label_children = self.layout_handler.get_user_ftu_labels(self.wsi,self.clicked_ftu)

        output_list.append(pop_label_children)

        return output_list

    def populate_cluster_tab(self, active_tab, cluster_butt, cluster_data_store, vis_sess_store, user_data_store):

        cluster_data_store = json.loads(cluster_data_store)
        vis_sess_store = json.loads(vis_sess_store)
        user_data_store = json.loads(user_data_store)

        get_data_div_children = dbc.Alert(
            'Data Aligned!',
            color = 'success',
            dismissable = True,
            is_open = True,
            className='d-grid col-12 mx-auto'
        )

        if active_tab == 'clustering-tab':
            if len(list(cluster_data_store.keys()))==0:
                current_ids = [i['_id'] for i in vis_sess_store]
                
                feature_select_data, feature_keys, label_select_options, filter_keys, filter_select_data = self.dataset_handler.get_plottable_keys(current_ids)
                label_select_value = []
                label_select_disable = False
                download_plot_data_style = {'display':'inline-block'}

            return get_data_div_children, feature_select_data, label_select_disable, label_select_options, label_select_value, filter_select_data, download_plot_data_style, cluster_data_store
        else:
            raise exceptions.PreventUpdate

    def populate_cluster_tab(self,active_tab, cluster_butt,cluster_data_store,vis_sess_store,user_data_store):

        cluster_data_store = json.loads(cluster_data_store)
        vis_sess_store = json.loads(vis_sess_store)
        user_data_store = json.loads(user_data_store)

        used_get_cluster_data_plugin = None
        clustering_data = pd.DataFrame.from_dict(cluster_data_store['clustering_data'],orient='index')

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
                #data_getting_response = self.dataset_handler.get_collection_annotation_meta([i['_id'] for i in vis_sess_store if i['included']])

                used_get_cluster_data_plugin = True
            
                self.dataset_handler.generate_feature_dict([i for i in vis_sess_store if i['included']])
                # Monitoring getting the data:
                #data_get_status = 0
                #while data_get_status<3:
                #    data_status, data_log = self.dataset_handler.get_job_status(data_getting_response['_id'])
                #    data_get_status=data_status
                #    print(f'data_get_status: {data_get_status}')
                #    time.sleep(1)

                #cluster_data_store['clustering_data'] = self.dataset_handler.load_clustering_data(user_data_store).to_dict('index')
                cluster_data_store['clustering_data'] = self.dataset_handler.get_clustering_data([i['_id'] for i in vis_sess_store if i['included']]).to_dict('index')

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
            #TODO: get rid of self.reports_generated, potential for user overlap
            if report_tab in self.reports_generated:
                # Report already generated, return the report
                #TODO: get rid of self.reports_generated, potential for user overlap
                report_tab_children = self.reports_generated[report_tab]
            else:
                # Report hasn't been generated yet, generate it
                report_tab_children = self.layout_handler.gen_report_child(cluster_data_store['feature_data'],report_tab)
                #TODO: get rid of self.reports_generated, potential for user overlap
                self.reports_generated[report_tab] = report_tab_children
            
        return report_tab_children

    def download_plot_data(self,download_button_clicked,cluster_data_store):

        cluster_data_store = json.loads(cluster_data_store)

        if not download_button_clicked:
            raise exceptions.PreventUpdate
        
        if not cluster_data_store['feature_data'] is None:
            feature_data = pd.DataFrame.from_dict(cluster_data_store['feature_data'],orient='index')
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
            feature_data = pd.DataFrame.from_dict(cluster_data_store['feature_data'],orient='index')
            
            # Saving current feature data to user public folder
            features_for_markering = self.dataset_handler.save_to_user_folder(
                save_object = {
                    'filename':f'marker_features.csv',
                    'content': feature_data
                },
                user_info = user_data_store
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
            
            #TODO: get rid of self.reports_generated, potential for user overlap
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

    def update_question_div(self,question_tab,user_store_data):

        user_store_data = json.loads(user_store_data)
        logged_in_username = user_store_data['login']
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

    def post_usability_response(self,butt_click,questions_inputs, user_store_data):

        # Updating usability info file in DSA after user clicks "Submit" button
        user_store_data = json.loads(user_store_data)

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
                usability_info['usability_study_users'][user_store_data["login"]]['responses'][f'{level_name}'] = questions_inputs

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

    def add_channel_color_select(self,channel_opts, slide_info_store):

        # Creating a new color selector thing for overlaid channels?
        slide_info_store = json.loads(slide_info_store)

        if not channel_opts is None and len(channel_opts)>0:
            current_channels = slide_info_store['current_channels']
            if type(channel_opts)==list:
                if len(channel_opts[0])>0:
                    channel_opts = channel_opts[0]
                    active_tab = channel_opts[0]
                    disable_butt = [False]
                else:
                    active_tab = None
                    disable_butt = [False]
                    channel_opts = channel_opts[0]
            
            # Removing any channels which aren't included from self.current_channels
            if not current_channels is None:
                intermediate_dict = current_channels.copy()
            else:
                intermediate_dict = {}

            current_channels = {}
            channel_tab_list = []
            for c_idx,channel in enumerate(channel_opts):
                
                if channel in intermediate_dict:
                    channel_color = intermediate_dict[channel]['color']
                else:
                    channel_color = 'rgba(255,255,255,255)'

                current_channels[channel] = {
                    'index': slide_info_store['frame_names'].index(channel),
                    'color': channel_color
                }

                channel_tab_list.append(
                    dbc.Tab(
                        id = {'type':'overlay-channel-tab','index':c_idx},
                        tab_id = channel,
                        label = channel,
                        activeTabClassName='fw-bold fst-italic',
                        label_style = {'color': channel_color if not channel_color=='rgba(255,255,255,255)' else 'rgb(0,0,0,255)'},
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

            slide_info_store['current_channels'] = current_channels
            slide_info_store = json.dumps(slide_info_store)

            return [channel_tabs],disable_butt, slide_info_store
        else:
            current_channels = {}

            slide_info_store['current_channels'] = current_channels

            slide_info_store = json.dumps(slide_info_store)
            disable_butt = [False]*len(ctx.outputs_list[1])

            return [],disable_butt, slide_info_store
    
    def add_channel_overlay(self,butt_click,channel_colors,channels, slide_info_store, user_info_store):

        # Adding an overlay for channel with a TileLayer containing stylized grayscale info
        if not ctx.triggered[0]['value']:
            raise exceptions.PreventUpdate
        
        slide_info_store = json.loads(slide_info_store)
        user_info_store = json.loads(user_info_store)
        current_channels = slide_info_store['current_channels']
        # self.current_channels contains a list of all the currently overlaid channels
        # Updating the color for this channel
        for ch, co in zip(channels,channel_colors):
            current_channels[ch]['color'] = co

        updated_channel_idxes = [current_channels[i]['index'] for i in current_channels]
        updated_channel_colors = [current_channels[i]['color'] for i in current_channels]

        update_style = {
            c_idx: color
            for c_idx,color in zip(updated_channel_idxes,updated_channel_colors)
        }

        updated_urls = []
        for c_idx, old_style in enumerate(slide_info_store['tile_url']):
            if c_idx not in updated_channel_idxes:
                if not slide_info_store['frame_names'][c_idx]=='Histology (H&E)':
                    base_style = {
                        c_idx: "rgba(255,255,255,255)"
                    }
                else:
                    rgb_dict, _ = self.slide_handler.get_rgb_url(slide_info_store, user_info_store)
                    base_style = {
                        band['framedelta']: band['palette'][-1]
                        for band in rgb_dict['bands']
                    }

                new_url = self.slide_handler.update_url_style(base_style | update_style, user_info_store, slide_info_store)

            else:

                new_url = self.slide_handler.update_url_style(update_style, user_info_store, slide_info_store)

            updated_urls.append(new_url)

        # Adding label style to tabs, just adding the overlay color to the text for that tab:
        tab_label_style = [
            {'color': co}
            for co in channel_colors
        ]

        slide_info_store['current_channels'] = current_channels
        slide_info_store = json.dumps(slide_info_store)

        return updated_urls, tab_label_style, slide_info_store

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

    def add_filter(self,butt_click, delete_click, slide_info_store, available_properties):
        """
        Adding a new filter to apply to current FTUs
        """

        slide_info_store = json.loads(slide_info_store)
        if len(list(slide_info_store.keys()))==0:
            raise exceptions.PreventUpdate
        else:
            if slide_info_store['overlay_prop'] is None and slide_info_store['filter_vals'] is None:
                raise exceptions.PreventUpdate

        if ctx.triggered_id=='add-filter-button':
            # Returning a new dropdown for selecting properties to use for a filter

            patched_list = Patch()
            # Find min and max vals
            filter_val = [i for i in available_properties if not 'Cell_Subtypes' in i and not i in ['Max Cell Type','Cell Type']][0]
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
            
            unique_values = self.slide_handler.get_overlay_value_list({'name':m_prop,'value':val,'sub_value': None},slide_info_store)
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
                                options = [i for i in available_properties if not i in ['Max Cell Type','Cell Type'] and not 'Cell_Subtypes' in i],
                                value = [i for i in available_properties if not i in ['Max Cell Type','Cell Type'] and not 'Cell_Subtypes' in i][0],
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

    def add_filter_slider(self, filter_drop,slide_info):

        slide_info = json.loads(slide_info)
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

        unique_values = self.slide_handler.get_overlay_value_list({'name':m_prop,'value':val,'sub_value': None}, slide_info)
        value_types = [type(i) for i in unique_values][0]
        if value_types in [int,float]:
            slider_style = {'display':'inline-block','margin':'auto','width':'100%'}
            return np.min(unique_values), np.max(unique_values), slider_style

        else:
            raise exceptions.PreventUpdate

    def cell_labeling_initialize(self, selectedData, label_butt, label_label, label_rationale, current_selectedData, viewport_data_features, slide_info_store, all_cell_geojson):
        """
        Takes as input some selected cells, a frame to plot their distribution of intensities, and then a Set button that labels those cells
        """

        if ctx.triggered_id is None:
            raise exceptions.PreventUpdate
        
        slide_info_store = json.loads(slide_info_store)
        all_cell_geojson = json.loads(all_cell_geojson)
        if all_cell_geojson is None or len(list(all_cell_geojson.keys()))==0:
            all_cell_geojson = {'type': 'FeatureCollection','features': []}

        label_label = get_pattern_matching_value(label_label)
        label_rationale = get_pattern_matching_value(label_rationale)
        viewport_data_features = get_pattern_matching_value(viewport_data_features)    
        updated_summary = [no_update]*len(ctx.outputs_list[4])

        new_label_label = ['']*len(ctx.outputs_list[5])
        new_label_rationale = ['']*len(ctx.outputs_list[6])

        if slide_info_store['slide_type']=='CODEX':
            viewport_data_features = [slide_info_store['frame_names'][i] for i in viewport_data_features]
        else:
            raise exceptions.PreventUpdate
        
        if ctx.triggered_id['type'] in ['ftu-cell-pie','cell-marker-apply']:
            
            # Pulling current selected data if caused by another trigger
            if ctx.triggered_id['type']=='cell-marker-apply':
                selectedData = current_selectedData

            if not label_label is None:
                cell_properties = {
                    'label': label_label,
                    'rationale': label_rationale,
                    'features': viewport_data_features
                }
            else:
                cell_properties = None

            all_cell_markers = []
            for sD in selectedData:
                if not sD is None:
                    # Pulling customdata from each point object
                    labeling_cells = [i['customdata'] for i in sD['points']]
                    updated_cell_geojson, cell_markers = make_marker_geojson(
                        [i[0]['Bounding_Box'] if type(i)==list else i['Bounding_Box'] for i in labeling_cells],
                        cell_properties
                    )
                    all_cell_markers.extend(cell_markers)

                    if not label_label is None and ctx.triggered_id['type'] in ['cell-marker-apply']:
                        if len(list(all_cell_geojson.keys()))==0:
                            all_cell_geojson = updated_cell_geojson
                        else:
                            all_cell_geojson['features'].extend(updated_cell_geojson['features'])
                    

            # Generating cell labels summary
            if 'features' in all_cell_geojson:
                if len(all_cell_geojson['features'])>0:
                    cell_label_info = [i['properties'] for i in all_cell_geojson['features']]
                    cell_label_df = pd.DataFrame.from_records(cell_label_info)
                    cell_label_df.drop(columns = ['type','rationale'],inplace=True)
                    cell_label_df = cell_label_df.value_counts(subset = ['label'],ascending = False).to_frame()
                    cell_label_df = cell_label_df.reset_index()
                    cell_label_df.columns = ['Cell Type', 'Count']

                    # Generating dash_table and returning underneath cell labeling portion:
                    cell_label_dash_table = dash_table.DataTable(
                        columns = [{'name':i,'id':i,'deletable':False,'selectable':True} for i in cell_label_df],
                        data = cell_label_df.to_dict('records'),
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
                            } for row in cell_label_df.to_dict('records')
                        ],
                        tooltip_duration = None
                    )

                    updated_summary = [
                        html.Div([
                            dbc.Row(
                                html.H5('Cell Labels Summary')
                            ),
                            html.Hr(),
                            dbc.Row(
                                cell_label_dash_table
                            ),
                            html.Hr(),
                            dbc.Row(
                                dbc.Button(
                                    'Export Cell Annotations',
                                    id = {'type': 'cell-marker-export','index': 0},
                                    className = 'd-grid col-12 mx-auto',
                                    disabled = True
                                )
                            )
                        ])
                    ] * len(ctx.outputs_list[4])


            all_cell_geojson = json.dumps(all_cell_geojson)
            updated_count = [html.H4(f'Selected Cells: {len(all_cell_markers)}')]*len(ctx.outputs_list[2])
            updated_rationale = [f'`{", ".join(viewport_data_features)}`']*len(ctx.outputs_list[3])

            return all_cell_markers, all_cell_geojson, updated_count, updated_rationale, updated_summary, new_label_label, new_label_rationale
        else:
            raise exceptions.PreventUpdate

    def update_annotation_session(self, session_tab, new_session, new_session_name, new_session_description, new_session_users, current_session_names, annotation_classes, annotation_colors, annotation_labels, annotation_users, annotation_user_types, user_data_store,slide_info_store):
        """
        Populating annotation station tab with necessary information
        """
        return_div = [no_update]
        new_annotation_tab_group = [no_update]

        user_data_store = json.loads(user_data_store)
        slide_info_store = json.loads(slide_info_store)

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
                    parent_path = f'/user/{user_data_store["login"]}/Public',
                    folder_name = 'FUSION Annotation Sessions',
                    metadata = {"Folder Description":"Data related to annotation sessions completed in FUSION"}
                )

            new_session_folder = self.dataset_handler.create_user_folder(
                parent_path = f'/user/{user_data_store["login"]}/Public/FUSION Annotation Sessions',
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
            current_ftu_names = [i['annotation']['name'] for i in slide_info_store['annotations']]
            updated_tabs,first_tab, first_session = self.layout_handler.gen_annotation_card(self.dataset_handler, current_ftu_names, user_data_store)
            session_progress, current_ann_session = self.dataset_handler.get_annotation_session_progress(first_session['name'], user_data_store)

            user_data_store['current_ann_session'] = current_ann_session
            user_data_store['current_ann_session']['session_progress'] = session_progress

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
                user_data_store['current_ann_session'] = {
                    'name':'Create New Session',
                    'Annotations':[],
                    'Labels': [],
                    'session_progress': {
                        'slides': 0,
                        'annotations': 0,
                        'labels': 0
                    }
                }

                current_ftu_names = [i['annotation']['name'] for i in slide_info_store['annotations']]
                first_tab = self.layout_handler.gen_annotation_content(new = True, current_ftus = current_ftu_names, classes = current_ann_session['Annotations'], labels = current_ann_session['Labels'], ann_progress = session_progress, user_type = current_ann_session['user_type'])
            else:
                session_progress, current_ann_session = self.dataset_handler.get_annotation_session_progress(session_name,user_data_store)
                current_ftu_names = [i['annotation']['name'] for i in slide_info_store['annotations']]
                first_tab = self.layout_handler.gen_annotation_content(new = False, current_ftus = current_ftu_names, classes = current_ann_session['Annotations'], labels = current_ann_session['Labels'], ann_progress = session_progress, user_type = current_ann_session['user_type'])

                user_data_store['current_ann_session'] = current_ann_session
                user_data_store['current_ann_session']['session_progress'] = session_progress


            return_div = [first_tab]

        
        return return_div, new_annotation_tab_group, json.dumps(user_data_store)
    
    def remove_cell_label(self,cell_marker_click, current_marker_geojson):
        """
        Removing marked cell prior to assigning specific cell type label
        """

        if not any([i['value'] for i in ctx.triggered]):
            raise exceptions.PreventUpdate
        
        patched_list = Patch()
        values_to_remove = []
        for i,val in enumerate(cell_marker_click):
            if val:
                values_to_remove.insert(0,i)

        for v in values_to_remove:
            del patched_list[v]

        # Updating number of cells
        updated_count = [html.H4(f'Selected Cells: {len(cell_marker_click)-len(values_to_remove)}')]*len(ctx.outputs_list[1])

        #TODO: Update the cell-marker-geojson with the updated cells
        updated_geojson = current_marker_geojson
        
        return patched_list, updated_count, updated_geojson

    def update_current_annotation(self, ftus, prev_click, next_click, save_click, set_click, delete_click,  line_slide, ann_class, all_annotations, class_label, image_label, ftu_names, ftu_idx, user_store_data, viewport_store_data, slide_info_store):
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

        user_store_data = json.loads(user_store_data)
        viewport_store_data = json.loads(get_pattern_matching_value(viewport_store_data))
        slide_info_store = json.loads(slide_info_store)

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

        ignore_ftus = ['Spots','Cells']

        if ctx.triggered_id['type']=='annotation-station-ftu':
            ftu_styles = [{'display':'inline-block','marginBottom':'15px'} if not i==ctx.triggered_id['index'] else selected_style for i in range(len(ctx.outputs_list[1]))]

            clicked_ftu_name = ftu_names[ctx.triggered_id['index']][0]
            if 'Manual' in clicked_ftu_name:
                clicked_ftu_name = 'Manual:'+clicked_ftu_name.split(':')[1].strip()
            elif 'Marked' in clicked_ftu_name:
                clicked_ftu_name = 'Marked:'+clicked_ftu_name.split(':')[1].strip()
            else:
                clicked_ftu_name = clicked_ftu_name.split(':')[0]

            if clicked_ftu_name in ignore_ftus:
                raise exceptions.PreventUpdate
            
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
                intersecting_ftu_props, intersecting_ftu_polys = self.slide_handler.find_intersecting_ftu(
                    viewport_store_data['current_slide_bounds'],
                    clicked_ftu_name,
                    slide_info = slide_info_store
                )
            elif 'Manual' in clicked_ftu_name:
                manual_idx = int(clicked_ftu_name.split(':')[-1])-1
                intersecting_ftu_polys = [shape(slide_info_store['manual_ROIs'][manual_idx]['geojson']['features'][0]['geometry'])]
            elif 'Marked' in clicked_ftu_name:
                intersecting_ftu_polys = [shape(i['geojson']['features'][0]['geometry']) for i in slide_info_store['marked_FTUs']]

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
            
            ftu_bbox = np.array(self.slide_handler.convert_map_coords(ftu_bbox_coords,slide_info_store))
            ftu_bbox = [np.min(ftu_bbox[:,0])-50,np.min(ftu_bbox[:,1])-50,np.max(ftu_bbox[:,0])+50,np.max(ftu_bbox[:,1])+50]
            #TODO: This method only grabs histology images that are single frame or multi-frame with RGB at 0,1,2
            ftu_image = self.dataset_handler.get_image_region(
                slide_info_store['slide_info']['_id'],
                user_store_data,
                [int(i) for i in ftu_bbox]
            )

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
            
            label_dict['Annotator'] = user_store_data["login"]
            
            if ctx.triggered_id['type']=='annotation-set-label':
                # Grab the first member of the clicked ftu
                if not 'Manual' in clicked_ftu_name or 'Marked' in clicked_ftu_name:
                    intersecting_ftu_props, intersecting_ftu_polys = self.slide_handler.find_intersecting_ftu(
                        viewport_store_data['current_slide_bounds'],
                        clicked_ftu_name,
                        slide_info_store
                    )
                elif 'Manual' in clicked_ftu_name:
                    manual_idx = int(clicked_ftu_name.split(':')[-1])
                    intersecting_ftu_polys = [shape(slide_info_store['manual_ROIs'][manual_idx]['geojson'])]
                elif 'Marked' in clicked_ftu_name:
                    intersecting_ftu_polys = [shape(i['geojson']) for i in slide_info_store['marked_FTUs']]

                ftu_coords = list(intersecting_ftu_polys[ftu_idx].exterior.coords)
                ftu_coords = np.array(self.slide_handler.convert_map_coords(ftu_coords, slide_info_store))
                ftu_bbox = [np.min(ftu_coords[:,0])-50,np.min(ftu_coords[:,1])-50,np.max(ftu_coords[:,0])+50,np.max(ftu_coords[:,1])+50]

                # Now saving image with label in metadata
                ftu_image = np.array(
                    self.dataset_handler.get_image_region(
                        slide_info_store['slide_info']['_id'],
                        [int(i) for i in ftu_bbox]
                    )
                )

                ftu_content = {
                    'content': ftu_image,
                    'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.png',
                    'metadata': label_dict
                }

                # Checking if slide folder is created
                slide_folder = self.dataset_handler.check_user_folder(
                    folder_name = 'FUSION Annotation Sessions',
                    subfolder = user_store_data['current_ann_session']['name'],
                    sub_sub_folder = slide_info_store['slide_info']['name']
                )
                if slide_folder is None:
                    # Creating slide folder in current_ann_session
                    new_folder = self.dataset_handler.create_user_folder(
                        parent_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}',
                        folder_name = slide_info_store['slide_info']['name']
                    )
                    new_folder = self.dataset_handler.create_user_folder(
                        parent_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}/{slide_info_store["slide_info"]["name"]}',
                        folder_name = 'Images'
                    )

                # Saving data
                self.dataset_handler.save_to_user_folder(ftu_content,user_store_data,output_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}/{slide_info_store["slide_info"]["name"]}/Images')
                
                # Updating image labels
                image_labels = ['']*len(ctx.outputs_list[3])

            else:
                #TODO: delete label from image metadata if that metadata is already attached to an image in the annotation session

                image_labels = [no_update if not i==ctx.triggered_id['index'] else '' for i in range(len(ctx.outputs_list[3]))]

        elif ctx.triggered_id['type'] == 'annotation-save-button':

            # Grab the first member of the clicked ftu
            if not 'Manual' in clicked_ftu_name and not 'Marked' in clicked_ftu_name:
                intersecting_ftu_props, intersecting_ftu_polys = self.slide_handler.find_intersecting_ftu(
                    viewport_store_data['current_slide_bounds'],
                    clicked_ftu_name,
                    slide_info_store
                )
            elif 'Manual' in clicked_ftu_name:
                manual_idx = int(clicked_ftu_name.split(':')[-1])
                intersecting_ftu_polys = [shape(slide_info_store['manual_ROIs'][manual_idx]['geojson'])]
            elif 'Marked' in clicked_ftu_name:
                intersecting_ftu_polys = [shape(i['geojson']) for i in slide_info_store['marked_FTUs']]

            ftu_coords = list(intersecting_ftu_polys[ftu_idx].exterior.coords)
            ftu_coords = np.array(self.slide_handler.convert_map_coords(ftu_coords,slide_info_store))
            ftu_bbox = [np.min(ftu_coords[:,0])-50,np.min(ftu_coords[:,1])-50,np.max(ftu_coords[:,0])+50,np.max(ftu_coords[:,1])+50]
            height = int(ftu_bbox[3]-ftu_bbox[1])
            width = int(ftu_bbox[2]-ftu_bbox[0])
            
            # Convert annotations relayoutData to a mask and save both image and mask to current annotation session
            #TODO: combined mask here should have a channel dimension equal to the number of classes (e.g. if two classes overlap)
            combined_mask = np.zeros((height,width,len(user_store_data["current_ann_session"]["Annotations"])))
            annotation_colors_list = [i['value'] for i in user_store_data["current_ann_session"]['Annotations']]
            for key in annotations:
                for i in range(len(annotations)):
                    #TODO: Add color property to this function for multiple classes
                    #TODO: Add lines to this mask
                    if 'path' in annotations[i]:
                        mask = path_to_mask(annotations[i]['path'],(combined_mask.shape[0],combined_mask.shape[1]))
                        mask_class = annotations[i]['line']['color']

                        combined_mask[:,:,annotation_colors_list.index(mask_class)] += 255*mask
            
            # Now saving both image and mask to the annotation session folder
            ftu_image = np.array(
                self.dataset_handler.get_image_region(
                    slide_info_store['slide_info']['_id'],
                    user_store_data,
                    [int(i) for i in ftu_bbox]
                )
            )
            
            mask_image = np.uint8(combined_mask)

            ftu_content = {
                'content': ftu_image,
                'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.png',
                'metadata': {
                    "Annotator": user_store_data['login']
                }
            }

            if mask_image.shape[-1] in [1,3]:
                mask_content = {
                    'content': mask_image,
                    'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.png',
                    'metadata': {
                        "Annotator": user_store_data['login']
                    }
                }
            else:
                mask_content = {
                    'content': mask_image,
                    'filename': f'{clicked_ftu_name}_{int(ftu_bbox[0])}_{int(ftu_bbox[1])}.tiff',
                    'metadata': {
                        "Annotator": user_store_data['login']
                    }
                }

            # Checking if slide folder is created
            slide_folder = self.dataset_handler.check_user_folder(
                folder_name = 'FUSION Annotation Sessions',
                user_info = user_store_data,
                subfolder = user_store_data["current_ann_session"]['name'],
                sub_sub_folder = slide_info_store['slide_info']['name']
            )
            if slide_folder is None:
                # Creating slide folder in current_ann_session
                new_folder = self.dataset_handler.create_user_folder(
                    parent_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}',
                    folder_name = slide_info_store['slide_info']['name']
                )
                new_folder = self.dataset_handler.create_user_folder(
                    parent_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}/{slide_info_store["slide_info"]["name"]}',
                    folder_name = 'Images'
                )
                new_folder = self.dataset_handler.create_user_folder(
                    parent_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}/{slide_info_store["slide_info"]["name"]}',
                    folder_name = 'Masks'
                )

            # Saving data
            self.dataset_handler.save_to_user_folder(ftu_content,user_store_data,output_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}/{slide_info_store["slide_info"]["name"]}/Images')
            self.dataset_handler.save_to_user_folder(mask_content,user_store_data,output_path = f'/user/{user_store_data["login"]}/Public/FUSION Annotation Sessions/{user_store_data["current_ann_session"]["name"]}/{slide_info_store["slide_info"]["name"]}/Masks')

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

    def download_annotation_session_log(self,butt_click,new_interval,user_data_store,is_open):
        """
        Checking if annotation session is done loading
        """

        user_data_store = json.loads(user_data_store)

        if not ctx.triggered:
            raise exceptions.PreventUpdate
        
        if ctx.triggered_id['type']=='download-ann-session':
            
            # Starting thread that downloads session files
            new_thread = threading.Thread(target = self.download_handler.extract_annotation_session, name = 'ann-session-download', args = [user_data_store['current_ann_session']['folder_id']])
            new_thread.daemon = True
            new_thread.start()

            # Creating interval modal object
            modal_div_children = [
                html.Div([
                    dbc.ModalHeader(html.H4(f'Preparing annotation session data: {user_data_store["current_ann_session"]["name"]}')),
                    dbc.ModalBody([
                        dbc.Progress(
                            id = {'type':'ann-session-progress','index':0},
                            value = 0,
                            label = '0%'
                        )
                    ])
                ])
            ]

            download_data = [no_update]
            interval_disable = [False]
            modal_open = [True]

        elif ctx.triggered_id['type']=='ann-session-interval':
            
            if not new_interval:
                raise exceptions.PreventUpdate

            thread_names = [i.name for i in threading.enumerate()]
            if 'ann-session-download' in thread_names:

                # Get total number of files downloaded
                total = 0
                for root, dirs, files in os.walk('./assets/FUSION_Download/'):
                    total += len(files)
                
                # Compare with total needed
                if not 'session_progress' in user_data_store['current_ann_session']:
                    session_files, current_ann_session = self.dataset_handler.get_annotation_session_progress(user_data_store['current_ann_session']['name'],user_data_store)
                else:
                    session_files = user_data_store['current_ann_session']['session_progress']
                n_session_files = 2*session_files['annotations'] if session_files['labels']==0 else session_files['labels']

                progress_val = np.minimum(int(100*(total/n_session_files)),100)

                modal_div_children = [
                    html.Div([
                        dbc.ModalHeader(html.H4(f'Preparing annotation session data: {user_data_store["current_ann_session"]["name"]}')),
                        dbc.ModalBody([
                            dbc.Progress(
                                id = {'type':'ann-session-progress','index':0},
                                value = progress_val,
                                label = f'{progress_val}%'
                            )
                        ])
                    ])
                ]

                download_data = [no_update]
                interval_disable = [False]
                modal_open = [True]

            else:
                
                modal_div_children = [
                    html.Div([
                        dbc.ModalHeader(html.H4(f'All Done!')),
                        dbc.ModalBody([
                            dbc.Progress(
                                id = {'type':'ann-session-progress','index':0},
                                value = 100,
                                label = f'100%'
                            )
                        ])
                    ])
                ]

                download_data = [dcc.send_file('./assets/FUSION_Download.zip')]
                interval_disable = [True]
                modal_open = [False]

        return download_data, interval_disable, modal_open, modal_div_children







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
        sys.exit(1)

    # Initializing GirderHandler
    dataset_handler = GirderHandler(
        apiUrl=dsa_url,
        username=username,
        password=p_word
    )
    initial_user_info = dataset_handler.user_details

    # Initial collection: can be specified as a single or multiple collections which will automatically be loaded into the visualization session
    try:
        default_items = os.environ.get('FUSION_INITIAL_ITEMS')
        default_items = default_items.split(',')
    except:
        # Can be one or more items
        default_items = [
            '6495a4e03e6ae3107da10dc5',
            '6495a4df3e6ae3107da10dc2'
        ]
    
    default_item_info = [dataset_handler.get_item_info(i) for i in default_items]

    # Saving & organizing relevant id's in GirderHandler
    print('Getting initial items metadata')
    dataset_handler.set_default_slides(default_item_info)

    # Going through fusion_configs.json, adding plugins to Prepper, initializing user studies
    # Step 1: Check for presence of required plugins
    # Step 2: Pull missing ones specified in fusion_configs.json
    # Step 3: If any user studies are specified:
    #   Step 3a: Create a collection called "FUSION User Studies"
    #   Step 3b: Create a separate item for each study containing a JSON file with questions, admins, and users (with user_type, name, and responses)
    #   Step 3c: Create a new group for each user study and add users who already have accounts to that group to enable edit access to responses file
    #   Step 3d: Specify location of associated study materials (if we want to continue to host those on FUSION (PowerPoint slides, etc.))

    # Getting usability study information
    #TODO: Generalize for other types of user studies
    print(f'Getting asset items')
    assets_path = '/collection/FUSION Assets/'
    dataset_handler.get_asset_items(assets_path)

    # Getting the slide data for DSASlide()
    slide_names = [
        {'label': i['name'],'value':i['_id']}
        for i in default_item_info
    ]

    # Required for Dash layouts, themes, and icons
    external_stylesheets = [
        dbc.themes.LUX,
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
        ]

    # Initializing slide datasets with public collections (edge parent folders of image items) and user upload folders
    slide_dataset = dataset_handler.update_slide_datasets(initial_user_info)

    print(f'Generating layouts')
    layout_handler = LayoutHandler()
    layout_handler.gen_initial_layout(slide_names,initial_user_info,dataset_handler.default_slides, slide_dataset)
    layout_handler.gen_vis_layout(GeneHandler(),None)
    _ = layout_handler.gen_builder_layout(dataset_handler,initial_user_info, None)
    layout_handler.gen_uploader_layout()

    download_handler = DownloadHandler(dataset_handler)

    prep_handler = Prepper(dataset_handler)
    
    print('Ready to rumble!')
    main_app = DashProxy(
        __name__,
        external_stylesheets=external_stylesheets,
        transforms = [MultiplexerTransform()]
    )
    
    # Passing main handlers to application object
    vis_app = FUSION(
        main_app,
        layout_handler,
        dataset_handler,
        download_handler,
        prep_handler
    )


if __name__=='__main__':
    #TODO: Can add path to configs here as an input argument
    app()
