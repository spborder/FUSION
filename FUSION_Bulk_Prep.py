"""

Running preprocessing steps on folders of slides in DSA

These should already have multi-compartment segmentation run and should have spots uploaded


"""

import os
import sys
import json
import girder_client
import time

import plotly.graph_objects as go
import plotly.express as px

import dash_bootstrap_components as dbc
from dash import dcc, ctx, MATCH, ALL, exceptions, no_update
from dash_extensions.enrich import DashProxy, html, Input, Output, State, MultiplexerTransform

from FUSION_Handlers import GirderHandler, LayoutHandler
from FUSION_Prep import Prepper


class BulkProcessApp:
    def __init__(self,
                 app,
                 layout,
                 dataset_handler,
                 prep_handler,
                 ):
        
        self.app = app
        self.app.layout = layout
        self.app.title = "Bulk Preprocessing App"
        self.app._favicon = './assets/favicon.ico'

        self.dataset_handler = dataset_handler
        self.prep_handler = prep_handler

        self.sub_compartment_params = self.prep_handler.initial_segmentation_parameters

        self.bulk_callbacks()

        self.app.run_server(host = '0.0.0.0', debug = False, use_reloader=False,port=8000)
    
    def bulk_callbacks(self):

        # Loading new slide either from folder drop, slide drop, or next-slide button
        self.app.callback(
            [Input('folder-drop','value'),
             Input('slide-drop','value')],
            [Output('ex-ftu-img','figure'),
             Output('ftu-select','options'),
             Output('feature-items','children'),
             Output('sub-thresh-slider','disabled'),
             Output('sub-comp-method','disabled'),
             Output('go-to-feat','disabled'),
             Output('slide-drop','options'),
             Output('slide-store','data')],
             State('slide-store','data')
        )(self.update_slide)

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
            [State('go-to-feat','disabled'),
             State('slide-store','data')],
            [Output('ex-ftu-img','figure'),
             Output('sub-thresh-slider','marks'),
             Output('feature-items','children'),
             Output('sub-thresh-slider','disabled'),
             Output('sub-comp-method','disabled'),
             Output('go-to-feat','disabled'),
             Output('slide-store','data')],
            prevent_initial_call=True
        )(self.update_sub_compartment)

        self.app.callback(
            Input({'type':'start-feat','index':ALL},'n_clicks'),
            [Output({'type':'feat-logs','index':ALL},'children'),
             Output('slide-drop','options'),Output('slide-store','data')],
             [State({'type': 'include-ftu-drop','index':ALL},'value'),
              State({'type': 'include-feature-drop','index':ALL},'value'),
              State('slide-store','data')],
            prevent_initial_call=True
        )(self.run_feature_extraction)

    def update_slide(self,folder,slide,slide_store):

        # no matter what the triggered_id, a new slide is loaded
        slide_store = json.loads(slide_store)
        layer_ann = slide_store['layer_ann']
        user_info = slide_store['user_info']

        if type(slide)==dict:
            slide = slide['value']
        
        if ctx.triggered_id == 'folder-drop':

            slide_store['current_slides'] = self.dataset_handler.get_folder_slides(folder)
            slide_store['done_slides'] = []
            done_slides = []
            current_slides = slide_store['current_slides']

            new_slides = [
                {'label':i['name'],'value':i['_id'],'disabled':False}
                if i['_id'] not in done_slides
                else {'label':i['name'],'value':i['_id'],'disabled':True}
                for i in current_slides
            ]
            slide = new_slides[0]['value']
        else:
            new_slides = no_update


        ex_ftu_img = go.Figure()
        ftu_options = []

        # Getting annotations for the new slide
        slide_annotations = self.dataset_handler.get_annotations(slide)
        for idx,i in enumerate(slide_annotations):
            if 'annotation' in i:
                if 'elements' in i['annotation']:
                    if not 'interstitium' in i['annotation']['name']:
                        if len(i['annotation']['elements'])>0:
                            ftu_options.append({
                                'label':i['annotation']['name'],
                                'value':idx,
                                'disabled':False
                            })
                        else:
                            ftu_options.append({
                                'label':i['annotation']['name']+' (None detected in slide)',
                                'value':idx,
                                'disabled':True
                            })
                    else:
                        ftu_options.append({
                            'label':i['annotation']['name']+' (Not implemented for interstitium)',
                            'value':idx,
                            'disabled':True
                        })
        
        print(ftu_options)
        if len([i for i in ftu_options if not i['disabled']])>0:
            layer_ann = {
                'current_layer':[i['value'] for i in ftu_options if not i['disabled']][0],
                'current_annotation':0,
                'previous_annotation':0,
                'max_layers':[len(i['annotation']['elements']) for i in slide_annotations]
            }

            image, mask = self.prep_handler.get_annotation_image_mask(slide,user_info,slide_annotations,layer_ann['current_layer'],layer_ann['current_annotation'])

            ex_ftu_img = go.Figure(
                data = px.imshow(image)['data'],
                layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
            )
        else:
            ex_ftu_img = go.Figure()

        slide_store['upload_wsi_id'] = slide
        slide_store['upload_annotations'] = slide_annotations
        slide_store['feature_extract_ftus'] = ftu_options
        slide_store['layer_ann'] = layer_ann

        slide_store = json.dumps(slide_store)

        return ex_ftu_img, ftu_options, [], False, False, False, new_slides, slide_store

    def update_sub_compartment(self,select_ftu,prev,next,go_to_feat,ex_ftu_view,ftu_slider,thresh_slider,sub_method,go_to_feat_state, slide_store):

        slide_store = json.loads(slide_store)

        sub_compartment_params = slide_store['sub_compartment_params']
        layer_ann = slide_store['layer_ann']
        feature_extract_ftus = slide_store['feature_extract_ftus']
        upload_wsi_id = slide_store['upload_wsi_id']
        upload_annotations = slide_store['upload_annotations']
        user_info = slide_store['user_info']

        new_ex_ftu = go.Figure()
        feature_extract_children = []
        go_to_feat_disabled = go_to_feat_state
        disable_slider = go_to_feat_state
        disable_method = go_to_feat_state

        slider_marks = {
            val:{'label':f'{sub_comp["name"]}: {val}','style':{'color':sub_comp["marks_color"]}}
            for val,sub_comp in zip(thresh_slider[::-1],sub_compartment_params)
        }

        for idx,ftu,thresh in zip(list(range(len(sub_compartment_params))),sub_compartment_params,thresh_slider[::-1]):
            ftu['threshold'] = thresh
            sub_compartment_params[idx] = ftu

        if ctx.triggered_id=='next-butt':
            # Moving to next annotation in current layer
            layer_ann['previous_annotation'] = layer_ann['current_annotation']

            if layer_ann['current_annotation']+1>=layer_ann['max_layers'][layer_ann['current_layer']]:
                layer_ann['current_annotation'] = 0
            else:
                layer_ann['current_annotation'] += 1

        elif ctx.triggered_id=='prev-butt':
            # Moving back to previous annotation in current layer
            layer_ann['previous_annotation'] = layer_ann['current_annotation']

            if layer_ann['current_annotation']==0:
                layer_ann['current_annotation'] = layer_ann['max_layers'][layer_ann['current_layer']]-1
            else:
                layer_ann['current_annotation'] -= 1
        
        elif ctx.triggered_id=='ftu-select':
            # Moving to next annotation layer, restarting annotation count
            if type(select_ftu)==dict:
                layer_ann['current_layer']=select_ftu['value']
            elif type(select_ftu)==int:
                layer_ann['current_layer'] = select_ftu

            layer_ann['current_annotation'] = 0
            layer_ann['previous_annotation'] = layer_ann['max_layers'][layer_ann['current_layer']]

        if ctx.triggered_id not in ['go-to-feat','ex-ftu-slider','sub-comp-method']:
            
            new_image, new_mask = self.prep_handler.get_annotation_image_mask(upload_wsi_id,user_info,upload_annotations,layer_ann['current_layer'],layer_ann['current_annotation'])

        if ctx.triggered_id not in ['go-to-feat']:
            
            new_image,new_mask = self.prep_handler.get_annotation_image_mask(upload_wsi_id,user_info,upload_annotations,layer_ann['current_layer'],layer_ann['current_annotation'])
            sub_compartment_image = self.prep_handler.sub_segment_image(new_image,new_mask,sub_compartment_params,ex_ftu_view,ftu_slider)

            new_ex_ftu = go.Figure(
                data = px.imshow(sub_compartment_image)['data'],
                layout = {'margin':{'t':0,'b':0,'l':0,'r':0}}
            )

        else:
            go_to_feat_disabled = True
            disable_slider = True
            disable_method = True

            new_ex_ftu = no_update
            feature_extract_children = self.prep_handler.gen_feat_extract_card(feature_extract_ftus)

        slide_store['layer_ann'] = layer_ann
        slide_store['sub_compartment_params'] = sub_compartment_params
        slide_store = json.dumps(slide_store)

        return new_ex_ftu, slider_marks, feature_extract_children, disable_slider, disable_method, go_to_feat_disabled, slide_store

    def run_feature_extraction(self,feat_butt,include_ftu,include_feature, slide_store):
        
        slide_store = json.loads(slide_store)
        done_slides = slide_store['done_slides']
        upload_wsi_id = slide_store['upload_wsi_id']
        feature_extract_ftus = slide_store['feature_extract_ftus']
        sub_compartment_params = slide_store['sub_compartment_params']
        current_slides = slide_store['current_slides']

        if not ctx.triggered_id is None:
            if type(feat_butt) == list:
                if len(feat_butt)>0:
                    if feat_butt[0]>0:
                        
                        include_feature = ','.join(include_feature[0])
                        include_ftu = [feature_extract_ftus[i]['label'] for i in include_ftu[0]]
                        ignore_ftus = ','.join([i['label'] for i in feature_extract_ftus if i['label'] not in include_ftu or '(' in i['label']])
                        
                        # Getting the file to send to feature extraction
                        #file_id = self.dataset_handler.gc.get(f'/item/{self.upload_wsi_id}/files')[0]["_id"]
                        feat_ext_job = self.prep_handler.run_feature_extraction(upload_wsi_id,sub_compartment_params,include_feature,ignore_ftus)
                        
                        done_slides.append(upload_wsi_id)

                        # Updating slide-drop options
                        updated_slide_drop = [
                            {'label':i['name'],'value':i['_id'], 'disabled':False}
                            if i['_id'] not in done_slides
                            else {'label':i['name'],'value':i['_id'],'disabled':True}
                            for i in current_slides
                        ]
                        
                        slide_store['done_slides'] = done_slides

                        slide_store = json.dumps(slide_store)

                        return ['Submitted!'], updated_slide_drop, slide_store
                    else:
                        slide_store = json.dumps(slide_store)
                        return [no_update], no_update,slide_store
                else:
                    slide_store = json.dumps(slide_store)
                    return [no_update], no_update,slide_store
            else:
                slide_store = json.dumps(slide_store)
                return [no_update], no_update,slide_store
        else:
            raise exceptions.PreventUpdate

# Special layout
def gen_bulk_prep_layout(dataset_handler,dataset_path,prep_handler):

    # Layout handler has the initial layout, just need to add the container stuff

    # Copying some stuff over from uploader_layout but removing file upload and multi-compartment prediction

    # Getting slide datasets
    resource_id = dataset_handler.get_resource_id(dataset_path)
    folder_slides = dataset_handler.get_folder_slides(resource_id)

    all_datasets = [{
        'label': dataset_path.split('/')[-1],
        'val': resource_id
        }]
    first_dataset = all_datasets[0]
    
    first_dataset_slides = folder_slides

    slide_store = {
        'all_datasets': all_datasets,
        'current_slides': folder_slides,
        'done_slides': [],
        'upload_wsi_id': '',
        'upload_annotations': {},
        'sub_compartment_params': [],
        'feature_extract_ftus': {},
        'layer_ann': {},
        'sub_compartment_params': prep_handler.initial_segmentation_parameters,
        'user_info': dataset_handler.gc.get('/user/me') | dataset_handler.gc.get('/token/session')
    }


    # Creating the row for folder/slide iteration
    folder_slide_select = [
        html.Div([
            'Select a folder and slide to process or click "Next Slide" button to go to the next up',
            dbc.Row([
                dbc.Col([
                    'Available Folders:',
                    dcc.Dropdown(
                        options = [
                            {'label':i['label'],'value':i['val'],'disabled':False}
                            for i in all_datasets
                        ],
                        value = first_dataset,
                        id = 'folder-drop'
                    )
                ],md=6),
                dbc.Col([
                    'Slides in current folder:',
                    dcc.Dropdown(
                        options = [
                            {'label':i['name'],'value':i['_id'],'disabled':False}
                            for i in first_dataset_slides
                        ],
                        value = {'label':first_dataset_slides[0]['name'],'value':first_dataset_slides[0]['_id']},
                        id = 'slide-drop'
                    )
                ],md=6)
            ],align='center')
        ]),
        html.Div(
            dcc.Store(
                id = 'slide-store',
                storage_type='memory',
                data = json.dumps(slide_store)
            )
        )
    ]

    # Sub-compartment segmentation card:
    sub_comp_methods_list = [
        {'label':'Manual','value':'Manual','disabled':False},
        {'label':'Use Plugin','value':'plugin','disabled':True}
    ]
    sub_comp_card = dbc.Card([
        dbc.CardHeader([
            'Sub-Compartment Segmentation',
            ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        dbc.Label('Select FTU',html_for='ftu-select'),
                        dcc.Dropdown(placeholder='FTU Options',id='ftu-select')
                    ]),md=12)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        children = [
                            dcc.Graph(figure=go.Figure(),id='ex-ftu-img')
                        ]
                    ),md=12)
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        children = [
                            dbc.Label('Example FTU Segmentation Options',html_for='ex-ftu-opts'),
                            html.Hr(),
                            html.Div(
                                id='ex-ftu-opts',
                                children = [
                                    dcc.RadioItems(
                                        [
                                            {'label':html.Span('Overlaid',style={'marginBottom':'5px','marginLeft':'5px','marginRight':'10px'}),'value':'Overlaid'},
                                            {'label':html.Span('Side-by-side',style={'marginLeft':'5px'}),'value':'Side-by-side'}
                                        ],
                                            value='Overlaid',inline=True,id='ex-ftu-view'),
                                    html.B(),
                                    dbc.Label('Overlaid Mask Transparency:',html_for='ex-ftu-slider',style={'marginTop':'10px'}),
                                    dcc.Slider(0,1,0.05,value=0,marks=None,vertical=False,tooltip={'placement':'bottom'},id='ex-ftu-slider'),
                                    html.B(),
                                    dbc.Row([
                                        dbc.Col(dbc.Button('Previous',id='prev-butt',outline=True,color='secondary',className='d-grid gap-2 col-6 mx-auto')),
                                        dbc.Col(dbc.Button('Next',id='next-butt',outline=True,color='secondary',className='d-grid gap-2 col-6 mx-auto'))
                                    ],style={'marginBottom':'15px','display':'flex'}),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col(dbc.Button('Go to Feature Extraction',id='go-to-feat',color='success',className='d-grid gap-2 col-12 mx-auto'))
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
                            dbc.Label('Sub-compartment Segmentation Method:',html_for='sub-comp-method')
                        ]
                    ),md=4),
                dbc.Col(
                    html.Div(
                        children = [
                            dcc.Dropdown(sub_comp_methods_list,placeholder='Available Methods',id='sub-comp-method')
                        ]
                    ),md=8
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        id='sub-comp-tabs',
                        children = [
                            dbc.Label('Sub-Compartment Thresholds',html_for='sub-thresh-slider'),
                            dcc.RangeSlider(
                                id = 'sub-thresh-slider',
                                min = 0.0,
                                max = 255.0,
                                step = 5.0,
                                value = [0.0,50.0,120.0],
                                marks = {
                                    0.0:{'label':'Luminal Space: 0','style':'rgb(0,255,0)'},
                                    50.0:{'label':'PAS: 50','style':'rgb(255,0,0)'},
                                    120.0:{'label':'Nuclei: 120','style':'rgb(0,0,255)'}
                                },
                                tooltip = {'placement':'top','always_visible':False},
                                allowCross=False
                            )
                        ]
                    ),md=12, style = {'marginLeft':'20px','marginRight':'20px'}
                ),
            ])
        ])
    ])

    # Feature extraction card:
    feat_extract_card = dbc.Card([
        dbc.CardHeader([
            'Morphometric Feature Extraction',
            ]),
        dbc.CardBody(
            dbc.Row([
                dbc.Col(html.Div(id='feature-items'))
            ])
        )
    ])

    # Assembling final layout
    bulk_prep_layout = [
        html.H1('Bulk dataset pre-processing'),
        html.Hr(),
        dbc.Row(
            children = [
                dbc.Col(folder_slide_select,md=12)
            ]
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
            id='post-segment-row'
        )
    ]

    return bulk_prep_layout



def main():

    dsa_url = os.environ.get('DSA_URL')
    username = os.environ.get('DSA_USER')
    p_word = os.environ.get('DSA_PWORD')

    dataset_handler = GirderHandler(apiUrl=dsa_url,username=username,password=p_word)

    dataset_path = '/collection/10x Xenium/Kidney'

    external_stylesheets = [
        dbc.themes.LUX,
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ]

    layout_handler = LayoutHandler()
    prep_handler = Prepper(dataset_handler)

    bulk_layout = gen_bulk_prep_layout(dataset_handler,dataset_path,prep_handler)

    app_layout = layout_handler.gen_single_page_layout('Application for bulk pre-processing of slides in collections',bulk_layout)


    main_app = DashProxy(__name__,
                         external_stylesheets=external_stylesheets,
                         transforms=[MultiplexerTransform()])

    prep_app = BulkProcessApp(
        main_app,
        app_layout,
        dataset_handler,
        prep_handler
    )


if __name__=='__main__':
    main()