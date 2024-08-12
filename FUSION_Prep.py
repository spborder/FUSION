"""

File containing FUSION pre-processing scripts and workflows

"""

import os
import sys

import numpy as  np

from PIL import Image
from skimage.draw import polygon
from skimage.color import rgb2hsv, rgb2lab, lab2rgb
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
#from skimage.filters import rank
from skimage import exposure

from matplotlib import colormaps

import dash_bootstrap_components as dbc
from dash import dcc, html

#from histomicstk.preprocessing import color_conversion
from wsi_annotations_kit import wsi_annotations_kit as wak
from shapely.geometry import Polygon, Point
import pandas as pd
import json
import lxml.etree as ET
import base64
import shutil
import uuid
from math import pi

from tqdm import tqdm

from typing_extensions import Union

from io import StringIO, BytesIO

#TODO: Remove some default settings
# sub-compartment segmentation and feature extraction should vary

class Prepper:
    def __init__(self, girder_handler):

        self.girder_handler = girder_handler

        # Dictionary containing model and item id's
        self.model_zoo = {
            'MultiCompartment_Model':{
                'plugin_name':'samborder2256_multicomp_latest/MultiCompartmentSegment',
                'model_id':'648123761019450486d13dce',
                'structures':['Cortical interstitium','Medullary interstitium','Glomeruli','Sclerotic Glomeruli','Tubules','Arteries and Arterioles']
            },
            'IFTA_Model':{
                'plugin_name':'dpraveen511_ifta_ifta_seg_aws_1/IFTASegmentation',
                'model_id':'64c9422a287cfdce1e9c2530',
                'structures':['IFTA']
            },
            'PTC_Model':{
                'plugin_name':'dpraveen511_ptc_ptc_seg_aws_1/PTCSegmentation',
                'model_id':'64b5d4ec5fd253763e671721',
                'structures':['PTC']
            }
        }

        self.feature_extraction_plugin = 'samborder2256_ftx_plugin_latest/Ftx_sc'

        self.color_map = colormaps['jet']

        self.padding_pixels = 50

        self.initial_segmentation_parameters = [
            {
                'name':'Nuclei',
                'threshold':110,
                'min_size':45,
                'color':[0,0,255],
                'marks_color':'rgb(0,0,255)'
            },            
            {
                'name':'Eosinophilic',
                'threshold':45,
                'min_size':20,
                'color':[255,0,0],
                'marks_color':'rgb(255,0,0)'
            },
            {
                'name':'Luminal Space',
                'threshold':0,
                'min_size':20,
                'color':[0,255,0],
                'marks_color':'rgb(0,255,0)'
            }
        ]

    def get_plugin_list(self)->list:
        """
        Return list of current plugins and info
        """

        plugin_list = self.girder_handler.gc.get('/slicer_cli_web/cli')

        return plugin_list

    def add_plugin(self,plugin_info:Union[list,dict],user_info:dict):
        """
        Adding new analysis/preprocessing plugin to available list
        """

        # Check if present already and delete old version
        if type(plugin_info)==dict:
            plugin_info = [plugin_info]

        self.delete_plugin(plugin_info)

        for p_info in plugin_info:
            put_response = self.girder_handler.gc.put(
                '/slicer_cli_web/docker_image',
                parameters = {
                    'name': p_info['image']
                }
            )

    def delete_plugin(self,plugin_info:Union[list,dict],user_info:dict):
        """
        Removing analysis/preprocessing plugin from available options        
        """

        current_clis = self.get_plugin_list()

        if type(plugin_info)==dict:
            plugin_info = [plugin_info]

        for p_info in plugin_info:
            overlap_cli = [i for i in current_clis if i['image']==p_info['image']]

            if len(overlap_cli)>0:
                cli_id = overlap_cli[0]['_id']

                del_cli = self.girder_handler.gc.delete(f'/slicer_cli_web/cli/{cli_id}')
                del_image = self.girder_handler.gc.delete(
                    '/slicer_cli_web/docker_image',
                    parameters = {
                        'name': p_info['image'],
                        'delete_from_local_repo': True
                    })

    def run_plugin(self,plugin_args: Union[list,dict],user_info:dict) -> list:
        """
        Running plugin with provided arguments
        """
        
        if type(plugin_args)==dict:
            plugin_args = [plugin_args]
        
        job_responses = []
        for p_args in plugin_args:
            
            # plugin_args only has to contain plugin specific arguments. Merged with girderApiUrl and girderToken here
            p_job = self.girder_handler.gc.post(
                f'/slicer_cli_web/{p_args["plugin_name"]}/run',
                parameters = p_args['arguments'] | {'girderApiUrl': self.girder_handler.gc.apiUrl} | {'girderToken': user_info['token']}
            )
            job_responses.append(p_job)

        return job_responses

    def get_annotation_image_mask(self,item_id: str, user_info: dict, annotations: list, layer_idx: int, ann_idx: int):
        """
        Pulling image region and returning boundary mask and image within bounding box + padding
        """
        filtered_annotations = [i for i in annotations if 'annotation' in i]
        current_item = filtered_annotations[layer_idx]['annotation']['elements'][ann_idx]

        if current_item['type']=='polyline':
            coordinates = np.squeeze(np.array(current_item['points']))

            # Defining bounding box
            min_x = np.min(coordinates[:,0])-self.padding_pixels
            min_y = np.min(coordinates[:,1])-self.padding_pixels
            max_x = np.max(coordinates[:,0])+self.padding_pixels
            max_y = np.max(coordinates[:,1])+self.padding_pixels

            # Getting image and mask
            image = np.uint8(
                np.array(
                    self.girder_handler.get_image_region(
                        item_id,
                        user_info,
                        [min_x,min_y,max_x,max_y]
                    )
                )
            )
            
            # Scaling coordinates to fit within bounding box
            scaled_coordinates = coordinates.tolist()
            scaled_coordinates = [[i[0]-min_x,i[1]-min_y] for i in scaled_coordinates]

            x_coords = [int(i[0]) for i in scaled_coordinates]
            y_coords = [int(i[1]) for i in scaled_coordinates]

            # Creating mask from scaled coordinates
            height = np.shape(image)[0]
            width = np.shape(image)[1]
            mask = np.zeros((height,width))
            cc,rr = polygon(y_coords,x_coords,(height,width))
            mask[cc,rr] = 1

            return image, mask

    def segment_image(self,item_id: str,structure_types: list):

        # Get folder id from item id
        item_info = self.girder_handler.gc.get(f'/item/{item_id}')
        folder_id = item_info['folderId']
        file_id = item_info['largeImage']['fileId']

        job_responses = []
        for model in self.model_zoo:

            # Testing if a structure from structure_types is included in that model's structures
            model_structures = self.model_zoo[model]['structures']
            selected_in_model = [1 if i in model_structures else 0 for i in structure_types]

            if any(selected_in_model):
                
                if model=='MultiCompartment_Model':
                    job_response = self.girder_handler.gc.post(f'/slicer_cli_web/{self.model_zoo[model]["plugin_name"]}/run',
                                                parameters={
                                                    'girderApiUrl':self.girder_handler.apiUrl,
                                                    'girderToken':self.girder_handler.user_token,
                                                    'files':file_id,
                                                    'base_dir':folder_id,
                                                    'modelfile':self.model_zoo[model]['model_id']
                                                })
                    job_responses.append(job_response)
                elif model=='IFTA_Model':
                    # The other models have slightly different input parameter names :/
                    job_response = self.girder_handler.gc.post(f'/slicer_cli_web/{self.model_zoo[model]["plugin_name"]}/run',
                                                parameters = {
                                                    'girderApiUrl':self.girder_handler.apiUrl,
                                                    'girderToken':self.girder_handler.user_token,
                                                    'input_files':file_id,
                                                    'basedir':folder_id,
                                                    'boxSizeHR':3000,
                                                    'overlap_percentHR':0.5,
                                                    'model':self.model_zoo[model]['model_id']
                                                })
                    job_responses.append(job_response)

                elif model=='PTC_Model':
                    # The other models have slightly different input parameter names :/
                    job_response = self.girder_handler.gc.post(f'/slicer_cli_web/{self.model_zoo[model]["plugin_name"]}/run',
                                                parameters = {
                                                    'girderApiUrl':self.girder_handler.apiUrl,
                                                    'girderToken':self.girder_handler.user_token,
                                                    'input_files':file_id,
                                                    'base_dir':folder_id,
                                                    'patch_size': 256,
                                                    'batch_size': 10,
                                                    'resize':1,
                                                    'model':self.model_zoo[model]['model_id']
                                                })
                    job_responses.append(job_response)
                                
        return job_responses

    def sub_segment_image(self,image:np.ndarray,mask:np.ndarray,seg_params:list,view_method:str,transparency_val:float):
        
        # Sub-compartment segmentation
        sub_comp_image = np.zeros((np.shape(image)[0],np.shape(image)[1],3))
        remainder_mask = np.ones((np.shape(image)[0],np.shape(image)[1]))

        hsv_image = np.uint8(255*rgb2hsv(image))
        hsv_image = hsv_image[:,:,1]

        for idx,param in enumerate(seg_params):

            # Check for if the current sub-compartment is nuclei
            if param['name'].lower()=='nuclei':
                # Using the inverse of the value channel for nuclei
                h_image = 255-np.uint8(255*rgb2hsv(image)[:,:,2])
                h_image = np.uint8(255*exposure.equalize_hist(h_image, mask = mask))

                remaining_pixels = np.multiply(h_image,remainder_mask)
                masked_remaining_pixels = np.multiply(remaining_pixels,mask)

                # Applying manual threshold
                masked_remaining_pixels[masked_remaining_pixels<=param['threshold']] = 0
                masked_remaining_pixels[masked_remaining_pixels>0] = 1

                # Area threshold for holes is controllable for this
                sub_mask = remove_small_holes(masked_remaining_pixels>0,area_threshold=10)
                sub_mask = sub_mask>0
                # Watershed implementation from: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
                distance = ndi.distance_transform_edt(sub_mask)
                labeled_mask, _ = ndi.label(sub_mask)
                coords = peak_local_max(distance,footprint=np.ones((3,3)),labels = labeled_mask)
                watershed_mask = np.zeros(distance.shape,dtype=bool)
                watershed_mask[tuple(coords.T)] = True
                markers, _ = ndi.label(watershed_mask)
                sub_mask = watershed(-distance,markers,mask=sub_mask)
                sub_mask = sub_mask>0

                # Filtering out small objects again
                sub_mask = remove_small_objects(sub_mask,param['min_size'])

            else:

                remaining_pixels = np.multiply(hsv_image,remainder_mask)
                masked_remaining_pixels = np.multiply(remaining_pixels,mask)

                # Applying manual threshold
                masked_remaining_pixels[masked_remaining_pixels<=param['threshold']] = 0
                masked_remaining_pixels[masked_remaining_pixels>0] = 1

                # Filtering by minimum size
                small_object_filtered = (1/255)*np.uint8(remove_small_objects(masked_remaining_pixels>0,param['min_size']))

                sub_mask = small_object_filtered

            sub_comp_image[sub_mask>0,:] = param['color']
            remainder_mask -= sub_mask>0

        # Assigning remaining pixels within the boundary mask to the last sub-compartment
        remaining_pixels = np.multiply(mask,remainder_mask)
        sub_comp_image[remaining_pixels>0,:] = param['color']

        # have to add the final mask thing for the lowest segmentation hierarchy
        if view_method=='Side-by-side':
            # Side-by-side view of sub-compartment segmentation
            sub_comp_image = np.concatenate((image,sub_comp_image),axis=1)
        elif view_method=='Overlaid':
            # Overlaid view of sub-compartment segmentation
            # Processing combined annotations to set black background to transparent
            zero_mask = np.where(np.sum(sub_comp_image.copy(),axis=2)==0,0,255*transparency_val)
            sub_comp_mask_4d = np.concatenate((sub_comp_image,zero_mask[:,:,None]),axis=-1)
            rgba_mask = Image.fromarray(np.uint8(sub_comp_mask_4d),'RGBA')
            
            image = Image.fromarray(np.uint8(image)).convert('RGBA')
            image.paste(rgba_mask, mask = rgba_mask)
            sub_comp_image = np.array(image.copy())[:,:,0:3]

        self.current_sub_comp_image = sub_comp_image

        return sub_comp_image

    def gen_feat_extract_card(self,ftu_names:list):

        # Generating layout of feature extraction card
        card_children = [
            dbc.Row([
                dbc.Col([
                    dbc.Label('Structures for Feature Extraction:',html_for='include-ftu-drop')
                ])
            ]),
            html.B(),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        options = ftu_names,
                        value = [i['value'] for i in ftu_names if not i['disabled']],
                        multi=True,
                        placeholder = 'Select FTUs for feature extraction',
                        id = {'type':'include-ftu-drop','index':0}
                    )
                ])
            ],style={'marginBottom':'20px'}),
            html.B(),
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader('Features to extract'),
                    dbc.CardBody([
                        html.Div('Selecting which types of features to extract here'),
                        dcc.Dropdown(
                            options = ['Distance Transform Features','Color Features','Texture Features', 'Morphological Features'],
                            value = ['Distance Transform Features','Color Features','Texture Features','Morphological Features'],
                            multi = True,
                            placeholder = 'Select Feature Types to extract',
                            id = {'type':'include-feature-drop','index':0}
                        )
                    ])
                ])
            ],style = {'marginBottom':'20px'}),
            html.B(),
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dbc.Button(
                            'Start Extracting!',
                            color = 'success',
                            className='d-grid gap-2 col-12 mx-auto',
                            id = {'type':'start-feat','index':0},
                            n_clicks = 0
                        )
                    )
                ],md=6),
                dbc.Col([
                    html.Div(
                        dbc.Button(
                            'Skip Feature Extraction',
                            color = 'primary',
                            className = 'd-grid gap-2 col-12 mx-auto',
                            id = {'type':'skip-feat','index':0},
                            n_clicks=0
                        )
                    )
                ],md=6)
            ],style = {'marginBottom':'10px'}),
            html.B(),
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader('Feature extraction progress'),
                    dbc.CardBody([
                        html.Div('Feature Extraction Logs',id={'type':'feat-logs','index':0})
                    ])
                ])
            ])
        ]


        return card_children

    def run_feature_extraction(self,image_id:str,sub_seg_params:list,feature_cats:list,ignore_anns:list):

        # Have to pass the file id to feature extraction
        file_id = self.girder_handler.gc.get(f'/item/{image_id}/files')[0]['_id']

        # Parsing through sub-seg-params
        _, thresh_nuc, minsize_nuc, _, _ = tuple(list([i for i in sub_seg_params if i['name']=='Nuclei'][0].values()))
        _, thresh_pas, minsize_pas, _, _ = tuple(list([i for i in sub_seg_params if i['name']=='Eosinophilic'][0].values()))
        _, thresh_las, minsize_ls, _, _ = tuple(list([i for i in sub_seg_params if i['name']=='Luminal Space'][0].values()))

        job_response = self.girder_handler.gc.post(f'/slicer_cli_web/{self.feature_extraction_plugin}/run',
                                        parameters = {
                                            'input_image':file_id,
                                            'threshold_nuclei':thresh_nuc,
                                            'minsize_nuclei':minsize_nuc,
                                            'threshold_PAS':thresh_pas,
                                            'minsize_PAS':minsize_pas,
                                            'threshold_LS':thresh_las,
                                            'minsize_LS':minsize_ls,
                                            'featureCats': feature_cats,
                                            'ignoreAnns': ignore_anns,
                                            'girderApiUrl':self.girder_handler.apiUrl,
                                            'girderToken':self.girder_handler.user_token
                                        })
        return job_response
    
    def process_uploaded_anns(self, filename:str, annotation_str:str,item_id:str, alignment = None):

        annotation_names = []
        annotation_str = base64.b64decode(annotation_str.split(',')[-1])
        annotation_info = None
        
        if not os.path.exists('./assets/conversion/'):
            os.makedirs('./assets/conversion/')
        
        if 'xml' in filename:
            ann = ET.fromstring(annotation_str)

            # Saving annotations locally and converting them to Histomics format
            xml_string = ET.tostring(ann,encoding='unicode',pretty_print=True)
            with open(f'./assets/conversion/{filename}','w') as f:
                f.write(xml_string)
                f.close()

            # Checking for annotation names
            structures_in_xml = ET.parse(f'./assets/conversion/{filename}').getroot().findall('Annotation')
            ann_dict = {}
            for s_idx,s in enumerate(structures_in_xml):
                if not s.attrib['Name']=='':
                    ann_dict[s.attrib['Name']] = s_idx+1
                else:
                    ann_dict[f'Layer_{s_idx+1}'] = s_idx+1

        elif 'json' in filename:
            ann = json.loads(annotation_str)
            ann_dict = {}

            with open(f'./assets/conversion/{filename}','w') as f:
                json.dump(ann,f)
                f.close()

        elif 'csv' in filename:
            # Creating cell polygons from coordinates/group
            csv_anns = pd.read_csv(BytesIO(annotation_str),sep=',')
            print(csv_anns.iloc[0:5,:])
            print(csv_anns.columns.tolist())

            if any(['vertex' in i for i in csv_anns.columns.tolist()]):
                # Creating annotations from csv of vertices
                if 'cell_id' in csv_anns.columns.tolist():
                    unique_cells = csv_anns['cell_id'].unique().tolist()
                    
                    annotation_data = {
                        "annotation": {
                            "name": "Cells",
                            "elements": []
                        }
                    }
                    print(f'Unique cell boundaries: {len(unique_cells)}')
                    for u_idx,u in tqdm(enumerate(unique_cells),total = len(unique_cells)):

                        cell_verts = csv_anns[csv_anns['cell_id'].str.match(u)]
                        cell_vert_x = cell_verts['vertex_x'].values[:,None]
                        cell_vert_y = cell_verts['vertex_y'].values[:,None]
                        cell_vert_z = np.zeros((cell_verts.shape[0],1))

                        vertex_array = np.concatenate((cell_vert_x,cell_vert_y,cell_vert_z),axis=-1)

                        if not alignment is None:
                            vertex_array = np.dot(vertex_array,alignment)

                        cell_element_dict = {
                            "type": "polyline",
                            "id": uuid.uuid4().hex[:24],
                            "points": vertex_array.tolist(),
                            "user": {
                                "cell_id": u
                            }
                        }
                        annotation_data['annotation']['elements'].append(cell_element_dict)

                    self.girder_handler.gc.post(
                        f'/annotation/item/{item_id}',
                        data = json.dumps(annotation_data),
                        headers = {
                            'X-HTTP-Method': "POST",
                            'Content-Type': 'application/json'
                        }
                    )

            if any(['centroid' in i for i in csv_anns.columns.tolist()]):

                # Creating annotations from csv containing centroid and area
                if 'cell_id' in csv_anns.columns.tolist():
                    unique_cells = csv_anns['cell_id'].unique().tolist()

                    annotation_data = {
                        "annotation": {
                            "name": "Cells",
                            "elements": []
                        }
                    }

                    print(f'Unique cell centroids: {len(unique_cells)}')
                    for u_idx, u in tqdm(enumerate(unique_cells),total = len(unique_cells)):

                        cell_center = csv_anns[csv_anns['cell_id'].str.match(u)]
                        cell_center_x = cell_center['x_centroid'].values[:,None]
                        cell_center_y = cell_center['y_centroid'].values[:,None]
                        cell_center_z = np.zeros((1,1))

                        center_array = np.concatenate((cell_center_x,cell_center_y,cell_center_z),axis=-1)

                        if not alignment is None:
                            center_array = np.dot(center_array,alignment)

                        cell_nucleus_area = cell_center['nucleus_area'].values
                        equivalent_radius = (cell_nucleus_area/pi)**(0.5)
                        nucleus_circle = Point(np.squeeze(center_array).tolist()).buffer(equivalent_radius)[0]

                        cell_element_dict = {
                            "type": "polyline",
                            "id": uuid.uuid4().hex[:24],
                            "points": [[i[0],i[1],0] for i in list(nucleus_circle.exterior.coords)],
                            "user": {
                                "cell_id": u,
                                "transcript_counts":cell_center["transcript_counts"].tolist()[0] if "transcript_counts" in cell_center.columns.tolist() else 0
                            }
                        }

                        annotation_data['annotation']['elements'].append(cell_element_dict)
                    
                    self.girder_handler.gc.post(
                        f'/annotation/item/{item_id}',
                        data = json.dumps(annotation_data),
                        headers = {
                            'X-HTTP-Method': "POST",
                            'Content-Type': 'application/json'
                        }
                    )

                    annotation_info = {
                        "Cells": len(unique_cells)
                    }


        if filename in os.listdir('./assets/conversion/'):
            converter_object = wak.Converter(
                starting_file = f'./assets/conversion/{filename}',
                ann_dict = ann_dict,
                verbose = False
            )
            
            annotation_names = converter_object.annotation.structure_names
            annotation_info = {}
            for a in annotation_names:
                annotation_info[a] = len(converter_object.annotation.objects[a])

            print(f'annotation_names: {annotation_names}')
            converted_annotations = wak.Histomics(converter_object.annotation)

            # Posting annotations to uploaded object
            self.girder_handler.gc.post(
                f'/annotation/item/{item_id}',
                data = json.dumps(converted_annotations.json),
                headers = {
                    'X-HTTP-Method':'POST',
                    'Content-Type':'application/json'
                }
            )

            # Removing temporary directory
            shutil.rmtree('./assets/conversion/')
        

        return annotation_info

    def post_segmentation(self, upload_wsi_id:str, upload_annotations:list):

        # What to do after segmentation for a Regular upload

        # Getting annotations and returning layer_anns
        ftu_names = []
        for idx, i in enumerate(upload_annotations):
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
                                'label': i['annotation']['name']+ ' (None detected in slide)',
                                'value':idx,
                                'disabled': True
                            })
                    else:
                        ftu_names.append({
                            'label': i['annotation']['name'] + ' (Not implemented for interstitium)',
                            'value': idx,
                            'disabled': True
                        })

        if not all([i['disabled'] for i in ftu_names]):
            # Initializing layer and annotation idxes (starting with the first one that isn't disabled)
            layer_ann = {
                'current_layer': [i['value'] for i in ftu_names if not i['disabled']][0],
                'current_annotation': 0,
                'previous_annotation': 0,
                'max_layers': [len(i['annotation']['elements']) for i in upload_annotations if 'annotation' in i]
            }
        else:
            layer_ann = None
            ftu_names = [{
                'label': 'No FTUs for Feature Extraction',
                'value':1,
                'disabled':False
            }]

        return ftu_names, layer_ann


class VisiumPrep(Prepper):
    def __init__(self, girder_handler):
        super().__init__(girder_handler)

        # Info for spot annotation plugin
        self.spot_annotation_info = {
            'definitions_file':'64fa0f782d82d04be3e5daa3',
            'plugin_name':'samborder2256_spot_annotation_latest/SpotAnnotation'
        }

        self.cell_deconvolution_plugin = {
            'plugin_name':'sayatmimar_atlasrds_t_7/AtlasRDSCSV',
            'atlas':'65159ea82d82d04be3e73f0a'
        }

        self.spot_aggregation_plugin = 'samborder2256_spot_aggregation_latest/spot_agg'

    def run_spot_aggregation(self,image_id, user_details):
        
        # Getting the fileId for the image item
        image_item = self.girder_handler.gc.get(f'/item/{image_id}')
        fileId = image_item['largeImage']['fileId']
        folderId = image_item['folderId']

        job_response = self.girder_handler.gc.post(f'/slicer_cli_web/{self.spot_aggregation_plugin}/run',
                                                   parameters = {
                                                       'input_image':fileId,
                                                       'basedir':folderId,
                                                       'girderApiUrl':self.girder_handler.apiUrl,
                                                       'girderToken':user_details['token']
                                                   })

        return job_response

    def run_cell_deconvolution(self,image_id,rds_id,user_details):

        # Getting the fileId for the image item
        image_item = self.girder_handler.gc.get(f'/item/{image_id}')
        fileId = image_item['largeImage']['fileId']
        folderId = image_item['folderId']

        # Getting fileId for rds item
        rds_item = self.girder_handler.gc.get(f'/item/{rds_id}/files')
        rds_file_id = rds_item[0]['_id']

        # Running cell deconvolution
        cell_deconv_job = self.girder_handler.gc.post(f'/slicer_cli_web/{self.cell_deconvolution_plugin["plugin_name"]}/run',
                                                   parameters = {
                                                       'inputRDSFile':rds_file_id,
                                                       'atlas':self.cell_deconvolution_plugin['atlas'],
                                                       'outputRDSFile':'output_cell_types.RDS',
                                                       'outputRDSFile_folder':folderId
                                                   })
        return cell_deconv_job

    def run_spot_annotation(self,image_id,omics_id, organ, gene_method, gene_n, gene_list, user_details):

        # Getting the fileId for the image item
        image_item = self.girder_handler.gc.get(f'/item/{image_id}')
        fileId = image_id
        folderId = image_item['folderId']

        omics_info = self.girder_handler.gc.get(f'/item/{omics_id}')
        omics_name = omics_info['name']

        if omics_name.split('.')[-1]=='rds':

            # Getting fileId for rds item
            # Looking for output_cell_types.RDS file
            output_folder_contents = self.girder_handler.gc.get(f'/resource/{folderId}/items',parameters={'limit':10000,'type':'folder'})
            output_folder_names = [i['name'] for i in output_folder_contents]
            if 'output_cell_types.RDS' in output_folder_names:
                rds_item = output_folder_contents[output_folder_names.index('output_cell_types.RDS')]
                rds_file_id = rds_item['_id']

                # Getting fileId for definitions file
                def_file_id = self.spot_annotation_info['definitions_file']

                # Generating spot annotations
                spot_ann_job = self.girder_handler.gc.post(f'/slicer_cli_web/{self.spot_annotation_info["plugin_name"]}/run',
                                                parameters = {
                                                    'counts_file':rds_file_id,
                                                    'definitions_file':def_file_id,
                                                    'input_files':fileId,
                                                    'organ': 'kidney',
                                                    'gene_selection_method': '',
                                                    'n': 0,
                                                    'list': '',
                                                    'girderApiUrl':self.girder_handler.apiUrl,
                                                    'girderToken': user_details['token']
                                                })
                

                return spot_ann_job
        
            else:
                return 'No output found :/'
            
        elif omics_name.split('.')[-1]=='h5ad':

            # Generating spot annotations using gene-selection-method stuff
            # Getting fileId for definitions file

            def_file_id = self.spot_annotation_info['definitions_file']
            if gene_method=='':
                gene_method = 'highly_variable'
                gene_n = 25

            # Generating spot annotations
            spot_ann_job = self.girder_handler.gc.post(f'/slicer_cli_web/{self.spot_annotation_info["plugin_name"]}/run',
                                            parameters = {
                                                'counts_file':omics_id,
                                                'definitions_file':def_file_id,
                                                'input_files':fileId,
                                                'organ': organ,
                                                'gene_selection_method': gene_method,
                                                'n': gene_n,
                                                'list': gene_list,
                                                'girderApiUrl':self.girder_handler.apiUrl,
                                                'girderToken': user_details['token']
                                            })
            
        else:
            print(f'Invalid omics type: {omics_name}')

    def post_segmentation(self, upload_wsi_id, upload_omics_id, upload_annotations, organ, gene_method, gene_n, gene_list, user_details):

        # What to do after segmentation for a Visium upload

        # Generate spot annotations and aggregate --omics info
        spot_annotation = self.run_spot_annotation(upload_wsi_id,upload_omics_id, organ, gene_method, gene_n, gene_list, user_details)
        spot_aggregation = self.run_spot_aggregation(upload_wsi_id, user_details)

        # Getting annotations and returning layer_anns
        ftu_names = []
        for idx, i in enumerate(upload_annotations):
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
                                'label': i['annotation']['name']+ ' (None detected in slide)',
                                'value':idx,
                                'disabled': True
                            })
                    else:
                        ftu_names.append({
                            'label': i['annotation']['name'] + ' (Not implemented for interstitium)',
                            'value': idx,
                            'disabled': True
                        })

        if not all([i['disabled'] for i in ftu_names]):
            # Initializing layer and annotation idxes (starting with the first one that isn't disabled)
            layer_ann = {
                'current_layer': [i['value'] for i in ftu_names if not i['disabled']][0],
                'current_annotation': 0,
                'previous_annotation': 0,
                'max_layers': [len(i['annotation']['elements']) for i in upload_annotations if 'annotation' in i]
            }
        else:
            layer_ann = None
            ftu_names = [{
                'label': 'No FTUs for Feature Extraction',
                'value':1,
                'disabled':False
            }]

        return ftu_names, layer_ann


class CODEXPrep(Prepper):
    def __init__(self, girder_handler):
        super().__init__(girder_handler)

        # Defining plugin information:

        # Registration plugin
        self.registration_plugin = {
            'plugin_name': 'dsarchive_histomicstk_extras_latest/RegisterImage'
        }

        # DeepCell plugin (with post-processing and feature extraction)
        self.cell_seg_plugin = {
            'plugin_name': 'samborder2256_deepcell_plugin_latest/DeepCell_Plugin'
        }

    def post_segmentation(self,upload_wsi_id, upload_codex_id, upload_annotations, user_details):
        """
        If a histology image is provided (with annotations), then show those for morphometrics extraction
        """

        # Registration:
        if not upload_wsi_id is None:
            registration_job = self.girder_handler.gc.post(f'/slicer_cli_web/{self.registration_plugin["plugin"]}/run',
                                                    parameters = {
                                                        'image1': upload_wsi_id,
                                                        'image2': upload_codex_id,
                                                        'girderApiUrl': self.girder_handler.apiUrl,
                                                        'girderToken': user_details['token']
                                                    })
        else:
            registration_job = None

        # Running cell segmentation:
        cell_seg_job = self.girder_handler.gc.post(f'/slicer_cli_web/{self.cell_seg_plugin["plugin_name"]}/run',
                                                   parameters = {
                                                       'input_image': upload_codex_id,
                                                       'input_region': [-1,-1,-1,-1],
                                                       'nuclei_frame': 0,
                                                       'get_features': True,
                                                       'girderApiUrl': self.girder_handler.apiUrl,
                                                       'girderToken': user_details['token']
                                                   })

        # Getting annotations and returning layer_anns
        ftu_names = []
        for idx, i in enumerate(upload_annotations):
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
                                'label': i['annotation']['name']+ ' (None detected in slide)',
                                'value':idx,
                                'disabled': True
                            })
                    else:
                        ftu_names.append({
                            'label': i['annotation']['name'] + ' (Not implemented for interstitium)',
                            'value': idx,
                            'disabled': True
                        })

        if not all([i['disabled'] for i in ftu_names]):
            # Initializing layer and annotation idxes (starting with the first one that isn't disabled)
            layer_ann = {
                'current_layer': [i['value'] for i in ftu_names if not i['disabled']][0],
                'current_annotation': 0,
                'previous_annotation': 0,
                'max_layers': [len(i['annotation']['elements']) for i in upload_annotations if 'annotation' in i]
            }
        else:
            layer_ann = None
            ftu_names = [{
                'label': 'No FTUs for Feature Extraction',
                'value':1,
                'disabled':False
            }]

        return ftu_names, layer_ann


class XeniumPrep(Prepper):
    def __init__(self,
                 girder_handler):
        super().__init__(girder_handler)

        self.girder_handler = girder_handler

        # Adjusting uploaded annotations (cell segmentations)
        self.alignment_plugin = {
            'plugin_name': ''
        }

    def post_segmentation(self, upload_wsi_id, dapi_image_id, upload_annotations, alignment, cell_info):
        """
        # Aligning cell_info centroids with upload_wsi (applying alignment matrix)
        # Processing any uploaded annotations for morphometrics extraction

        """

        # Getting annotations and returning layer_anns
        ftu_names = []
        for idx, i in enumerate(upload_annotations):
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
                                'label': i['annotation']['name']+ ' (None detected in slide)',
                                'value':idx,
                                'disabled': True
                            })
                    else:
                        ftu_names.append({
                            'label': i['annotation']['name'] + ' (Not implemented for interstitium)',
                            'value': idx,
                            'disabled': True
                        })

        if not all([i['disabled'] for i in ftu_names]):
            # Initializing layer and annotation idxes (starting with the first one that isn't disabled)
            layer_ann = {
                'current_layer': [i['value'] for i in ftu_names if not i['disabled']][0],
                'current_annotation': 0,
                'previous_annotation': 0,
                'max_layers': [len(i['annotation']['elements']) for i in upload_annotations if 'annotation' in i]
            }
        else:
            layer_ann = None
            ftu_names = [{
                'label': 'No FTUs for Feature Extraction',
                'value':1,
                'disabled':False
            }]

        return ftu_names, layer_ann


