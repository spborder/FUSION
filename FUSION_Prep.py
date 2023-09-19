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

class PrepHandler:
    def __init__(self,
                 girder_handler
                 ):

        self.girder_handler = girder_handler

        # Dictionary containing model and item id's
        self.model_zoo = {
            'Kidney':'64f0b3f82d82d04be3e2b4ba'
        }

        # Info for spot annotation plugin
        self.spot_annotation_info = {
            'definitions_file':'64fa0f782d82d04be3e5daa3',
            'plugin_name':'dpraveen511_spot_spot_ec2/SpotAnnotation'
        }

        self.feature_extraction_plugin = 'sumanthdevarasetty_ftx_ftx_19/Ftx_sc'

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
                'name':'PAS',
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

    def get_annotation_image_mask(self,item_id,annotations,layer_idx,ann_idx):

        # Codes taken from Annotaterator
        print(f'item_id: {item_id}')
        print(f'layer_idx: {layer_idx}')
        print(f'ann_idx: {ann_idx}')
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
            image = np.uint8(np.array(self.girder_handler.get_image_region(item_id,[min_x,min_y,max_x,max_y])))
            
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

    def segment_image(self,item_id,model_type):

        # Get folder id from item id
        item_info = self.girder_handler.gc.get(f'/item/{item_id}')
        folder_id = item_info['folderId']
        file_id = item_info['largeImage']['fileId']

        if model_type=='Kidney':
            job_response = self.girder_handler.gc.post('/slicer_cli_web/sayatmimar_histo-cloud_MultiCompartmentSegment_2/MultiCompartmentSegment/run',
                                        parameters={
                                            'girderApiUrl':self.girder_handler.apiUrl,
                                            'girderToken':self.girder_handler.user_token,
                                            'input_file':file_id,
                                            'base_dir':folder_id,
                                            'modelfile':self.model_zoo[model_type]
                                        })

            print(f'job_response: {job_response}')

        else:
            job_response = 'Oopsie Poopsie!'

        return job_response

    def sub_segment_image(self,image,mask,seg_params,view_method,transparency_val):
        
        # Sub-compartment segmentation
        sub_comp_image = np.zeros((np.shape(image)[0],np.shape(image)[1],3))
        remainder_mask = np.ones((np.shape(image)[0],np.shape(image)[1]))

        hsv_image = np.uint8(255*rgb2hsv(image)[:,:,1])

        # Applying adaptive histogram equalization
        #hsv_image = rank.equalize(hsv_image,footprint=disk(30))
        hsv_image = np.uint8(255*exposure.equalize_hist(hsv_image))

        for idx,param in enumerate(seg_params):

            remaining_pixels = np.multiply(hsv_image,remainder_mask)
            masked_remaining_pixels = np.multiply(remaining_pixels,mask)

            # Applying manual threshold
            masked_remaining_pixels[masked_remaining_pixels<=param['threshold']] = 0
            masked_remaining_pixels[masked_remaining_pixels>0] = 1

            # Check for if the current sub-compartment is nuclei
            if param['name'].lower()=='nuclei':
                
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

    def gen_feat_extract_card(self,ftu_names):

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
                        id = 'include-ftu-drop'
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
                            id = 'include-feature-drop'
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
                            id = {'type':'start-feat','index':0}
                        )
                    )
                ])
            ],style = {'marginBottom':'10px'}),
            html.B(),
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader('Feature extraction progress'),
                    dbc.CardBody([
                        html.Div('Record logs/progress here',id={'type':'feat-logs','index':0})
                    ])
                ])
            ])
        ]


        return card_children

    def gen_spot_annotations(self,image_id,rds_id):

        # Getting the fileId for the image item
        image_item = self.girder_handler.gc.get(f'/item/{image_id}')
        fileId = image_item['largeImage']['fileId']
        folderId = image_item['folderId']

        # Getting fileId for rds item
        rds_item = self.girder_handler.gc.get(f'/item/{rds_id}/files')
        rds_file_id = rds_item[0]['_id']

        # Getting fileId for definitions file
        def_item = self.girder_handler.gc.get(f'/item/{self.spot_annotation_info["definitions_file"]}/files')
        def_file_id = def_item[0]['_id']

        job_response = self.girder_handler.gc.post(f'/slicer_cli_web/{self.spot_annotation_info["plugin_name"]}/run',
                                        parameters = {
                                            'rds_file':rds_file_id,
                                            'definitions_file':def_file_id,
                                            'input_files':fileId,
                                            'basedir':folderId,
                                            'girderApiUrl':self.girder_handler.apiUrl,
                                            'girderToken':self.girder_handler.user_token
                                        })
        

        return job_response

    def run_feature_extraction(self,image_id,sub_seg_params):
        
        # Getting the fileId for the image item
        image_item = self.girder_handler.gc.get(f'/item/{image_id}')
        fileId = image_item['largeImage']['fileId']
        folderId = image_item['folderId']

        # Parsing through sub-seg-params
        _, thresh_nuc, minsize_nuc, _, _ = tuple(list([i for i in sub_seg_params if i['name']=='Nuclei'][0].values()))
        _, thresh_pas, minsize_pas, _, _ = tuple(list([i for i in sub_seg_params if i['name']=='PAS'][0].values()))
        _, thresh_las, minsize_las, _, _ = tuple(list([i for i in sub_seg_params if i['name']=='Luminal Space'][0].values()))

        job_response = self.girder_handler.gc.post(f'/slicer_cli_web/{self.feature_extraction_plugin}/run',
                                        parameters = {
                                            'input_image':fileId,
                                            'basedir':folderId,
                                            'threshold_nuclei':thresh_nuc,
                                            'minsize_nuclei':minsize_nuc,
                                            'threshold_PAS':thresh_pas,
                                            'minsize_PAS':minsize_pas,
                                            'threshold_LAS':thresh_las,
                                            'minsize_LAS':minsize_las,
                                            'girderApiUrl':self.girder_handler.apiUrl,
                                            'girderToken':self.girder_handler.user_token
                                        })
        
        return job_response
    


class CellAnnotations:
    def __init__(self,
                 item_id,
                 ftus,
                 spots,
                 counts_definitions,
                 gc,
                 output_file,
                 trial_run=False,
                 names_key=None
                 ):
        
        self.item_id = item_id
        self.ftus = ftus
        self.spots = spots
        self.counts_definitions = counts_definitions
        self.gc = gc
        self.names_key = names_key
        self.output_file = output_file
        self.trial_run = trial_run

        self.user_token = self.gc.get('token/session')['token']

        self.spot_polys, self.spot_properties = self.process_spots()
        self.ftu_annotations = self.process_ftus()

        self.post_outputs()

    def process_spots(self):

        # Iterating through spot annotation elements
        spot_polys = []
        spot_properties = []
        for s_idx, s in enumerate(self.spots['annotation']['elements']):
            og_coords = np.squeeze(np.array(s['points'])).tolist()

            # Converting spot into shapely Polygon
            coords = [(i[0],i[1]) for i in og_coords]
            spot_poly = Polygon(coords)

            # Adding spot properties to list
            if 'user' in s:
                spot_props = s['user']
            else:
                spot_props = {}

            spot_polys.append(spot_poly)
            spot_properties.append(spot_props)

        return spot_polys, spot_properties

    def process_ftus(self):

        # Initialize annotations object
        ftu_annotations = wak.Annotation()

        # Iterate through annotations
        for a in self.ftus:
            # If someone runs nuclei segmentation this doesn't have a "name" so we don't have to aggregate over each nucleus
            if 'name' in a['annotation']:

                # Changing the name if it's in the names key
                ftu_name = a['annotation']['name']
                if not self.names_key is None:
                    if a['annotation']['name'] in self.names_key:
                        ftu_name = self.names_key[a['annotation']['name']]
                    
                ftu_annotations.add_names([ftu_name])

                if 'elements' in a['annotation']:

                    for f_idx,f in enumerate(a['annotation']['elements']):

                        # For polyline annotations
                        if f['type']=='polyline':
                            og_coords = np.squeeze(np.array(f['points']))
                            coords = [(i[0],i[1]) for i in og_coords]

                            ftu_poly = Polygon(coords)
                        
                        elif f['type'] =='rectangle':
                            width = f['width']
                            height = f['height']
                            center = f['center'][0:-1]

                            bbox_coords = [
                                [int(center[0])-int(width/2),int(center[1])-int(height/2)],
                                [int(center[0])+int(width/2),int(center[1])-int(height/2)],
                                [int(center[0])+int(width/2),int(center[1])+int(height/2)],
                                [int(center[0])-int(width/2),int(center[1])+int(height/2)]
                            ]

                            ftu_poly = Polygon([
                                (bbox_coords[0],bbox_coords[3]),
                                (bbox_coords[0],bbox_coords[1]),
                                (bbox_coords[2],bbox_coords[1]),
                                (bbox_coords[2],bbox_coords[3]),
                                (bbox_coords[0],bbox_coords[3])
                            ])

                        intersecting_spot_props = []
                        intersecting_spot_areas = []
                        intersecting_spot_keys = []
                        for s_idx,s in enumerate(self.spot_polys):
                            
                            # Checking if a given FTU intersects with any spots
                            if ftu_poly.intersects(s):
                                try:
                                    intersecting_spot_areas.append(ftu_poly.intersection(s).area)
                                    intersecting_spot_props.append(self.spot_properties[s_idx])
                                    intersecting_spot_keys.extend(list(self.spot_properties[s_idx].keys()))
                                except:
                                    print(f'shapely error')
                                    continue
                        if len(intersecting_spot_areas)>0:
                            
                            agg_props = np.unique(intersecting_spot_keys)

                            ftu_props = {}
                            for a in agg_props:
                                
                                # This should be a list of either dictionaries or single valuesH
                                i_prop_list = [i[a] for i in intersecting_spot_props if a in i]
                                has_sub_props = False
                                str_prop = False
                                # Convert the prop to some kind of table
                                if type(i_prop_list[0])==dict:
                                    if not type(i_prop_list[0][list(i_prop_list[0].keys())[0]])==dict:
                                        # Where each shared key becomes a column in the dataframe
                                        i_prop_table = pd.DataFrame.from_records(i_prop_list)
                                    else:
                                        has_sub_props = True

                                elif type(i_prop_list[0])==float or type(i_prop_list[0])==int:
                                    # Where each value in the list becomes a single row in the dataframe (This might need to be a series).
                                    i_prop_table = pd.DataFrame(data = i_prop_list, columns=[a])
                                elif type(i_prop_list[0])==str:
                                    # Where each value in the list becomes a single row in the dataframe
                                    i_prop_table = pd.DataFrame(data = i_prop_list,columns=[a])
                                    str_prop = True
                                else:
                                    print(f'Unsupported property type: {a} has type: {type(i_prop_list[0])}')
                                    raise TypeError
                                                                
                                if not has_sub_props and not str_prop:
                                    # Multiplying rows of the prop table by overlap area and re-normalizing to sum to 1
                                    weighted_prop_table = i_prop_table.mul(intersecting_spot_areas,axis=0)
                                    # Summing on rows axis
                                    sum_prop = weighted_prop_table.sum(axis=0)
                                    # Normalizing by sum of sums
                                    norm_prop = (sum_prop/(sum_prop.sum())).fillna(0)
                                    # Adding weighted and normalized property to ftu properties dictionary
                                    ftu_props[a] = norm_prop.to_dict()
                                
                                if has_sub_props and not str_prop:
                                    ftu_props[a] = {}
                                    # Iterating through sub-props
                                    sub_props_list = []
                                    for i_s in i_prop_list:
                                        sub_props_list.extend(list(i_s.keys()))

                                    sub_props_list = np.unique(sub_props_list)
                                    for s_p in sub_props_list:
                                        s_prop = [i[s_p] for i in i_prop_list if s_p in i]

                                        if type(s_prop[0])==dict:
                                            if not type(s_prop[0][list(s_prop[0].keys())[0]])==dict:
                                                s_prop_table = pd.DataFrame.from_records(s_prop)
                                            else:
                                                print(f'Property: {a} exceeds number of allowed subproperties (2)')
                                                raise TypeError
                                        elif type(s_prop[0])==float or type(s_prop[0])==int:
                                            s_prop_table = pd.DataFrame(data = s_prop,columns=[s_p])
                                        elif type(s_prop[0])==str:
                                            s_prop_table = pd.DataFrame(data = s_prop,columns=[s_p])
                                            str_prop = True

                                    if not str_prop:
                                        weighted_prop_table = s_prop_table.mul(intersecting_spot_areas,axis=0)
                                        sum_prop = weighted_prop_table.sum(axis=0)
                                        norm_prop = (sum_prop/(sum_prop.sum())).fillna(0)
                                        ftu_props[a][s_p] = norm_prop.to_dict()

                                    #TODO: Some method for string properties in spots, maybe just present a list of all those?

                        else:
                            ftu_props = {}
                            
                        # Adding shape to full ftu_annotations object
                        ftu_annotations.add_shape(
                            poly = ftu_poly,
                            box_crs = [0,0],
                            structure = ftu_name,
                            name = f'{ftu_name.strip()}_{f_idx}',
                            properties = ftu_props
                        )

        return ftu_annotations

    def post_outputs(self):

        # Saving outputs, reading them, then posting them
        # Don't have a direct return json formatted method for wak yet

        self.ftu_annotations.json_save(self.output_file)
        #self.spots.json_save(self.output_file.replace('.json','_Spots.json'))

        new_annotations = json.load(open(self.output_file))
        new_annotations.append(json.load(open(self.output_file.replace('.json','_Spots.json'))))
        if not self.trial_run:
            # Deleting old annotations
            self.gc.delete(f'/annotation/item/{self.item_id}?token={self.user_token}')
            self.gc.post(f'/annotation/item/{self.item_id}?token={self.user_token}',
                         data = json.dumps(new_annotations),
                         headers={'X-HTTP-Method':'POST','Content-Type':'application/json'}
                         )
        else:
            print(f'Generated annotations for: {len(new_annotations)} structures')
            for n_idx,n_a in enumerate(new_annotations):
                print(f'Structure {n_idx}: {n_a["name"]}')
                print(f'{len(n_a["elements"])} elements')

    def post_outputs(self):

        # Saving outputs, reading them, then posting them
        # Don't have a direct return json formatted method for wak yet

        self.ftu_annotations.json_save(self.output_file)
        #self.spots.json_save(self.output_file.replace('.json','_Spots.json'))

        new_annotations = json.load(open(self.output_file))
        new_annotations.append(self.spots)
        if not self.trial_run:
            # Deleting old annotations
            self.gc.delete(f'/annotation/item/{self.item_id}?token={self.user_token}')
            self.gc.post(f'/annotation/item/{self.item_id}?token={self.user_token}',
                         data = json.dumps(new_annotations),
                         headers={'X-HTTP-Method':'POST','Content-Type':'application/json'}
                         )
        else:
            print(f'Generated annotations for: {len(new_annotations)} structures')
            for n_idx,n_a in enumerate(new_annotations):
                if 'name' in n_a:
                    print(f'Structure {n_idx}: {n_a["name"]}')
                    print(f'{len(n_a["elements"])} elements')




