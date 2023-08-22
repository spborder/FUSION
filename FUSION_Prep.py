"""

File containing FUSION pre-processing scripts and workflows

"""

import os
import sys

import numpy as  np

from PIL import Image
from skimage.draw import polygon
from skimage.color import rgb2hsv
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage import exposure

from matplotlib import colormaps

import dash_bootstrap_components as dbc
from dash import dcc, html


class PrepHandler:
    def __init__(self,
                 girder_handler
                 ):

        self.girder_handler = girder_handler

        # Dictionary containing model and item id's
        self.model_zoo = {
            'Kidney':'648123751019450486d13dcd'
        }

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
        folder_id = self.girder_handler.gc.get(f'/item/{item_id}')
        print(f'item info: {folder_id}')
        folder_id = folder_id['folderId']

        if model_type=='Kidney':
            job_response = self.girder_handler.gc.post('/slicer_cli_web/sarderlab_histo-cloud_Segmentation/SegmentWSI/run',
                                        parameters={
                                            'inputImageFile':item_id,
                                            'outputAnnotationFile_folder':folder_id
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
            masked_remaining_pixels[masked_remaining_pixels<param['threshold']] = 0
            masked_remaining_pixels[masked_remaining_pixels>0] = 1

            # Filtering by minimum size
            small_object_filtered = (1/255)*np.uint8(remove_small_objects(masked_remaining_pixels>0,param['min_size']))
            # Check for if the current sub-compartment is nuclei
            if param['name'].lower()=='nuclei':
                
                # Area threshold for holes is controllable for this
                sub_mask = remove_small_holes(small_object_filtered>0,area_threshold=10)
                sub_mask = sub_mask>0
                # Watershed implementation from: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
                distance = ndi.distance_transform_edt(sub_mask)
                coords = peak_local_max(distance,footprint=np.ones((3,3)),labels = label(sub_mask))
                watershed_mask = np.zeros(distance.shape,dtype=bool)
                watershed_mask[tuple(coords.T)] = True
                markers, _ = ndi.label(watershed_mask)
                sub_mask = watershed(-distance,markers,mask=sub_mask)
                sub_mask = sub_mask>0

            else:
                sub_mask = small_object_filtered

            sub_comp_image[sub_mask>0,:] = param['color']
            remainder_mask -= sub_mask>0

        # Assigning remaining pixels within the boundary mask to the last sub-compartment
        masked_remaining_pixels = np.multiply(remaining_pixels,mask)
        sub_comp_image[masked_remaining_pixels>0] = param['color']

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
                        html.Div('Selecting which types of features to extract here')
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
                            id = 'start-feat'
                        )
                    )
                ])
            ],style = {'marginBottom':'10px'}),
            html.B(),
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader('Feature extraction progress'),
                    dbc.CardBody([
                        html.Div('Record logs/progress here')
                    ])
                ])
            ])
        ]


        return card_children




