"""

File containing FUSION pre-processing scripts and workflows

"""

import os
import sys

import numpy as  np

from PIL import Image
from skimage.draw import polygon
from skimage.color import rgb2hsv
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed
from skimage.measure import label

class PrepHandler:
    def __init__(self,
                 girder_handler
                 ):

        self.girder_handler = girder_handler

        # Dictionary containing model and item id's
        self.model_zoo = {
            'Kidney':'648123751019450486d13dcd'
        }


    def get_annotation_image_mask(self,item_id,annotations,layer_idx,ann_idx):

        # Codes taken from Annotaterator
        current_item = annotations[layer_idx]['annotation']['elements'][ann_idx]

        if current_item['type']=='polyline':
            coordinates = np.squeeze(np.array(current_item['points']))

            # Defining bounding box
            min_x = np.min(coordinates[:,0])
            min_y = np.min(coordinates[:,1])
            max_x = np.max(coordinates[:,0])
            max_y = np.max(coordinates[:,1])

            # Getting image and mask
            image = np.uint8(np.array(self.girder_handler.get_image_region(item_id,[min_x,min_y,max_x,max_y])))
            
            # Scaling coordinates to fit within bounding box
            scaled_coordinates = coordinates.tolist()
            scaled_coordinates = [[i[0]-min_x,i[1]-min_y] for i in scaled_coordinates]

            x_coords = [int(i[0]) for i in scaled_coordinates]
            y_coords = [int(i[1]) for i in scaled_coordinates]

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

    def sub_segment_image(self,image,mask,seg_params):
        
        # Sub-compartment segmentation
        sub_comp_image = np.zeros((np.shape(image)[0],np.shape(image)[1],len(seg_params)))
        remainder_mask = np.ones((np.shape(image)[0],np.shape(image)[1]))
        hsv_image = rgb2hsv(sub_comp_image)[:,:,1]
        for idx,param in enumerate(seg_params):

            remaining_pixels = np.multiply(hsv_image,remainder_mask)
            masked_remaining_pixels = np.multiply(remaining_pixels,mask)

            # Applying manual threshold
            masked_remaining_pixels[masked_remaining_pixels<param['threshold']] = 0
            masked_remaining_pixels[masked_remaining_pixels>0] = 1

            # Filtering by minimum size
            small_object_filtered = (1/255)*np.uint8(remove_small_objects(masked_remaining_pixels,param['min_size']))

            # Check for if the current sub-compartment is nuclei
            if param['name'].lower()=='nuclei':
                
                # Area threshold for holes is controllable for this
                sub_mask = remove_small_holes(small_object_filtered,area_threshold=64)
                sub_mask = watershed(sub_mask,label(sub_mask))
                sub_mask = sub_mask>0

            else:
                sub_mask = small_object_filtered

            sub_comp_image[:,:,idx] += sub_mask
            remainder_mask -= sub_mask

            # have to add the final mask thing for the lowest segmentation hierarchy

        return sub_comp_image






