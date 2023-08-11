"""

File containing FUSION pre-processing scripts and workflows

"""

import os
import sys

import numpy as  np

from PIL import Image
from skimage.draw import polygon

class PrepHandler:
    def __init__(self,
                 girder_handler
                 ):

        self.girder_handler = girder_handler


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

    






















