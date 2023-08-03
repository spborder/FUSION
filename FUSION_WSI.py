"""

External file to hold the WholeSlide class used in FUSION


"""

import geojson
import json

import numpy as np

import shapely
from shapely.geometry import Polygon, Point, shape
from skimage.draw import polygon


class DSASlide:
    def __init__(self,
                 slide_name,
                 item_id,
                 tile_url,
                 geojson_annotations,
                 image_dims,
                 base_dims,
                 x_scale,
                 y_scale,
                 manual_rois = [],
                 marked_ftus = []):

        self.slide_name = slide_name
        self.item_id = item_id
        self.tile_url = tile_url
        self.geojson_annotations = geojson_annotations
        self.image_dims = image_dims
        self.base_dims = base_dims
        self.manual_rois = manual_rois
        self.marked_ftus = marked_ftus

        # Initializing conversion scales
        self.x_scale = x_scale
        self.y_scale = y_scale

        # Processing geojson_annotations:
        self.ftu_names = np.unique([f['properties']['name'] for f in self.geojson_annotations['features']])
        self.ftu_names = [i for i in self.ftu_names if not i=='Spots']

        #self.geojson_ftus = {'type':'FeatureCollection', 'features': []}
        self.ftu_polys = {
            ftu: [shape(f['geometry']) for f in self.geojson_annotations['features'] if f['properties']['name']==ftu]
            for ftu in self.ftu_names
        }

        self.geojson_ftus = {'type':'FeatureCollection','features':[i for i in self.geojson_annotations['features'] if not i['properties']['name']=='Spots']}

        self.ftu_props = {
            ftu: [f['properties'] for f in self.geojson_annotations['features'] if f['properties']['name']==ftu]
            for ftu in self.ftu_names
        }

        self.spot_polys = [shape(f['geometry']) for f in self.geojson_annotations['features'] if f['properties']['name']=='Spots']
        self.spot_props = [f['properties'] for f in self.geojson_annotations['features'] if f['properties']['name']=='Spots']

        self.geojson_spots = {'type':'FeatureCollection','features':[i for i in self.geojson_annotations['features'] if i['properties']['name']=='Spots']}

        # Getting a list of all the properties present in this slide:
        self.properties_list = []
        for ftu in self.ftu_names:
            ftu_props = self.ftu_props[ftu]

            # Iterating through each ftu's properties, checking for sub-dicts
            f_prop_list = []
            for f in ftu_props:
                f_keys = list(f.keys())

                for f_k in f_keys:
                    # Limiting depth to 1 sub-property (c'mon)
                    if type(f[f_k])==dict:
                        f_prop_list.extend([f'{f_k} --> {i}' for i in list(f[f_k].keys())])
                    else:
                        f_prop_list.append(f_k)
            
            self.properties_list.extend(f_prop_list)
        
        self.properties_list = np.unique(self.properties_list)


    def find_intersecting_spots(self,box_poly):

        # Finging intersecting spots
        intersecting_spot_idxes = [i for i in range(0,len(self.spot_polys)) if self.spot_polys[i].intersects(box_poly)]
        
        # Returning list of dictionaries using original keys
        intersecting_spot_props = []
        if len(intersecting_spot_idxes)>0:
            intersecting_spot_props = [self.spot_props[i] for i in intersecting_spot_idxes]

        return intersecting_spot_props

    def find_intersecting_ftu(self, box_poly, ftu: str):

        if ftu in self.ftu_names:
            
            # Finding which members of a specfied ftu group intersect with the provided box_poly
            ftu_intersect_idx = [i for i in range(0,len(self.ftu_polys[ftu])) if self.ftu_polys[ftu][i].intersects(box_poly)]
            
            # Returning list of dictionaries that use original keys in properties
            intersecting_ftu_props = []
            if len(ftu_intersect_idx)>0:
                intersecting_ftu_props = [self.ftu_props[ftu][i] for i in ftu_intersect_idx]

            return intersecting_ftu_props
        else:
            raise ValueError
        
    def convert_map_coords(self, input_coords):
        
        print(f'x_scale: {self.x_scale}')
        print(f'y_scale: {self.y_scale}')

        # Convert map coordinates to slide coordinates
        # input_coords are in terms of the tile map and returned coordinates are relative to the slide pixel dimensions
        return_coords = []
        for i in input_coords:
            return_coords.append([i[0]/self.x_scale,i[1]/self.y_scale])

        return return_coords



