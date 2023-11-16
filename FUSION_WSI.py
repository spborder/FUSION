"""

External file to hold the WholeSlide class used in FUSION


"""
import os
import numpy as np

from shapely.geometry import shape
import random
import json
import shutil

from tqdm import tqdm

class DSASlide:
    def __init__(self,
                 slide_name,
                 item_id,
                 girder_handler,
                 ftu_colors,
                 manual_rois = [],
                 marked_ftus = []):

        self.slide_name = slide_name
        self.slide_ext = slide_name.split('.')[-1]
        self.item_id = item_id
        self.girder_handler = girder_handler
        self.ftu_colors = ftu_colors

        self.manual_rois = manual_rois
        self.marked_ftus = marked_ftus

        self.visualization_properties = [
            'Area', 'Arterial Area', 'Average Cell Thickness', 'Average TBM Thickness', 'Cluster',
            'Luminal Fraction','Main_Cell_Types','Mesangial Area','Mesangial Fraction'
        ]

        self.get_slide_map_data(self.item_id)
    
    def __str__(self):
        
        return f'{self.slide_name}'

    def get_slide_map_data(self,resource):
        # Getting all of the necessary materials for loading a new slide

        # Step 1: get resource item id
        # lol
        item_id = resource

        # Step 2: get resource tile metadata
        tile_metadata = self.girder_handler.get_tile_metadata(item_id)
        # Step 3: get tile, base, zoom, etc.
        # Number of zoom levels for an image
        self.zoom_levels = tile_metadata['levels']
        # smallest level dimensions used to generate initial tiles
        self.base_dims = [
            tile_metadata['sizeX']/(2**(self.zoom_levels-1)),
            tile_metadata['sizeY']/(2**(self.zoom_levels-1))
        ]
        # Getting the tile dimensions (used for all tiles)
        self.tile_dims = [
            tile_metadata['tileWidth'],
            tile_metadata['tileHeight']
        ]
        # Original image dimensions used for scaling annotations
        self.image_dims = [
            tile_metadata['sizeX'],
            tile_metadata['sizeY']
        ]

        # Step 4: Defining bounds of map
        self.map_bounds = [[0,self.image_dims[1]],[0,self.image_dims[0]]]

        # Step 5: Getting annotations for a resource
        self.annotations = self.girder_handler.get_annotations(item_id)
        print(f'Found: {len(self.annotations)} Annotations')
        # Step 6: Getting user token and tile url
        self.user_token = self.girder_handler.get_token()
        self.tile_url = self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+self.user_token

        # Step 7: Converting Histomics/large-image annotations to GeoJSON
        self.x_scale, self.y_scale = self.convert_json()

        self.map_bounds[0][1]*=self.x_scale
        self.map_bounds[1][1]*=self.y_scale

    def convert_json(self):

        # Translation step
        base_x_scale = self.base_dims[0]/self.tile_dims[0]
        base_y_scale = self.base_dims[1]/self.tile_dims[1]
        print(f'base_x_scale: {base_x_scale}, base_y_scale: {base_y_scale}')

        # image bounds [maxX, maxY]
        # bottom_right[0]-top_left[0] --> range of x values in target crs
        # bottom_right[1]-top_left[1] --> range of y values in target crs
        # scaling values so they fall into the current map (pixels)
        x_scale = self.tile_dims[0]/self.image_dims[0]
        y_scale = self.tile_dims[1]/self.image_dims[1]
        y_scale*=-1

        print(f'x_scale: {x_scale}, y_scale: {y_scale}')
        # y_scale has to be inverted because y is measured upward in these maps

        ## Error occurs with tile sizes = 256, all work with tile size=240 ##

        print('Processing annotations')
        self.ftu_names = []
        self.ftu_polys = {}
        self.ftu_props = {}
        self.spot_polys = []
        self.spot_props = []
        self.properties_list = []

        self.map_dict = {
            'url': self.tile_url,
            'FTUs':{}
        }

        included_props = []
        # Emptying current ./assets/slide_annotations/ folder
        if os.path.exists('./assets/slide_annotations/'):
            shutil.rmtree('./assets/slide_annotations/')

        for a in tqdm(self.annotations):
            if 'elements' in a['annotation']:
                f_name = a['annotation']['name']
                self.ftu_names.append(f_name)

                if not f_name=='Spots':
                    self.ftu_polys[f_name] = []
                    self.ftu_props[f_name] = []

                # Checking if this ftu is in the ftu_colors list
                if f_name not in self.ftu_colors:
                    self.ftu_colors[f_name] = '#%02x%02x%02x' % (random.randint(0,255),random.randint(0,255),random.randint(0,255))

                individual_geojson = {'type':'FeatureCollection','features':[]}
                for f in tqdm(a['annotation']['elements']):
                    f_dict = {'type':'Feature','geometry':{'type':'Polygon','coordinates':[]},'properties':{}}
                    
                    # This is only for polyline type elements
                    if f['type']=='polyline':
                        og_coords = np.squeeze(np.array(f['points']))
                        
                        scaled_coords = og_coords.tolist()
                        scaled_coords = [i[0:-1] for i in scaled_coords]
                        scaled_coords = [[base_x_scale*((i[0]*x_scale)),base_y_scale*((i[1]*y_scale))] for i in scaled_coords]
                        f_dict['geometry']['coordinates'] = [scaled_coords]

                    elif f['type']=='rectangle':
                        width = f['width']
                        height = f['height']
                        center = f['center'][0:-1]
                        # Coords: top left, top right, bottom right, bottom left
                        bbox_coords = [
                            [int(center[0])-int(width/2),int(center[1])-int(height/2)],
                            [int(center[0])+int(width/2),int(center[1])-int(height/2)],
                            [int(center[0])+int(width/2),int(center[1])+int(height/2)],
                            [int(center[0])-int(width/2),int(center[1])+int(height/2)]
                        ]
                        scaled_coords = [[base_x_scale*(i[0]*x_scale),base_y_scale*(i[1]*y_scale)] for i in bbox_coords]
                        f_dict['geometry']['coordinates'] = [scaled_coords]

                    # Who even cares about circles and ellipses??
                    # If any user-provided metadata is provided per element add it to "properties" key                       
                    if 'user' in f:
                        f_dict['properties'] = f['user']

                    f_dict['properties']['name'] = f_name
                    individual_geojson['features'].append(f_dict)

                    if not f_name=='Spots':
                        self.ftu_polys[f_name].append(shape(f_dict['geometry']))
                        self.ftu_props[f_name].append(f_dict['properties'])
                    else:
                        self.spot_polys.append(shape(f_dict['geometry']))
                        self.spot_props.append(f_dict['properties'])

                    for p in f_dict['properties']:
                        if p not in included_props:
                            if p in self.visualization_properties:
                                
                                if type(f_dict['properties'][p])==dict:
                                    included_props.append(p)
                                    f_k = list(f_dict['properties'][p].keys())
                                    if p=='Main_Cell_Types':
                                        f_prop_list = [f'{p} --> {self.girder_handler.cell_graphics_key[i]["full"]}' for i in f_k]
                                    else:
                                        f_prop_list = [f'{p} --> {i}' for i in f_k]
                                else:
                                    included_props.append(p)
                                    f_prop_list = [p]

                                self.properties_list.extend(f_prop_list)

                self.map_dict['FTUs'][f_name] = {
                    'id':{'type':'ftu-bounds','index':len(self.ftu_names)-1},
                    'popup_id':{'type':'ftu-popup','index':len(self.ftu_names)-1},
                    'color':self.ftu_colors[f_name],
                    'hover_color':'#9caf00'
                }

                # Writing annotations to local assets
                # Emptying current ./assets/slide_annotations/ folder
                if not os.path.exists('./assets/slide_annotations/'):
                    os.makedirs('./assets/slide_annotations/')
                    
                with open(f'./assets/slide_annotations/{f_name}.json','w') as f:
                    json.dump(individual_geojson,f)
        
        self.properties_list = np.unique(self.properties_list).tolist()
        main_cell_types_test = [1 if 'Main_Cell_Types' in i else 0 for i in self.properties_list]
        if any(main_cell_types_test):
            self.properties_list.append('Max Cell Type')


        return base_x_scale*x_scale, base_y_scale*y_scale

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
        elif ftu=='all':

            # Specific check for when adding a marker to the map, returns both the props and poly
            intersecting_ftu_props = {}
            intersecting_ftu_poly = None
            for ftu in self.ftu_names:
                ftu_intersect_idx = [i for i in range(0,len(self.ftu_polys[ftu])) if self.ftu_polys[ftu][i].intersects(box_poly)]

                if len(ftu_intersect_idx)==1:
                    intersecting_ftu_props = self.ftu_props[ftu][ftu_intersect_idx[0]]
                    intersecting_ftu_poly = self.ftu_polys[ftu][ftu_intersect_idx[0]]

            return intersecting_ftu_props, intersecting_ftu_poly
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

