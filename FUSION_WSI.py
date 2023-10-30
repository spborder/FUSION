"""

External file to hold the WholeSlide class used in FUSION


"""
import os
import numpy as np

from shapely.geometry import shape
import random

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
        self.process_annotations()
    
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
        # Step 6: Converting Histomics/large-image annotations to GeoJSON
        self.geojson_annotations, self.x_scale, self.y_scale = self.convert_json()

        self.map_bounds[0][1]*=self.x_scale
        self.map_bounds[1][1]*=self.y_scale

        # Step 7: Getting user token and tile url
        self.user_token = self.girder_handler.get_token()
        self.tile_url = self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+self.user_token

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
        for a in tqdm(self.annotations):
            if 'elements' in a['annotation']:
                f_name = a['annotation']['name']
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

                # Writing annotations to local assets
                with open(f'./assets/{f_name}.json','w') as f:
                    json.dump(individual_geojson,f)

        return base_x_scale*x_scale, base_y_scale*y_scale

    def process_annotations(self):

        # Processing geojson_annotations:
        self.ftu_names = np.unique([f['properties']['name'] for f in self.geojson_annotations['features']])
        self.ftu_names = [i for i in self.ftu_names if not i=='Spots']

        # Checking if all ftu_names are in the ftu_colors list
        for f in self.ftu_names:
            if f not in self.ftu_colors:
                self.ftu_colors[f] = '#%02x%02x%02x' % (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        #self.geojson_ftus = {'type':'FeatureCollection', 'features': []}
        self.ftu_polys = {}
        for ftu in self.ftu_names:
            self.ftu_polys[ftu] = []
            for f in self.geojson_annotations['features']:
                if f['properties']['name']==ftu:
                    try:
                        self.ftu_polys[ftu].append(shape(f['geometry']))
                    except:
                        print(f'Bad annotation detected')

        self.geojson_ftus = {
            'type':'FeatureCollection',
            'features':[i for i in self.geojson_annotations['features'] if not i['properties']['name']=='Spots']
            }

        self.ftu_props = {
            ftu: [f['properties'] for f in self.geojson_annotations['features'] if f['properties']['name']==ftu]
            for ftu in self.ftu_names
        }

        self.spot_polys = [shape(f['geometry']) for f in self.geojson_annotations['features'] if f['properties']['name']=='Spots']
        self.spot_props = [f['properties'] for f in self.geojson_annotations['features'] if f['properties']['name']=='Spots']

        self.geojson_spots = {
            'type':'FeatureCollection',
            'features':[i for i in self.geojson_annotations['features'] if i['properties']['name']=='Spots']
            }

        # Getting a list of all the properties present in this slide:
        self.properties_list = []
        for ftu in self.ftu_names:
            ftu_props = self.ftu_props[ftu]

            # Iterating through each ftu's properties, checking for sub-dicts
            f_prop_list = []
            for f in ftu_props:
                f_keys = list(f.keys())

                for f_k in f_keys:
                    if f_k in self.visualization_properties:
                        # Limiting depth to 1 sub-property (c'mon)
                        if type(f[f_k])==dict:
                            f_prop_list.extend([f'{f_k} --> {self.girder_handler.cell_graphics_key[i]["full"]}' for i in list(f[f_k].keys())])
                        else:
                            f_prop_list.append(f_k)
            
            self.properties_list.extend(f_prop_list)

        s_prop_list = []
        for s in self.spot_props:
            s_keys = list(s.keys())
            for s_k in s_keys:
                if s_k in self.visualization_properties:
                    if type(s[s_k])==dict:
                        s_prop_list.extend([f'{s_k} --> {self.girder_handler.cell_graphics_key[i]["full"]}' for i in list(s[s_k].keys())])
                    else:
                        s_prop_list.append(s_k)

        self.properties_list.extend(s_prop_list)
        
        self.properties_list = np.unique(self.properties_list).tolist()

        # Adding max cell type if main_cell_types are in the properties list
        main_cell_types_test = [1 if 'Main_Cell_Types' in i else 0 for i in self.properties_list]
        if any(main_cell_types_test):
            self.properties_list.append('Max Cell Type')

        # Making dictionaries for input into vis layout
        self.map_dict = {
            'url': self.tile_url,
            'FTUs':{
                struct: {
                    'geojson': {'type':'FeatureCollection','features':[i for i in self.geojson_annotations['features'] if i['properties']['name']==struct]},
                    'id':{'type':'ftu-bounds','index':idx},
                    'popup_id':{'type':'ftu-popup','index':idx},
                    'color':self.ftu_colors[struct],
                    'hover_color':'#9caf00'
                }
                for idx,struct in enumerate(self.ftu_names)
            }
        }

        self.spot_dict = {
            'geojson':self.geojson_spots,
            'id':{'type':'ftu-bounds','index':len(self.ftu_names)},
            'popup_id':{'type':'ftu-popup','index':len(self.ftu_names)},
            'color':self.ftu_colors['Spots'],
            'hover_color':'#9caf00'
        }

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

