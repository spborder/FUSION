"""

External file to hold the WholeSlide class used in FUSION


"""
import os
import numpy as np

from shapely.geometry import shape
import random
import json
import geojson
import shutil

from tqdm import tqdm
from FUSION_Utils import extract_overlay_value

class DSASlide:

    spatial_omics_type = 'Regular'

    def __init__(self,
                 item_id,
                 girder_handler,
                 ftu_colors,
                 manual_rois = [],
                 marked_ftus = []):

        self.item_id = item_id
        self.girder_handler = girder_handler

        self.item_info = self.girder_handler.gc.get(f'/item/{self.item_id}?token={self.girder_handler.user_token}')
        self.slide_name = self.item_info['name']
        self.slide_ext = self.slide_name.split('.')[-1]
        self.ftu_colors = ftu_colors

        self.manual_rois = manual_rois
        self.marked_ftus = marked_ftus

        self.n_frames = 1

        self.visualization_properties = [
            'Main_Cell_Types','Gene Counts','Morphometrics','Cluster'
        ]

        # Adding ftu hierarchy property. This just stores which structures contain which other structures.
        self.ftu_hierarchy = {}

        self.get_slide_map_data()
    
    def __str__(self):
        
        return f'{self.slide_name}'

    def add_label(self,ftu,label,mode):
        
        # Updating geojson for an FTU to include the provided label
        ftu_name = ftu['properties']['name']
        ftu_index = ftu['properties']['unique_index']

        print(f'ftu_name: {ftu_name}, ftu_index: {ftu_index}, mode: {mode}, label: {label}')
        
        if mode=='add':

            # Adding to props
            if ftu_name == 'Spots':
                if 'user_labels' in self.spot_props[ftu_index]:
                    self.spot_props[ftu_index]['user_labels'].append(label)
                else:
                    self.spot_props[ftu_index]['user_labels'] = [label]
            else:
                if 'user_labels' in self.ftu_props[ftu_name][ftu_index]:
                    self.ftu_props[ftu_name][ftu_index]['user_labels'].append(label)
                else:
                    self.ftu_props[ftu_name][ftu_index]['user_labels'] = [label]

        elif mode=='remove':

            # Removing from props
            if ftu_name=='Spots':
                self.spot_props[ftu_index]['user_labels'].pop(label)
            else:
                self.ftu_props[ftu_name][ftu_index]['user_labels'].pop(label)

    def get_slide_map_data(self):
        # Getting all of the necessary materials for loading a new slide

        # Step 1: get resource tile metadata
        tile_metadata = self.girder_handler.get_tile_metadata(self.item_id)
        # Step 2: adding n_frames if "frames" in metadata (normal rgb images just have 1 frame but ome-tiff images record each color channel as a "frame")
        if 'frames' in tile_metadata:
            self.n_frames = len(tile_metadata['frames'])
        else:
            self.n_frames = 1

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

        # Step 4: Defining bounds of map (size of the first tile)
        self.map_bounds = [[0,self.image_dims[1]],[0,self.image_dims[0]]]

        # Step 5: Getting annotation ids for an item
        self.annotation_ids = self.girder_handler.get_available_annotation_ids(self.item_id)
        print(f'Found: {len(self.annotation_ids)} Annotations')

        # Step 6: Getting user token and tile url
        self.user_token = self.girder_handler.get_token()
        if not 'frames' in tile_metadata:
            self.tile_url = self.girder_handler.gc.urlBase+f'item/{self.item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+self.user_token
        else:
            # Set first 3 frames to RGB
            self.tile_url = self.girder_handler.gc.urlBase+f'item/{self.item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+self.user_token+'&style={"bands": [{"framedelta":0,"palette":"rgba(255,0,0,255)"},{"framedelta":1,"palette":"rgba(0,255,0,255)"},{"framedelta":2,"palette":"rgba(0,0,255,255)"}]}'

        # Step 7: Converting Histomics/large-image annotations to GeoJSON
        base_x_scale = self.base_dims[0]/self.tile_dims[0]
        base_y_scale = self.base_dims[1]/self.tile_dims[1]

        self.x_scale = (self.tile_dims[0])/(self.image_dims[0]*(self.tile_dims[0]/240))
        self.y_scale = (self.tile_dims[1])/(self.image_dims[1]*(self.tile_dims[1]/240))
        self.y_scale*=-1

        self.x_scale *= base_x_scale
        self.y_scale *= base_y_scale

        self.map_bounds[0][1]*=self.x_scale
        self.map_bounds[1][1]*=self.y_scale

        # Initializing ftu info
        self.ftu_names = []
        self.ftu_polys = {}
        self.ftu_props = {}
        self.properties_list = []

        self.map_dict = {
            'url': self.tile_url,
            'FTUs':{}
        }

    def get_annotation_geojson(self,idx):
        """
        Get annotation in geojson format, extract polygon and properties information, save locally
        """
        if idx>=len(self.annotation_ids):
            print(f'Uh oh! Tried to get annotation index {idx} when there are only {len(self.annotation_ids)} annotations available!')
            raise IndexError
        
        this_annotation = self.annotation_ids[idx]
        f_name = this_annotation['annotation']['name']

        save_path = f'./assets/slide_annotations/{self.item_id}/{this_annotation["_id"]}.json'
        if not os.path.exists(save_path):
            if not os.path.exists(f'./assets/slide_annotations/{self.item_id}'):
                os.makedirs(f'./assets/slide_annotations/{self.item_id}')

                # Create placeholder file for timekeeping
                with open(f'./assets/slide_annotations/{self.item_id}/rock.txt','w') as f:
                    f.write('Hey look a cool rock!')
                f.close()

            # Step 1: Get annotation in geojson form
            annotation_geojson = self.girder_handler.gc.get(f'/annotation/{self.annotation_ids[idx]["_id"]}/geojson?token={self.user_token}')
            
            # Step 2: Scale coordinates of geojson object
            scaled_annotation = geojson.utils.map_geometries(lambda g: geojson.utils.map_tuples(lambda c: (c[0]*self.x_scale, c[1]*self.y_scale, c[2]), g), annotation_geojson)
            
            if len(scaled_annotation['features'])>0:
                for f in scaled_annotation['features']:
                    f['properties']['name'] = f_name

                    if not 'user' in f['properties']:
                        f['user'] = {}

                self.ftu_names.append(f_name)

                if f_name not in self.ftu_colors:
                    self.ftu_colors[f_name] = '#%02x%02x%02x' % (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                
        else:
            try:
                with open(save_path,'r') as f:
                    scaled_annotation = geojson.load(f)

                f.close()

                self.ftu_names.append(f_name)

                if f_name not in self.ftu_colors:
                    self.ftu_colors[f_name] = '#%02x%02x%02x' % (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                
            except json.decoder.JSONDecodeError:
                scaled_annotation = None

        if not scaled_annotation is None and len(scaled_annotation['features'])>0:
            # Step 3: Recording properties and polys for that annotation
            self.ftu_polys[f_name] = [
                shape(g['geometry'])
                for g in scaled_annotation['features']
            ]
            start_idx = sum([len(i)-1 for i in self.ftu_props])
            self.ftu_props[f_name] = [
                g['properties']['user'] | {'name': f_name, 'unique_index': start_idx+idx}
                for idx,g in enumerate(scaled_annotation['features'])
                if 'user' in g['properties']
            ]

            ftu_prop_list = []
            if len(self.ftu_props[f_name])>0:
                this_ftu_props = list(self.ftu_props[f_name][0].keys())
                for p in this_ftu_props:
                    if p in self.visualization_properties:
                        if type(self.ftu_props[f_name][0][p])==dict:
                            ftu_prop_list.extend([
                                f'{p} --> {i}' if not p=='Main_Cell_Types' else f'{p} --> {self.girder_handler.cell_graphics_key[i]["full"]}'
                                for i in list(self.ftu_props[f_name][0][p].keys())
                            ])
                        else:
                            ftu_prop_list.append(p)

                if len(self.properties_list)>0:
                    self.properties_list.extend(list(set(ftu_prop_list) - set(self.properties_list)))
                else:
                    self.properties_list = ftu_prop_list

                if any(['Main_Cell_Types' in i for i in self.properties_list]) and 'Max Cell Type' not in self.properties_list:
                    self.properties_list.append('Max Cell Type')

            self.map_dict['FTUs'][f_name] = {
                'id':{'type':'ftu-bounds','index':len(self.ftu_names)-1},
                'url':f'./assets/slide_annotations/{self.item_id}/{this_annotation["_id"]}.json',
                'popup_id':{'type':'ftu-popup','index':len(self.ftu_names)-1},
                'color':self.ftu_colors[f_name],
                'hover_color':'#9caf00'
            }

            # Step 4: Save geojson locally
            save_path = f'./assets/slide_annotations/{self.item_id}/{this_annotation["_id"]}.json'
            if not os.path.exists(save_path):
                with open(save_path,'w') as f:
                    geojson.dump(scaled_annotation,f)

                f.close()

    def find_intersecting_ftu(self, box_poly, ftu: str):

        if ftu in self.ftu_names:
            # Finding which members of a specfied ftu group intersect with the provided box_poly
            ftu_intersect_idx = [i for i in range(0,len(self.ftu_polys[ftu])) if self.ftu_polys[ftu][i].intersects(box_poly)]
            
            # Returning list of dictionaries that use original keys in properties
            intersecting_ftu_props = []
            intersecting_ftu_polys = []

            if len(ftu_intersect_idx)>0:
                intersecting_ftu_props = [self.ftu_props[ftu][i] for i in ftu_intersect_idx]
                intersecting_ftu_polys = [self.ftu_polys[ftu][i] for i in ftu_intersect_idx]


            return intersecting_ftu_props, intersecting_ftu_polys
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

        # Convert map coordinates to slide coordinates
        # input_coords are in terms of the tile map and returned coordinates are relative to the slide pixel dimensions
        return_coords = []
        for i in input_coords:
            return_coords.append([i[0]/self.x_scale,i[1]/self.y_scale])

        return return_coords

    def convert_slide_coords(self, input_coords):

        # Inverse of convert_map_coords, takes a set of slide pixel coordinates
        # and converts to be map coordinates
        return_coords = []
        for i in input_coords:
            return_coords.append([i[0]*self.x_scale,i[1]*self.y_scale])

        return return_coords
    
    def gen_structure_hierarchy(self, structure_name:str):
        """
        Determine containment of other structures within a given structure 
        """
        if not structure_name in self.ftu_names:
            print('Structure not present in this slide')
            raise ValueError
        else:

            structure_intersect = []
            if not structure_name=='Spots':
                for f in self.ftu_names:
                    if not f==structure_name and f in self.ftu_polys:
                        for g in self.ftu_polys[f]:
                            if any([g.intersects(i) for i in self.ftu_polys[structure_name]]):
                                structure_intersect.append(f)
                                break

                
                if 'Spots' in self.ftu_polys:
                    for g in self.spot_polys:
                        if any([g.intersects(i) for i in self.ftu_polys[structure_name]]):
                            structure_intersect.append(f)
                            break
            
            elif structure_name == 'Spots':
                for f in self.ftu_names:
                    if not f==structure_name:
                        for g in self.ftu_polys[f]:
                            if any([g.intersects(i) for i in self.spot_polys]):
                                structure_intersect.append(f)
                                break

                                
            tree_dict = {
                'title': structure_name,
                'key': '0',
                'children':[
                    {
                        'title': i,
                        'key': f'0-{idx}'
                    }
                    for idx,i in enumerate(structure_intersect)
                ]
            }

            self.ftu_hierarchy[structure_name] = tree_dict

            return tree_dict

    def get_overlay_value_list(self, overlay_prop):
        """
        Pulling out overlay props and returning list of raw values
        """

        raw_values_list = []
        # Iterating through each ftu in each group:
        for f in self.ftu_props:
            # Grabbing values using utils function
            f_raw_vals = extract_overlay_value(self.ftu_props[f],overlay_prop)
            raw_values_list.extend(f_raw_vals)
        
        """
        if self.spatial_omics_type=='Visium':
            # Iterating through spots
            spot_raw_vals = extract_overlay_value(self.spot_props,overlay_prop)
            raw_values_list.extend(spot_raw_vals)
        """

        # Check for manual ROIs
        if len(self.manual_rois)>0:

            #TODO: Record manual ROI properties in a more sane manner.
            manual_props = [i['geojson']['features'][0]['properties'] for i in self.manual_rois if 'properties' in i['geojson']['features'][0]]
            manual_raw_vals = extract_overlay_value(manual_props,overlay_prop)
            raw_values_list.extend(manual_raw_vals)

        raw_values_list = np.unique(raw_values_list).tolist()

        return raw_values_list

    








class VisiumSlide(DSASlide):
    # Additional properties for Visium slides are:
    # id of RDS object
    spatial_omics_type = 'Visium'

    def __init__(self,
                 item_id:str,
                 girder_handler,
                 ftu_colors,
                 manual_rois:list,
                 marked_ftus:list):
        super().__init__(item_id,girder_handler,ftu_colors,manual_rois,marked_ftus)

        self.change_level_plugin = {
            'plugin_name': 'samborder2256_change_level_latest/ChangeLevel',
            'definitions_file': '64fa0f782d82d04be3e5daa3'
        }

    def run_change_level(self,main_cell_types):
        """
        Getting cell sub-types for a set of main cell types

        """
        job_id = None
        change_type = {
            "ACTION":"SUBTYPE",
            "SELECTOR": [
                {
                    "MAINS":main_cell_types
                }
            ]
        }


        item_files = self.girder_handler.gc.get(f'/item/{self.item_id}/files')
        counts_file = [i for i in item_files if 'rds' in i['exts'] or 'h5ad' in i['exts']][0]['_id']

        job_id = self.girder_handler.gc.post(f'/slicer_cli_web/{self.change_level_plugin["plugin_name"]}/run',
                                    parameters = {
                                        'rds_file': counts_file,
                                        'definitions_file': self.change_level_plugin["plugin_name"],
                                        'image_id': self.item_id,
                                        'change_type': json.dumps(change_type),
                                        'girderApiUrl': self.girder_handler.apiUrl,
                                        'girderToken': self.girder_handler.user_token
                                    }
                                )['_id']



        return job_id


class CODEXSlide(DSASlide):
    # Additional properties needed for CODEX slides are:
    # names for each channel

    spatial_omics_type = 'CODEX'

    def __init__(self,
                 item_id:str,
                 girder_handler,
                 ftu_colors:dict,
                 manual_rois:list,
                 marked_ftus:list
                 ):
        super().__init__(item_id,girder_handler,ftu_colors,manual_rois,marked_ftus)

        # Updating tile_url so that it includes the different frames
        self.channel_names = []
        
        # Getting image metadata which might contain frame names
        image_metadata = self.girder_handler.get_tile_metadata(self.item_id)
        if 'frames' in image_metadata:
            for f in image_metadata['frames']:
                if 'Channel' in f:
                    self.channel_names.append(f['Channel'])

        if 'Histology Id' in self.item_info['meta']:
            self.channel_names = ['Histology (H&E)'] + self.channel_names

            # Checking if histology image is multi-frame
            histo_id = self.item_info["meta"]["Histology Id"]
            histo_img_info = self.girder_handler.get_tile_metadata(histo_id)
            if not 'frames' in histo_img_info:
                self.histology_url = self.girder_handler.gc.urlBase+f'item/{histo_id}/tiles/zxy/'+'{z}/{x}/{y}?token='+self.user_token
            else:
                self.histology_url = self.girder_handler.gc.urlBase+f'item/{histo_id}/tiles/zxy/'+'{z}/{x}/{y}?token='+self.user_token+'&style={"bands": [{"framedelta":0,"palette":"rgba(255,0,0,255)"},{"framedelta":1,"palette":"rgba(0,255,0,255)"},{"framedelta":2,"palette":"rgba(0,0,255,255)"}]}'

        if self.channel_names == []:
            # Fill in with dummy channel_names (test case with 16 or 17 channels)
            self.channel_names = [f'Channel_{i}' for i in range(0,self.n_frames)]

        if not 'Histology (H&E)' in self.channel_names:
            self.channel_tile_url = [
                self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/'+'/{z}/{x}/{y}?token='+self.user_token+'&style={"bands": [{"palette":["rgba(0,0,0,0)","rgba(255,255,255,255)"],"framedelta":'+str(i)+'}]}'
                for i in range(len(self.channel_names))
            ]
        else:
            self.channel_tile_url = [
                self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/'+'/{z}/{x}/{y}?token='+self.user_token+'&style={"bands": [{"palette":["rgba(0,0,0,0)","rgba(255,255,255,255)"],"framedelta":'+str(i-1)+'}]}'
                if not self.channel_names[i] == 'Histology (H&E)' else self.histology_url
                for i in range(len(self.channel_names))
            ]

    def intersecting_frame_intensity(self,box_poly,frame_list):
        # Finding the intensity of each "frame" representing a channel in the original CODEX image within a region
        
        if frame_list=='all':
            frame_list = [i for i in self.wsi.channel_names if not i=='Histology (H&E)']

        box_coordinates = np.array(self.convert_map_coords(list(box_poly.exterior.coords)))
        min_x = int(np.min(box_coordinates[:,0]))
        min_y = int(np.min(box_coordinates[:,1]))
        max_x = int(np.max(box_coordinates[:,0]))
        max_y = int(np.max(box_coordinates[:,1]))
        
        # Box size then can be determined by (maxx-minx)*(maxy-miny)
        box_size = (max_x-min_x)*(max_y-min_y)
        # Or probably also by multiplying some scale factors by box_poly.area
        # Pulling out those regions of the image
        frame_indices = [self.channel_names.index(i) for i in frame_list]

        # Slide coordinates list should be [minx,miny,maxx,maxy]
        slide_coords_list = [min_x,min_y,max_x,max_y]
        frame_properties = {}
        for frame in frame_indices:
            print(f'Working on frame {frame} of {self.n_frames}')
            # Get the image region associated with that frame
            # Or just get the histogram for that channel? not sure if this can be for a specific image region
            image_histogram = self.girder_handler.gc.get(f'/item/{self.item_id}/tiles/histogram',
                                                         parameters = {
                                                            'top': min_y,
                                                            'left': min_x,
                                                            'bottom': max_y,
                                                            'right': max_x,
                                                            'frame': frame,
                                                            'roundRange': True, 
                                                            'density': True
                                                            }
                                                        )

            # Fraction of total intensity (maximum intensity = every pixel is 255 for uint8)
            frame_properties[self.channel_names[frame]] = image_histogram[0]

        
        return frame_properties

    def update_url_style(self,color_options: dict):

        # Color options is a dict containing "bands" followed by each styled frame
        # Assembling the style dict 
        style_dict = {
            "bands": [
                {
                    "palette": ["rgba(0,0,0,0)",color_options[c]],
                    "framedelta": c
                }
                for c in color_options
            ]
        }

        styled_url = self.girder_handler.gc.urlBase+f'item/{self.item_id}/tiles/zxy/'+'/{z}/{x}/{y}?token='+self.user_token+'&style='+json.dumps(style_dict)

        return styled_url

