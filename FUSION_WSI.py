"""

External file to hold the WholeSlide class used in FUSION


"""
import os
import numpy as np

from shapely.geometry import shape, Polygon, box
import random
import json
import geojson
import shutil
import requests

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc
import dash_bootstrap_components as dbc
from dash_extensions.enrich import html

from tqdm import tqdm
from FUSION_Utils import (
    extract_overlay_value,
    gen_violin_plot,
    gen_umap)

class DSASlide:

    spatial_omics_type = 'Regular'

    def __init__(self,
                 item_id,
                 user_details,
                 girder_handler,
                 ftu_colors,
                 manual_rois = [],
                 marked_ftus = []):

        self.item_id = item_id
        self.girder_handler = girder_handler
        self.user_details = user_details

        self.item_info = self.girder_handler.gc.get(f'/item/{self.item_id}?token={self.user_details["token"]}')
        self.slide_name = self.item_info['name']
        self.slide_ext = self.slide_name.split('.')[-1]
        self.ftu_colors = ftu_colors

        self.manual_rois = manual_rois
        self.marked_ftus = marked_ftus

        self.n_frames = 1

        self.visualization_properties = [
            'Main_Cell_Types',
            'Gene Counts',
            'Morphometrics',
            'Cluster',
            'Cell_Subtypes',
            'Cell Label',
            'Cell Type',
            'Transcript Counts'
        ]

        # Adding ftu hierarchy property. This just stores which structures contain which other structures.
        self.ftu_hierarchy = {}

        self.get_slide_map_data()

        self.default_view_type = {
            'name': 'features',
            #'features': [self.properties_list[0]]
        }
    
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
        self.user_token = self.user_details['token']
        if not 'frames' in tile_metadata:
            self.tile_url = self.girder_handler.gc.urlBase+f'item/{self.item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+self.user_token
        else:
            # Set first 3 frames to RGB
            self.tile_url = self.girder_handler.gc.urlBase+f'item/{self.item_id}'+'/tiles/zxy/{z}/{x}/{y}?token='+self.user_token+'&style={"bands": [{"framedelta":0,"palette":"rgba(255,0,0,255)"},{"framedelta":1,"palette":"rgba(0,255,0,255)"},{"framedelta":2,"palette":"rgba(0,0,255,255)"}]}'

        # Step 7: Converting Histomics/large-image annotations to GeoJSON
        self.x_scale = self.base_dims[0]/self.image_dims[0]
        self.y_scale = self.base_dims[1]/self.image_dims[1]
        self.y_scale*=-1

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
            try:
                annotation_geojson = self.girder_handler.gc.get(f'/annotation/{self.annotation_ids[idx]["_id"]}/geojson?token={self.user_token}')
            except requests.exceptions.ChunkedEncodingError:
                # Some error converting to geojson
                annotation_json = self.girder_handler.gc.get(f'/annotation/{self.annotation_ids[idx]["_id"]}?token={self.user_token}')

                # Converting to geojson
                annotation_geojson = {
                    'type': 'FeatureCollection',
                    'features': [
                        {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Polygon',
                                'coordinates': [el['points']]
                            },
                            'properties': {
                                'user': el['user']
                            }
                        }
                        for el in annotation_json['annotation']['elements']
                    ]
                }

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
                                f'{p} --> {i}' if not p in ['Main_Cell_Types','Cell_Subtypes'] else f'{p} --> {self.girder_handler.cell_graphics_key[i]["full"]}'
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

    def update_annotation_geojson(self,new_annotations):
        """
        Updating local annotations with new ones (converting from large_image to GeoJSON)
        """
        for new_ann in new_annotations:
            new_ann_id = new_ann['_id']
            save_path = f'./assets/slide_annotations/{self.item_id}/{new_ann_id}.json'

            geojson_root = {
                'type':'FeatureCollection',
                'features': []
            }

            for el in new_ann['annotation']['elements']:

                feature_dict = {
                    'type':'Feature',
                    'geometry': {
                        'type':'Polygon',
                        'coordinates': [[[i[0] * self.x_scale,i[1] * self.y_scale] for i in el['points']]]
                    },
                    'properties': {
                        'user': el['user'],
                        'name': el['name']
                    }
                }

                geojson_root['features'].append(feature_dict)
            
            with open(save_path,'w') as f:
                geojson.dump(geojson_root,f)

            f.close()

    def find_intersecting_ftu(self, box_poly, ftu: str):

        if type(box_poly)==list:
            box_poly = box(*box_poly)

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

        # Check for manual ROIs
        if len(self.manual_rois)>0:

            #TODO: Record manual ROI properties in a more sane manner.
            manual_props = [i['geojson']['features'][0]['properties'] for i in self.manual_rois if 'properties' in i['geojson']['features'][0]]
            manual_raw_vals = extract_overlay_value(manual_props,overlay_prop)
            raw_values_list.extend(manual_raw_vals)

        raw_values_list = np.unique(raw_values_list).tolist()

        return raw_values_list

    def update_ftu_props(self,new_annotation):
        """
        Updating ftu_props according to changelevel runs (adding cell subtypes or readcounts)
        """

        new_annotation_names = [i['annotation']['name'] for i in new_annotation]
        new_properties_list = []
        for f in new_annotation_names:
            self.ftu_props[f] = [
                f_prop | new_annotation[new_annotation_names.index(f)]['annotation']['elements'][f_idx]['user']
                for f_idx,f_prop in enumerate(self.ftu_props[f])
            ]

            for el in self.ftu_props[f]:
                add_keys = list(el.keys())
                edited_keys = []
                for a_k in add_keys:
                    if a_k in self.visualization_properties:
                        if type(el[a_k])==dict:
                            for sub_key in list(el[a_k].keys()):
                                if not a_k in ['Main_Cell_Types','Cell_Subtypes']:
                                    edited_keys.append(f'{a_k} --> {sub_key}')
                                else:
                                    edited_keys.append(f'{a_k} --> {self.girder_handler.cell_graphics_key[sub_key]["full"]}')
                        else:
                            edited_keys.append(a_k)
                    
                new_properties_list.extend(edited_keys)

        self.properties_list = np.unique(self.properties_list+np.unique(new_properties_list).tolist()).tolist()

        if not 'Max Cell Type' in self.properties_list:
            if any(['Main_Cell_Types' in i for i in self.properties_list]):
                self.properties_list.append('Max Cell Type')

    def update_viewport_data(self,bounds_box:list,view_type:dict):
        """
        Grabbing info for intersecting ftus        
        """
        
        if view_type['name'] is None:
            view_type = self.default_view_type

        viewport_data_components = []
        viewport_data = None
        
        pass
        
        return viewport_data_components, viewport_data

    def spatial_aggregation(self, agg_polygon):
        """
        Generalized aggregation of underlying structure properties for a given polygon
        """

        ignore_columns = ['unique_index','name','structure','ftu_name','image_id','ftu_type',
                          'Min_x_coord','Max_x_coord','Min_y_coord','Max_y_coord',
                          'x_tsne','y_tsne','x_umap','y_umap']

        # Step 1: Find intersecting structures with polygon
        aggregated_properties = {}
        for ftu_idx, ftu in enumerate(self.ftu_names):
            overlap_properties, overlap_polys = self.find_intersecting_ftu(agg_polygon, ftu)

            overlap_area = [(i.intersection(agg_polygon).area)/(agg_polygon.area) for i in overlap_polys]
            # overlap_properties is a list of properties for each intersecting polygon
            agg_prop_df = pd.DataFrame.from_records(overlap_properties)
            agg_prop_df = agg_prop_df.drop(columns = [i for i in ignore_columns if i in agg_prop_df.columns.tolist()])

            # string and dict types will be called "object" dtypes in pandas
            agg_numeric_props = agg_prop_df.select_dtypes(exclude = 'object')

            # Scaling numeric props by area
            for row_idx, area in enumerate(overlap_area):
                #print(f'area: {area}')
                agg_numeric_props.iloc[row_idx,:] *= area

            agg_numeric_dict = agg_numeric_props.sum(axis=0).to_dict()
            
            aggregated_properties[ftu] = agg_numeric_dict
            aggregated_properties[ftu][f'{ftu} Count'] = len(overlap_area)

            agg_object_props = agg_prop_df.select_dtypes(include='object')
            for col_idx, col_name in enumerate(agg_object_props.columns.tolist()):
                col_values = agg_object_props[col_name].tolist()
                col_vals_dict = {col_name: {}}
                
                print(col_name)
                print(col_values)

                if type(col_values[0])==dict:
                    
                    # Test for single nested dictionary
                    sub_values = list(col_values[0].keys())
                    if not type(col_values[0][sub_values[0]])==dict:
                        col_df = pd.DataFrame.from_records(col_values).astype(float)

                        # Scaling by intersection area
                        for row_idx, area in enumerate(overlap_area):
                            col_df.iloc[row_idx,:] *= area

                        col_df_norm = col_df.sum(axis=0).to_frame()
                        col_df_norm = (col_df_norm/col_df_norm.sum()).fillna(0.000).round(decimals=18)
                        col_df_norm[0] = col_df_norm[0].map('{:.19f}'.format)
                        col_vals_dict[col_name] = col_df_norm.astype(float).to_dict()[0]
                    
                    else:
                        for sub_val in sub_values:
                            if type(col_values[0][sub_val])==dict:
                                col_df = pd.DataFrame.from_records([i[sub_val] for i in col_values]).astype(float)
                                
                                # Scaling by intersection area
                                for row_idx, area in enumerate(overlap_area):
                                    col_df.iloc[row_idx,:] *= area

                                col_df_norm = col_df.sum(axis=0).to_frame()
                                col_df_norm = (col_df_norm/col_df_norm.sum()).fillna(0.000).round(decimals=18)
                                col_df_norm[0] = col_df_norm[0].map('{:.19f}'.format)
                                col_vals_dict[col_name][sub_val] = col_df_norm.astype(float).to_dict()[0]
                    
                elif type(col_values[0])==str:
                    # Just getting the count of each unique value here
                    col_vals_dict[col_name] = {i:col_values.count(i) for i in np.unique(col_values).tolist()}

                aggregated_properties[ftu] = aggregated_properties[ftu] | col_vals_dict

        return aggregated_properties





class VisiumSlide(DSASlide):
    # Additional properties for Visium slides are:
    # id of RDS object
    spatial_omics_type = 'Visium'

    def __init__(self,
                 item_id:str,
                 user_details,
                 girder_handler,
                 ftu_colors,
                 manual_rois:list,
                 marked_ftus:list):
        super().__init__(item_id,user_details,girder_handler,ftu_colors,manual_rois,marked_ftus)

        self.change_level_plugin = {
            'plugin_name': 'samborder2256_change_level_latest/ChangeLevel',
            'definitions_file': '64fa0f782d82d04be3e5daa3'
        }

        self.default_view_type = {
            'name': 'Main_Cell_Types',
        }

        self.n_frames = 0

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
                                        'definitions_file': self.change_level_plugin["definitions_file"],
                                        'image_id': self.item_id,
                                        'change_type': json.dumps(change_type),
                                        'girderApiUrl': self.girder_handler.apiUrl,
                                        'girderToken': self.user_token
                                    }
                                )['_id']

        return job_id

    def update_viewport_data(self,bounds_box:list, view_type:dict):
        """
        Find intersecting structures, including marked and manual.

        Grabbing Main_Cell_Types, Cell_States, and Gene_Counts
        """

        if view_type['name'] is None:
            view_type = self.default_view_type

        viewport_data_components = []
        viewport_data = None

        intersecting_ftus = {}
        intersecting_ftu_polys = {}

        for ftu in self.ftu_names:
            intersecting_ftus[ftu], intersecting_ftu_polys[ftu] = self.find_intersecting_ftu(bounds_box,ftu)

        for m_idx,m_ftu in enumerate(self.manual_rois):
            manual_intersect_ftus = list(m_ftu['geojson']['features'][0]['properties']['user'])
            for int_ftu in manual_intersect_ftus:
                intersecting_ftus[f'Manual ROI: {m_idx+1}, {int_ftu}'] = [m_ftu['geojson']['features'][0]['properties']['user'][int_ftu]]

        for marked_idx, marked_ftu in enumerate(self.marked_ftus):
            intersecting_ftus[f'Marked FTUs: {marked_idx+1}'] = [i['properties']['user'] for i in marked_ftu['geojson']['features']]


        if len(list(intersecting_ftus.keys()))>0:
            
            viewport_data = {}
            tab_list = []

            for f_idx,f in enumerate(list(intersecting_ftus.keys())):
                viewport_data[f] = {}
                if view_type['name'] in ['Main_Cell_Types','Cell_Subtypes']:
                    if view_type['name']=='Main_Cell_Types':
                        counts_dict_list = [i['Main_Cell_Types'] for i in intersecting_ftus[f] if 'Main_Cell_Types' in i]
                        if len(counts_dict_list)>0:
                            counts_data = pd.DataFrame.from_records(counts_dict_list).sum(axis=0).to_frame()
                            counts_data.columns = [f]

                            counts_data = (counts_data[f]/counts_data[f].sum()).to_frame()
                            counts_data.columns = [f]
                            counts_data = counts_data.sort_values(by = f, ascending = False)
                            counts_data = counts_data.reset_index()

                            viewport_data[f]['data'] = counts_data.to_dict('records')
                            if not 'Manual' in f:
                                viewport_data[f]['count'] = len(counts_dict_list)
                            else:
                                viewport_data[f]['count'] = intersecting_ftus[f][0][f.split(', ')[-1]+ ' Count']

                            # Getting cell state info:
                            viewport_data[f]['states'] = {}
                            for m in counts_data['index'].tolist():
                                cell_states_data = [i['Cell_States'][m] for i in intersecting_ftus[f] if 'Cell_States' in i]
                                cell_states_data = pd.DataFrame.from_records(cell_states_data).sum(axis=0).to_frame()
                                cell_states_data = cell_states_data.reset_index()
                                cell_states_data.columns = ['Cell State','Proportion']
                                cell_states_data['Proportion'] = cell_states_data['Proportion']/cell_states_data['Proportion'].sum()

                                viewport_data[f]['states'][m] = cell_states_data.to_dict('records')
                        else:
                            continue
                    elif view_type['name']=='Cell_Subtypes':

                        viewport_data[f] = {}
                        counts_dict_list = []
                        for ftu in intersecting_ftus[f]:
                            if 'Cell_Subtypes' in ftu and 'Main_Cell_Types' in ftu:
                                main_cell_types = list(ftu['Main_Cell_Types'].keys())

                                ftu_dict = {}
                                for m in main_cell_types:
                                    main_pct = ftu['Main_Cell_Types'][m]
                                    if m in ftu['Cell_Subtypes']:
                                        subtype_pct = ftu['Cell_Subtypes'][m]

                                        ftu_dict = ftu_dict | {i: main_pct*subtype_pct[i] for i in subtype_pct} 
                                        
                                counts_dict_list.append(ftu_dict)

                        if len(counts_dict_list)>0:
                            counts_data = pd.DataFrame.from_records(counts_dict_list).sum(axis=0).to_frame()
                            counts_data.columns = [f]

                            counts_data = (counts_data[f]/counts_data[f].sum()).to_frame()
                            counts_data.columns = [f]
                            counts_data = counts_data.sort_values(by = f, ascending = False)
                            counts_data = counts_data.reset_index()

                            viewport_data[f]['data'] = counts_data.to_dict('records')
                            viewport_data[f]['count'] = len(counts_dict_list)
                        else:
                            continue

                    elif view_type['name']=='Gene Counts':
                        counts_dict_list = [i['Gene Counts'][view_type['gene']] for i in intersecting_ftus[f] if 'Gene Counts' in i]

                        if len(counts_dict_list)>0:
                            viewport_data[f]['data'] = counts_dict_list
                            viewport_data[f]['count'] = len(counts_dict_list)
                        else:
                            continue
                    
                elif view_type['name'] == 'Morphometrics':

                    counts_dict_list = [i['Morphometrics'][view_type['type']] for i in intersecting_ftus[f] if 'Morphometrics' in i]
                    if len(counts_dict_list)>0:
                        viewport_data[f]['data'] = counts_dict_list
                        viewport_data[f]['count'] = len(counts_dict_list)
                    else:
                        continue
                
                if view_type['name'] in ['Main_Cell_Types','Cell_Subtypes','Gene Counts']:
                    chart_label = f'{f} Cell Composition'

                    if view_type['name'] in ['Main_Cell_Types','Cell_Subtypes']:
                        f_tab_plot = px.pie(counts_data,values=f,names='index')
                        f_tab_plot.update_traces(textposition='inside')
                        f_tab_plot.update_layout(uniformtext_minsize=12,uniformtext_mode='hide')

                        if view_type['name']=='Main_Cell_Types':
                            top_cell = counts_data['index'].tolist()[0]

                            pct_states = pd.DataFrame.from_records(viewport_data[f]['states'][top_cell])
                            state_bar = px.bar(pct_states,x='Cell State',y = 'Proportion',title = f'Cell State Proportions for:<br><sup>{top_cell} in:</sup><br><sup>{f}</sup>')
                            second_chart_label = f'{f} Cell State Proportions'

                            f_tab = dbc.Tab(
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label(chart_label),
                                        dcc.Graph(
                                            id = {'type': 'ftu-cell-pie','index':f_idx},
                                            figure = go.Figure(f_tab_plot)
                                        )
                                    ],md=6),
                                    dbc.Col([
                                        dbc.Label(second_chart_label),
                                        dcc.Graph(
                                            id = {'type': 'ftu-state-bar','index': f_idx},
                                            figure = go.Figure(state_bar)
                                        )
                                    ],md=6)
                                ]),label = f+f' ({viewport_data[f]["count"]})',tab_id = f'tab_{f_idx}'
                            )

                            tab_list.append(f_tab)
                        elif view_type['name'] == 'Cell_Subtypes':

                            f_tab = dbc.Tab(
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label(chart_label),
                                        dcc.Graph(
                                            id = {'type': 'ftu-cell-pie','index':f_idx},
                                            figure = go.Figure(f_tab_plot)
                                        )
                                    ],md=12)
                                ]),label = f+f' ({viewport_data[f]["count"]})',tab_id = f'tab_{len(tab_list)}'
                            )

                            tab_list.append(f_tab)

                    elif view_type['name'] == 'Gene Counts':
                        f_tab_plot = gen_violin_plot(
                            feature_data = pd.DataFrame.from_records(counts_dict_list),
                            label_col = f,
                            label_name = f'{f} Gene Counts',
                            feature_col = view_type['gene'],
                            custom_col = None
                            )

                        f_tab = dbc.Tab(
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label(chart_label),
                                    dcc.Graph(
                                        id = {'type':'ftu-cell-pie','index':f_idx},
                                        figure = go.Figure(f_tab_plot)
                                    )
                                ])
                            ]),label = f+f' ({viewport_data[f]["count"]})', tab_id = f'tab_{f_idx}'
                        )

                elif view_type['name'] == 'Morphometrics':
                    
                    chart_label = f'{f} Morphometrics'

                    f_tab_plot = gen_violin_plot(
                        feature_data = pd.DataFrame.from_records(counts_dict_list),
                        label_col = f,
                        label_name = f'{f} {view_type["type"]}',
                        feature_col = view_type['type'],
                        custom_col = None
                    )

                    f_tab = dbc.Tab(
                        dbc.Row([
                            dbc.Col([
                                dbc.Label(chart_label),
                                dcc.Graph(
                                    id = {'type':'ftu-cell-pie','index':f_idx},
                                    figure = go.Figure(f_tab_plot)
                                )
                            ])
                        ]),label = f+f' ({viewport_data[f]["count"]})', tab_id = f'tab_{f_idx}'
                    )

        viewport_data_components = html.Div([
            dbc.Row([
                dbc.Col(
                    children = [
                        dbc.Label('Select View Type: ')
                    ],
                    md = 4
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options = [
                            {'label': 'Main Cell Types','value': 'Main_Cell_Types', 'disabled': True if not any(['Main_Cell_Types' in i for i in self.properties_list]) else False},
                            {'label': 'Cell Subtypes','value': 'Cell_Subtypes', 'disabled': True if not any(['Cell_Subtypes' in i for i in self.properties_list]) else False},
                            {'label': 'Gene Counts','value': 'Gene Counts', 'disabled': True if not any(['Gene Counts' in i for i in self.properties_list]) else False},
                            {'label': 'Morphometrics','value': 'Morphometrics','disabled': True if not any(['Morphometrics' in i for i in self.properties_list]) else False}
                        ],
                        value = view_type['name'],
                        multi = False,
                        id = {'type': 'roi-view-data','index':0}
                    )
                )
            ]),
            html.Hr(),
            dbc.Row(
                dbc.Tabs(tab_list,active_tab = f'tab_0')
            )
        ])


        return viewport_data_components, viewport_data




class CODEXSlide(DSASlide):
    # Additional properties needed for CODEX slides are:
    # names for each channel

    spatial_omics_type = 'CODEX'

    def __init__(self,
                 item_id:str,
                 user_details,
                 girder_handler,
                 ftu_colors:dict,
                 manual_rois:list,
                 marked_ftus:list
                 ):
        super().__init__(item_id,user_details,girder_handler,ftu_colors,manual_rois,marked_ftus)

        # Updating tile_url so that it includes the different frames
        self.channel_names = []
        
        # Getting image metadata which might contain frame names
        image_metadata = self.girder_handler.get_tile_metadata(self.item_id)
        if 'frames' in image_metadata:
            for f in image_metadata['frames']:
                if 'Channel' in f:
                    self.channel_names.append(f['Channel'])
            
            if all([i in self.channel_names for i in ['red','green','blue']]):

                self.rgb_style_dict = {
                    "bands": [
                        {
                            "palette": ["rgba(0,0,0,0)",'rgba('+','.join(['255' if i==c_idx else '0' for i in range(3)]+['0'])+')'],
                            "framedelta": self.channel_names.index(c)
                        }
                        for c_idx,c in enumerate(['red','green','blue'])
                    ]
                }

                self.histology_url = self.girder_handler.gc.urlBase+f'item/{self.item_id}/tiles/zxy/'+'/{z}/{x}/{y}?token='+self.user_token+'&style='+json.dumps(self.rgb_style_dict)
                self.channel_names.append('Histology (H&E)')
                self.channel_names = [i for i in self.channel_names if i not in ['red','green','blue']]

        if len(self.channel_names) == 0:
            # Fill in with dummy channel_names (test case with 16 or 17 channels)
            self.channel_names = [f'Channel_{i}' for i in range(0,self.n_frames)]

        if not 'Histology (H&E)' in self.channel_names:
            self.channel_tile_url = [
                self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/'+'/{z}/{x}/{y}?token='+self.user_token+'&style={"bands": [{"palette":["rgba(0,0,0,0)","rgba(255,255,255,255)"],"framedelta":'+str(i)+'}]}'
                for i in range(len(self.channel_names))
            ]
        else:
            self.channel_tile_url = [
                self.girder_handler.gc.urlBase+f'item/{item_id}'+'/tiles/zxy/'+'/{z}/{x}/{y}?token='+self.user_token+'&style={"bands": [{"palette":["rgba(0,0,0,0)","rgba(255,255,255,255)"],"framedelta":'+str(i)+'}]}'
                if not self.channel_names[i] == 'Histology (H&E)' else self.histology_url
                for i in range(len(self.channel_names))
            ]

        self.default_view_type = {
            'name': 'channel_hist',
            'frame_list': [[self.channel_names[0]]],
            'frame_label': [None]
        }

    def intersecting_frame_intensity(self,box_poly,frame_list):
        # Finding the intensity of each "frame" representing a channel in the original CODEX image within a region
        
        if type(box_poly)==list:
            box_poly = box(*box_poly)

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
            # Get the image region associated with that frame
            # Or just get the histogram for that channel? not sure if this can be for a specific image region
            image_histogram = self.girder_handler.gc.get(f'/item/{self.item_id}/tiles/histogram',
                                                         parameters = {
                                                            'top': min_y,
                                                            'left': min_x,
                                                            'bottom': max_y,
                                                            'right': max_x,
                                                            'frame': frame,
                                                            'bins':100,
                                                            'rangeMax':255,
                                                            'roundRange': True, 
                                                            #'density': True
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

    def update_viewport_data(self,bounds_box: list,view_type:dict):
        """
        Finding viewport data according to view selection
        view_types available: cell, features, channel_hist
        """

        if view_type['name'] is None:
            view_type = self.default_view_type

        viewport_data_components = []
        viewport_data = None

        intersecting_ftus = {}
        intersecting_ftu_polys = {}
        if view_type['name'] in ['cell','features']:
            
            for ftu in self.ftu_names:
                intersecting_ftus[ftu], intersecting_ftu_polys[ftu] = self.find_intersecting_ftu(bounds_box,ftu)
            
            for m_idx, m_ftu in enumerate(self.manual_rois):
                for int_ftu in m_ftu:
                    intersecting_ftus[f'Manual ROI: {m_idx+1}, {int_ftu}'] = [m_ftu['geojson']['features'][0]['properties']['user'][int_ftu]]

            for marked_idx, marked_ftu in enumerate(self.marked_ftus):
                intersecting_ftus[f'Marked FTUs: {marked_idx+1}'] = [i['properties']['user'] for i in marked_ftu['geojson']['features']]

        elif view_type['name']=='channel_hist':
            intersecting_ftus['Tissue'] = self.intersecting_frame_intensity(bounds_box,view_type['frame_list'][0])

        
        if len(list(intersecting_ftus.keys()))>0:
            viewport_data = {}
            tab_list = []

            for f_idx,f in enumerate(list(intersecting_ftus.keys())):
                
                if len(intersecting_ftus[f])==0:
                    continue

                if f_idx<len(view_type['frame_list']):
                    f_frame_list = view_type['frame_list'][f_idx]
                    f_label = view_type['frame_label'][f_idx]
                else:
                    f_frame_list = view_type['frame_list'][0]
                    f_label = view_type['frame_label'][0]

                if not type(f_frame_list)==list:
                    f_frame_list = [f_frame_list]

                if view_type['name']=='channel_hist':
                    viewport_data[f] = {}
                    counts_data = pd.DataFrame()

                    for frame_name in intersecting_ftus[f]:
                        frame_data = intersecting_ftus[f][frame_name]

                        frame_data_df = pd.DataFrame({'Frequency':[0]+frame_data['hist'],'Intensity':frame_data['bin_edges'],'Channel':[frame_name]*len(frame_data['bin_edges'])})
                        
                        if counts_data.empty:
                            counts_data = frame_data_df
                        else:
                            counts_data = pd.concat([counts_data,frame_data_df],axis=0,ignore_index=True)

                    viewport_data[f]['data'] = counts_data.to_dict('records')
                    viewport_data[f]['count'] = 1

                    chart_label = 'Channel Intensity Histogram'
                    f_tab_plot = px.bar(
                        data_frame = counts_data,
                        x = 'Intensity',
                        y = 'Frequency',
                        color = 'Channel'
                    )
                    
                elif view_type['name']=='features':

                    chart_label = 'Cell Marker Cell Expression'

                    viewport_data[f] = {}
                    
                    counts_data_list = []
                    counts_data = pd.DataFrame()
                    if type(intersecting_ftus[f])==list:
                        for ftu_idx,ind_ftu in enumerate(intersecting_ftus[f]):
                            cell_features = {}

                            if 'Channel Means' in ind_ftu:
                                for frame in f_frame_list:
                                    cell_features[f'Channel {self.channel_names.index(frame)}'] = ind_ftu['Channel Means'][self.channel_names.index(frame)]
                            else:
                                for frame in f_frame_list:
                                    cell_features[f'Channel {self.channel_names.index(frame)}'] = 0.0


                            if not f_label is None and f_label in self.channel_names:
                                cell_features[f_label] = ind_ftu['Channel Means'][self.channel_names.index(frame)]
                            
                            cell_features['label'] = 'Cell Nucleus'

                            if f in intersecting_ftu_polys:
                                cell_features['Hidden'] = {'Bbox':list(intersecting_ftu_polys[f][ftu_idx].bounds)}

                            counts_data_list.append(cell_features)

                    if len(counts_data_list)>0:
                        counts_data = pd.DataFrame.from_records(counts_data_list)
                        viewport_data[f]['data'] = counts_data.to_dict('records')
                    else:

                        viewport_data[f]['data'] = []
                    
                    viewport_data[f]['count'] = len(counts_data_list)

                    if len(f_frame_list)==1:
                        f_tab_plot = gen_violin_plot(
                            feature_data = counts_data,
                            label_col = f_label if not f_label is None else 'label',
                            label_name = f_label if not f_label is None else 'Unlabeled',
                            feature_col = f'Channel {self.channel_names.index(f_frame_list[0])}',
                            custom_col = 'Hidden'
                        )
                        f_tab_plot.update_layout({
                            'yaxis': {
                                'title': {
                                    'text': f_frame_list[0]
                                }
                            },
                            'title': {
                                'text': f_frame_list[0]
                            }
                        })
                    elif len(f_frame_list)==2:
                        f_names = [i for i in counts_data.columns.tolist() if 'Channel' in i]
                        f_tab_plot = px.scatter(
                            data_frame = counts_data,
                            x = f_names[0],
                            y = f_names[1],
                            color = f_label if not f_label is None else 'label',
                            custom_data = 'Hidden'
                        )
                        f_tab_plot.update_layout({
                            'xaxis': {
                                'title': {
                                    'text': f'{f_frame_list[0]}'
                                }
                            },
                            'yaxis': {
                                'title': {
                                    'text': f'{f_frame_list[1]}'
                                }
                            }
                        })
                    elif len(f_frame_list)>2:
                        # Generating UMAP dimensional reduction 
                        f_umap_data = gen_umap(
                            feature_data = counts_data,
                            feature_cols = [i for i in counts_data.columns.tolist() if 'Channel' in i],
                            label_and_custom_cols = [i for i in counts_data.columns.tolist() if not 'Channel' in i]
                        )

                        viewport_data[f]['UMAP'] = f_umap_data.to_dict('records')

                        f_tab_plot = px.scatter(
                            data_frame = f_umap_data,
                            x = 'UMAP1',
                            y = 'UMAP2',
                            color = f_label if not f_label is None else 'label',
                            custom_data = 'Hidden'
                        )

                elif view_type['name'] == 'cell':
                    
                    chart_label = 'Cell Composition'
                    viewport_data[f] = {}
                    counts_data_list = []
                    counts_data = pd.DataFrame()
                    if type(intersecting_ftus[f])==list:
                        for ind_ftu in intersecting_ftus[f]:

                            if 'Cell' in ind_ftu:
                                counts_data_list.append({
                                    'Cell': ind_ftu["Cell"]
                                })
                            else:
                                counts_data_list.append({
                                    'Cell': 'Unlabeled'
                                })
                    
                    if len(counts_data_list)>0:
                        counts_data = pd.DataFrame.from_records(counts_data_list)
                        viewport_data[f]['data'] = counts_data.to_dict('records')
                    
                    else:
                        viewport_data[f]['data'] = []
                    
                    viewport_data[f]['count'] = len(counts_data_list)

                    counts_value_dict = counts_data['Cluster'].value_counts().to_dict()
                    pie_chart_data = []
                    for key,val in counts_value_dict.items():
                        pie_chart_data.append(
                            {'Cell': key, 'Count': val}
                        )
                    pie_chart_df = pd.DataFrame.from_records(pie_chart_data)
                    f_tab_plot = px.pie(
                        data_frame = pie_chart_df,
                        values = 'Count',
                        names = 'Cell'
                    )

                    f_tab_plot.update_traces(textposition='inside')
                    f_tab_plot.update_layout(uniformtext_minsize=12,uniformtext_mode='hide')


                f_tab = dbc.Tab([
                    dbc.Row([
                        dbc.Col(
                            dbc.Label('Select a Channel Name:',html_for={'type':'frame-histogram-drop','index':0}),
                            md = 3, align = 'center'
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                options = [i for i in self.channel_names],
                                value = f_frame_list,
                                multi = True,
                                id = {'type':'frame-histogram-drop','index':f_idx}
                            ),
                            md = 6, align = 'center'
                        ),
                        dbc.Col(
                            dbc.Button(
                                'Plot it!',
                                className = 'd-grid col-12 mx-auto',
                                n_clicks = 0,
                                style = {'width': '100%'},
                                id = {'type':'frame-data-butt','index': f_idx}
                            ),
                            md = 3
                        )
                    ],style = {'display': 'none'} if not view_type['name'] in ['channel_hist','features'] else {}),
                    dbc.Row([
                        dbc.Col(
                            dbc.Label('Select a label: ',html_for = {'type': 'frame-label-drop','index': f_idx}),
                            md = 3
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                options = self.channel_names,
                                value = f_label if len(f_frame_list)>2 else None,
                                id = {'type': 'frame-label-drop','index': f_idx},
                                disabled = True if len(f_frame_list)<2 or not view_type['name']=='features' else False
                            )
                        )
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label(chart_label),
                            dcc.Graph(
                                id = {'type': 'ftu-cell-pie','index':f_idx},
                                figure = go.Figure(f_tab_plot)
                            )
                        ],md = 12)
                    ])
                ], label = f+f' ({viewport_data[f]["count"]})',tab_id = f'tab_{len(tab_list)}')

                tab_list.append(f_tab)


        viewport_data_components = html.Div([
            dbc.Row([
                dbc.Col(
                    children = [
                        dbc.Label('Select View Type: '),
                    ],
                    md = 4
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options = [
                            {'label':'Channel Intensity Histograms','value':'channel_hist'},
                            {'label':'Nucleus Features','value':'features'},
                            {'label':'Cell Type Composition','value':'cell'}
                        ],
                        value = view_type['name'],
                        multi = False,
                        id = {'type':'roi-view-data','index':0}
                    )
                )
            ]),
            html.Hr(),
            dbc.Row(
                dbc.Tabs(tab_list,active_tab=f'tab_{len(tab_list)-1}')
            )
        ])


        return viewport_data_components, viewport_data

    def populate_cell_annotation(self, region_selection, current_data):
        """
        Populating cell annotation tab based on previous/new selections
        """
        # region_selection = whether running for the whole slide or just a specific region (current viewport)
        # current_data = dictionary containing any previous cell annotation information
        # Main component should be a big plot showing dimensional reduction, either select points
        # or run some clustering to pull out groups of points, see violin plots of specific channels, parent structures,
        # and some connectivity/graph-based measurement

        print(f'region_selection: {region_selection}')
        print(f'current_data: {current_data}')

        cell_annotation_components = []

        cell_annotation_components = [html.Div([
                                        dbc.Row('Some header here to describe the tab'),
                                        html.Hr(),
                                        dbc.Row('Then something about either using all data or the current viewport'),
                                        html.Hr(),
                                        dbc.Row('Then something that will show the big plot of all all/select cell features'),
                                        html.Hr(),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Row(
                                                    'Maybe something here where you can specify cell type names or view current ontologies'
                                                )
                                            ],md = 6),
                                            dbc.Col([
                                                dbc.Row(
                                                    'Then something here where you can pick and choose labels for individual cells that were selected'
                                                )
                                            ],md=6)
                                        ]),
                                        html.Hr(),
                                        dbc.Row(
                                            'Should be some kind of progress indicator down here'
                                        )
                                    ])]
        

        return cell_annotation_components



class XeniumSlide(DSASlide):
    def __init__(self,
                 item_id: str,
                 user_details,
                 girder_handler,
                 ftu_colors:dict,
                 manual_rois:list,
                 marked_ftus:list):
        super().__init__(item_id,user_details,girder_handler,ftu_colors,manual_rois,marked_ftus)

        self.default_view_type = {
            'name': 'cell_composition',
        }

        self.n_frames = 0

    def update_viewport_data(self,bounds_box,view_type):
        """
        Updating data passed to charts based on change in viewport position

        view_types available: cell_composition, features
        """
        
        if view_type['name'] is None:
            view_type = self.default_view_type

        viewport_data_components = []
        viewport_data = None

        #TODO: Collect all FTUs within the bounds_box



        return viewport_data_components, viewport_data















