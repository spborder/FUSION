"""

Convenience script for installing multiple plugins at the same time

"""

import os
import girder_client



def main():

    dsa_url = os.environ.get('DSA_URL')
    username = os.environ.get('DSA_USER')
    p_word = os.environ.get('DSA_PWORD')

    gc = girder_client.GirderClient(apiUrl=dsa_url)
    gc.authenticate(username,p_word)

    image_name_list = [
        "dsarchive/histomicstk_extras:latest",
        "samborder2256/spot_annotation:latest",
        "samborder2255/spot_aggregation:latest",
        "samborder2256/multicomp:latest",
        "dpraveen511/ifta:ifta_seg_aws_1",
        "dpraveen511/ptc:ptc_seg_aws_1",
        "samborder2256/ann_hierarchy:latest",
        "samborder2256/deepcell_plugin:latest",
        "samborder2256/ftx_plugin:latest",
        "sayatmimar/atlasrds:t_7",
        "samborder2256/get_cluster_data:latest",
        "samborder2256/clustermarkers_fusion:latest"
    ]

    for image_name in image_name_list:
        cli_id = [i for i in gc.get('/slicer_cli_web/cli') if i['image']==image_name]
        if len(cli_id)>0:
            cli_id = cli_id[0]['_id']

            # Deleting previous CLI by id
            gc.delete(f'/slicer_cli_web/cli/{cli_id}')
            print('Deleting current version of CLI')
            # Deleting docker image (including local repo)
            del_response = gc.delete('/slicer_cli_web/docker_image',parameters={'name':image_name,'delete_from_local_repo':True})

        put_response = gc.put('/slicer_cli_web/docker_image',parameters={'name':image_name})

if __name__=='__main__':
    main()


