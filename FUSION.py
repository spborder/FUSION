"""

Running FUSION from fusion-tools

"""

import os

from fusion_tools.fusion import vis


def main():

    dsa_url = 'https://3-230-122-132.nip.io/api/v1'
    dsa_user = 'fusionguest'
    dsa_pword = 'Fus3yWasHere'

    host = os.environ.get('FUSION_HOST','0.0.0.0')
    port = os.environ.get('FUSION_PORT',8000)
    db_path = os.environ.get('DATABASE_PATH','./.fusion_assets/fusion_database.db')

    if all([i is None for i in [dsa_url,dsa_user,dsa_pword]]):
        raise Exception('Need to initialize with at least the environment variable: DSA_URL')

    initial_items = [
        '64ef9c712d82d04be3e2b330',
        '64f542322d82d04be3e39e94',
        '6495a4df3e6ae3107da10dc2',
        '6495a4e03e6ae3107da10dc5'
    ] 

    args_dict = {
        'girderApiUrl': dsa_url,
        'user': dsa_user,# Optional
        'pword': dsa_pword,# Optional,
        'database': db_path,
        'initialItems': initial_items,
        'app_options': {
            'host': host,
            'port': port
        }
    }

    fusion_vis = vis.get_layout(args_dict)
    fusion_vis.start()

if __name__=='__main__':
    main()

