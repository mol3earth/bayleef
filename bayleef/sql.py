import traceback
from glob import glob
import pvl

import geopandas as gpd
import pandas as pd
from datetime import datetime
from geoalchemy2 import Geometry, WKTElement
from geoalchemy2.shape import from_shape
import shapely
from shapely.geometry import Polygon
from sqlalchemy import *

from bayleef.utils import get_path
from bayleef.utils import keys_to_lower
from bayleef.utils import apply_dict


def landsat_8_c1_to_sql(folder, engine):
    """
    """
    try:
        metafile = glob(folder+'/*MTL.txt')[0]
        ang = glob(folder+'/*ANG.txt')[0]
        bqa = glob(folder+'/*BQA.TIF')[0]
        bands = [
             glob(folder+'/*B1.TIF')[0],
             glob(folder+'/*B2.TIF')[0],
             glob(folder+'/*B3.TIF')[0],
             glob(folder+'/*B4.TIF')[0],
             glob(folder+'/*B5.TIF')[0],
             glob(folder+'/*B6.TIF')[0],
             glob(folder+'/*B7.TIF')[0],
             glob(folder+'/*B8.TIF')[0],
             glob(folder+'/*B9.TIF')[0],
             glob(folder+'/*B10.TIF')[0],
             glob(folder+'/*B11.TIF')[0]
        ]
    except IndexError as e:
        traceback.print_exc()
        raise ValueError("{} is not a valid landsat folder: {}".format(folder, e))

    metadata = pvl.load(metafile)[0][1]

    # Get the Primary Key
    pk = metadata['METADATA_FILE_INFO']['LANDSAT_SCENE_ID']

    # Change all keys to lowercase since postgres makes uppercase names a pain
    keys_to_lower(metadata)

    # Strip Timestamp info from all dates, causes to_sql errors
    apply_dict(metadata, lambda x: str(x).split('+')[0] if isinstance(x, datetime) else x)

    # Process Metadata into the database
    for key in metadata.keys():
        # change the ID name
        metadata[key]['landsat_scene_id'] = pk
        df = gpd.GeoDataFrame(metadata[key], index=[0])

        df.to_sql(key, engine, index=False, schema='landsat_8_c1', if_exists='append')

    # Get spatiotemporal data
    pm = metadata['product_metadata']

    ll = pm['corner_ll_lon_product'], pm['corner_ll_lat_product']
    lr = pm['corner_lr_lon_product'], pm['corner_lr_lat_product']
    ul = pm['corner_ul_lon_product'], pm['corner_ul_lat_product']
    ur = pm['corner_ur_lon_product'], pm['corner_ur_lat_product']

    footprint = shapely.geometry.Polygon([ll, ul, ur, lr])
    time = metadata['product_metadata']['date_acquired']
    image_record = {
        'geom' : WKTElement(Polygon([ll, ul, ur, lr]), srid=4326),
        'time' : metadata['product_metadata']['date_acquired'],
        'landsat_scene_id' : pk,
        'b1' : bands[0],
        'b2' : bands[1],
        'b3' : bands[2],
        'b4' : bands[3],
        'b5' : bands[4],
        'b6' : bands[5],
        'b7' : bands[6],
        'b8' : bands[7],
        'b9' : bands[8],
        'b10' : bands[9],
        'b11' : bands[10],
        'bqa' : bqa,
        'metafile' : metafile,
        'ang' : ang
    }
    gpd.GeoDataFrame(image_record, index=[0]).to_sql('images', engine, schema='landsat_8_c1', if_exists='append', index=False,
                                                        dtype={'geom': Geometry('POLYGON', srid=4326)})



func_map = {
    'LANDSAT_8_C1' : landsat_8_c1_to_sql
}
