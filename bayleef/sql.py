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
    For tagging day night, the following info is used from USGS EROS:

    Field Definition:  Nearest WRS Row to the Line-Of-Sight scene center.
    This is used primarily for scenes with off-nadir look angles.
    The center of scene for off-nadir imaging may be several paths
    left or right of the orbital path and the center may even be
    off the WRS-2 grid when near the poles. This is an estimated value, for reference.

    Values:
    001-059 = Northern Hemisphere (Descending)
    060 = Equator (Descending)
    061-119 = Southern Hemisphere (Descending)
    120-122 = Southern Polar Zone (Descending)
    123-183 = Southern Hemisphere (Ascending)
    184 = Equator (Ascending)
    185-246 = Northern Hemisphere (Ascending)
    247-248 = Northern Polar Zone (Descending)

    For acquisitions near the poles, it is possible to
    look off-nadir toward the pole, into an area not defined by the WRS-2 grid
    (above 82.61 degrees). To allow unique Target Row assignments, the North Pole
    area is assigned a row of 88n, and the South Pole area is assigned a row of 99n,
    where n is a sequential number. Up to seven scenes can be covered in these areas;
    therefore, the scenes are assigned row numbers 880 to 886, or 990 to 996.

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
        print(traceback.format_exc())
        raise ValueError("{} is not a valid landsat folder: {}".format(folder, e))

    metadata = pvl.load(metafile)[0][1]
    # Change all keys to lowercase since postgres makes uppercase names a pain
    keys_to_lower(metadata)

    # Get the Primary Key
    pk = metadata['metadata_file_info']['landsat_scene_id']
    is_day = not int(metadata['product_metadata']['wrs_row']) in range(123,245)
    metadata['image_attributes']['is_daytime'] = is_day

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
