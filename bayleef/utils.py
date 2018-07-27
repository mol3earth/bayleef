import os
import wget
import re

import gdal
import numpy as np
from datetime import datetime
from plio.io.io_gdal import GeoDataset


def get_path(response, root, dataset):
    """
    """
    if isinstance(response, dict):
        response = [response]

    try:
        for data in response:
            scene_id = data['entityId']
            s = data['summary']
            date = datetime.strptime(data['acquisitionDate'], '%Y-%m-%d')
            pathstr, rowstr = re.findall(r'Path: \b\d+\b|Row: \b\d+\b', s)
            path = pathstr.split(' ')[1]
            row = rowstr.split(' ')[1]
            return os.path.join(root, dataset, str(date.year), row, path, scene_id)

    except Exception as e:
        print('Failed to process request: {}'.format(e))


def keys_to_lower(dictionary):
    """
    """
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            keys_to_lower(dictionary[key])
        dictionary[key.lower()] = dictionary.pop(key)


def apply_dict(dictionary, func, *args, **kwargs):
    """
    """
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            apply_dict(dictionary[key], func)
        dictionary[key] = func(dictionary[key], *args, **kwargs)

def geolocate(infile, outfile, lats, lons, dstSRS="EPSG:4326", format="GTiff", woptions={}, toptions={}):
    """
    """
    image = gdal.Open(infile, gdal.GA_Update)
    geoloc= {
        'X_DATASET' : lons,
        'X_BAND' : '1',
        'Y_DATASET' : lats,
        'Y_BAND' : '1',
        'PIXEL_OFFSET' : '0',
        'LINE_OFFSET' : '0',
        'PIXEL_STEP' : '1',
        'LINE_STEP' : '1'
    }

    image.SetMetadata(geoloc, 'GEOLOCATION')
    # explicity close image
    del image
    gdal.Warp(outfile, infile, format=format, dstSRS=dstSRS)
    return GeoDataset(outfile)


def master_isvalid(file):

    if len(gdal.Open(file).GetSubDatasets()) != 17:
        return False

    calibrated_image = GeoDataset('HDF4_SDS:UNKNOWN:"{}":37'.format(file))
    lats = GeoDataset('HDF4_SDS:UNKNOWN:"{}":30'.format(file))
    lons = GeoDataset('HDF4_SDS:UNKNOWN:"{}":31'.format(file))

    res = []
    for ds in [calibrated_image, lats, lons]:
        arr = ds.read_array()
        test = np.empty(arr.shape)
        test[:] = ds.metadata['_FillValue']
        res.append(not (test == arr).all())
    return all(res)
