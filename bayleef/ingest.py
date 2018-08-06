import json
from os import path

import bayleef
import geopandas as gpd

import plio
from plio.io.io_gdal import GeoDataset
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
from pylab import rcParams
import numpy as np
import gdal
from os import path
import os
import math
import osr
import hashlib
import pvl
from glob import glob
import geopandas as gpd
import pandas as pd
from datetime import date
from geoalchemy2 import Geometry, WKTElement
from geoalchemy2.shape import from_shape
import shapely
from shapely.geometry import Polygon
from sqlalchemy import *
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
import errno
from shutil import copyfile

from datetime import datetime

def master(root, masterhdf):
    fd = GeoDataset(masterhdf)
    meta = fd.metadata

    meta['bayleef_name'] = 'MASTER'

    for key in meta.keys():
        val = meta.pop(key)
        meta[key.lower()] = val

    date = datetime.strptime(meta['completiondate'] , "%d-%b-%Y %H:%M:%S")
    line = meta['flightlinenumber']
    daytime_flag = meta['day_night_flag']
    ID = meta['producer_granule_id'].split('.')[0]

    basepath = path.join(root, 'MASTER',str(date.year), str(line), daytime_flag,ID)
    ogdatapath = path.join(basepath, 'original')
    imagedatapath = path.join(basepath, 'imagedata')

    newhdf = path.join(ogdatapath, path.basename(masterhdf))

    # Try making the directory
    try:
        os.makedirs(basepath)
        os.makedirs(ogdatapath)
        os.makedirs(imagedatapath)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.isdir(basepath):
            pass
        else:
            raise

    # copy original hdf
    try:
        copyfile(masterhdf, newhdf)
    except shutil.SameFileError:
        pass

    # explicitly close file descriptor
    del fd

    fd = GeoDataset(newhdf)
    subdatasets = fd.dataset.GetSubDatasets()

    for dataset in subdatasets:
        ofilename = '{}.tif'.format(dataset[1].split()[1])
        ofilename_abspath = path.join(imagedatapath, ofilename)
        gdal.Translate(ofilename_abspath, dataset[0], format="GTiff")

    # create geo corrected calibrated image
    lats = path.join(imagedatapath, 'PixelLatitude.tif')
    lons = path.join(imagedatapath, 'PixelLongitude.tif')
    image = path.join(imagedatapath, 'CalibratedData.tif')

    geocorrected_image = path.join(imagedatapath, 'CalibratedData_Geo.tif')
    geolocate(image, geocorrected_image, lats, lons)


    # Master has 50 bands, extract them as seperate files
    for i in range(1,51):
        gdal.Translate(path.join(imagedatapath, 'b{}.tif'.format(i)), path.join(imagedatapath, 'CalibratedData_Geo.tif'), bandList=[i])

    index_meta = {}

    ll = float(meta['lon_ll']), float(meta['lat_ll'])
    lr = float(meta['lon_lr']), float(meta['lon_lr'])
    ul = float(meta['lon_ul']), float(meta['lon_ul'])
    ur = float(meta['lon_ur']), float (meta['lon_ur'])

    index_meta['geom'] = Polygon([ll, ul, ur, lr]).wkt
    index_meta['id'] = ID

    index_meta['time'] = {}
    index_meta['time']['year'] = date.year
    index_meta['time']['month'] = date.month
    index_meta['time']['day'] = date.day
    index_meta['time']['hour'] = date.hour

    index_json_file = path.join(basepath, 'index.json')
    meta_json_file = path.join(basepath,'meta.json')
    json.dump(index_meta, open(index_json_file, 'w+'))
    json.dump(meta, open(meta_json_file, 'w+'))
    
