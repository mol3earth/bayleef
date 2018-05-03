import sys
import os, json
import click
import logging
import re
import wget
import tarfile

from threading import Thread
from glob import glob
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from bayleef import api
from bayleef.utils import get_path
from bayleef.sql import func_map


LOG_FORMAT = '%(asctime)-15s ->> %(message)s'
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger('bayleef-cli')
logger.setLevel(logging.INFO)

def get_node(dataset, node=None):
    """
    .. todo:: Move to more appropriate place in module.
    """

    if node is None:

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(cur_dir, "..", "data")
        dataset_path = os.path.join(data_dir, "datasets.json")

        with open(dataset_path, "r") as f:
            datasets = json.loads(f.read())

        node = datasets[dataset].upper()

    return node


def to_coordinates(bounds):
    xmin, ymin, xmax, ymax = bounds

    return [[
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin],
        [xmin, ymin]
    ]]


def to_geojson_feature(entry):

    # TODO: This key may not be present in all datasets.
    bounds = list(map(float, entry.pop("sceneBounds").split(',')))

    coordinates = to_coordinates(bounds)

    return {
        "type": "Feature",
        "properties": entry,
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        }
    }


def to_geojson(result):
    gj = {
        'type': 'FeatureCollection'
    }

    if type(result['data']) is list:
        features = list(map(to_geojson_feature, result['data']))
    else:
        features = list(map(to_geojson_feature, result['data']['results']))
        for key in result['data']:
            if key == "results":
                continue
            gj[key] = result['data'][key]

    gj['features'] = features
    for key in result:
        if key == "data":
            continue
        gj[key] = result[key]

    return gj


def explode(coords):
    for e in coords:
        if isinstance(e, (float, int, long)):
            yield coords
            break
        else:
            for f in explode(e):
                yield f


def get_bbox(f):
    x, y = zip(*list(explode(f['geometry']['coordinates'])))
    return min(x), min(y), max(x), max(y)


api_key_opt = click.option("--api-key", help="API key returned from USGS servers after logging in.", default=None)
node_opt = click.option("--node", help="The node corresponding to the dataset (CWIC, EE, HDDS, LPVS).", default=None)


@click.group()
def bayleef():
    pass


@click.command()
@click.argument("username", envvar='USGS_USERNAME')
@click.argument("password", envvar='USGS_PASSWORD')
def login(username, password):
    api_key = api.login(username, password)
    click.echo(api_key)


@click.command()
def logout():
    click.echo(api.logout())


@click.command()
@click.argument("node")
@click.option("--start-date")
@click.option("--end-date")
def datasets(node, start_date, end_date):
    data = api.datasets(None, node, start_date=start_date, end_date=end_date)
    click.echo(json.dumps(data))


@click.command()
@click.argument("dataset")
@click.argument("scene-ids", nargs=-1)
@node_opt
@click.option("--extended", is_flag=True, help="Probe for more metadata.")
@click.option('--geojson', is_flag=True)
@api_key_opt
def metadata(dataset, scene_ids, node, extended, geojson, api_key):

    if len(scene_ids) == 0:
        scene_ids = map(lambda s: s.strip(), click.open_file('-').readlines())

    node = get_node(dataset, node)
    result = api.metadata(dataset, node, scene_ids, extended=extended, api_key=api_key)

    if geojson:
        result = to_geojson(result)

    click.echo(json.dumps(result))


@click.command()
@click.argument("dataset")
@node_opt
def dataset_fields(dataset, node):
    node = get_node(dataset, node)
    data = api.dataset_fields(dataset, node)
    click.echo(json.dumps(data))


@click.command()
@click.argument("dataset")
@node_opt
@click.argument("aoi", default="-", required=False)
@click.option("--start-date")
@click.option("--end-date")
@click.option("--lng")
@click.option("--lat")
@click.option("--dist", help="Radius - in units of meters - used to search around the specified longitude/latitude.", default=100)
@click.option("--lower-left", nargs=2, help="Longitude/latitude specifying the lower left of the search window")
@click.option("--upper-right", nargs=2, help="Longitude/latitude specifying the lower left of the search window")
@click.option("--where", nargs=2, multiple=True, help="Supply additional search criteria.")
@click.option('--geojson', is_flag=True)
@click.option("--extended", is_flag=True, help="Probe for more metadata.")
@api_key_opt
def search(dataset, node, aoi, start_date, end_date, lng, lat, dist, lower_left, upper_right, where, geojson, extended, api_key):

    node = get_node(dataset, node)

    if aoi == "-":
        src = click.open_file('-')
        if not src.isatty():
            lines = src.readlines()

            if len(lines) > 0:

                aoi = json.loads(''.join([ line.strip() for line in lines ]))

                bbox = map(get_bbox, aoi.get('features') or [aoi])[0]
                lower_left = bbox[0:2]
                upper_right = bbox[2:4]

    if where:
        # Query the dataset fields endpoint for queryable fields
        resp = api.dataset_fields(dataset, node)

        def format_fieldname(s):
            return ''.join(c for c in s if c.isalnum()).lower()

        field_lut = { format_fieldname(field['name']): field['fieldId'] for field in resp['data'] }
        where = { field_lut[format_fieldname(k)]: v for k, v in where if format_fieldname(k) in field_lut }


    if lower_left:
        lower_left = dict(zip(['longitude', 'latitude'], lower_left))
        upper_right = dict(zip(['longitude', 'latitude'], upper_right))

    result = api.search(dataset, node, lat=lat, lng=lng, distance=dist, ll=lower_left, ur=upper_right, start_date=start_date, end_date=end_date, where=where, extended=extended, api_key=api_key)

    if geojson:
        result = to_geojson(result)

    print(json.dumps(result))


@click.command()
@click.argument("dataset")
@click.argument("scene-ids", nargs=-1)
@node_opt
@api_key_opt
def download_options(dataset, scene_ids, node, api_key):
    node = get_node(dataset, node)

    data = api.download_options(dataset, node, scene_ids)
    print(json.dumps(data))


@click.command()
@click.argument("dataset")
@click.argument("scene_ids", nargs=-1)
@click.option("--product", nargs=1, required=True)
@node_opt
@api_key_opt
def download_url(dataset, scene_ids, product, node, api_key):
    node = get_node(dataset, node)

    data = api.download(dataset, node, scene_ids, product)
    click.echo(json.dumps(data))


@click.command()
@click.argument("root")
@node_opt
def batch_download(root, node):
    def download_from_result(scene, root):
        scene_id = scene['entityId']
        temp_file = '{}.tar.gz'.format(scene_id)

        dataset = re.findall(r'dataset_name=[A-Z0-9_]*', scene['orderUrl'])[0]
        dataset = dataset.split('=')[1]

        path = get_path(scene, root, dataset)
        if os.path.exists(path):
            logger.warning('{} already in cache, skipping'.format(path))
            return

        download_info = api.download(dataset, get_node(dataset, node), scene_id)
        download_url = download_info['data'][0]

        logger.info('Downloading: {} from {}'.format(scene_id, download_url))
        wget.download(download_url, temp_file)
        print()
        logger.info('Extracting to {}'.format(path))
        tar = tarfile.open(temp_file)
        tar.extractall(path=path)
        tar.close()
        logger.info('Removing {}'.format(temp_file))
        os.remove(temp_file)
        logger.info('{} complete'.format(scene_id))


    # get pipped in response
    resp = ""
    for line in sys.stdin:
        resp += line

    # convert string into dict
    resp = json.loads(resp)
    logger.info("Number of files: {}".format(resp['data']['numberReturned']))
    results = resp['data']['results']

    for result in results:
        # Run Downloads as threads so they keyboard interupts are
        # deferred until download is complaete
        job = Thread(target=download_from_result, args=(result, root))
        job.start()
        job.join()


@click.command()
@click.argument("dataset")
@click.argument("root")
@click.argument("db")
@click.option("--host", default="localhost")
@click.option("--port", default="5432")
@click.option("--user")
@click.option("--password")
def to_sql(db, dataset, root, host, port, user, password):
    if not dataset in func_map.keys():
        logger.error("{} is not a valid dataset".format(dataset))

    dataset_root = os.path.join(root, dataset)

    # The dirs with important files will always be in leaf nodes
    leaf_dirs = list()
    for root, dirs, files in os.walk(dataset_root):
        if files and [f for f in files if not f.startswith('.')]:
            leaf_dirs.append(root)

    logger.info("{} Folders found".format(len(leaf_dirs)))

    # only suppoort postgres for now
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user,password,host,port,db))
    for dir in leaf_dirs:
        logger.info("Uploading {}".format(dir))
        try:
            func_map[dataset](dir, engine)
        except Exception as e:
            logger.error("{}".format(e))


bayleef.add_command(to_sql, "to-sql")
bayleef.add_command(login)
bayleef.add_command(logout)
bayleef.add_command(datasets)
bayleef.add_command(dataset_fields, "dataset-fields")
bayleef.add_command(metadata)
bayleef.add_command(search)
bayleef.add_command(download_options, "download-options")
bayleef.add_command(download_url, "download-url")
bayleef.add_command(batch_download, "batch-download")
