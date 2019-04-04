
import os
from os.path import expanduser
from xml.etree import ElementTree
import requests
from requests_futures.sessions import FuturesSession

from bayleef import USGS_API, USGSError
from bayleef import xsi, payloads
from bayleef.utils import geolocate, master_isvalid

TMPFILE = os.path.join(expanduser("~"), ".usgs")
NAMESPACES = {
    "eemetadata": "http://earthexplorer.usgs.gov/eemetadata.xsd"
}


def _get_api_key(api_key):
    if os.path.exists(TMPFILE):
        with open(TMPFILE, "r") as f:
            api_key = f.read()

    return api_key


def _check_for_usgs_error(data):

    error_code = data['errorCode']
    if error_code is None:
        return

    error = data['error']

    raise USGSError('%s: %s' % (error_code, error))


def _get_extended(scene, resp):
    """
    Parse metadata returned from the metadataUrl of a USGS scene.

    Parameters
    ----------
    scene : dict
        Dictionary representation of a USGS scene
    resp :
        Response object from requests/grequests
    Returns
    -------
    scene :
    """
    root = ElementTree.fromstring(resp.text)
    items = root.findall("eemetadata:metadataFields/eemetadata:metadataField", NAMESPACES)
    scene['extended'] = {item.attrib.get('name').strip(): xsi.get(item[0]) for item in items}

    return scene


def _async_requests(urls):
    """
    Sends multiple non-blocking requests. Returns
    a list of responses.

    Parameters
    ----------
    urls : list
        List of urls
    Returns
    -------
    responses : list
        list of responses
    """
    session = FuturesSession(max_workers=30)
    futures = [
        session.get(url)
        for url in urls
    ]
    return [ future.result() for future in futures ]


def _get_metadata_url(scene):
    return scene['metadataUrl']


def clear_bulk_download_order():
    raise NotImplementedError


def clear_order():
    raise NotImplementedError


def dataset_fields(dataset, node, api_key=None):

    api_key = _get_api_key(api_key)

    payload = {
        "jsonRequest": payloads.dataset_fields(dataset, node, api_key=api_key)
    }
    url = '{}/datasetfields'.format(USGS_API)
    r = requests.post(url, payload)
    response = r.json()

    _check_for_usgs_error(response)

    return response


def datasets(dataset, node, ll=None, ur=None, start_date=None, end_date=None, api_key=None):

    api_key = _get_api_key(api_key)

    url = '{}/datasets'.format(USGS_API)

    payload = {
        "jsonRequest": payloads.datasets(dataset, node, ll=ll, ur=ur, start_date=start_date, end_date=end_date, api_key=api_key)
    }
    r = requests.post(url, payload)
    response = r.json()

    _check_for_usgs_error(response)

    return response


def download(dataset, node, entityids, product='STANDARD', api_key=None):
    """
    Though USGS supports multiple products in a single request, there's
    ambiguity in the returned list. This wrapper only allows a single
    product per request.

    Additionally, the response has no indiction which URL is associated
    with which scene/entity id. The URL can be parsed, but the structure
    varies depending on the product.

    Parameters
    ----------
    dataset : str
        The specific collection to download from (e.g. LANDSAT_8_C1 for landsat 8 collection 1), see docs for a list of available datasets
    node : str
        Name of the catalog to search in (e.g. cwic for CWIC/LSI Explorer, ee for Earth Explorer, ect.), see docs for a list of available nodes
    entityids :
    product : str, optional

    Returns
    -------
    response : json
        json
    """

    api_key = _get_api_key(api_key)

    url = '{}/download'.format(USGS_API)
    payload = {
        "jsonRequest": payloads.download(dataset, node, entityids, [product], api_key=api_key)
    }

    r = requests.post(url, payload)
    response = r.json()

    _check_for_usgs_error(response)

    return response


def download_options(dataset, node, entityids, api_key=None):
    """
    Parameters
    ----------
    dataset : str
        The specific collection to download from (e.g. LANDSAT_8_C1 for landsat 8 collection 1), see docs for a list of available datasets
    node : str
        Name of the catalog to search in (e.g. cwic for CWIC/LSI Explorer, ee for Earth Explorer, ect.), see docs for a list of available nodes
    entityids :

    Returns
    -------
    response : json
        json
    """
    api_key = _get_api_key(api_key)

    url = '{}/downloadoptions'.format(USGS_API)
    payload = {
        "jsonRequest": payloads.download_options(dataset, node, entityids, api_key=api_key)
    }

    r = requests.post(url, payload)
    response = r.json()

    _check_for_usgs_error(response)

    return response


def get_bulk_download_products():
    raise NotImplementedError


def get_order_products():
    raise NotImplementedError


def hits():
    raise NotImplementedError


def item_basket():
    raise NotImplementedError

def login(username, password, save=True):
    """
    Logs a user in using their username and password for the USGS’s EROS service

    Parameters
    ----------
    username : str
        username for an account with the USGS’s EROS service
    password : str
        password for an account with the USGS’s EROS service
    save : bool, optional
        whether the credentials of the session should be saved as a temp file
    Returns
    -------
    response : json
        json with response from server after login request
    """
    url = '{}/login'.format(USGS_API)
    payload = {
        "jsonRequest": payloads.login(username, password)
    }

    r = requests.post(url, payload)
    if r.status_code is not 200:
        raise USGSError(r.text)

    response = r.json()
    api_key = response["data"]

    if api_key is None:
        raise USGSError(response["error"])

    if save:
        with open(TMPFILE, "w") as f:
            f.write(api_key)

    return response


def logout(api_key=None):
    """
    Logs out a user and removes credentials from session.

    Returns
    -------
    response : json
        json with response from server after logout request
    """
    api_key = _get_api_key(api_key)

    url = '{}/logout'.format(USGS_API)
    payload = {
        "jsonRequest": payloads.logout(api_key)
    }
    r = requests.post(url, payload)
    response = r.json()

    _check_for_usgs_error(response)

    if os.path.exists(TMPFILE):
        os.remove(TMPFILE)

    return response


def metadata(dataset, node, entityids, extended=False, api_key=None):
    """
    Request metadata for a given scene in a USGS dataset.

    Parameters
    ----------
    dataset : str
        The specific collection to download from (e.g. LANDSAT_8_C1 for landsat 8 collection 1), see docs for a list of available datasets
    node : str
        Name of the catalog to search in (e.g. cwic for CWIC/LSI Explorer, ee for Earth Explorer, ect.), see docs for a list of available nodes
    entityids :
    extended : bool, optional
        Send a second request to the metadata url to get extended metadata on the scene.
    Returns
    -------
    response : json
        json containing metadata
    """
    api_key = _get_api_key(api_key)

    url = '{}/metadata'.format(USGS_API)
    payload = {
        "jsonRequest": payloads.metadata(dataset, node, entityids, api_key=api_key)
    }
    r = requests.post(url, payload)
    response = r.json()

    _check_for_usgs_error(response)

    if extended:
        metadata_urls = map(_get_metadata_url, response['data'])
        results = _async_requests(metadata_urls)
        data = map(lambda idx: _get_extended(response['data'][idx], results[idx]), range(len(response['data'])))

    return response


def remove_bulk_download_scene():
    raise NotImplementedError


def remove_order_scene():
    raise NotImplementedError


def search(dataset, node, lat=None, lng=None, distance=100, ll=None, ur=None, start_date=None, end_date=None,
           where=None, max_results=50000, starting_number=1, sort_order="DESC", extended=False, api_key=None):
    """
    Search the database for images that match parameters passed.

    Parameters
    ----------
    dataset : str
        USGS dataset (e.g. EO1_HYP_PUB, LANDSAT_8)
    node : str
        USGS node representing a dataset catalog (e.g. CWIC, EE, HDDS, LPVS)
    lat : double, optional
        Latitude
    lng : double, optional
        Longitude
    distance : int, optional
        Distance in meters used to for a radial search
    ll : dict, optional
        Dictionary of longitude/latitude coordinates for the lower left corner
        of a bounding box search. e.g. { "longitude": 0.0, "latitude": 0.0 }
    ur : dict, optional
        Dictionary of longitude/latitude coordinates for the upper right corner
        of a bounding box search. e.g. { "longitude": 0.0, "latitude": 0.0 }
    start_date : str, optional
        Start date for when a scene has been acquired. In the format of yyyy-mm-dd
    end_date : str, optional
        End date for when a scene has been acquired. In the format of yyyy-mm-dd
    where : dict, optional
        Dictionary representing key/values for finer grained conditional
        queries. Only a subset of metadata fields are supported. Available
        fields depend on the value of `dataset`, and maybe be found by
        submitting a dataset_fields query.
    max_results : int, optional
        Maximum results returned by the server
    starting_number : int, optional
        Starting offset for results of a query.
    sort_order : str, optional
        Order in which results are sorted. Ascending ("ASC") or descending("DESC") with reference to the acquisition date.
    extended : bool, optional
        Boolean flag. When true a subsequent query will be sent to the `metadataUrl` returned by
        the first query.
    Returns
    -------
    response : json
        json response from the search that matches the parameters given
    """
    api_key = _get_api_key(api_key)

    url = '{}/search'.format(USGS_API)
    payload = {
        "jsonRequest": payloads.search(dataset, node,
        lat=lat, lng=lng,
        distance=100,
        ll=ll, ur=ur,
        start_date=start_date, end_date=end_date,
        where=where,
        max_results=max_results,
        starting_number=starting_number,
        sort_order=sort_order,
        api_key=api_key
    )}
    r = requests.post(url, payload)
    response = r.json()

    _check_for_usgs_error(response)

    if extended:
        metadata_urls = map(_get_metadata_url, response['data']['results'])
        results = _async_requests(metadata_urls)
        data = map(lambda idx: _get_extended(response['data']['results'][idx], results[idx]), range(len(response['data']['results'])))

    return response

def submit_bulk_order():
    raise NotImplementedError


def submit_order():
    raise NotImplementedError


def update_bulk_download_scene():
    raise NotImplementedError


def update_order_scene():
    raise NotImplementedError

def upload_sql(db, dataset, root, host, port, user, password):
    """
    Upload the dataset to a database
    """
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
            logger.error("ERROR: {}".format(e))
            import traceback
            traceback.print_exc()

def download(root, node, resp=None):
    """
    Used with the search function to download the results returned from search
    """
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


    if(resp == None):
        # get piped in response
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
