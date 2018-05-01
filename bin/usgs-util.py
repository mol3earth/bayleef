from datetime import date
import argparse

# No need to make these new requirements if they dont have to be
try:
    from usgs import api
except:
    print('You must install usgs-util.')

try:
    from geoalchemy2 import Geometry, WKTElement
except Exception as e:
    print(e)
    print('You must install geoalchemy')

try:
    import wget
except:
    print('You must install pywget')


class KwargsAction(argparse.Action):
    """
    Allows for makeshift kwargs option in argparse. Nested dicts work as strings.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        keyword_dict = {}

        for arg in values:  #values => The args found for keyword_args
            pieces = arg.split('=')
            try:
                keyword_dict[pieces[0]] = eval(pieces[1])
            except:
                keyword_dict[pieces[0]] = eval('\''+pieces[1]+'\'')

        setattr(namespace, self.dest, keyword_dict)




def bulk_download(response, root, dataset):
    results = response['data']['results']

    for scene in results:
        scene_id = scene['entityId']
        temp_file = '{}.tar.gz'.format(scene_id)

        path = get_path(scene, root, dataset)
        if os.path.exists(path):
            print('{} already in cache, skipping'.format(path))
            continue

        download_info = api.download('LANDSAT_8_C1', 'EE', scene_id)
        download_url = download_info['data'][0]

        print('Downloading: {} from {}'.format(scene_id, download_url))
        wget.download(download_url, temp_file)

        print('Extracting to {}'.format(path))
        tar = tarfile.open(temp_file)
        tar.extractall(path=path)
        tar.close()
        print('Removing {}'.format(temp_file))
        os.remove(temp_file)
        print()

    print('done')

if __name__ == '__main__':
    function_help_message = "\nLOGIN:\n  {}\n".format("--login username=user password=pass")
    function_help_message += "SEARCH:Params should be passed in '<param>=<val>'' format  {}\n".format(api.search.__doc__)
    function_help_message += "DOWNLOAD:\n  Same as search, except successful searches are downloaded to the specified directory"

    parser = argparse.ArgumentParser(description='Bulk download usgs landsat images in a query-able format.', usage=function_help_message)
    parser.add_argument('function', choices=['login','search', 'download'])
    parser.add_argument("args", help="extra args to pass into function", nargs='*', action=KwargsAction)
    opts = parser.parse_args()

    if opts.function == 'login':
        print(api.login(**opts.args))
        exit(0)

    if opts.function == 'search':
        # Only support Landsat8 Earth Engine stuff
        print(opts.args)
        print(api.search('LANDSAT_8_C1', 'EE', **opts.args))
        exit(0)

    if opts.function == 'download':
        location = opts.args.pop('root')

        print(api.search(**opts.args))
        exit(0)
