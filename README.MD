
![](logo.png)

Interface to REST services for planetary imagery with the goal of affording the transfer of mission data into an openly available and homogeneous data store.

Documentation is available at: https://kelvinrr.github.io/bayleef/

## CLI Overview

The CLI allows for the programatic searching, downloading and uploading (to a Postgres server) of USGS data.

The workflow is :

login -> search -> download -> upload  

#### Login

Before you can do anything, you must login using the CLI, logging in requires that you have a USGS EROS account https://ers.cr.usgs.gov/login/ and have authorization to download orders from the RESTful api.   

```
bayleef login [user] [pass]
```

#### Search

```
usgs search [dataset] --node [node] --start-date [start date] --end-date [end date] --longitude [lng] --latitude [lat] 
```

Breakdown:

 - [dataset] :

## Example

```
# Get night time Landsat 8 images given a bounding box and time range
bayleef search LANDSAT_8_C1 --lower-left -165 15 --upper-right -150.0 28.0 --start-date 2013-04-01 --end-date 2013-05-1 where 20513 Night --geojson
```

[<img src="footprints.png">](https://render.githubusercontent.com/view/geojson?commit=796cfecce7dbfaaf34664506c3f98d063f57c587&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f676973742f4b656c76696e72722f34303439363865643535613864616261613131356533313333623730343832652f7261772f373936636665636365376462666161663334363634353036633366393864303633663537633538372f746573742e67656f6a736f6e&nwo=Kelvinrr%2F404968ed55a8dabaa115e3133b70482e&path=test.geojson&repository_id=89327932&repository_type=Gist#7396ae8f-0d23-4319-bc9c-64a4e15e14d7)
