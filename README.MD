# Bayleef

Interface to REST services for planetary imagery and relevant data.

Documentation is available at:

https://kelvinrr.github.io/bayleef/

```
# Get all night time images given a bounding box and time series
bayleef search LANDSAT_8_C1 --lower-left -165 15 --upper-right -150.0 28.0 --start-date 2013-04-01 --end-date 2013-05-1 where 20513 Night --geojson
```

<div style="margin-top:10px; margin-bottom:20px">
  <iframe class='ghmap' width="680" height="400" src="https://render.githubusercontent.com/view/geojson?commit=796cfecce7dbfaaf34664506c3f98d063f57c587&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f676973742f4b656c76696e72722f34303439363865643535613864616261613131356533313333623730343832652f7261772f373936636665636365376462666161663334363634353036633366393864303633663537633538372f746573742e67656f6a736f6e&nwo=Kelvinrr%2F404968ed55a8dabaa115e3133b70482e&path=test.geojson&repository_id=89327932&repository_type=Gist#7396ae8f-0d23-4319-bc9c-64a4e15e14d7" frameborder="0"></iframe>
</div>
