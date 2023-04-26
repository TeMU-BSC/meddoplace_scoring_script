#Run

$input is the directory where there is the reference folder and result folder
the name of the reference folder should be ref and result res

allCountries.txt can be download from here http://download.geonames.org/export/dump/allCountries.zip

```
python meddoplace_normalization.py -i ./input -o ./ -d <dir/to/allCountries.txt> -p GN
```

-p can be [GN,PC,SCTID,task1,task3,all]
