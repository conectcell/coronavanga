import os
import urllib.request
import zipfile

if not os.path.exists('.download'):
    os.makedirs('./download')


url = "https://github.com/CSSEGISandData/COVID-19/archive/master.zip"
print ("download start!")
filename, headers = urllib.request.urlretrieve(url, filename="./download/master.zip")
print ("download complete!")





with zipfile.ZipFile('./download/master.zip', 'r') as zip_ref:
    zip_ref.extractall('./download')


print ("Unziped!")