from bs4 import BeautifulSoup
import requests

url = 'http://cdimage.debian.org/debian-cd/9.9.0/arm64/'
ext = 'iso'

def listFD(url, ext=''):
    page = requests.get(url).text
    print (page)
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

for file in listFD(url, ext):
    print (file)