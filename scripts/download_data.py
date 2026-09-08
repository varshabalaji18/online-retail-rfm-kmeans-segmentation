"""Download the public UCI workbook; do not commit the raw dataset."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

if __name__ == '__main__':
    target = Path('data')
    target.mkdir(exist_ok=True)
    archive = target / 'online_retail.zip'
    urlretrieve('https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip', archive)
    with ZipFile(archive) as bundle:
        with bundle.open('Online Retail.xlsx') as source:
            (target / 'Online Retail.xlsx').write_bytes(source.read())
    print('Downloaded data/Online Retail.xlsx (UCI Online Retail, CC BY 4.0)')
