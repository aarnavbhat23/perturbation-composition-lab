"""Retrieve the exact upstream GEARS export and extract only its expression file."""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
import truststore

URL='https://dataverse.harvard.edu/api/access/datafile/6154020'
SHA256='c1938353f6f41829c9137f073ecb843ce4d4114528b001b48312b019f739d877'


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


if __name__=='__main__':
    truststore.inject_into_ssl();folder=Path('data');folder.mkdir(exist_ok=True);archive=folder/'norman.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=120) as source,archive.open('wb') as target:
            shutil.copyfileobj(source,target)
    if digest(archive)!=SHA256:raise ValueError('Dataset checksum mismatch')
    with zipfile.ZipFile(archive) as source,source.open('norman/perturb_processed.h5ad') as handle,(folder/'norman.h5ad').open('wb') as target:
        shutil.copyfileobj(handle,target)
    print('Verified 161 MiB archive; extracted approximately 2.1 GiB expression file')
