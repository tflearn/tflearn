from __future__ import print_function
import gzip
import os
from six.moves import urllib


def download_dataset(filename='titanic_dataset.csv', work_directory='./'):
    """Download the data, unless it's already here."""
    url = 'http://tflearn.org/resources/titanic_dataset.csv'
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print('Downloading Titanic dataset...')
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def load_dataset():
    raise NotImplementedError
