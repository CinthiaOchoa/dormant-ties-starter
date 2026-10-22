"""Download the original SNAP releases.

Author: Cinthia Ochoa Torre

The static release (email-Eu-core) is needed only to recover the node-ID
crosswalk to the department labels; see crosswalk.py.
"""
from urllib.request import urlretrieve

from config import RAW

SOURCES = {
    'email-Eu-core-temporal.txt.gz': 'https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz',
    'email-Eu-core.txt.gz': 'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
    'email-Eu-core-department-labels.txt.gz': 'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz',
}

if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = RAW / name
        if dest.exists():
            print('Already downloaded:', dest)
            continue
        print('Downloading', url)
        urlretrieve(url, dest)
        print('Saved', dest)
