"""Fetch pinned training inputs only; never executes downloaded repository code."""
from __future__ import annotations
import hashlib
import json
import urllib.request
from .evo_public_data import REPOSITORIES, SOURCES, REPORTS

FILES = {
    'meetingqa':['LICENSE','AllData/Dataset/final-AMI-train.json'],
    'MeeQA':['Data/original/train_data.json.zip','Data/train_meetings.csv'],
    'MISeD':['mised/train.jsonl'],
}


def fetch_training():
    expected = json.loads((REPORTS/'catalog.json').read_text())['sources'] if (REPORTS/'catalog.json').exists() else {}
    for dataset,paths in FILES.items():
        repository,commit = REPOSITORIES[dataset]
        for relative in paths:
            target = SOURCES/dataset/relative
            if not target.exists():
                request = urllib.request.Request(f'https://raw.githubusercontent.com/{repository}/{commit}/{relative}',headers={'User-Agent':'Tokenese-research'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    content = response.read()
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary = target.with_suffix(target.suffix+'.download')
                temporary.write_bytes(content)
                temporary.replace(target)
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            entry = expected.get(dataset,{})
            if entry.get('path','').endswith('/'+relative) and sha != entry['file_sha256']:
                raise ValueError(f'Pinned source checksum mismatch: {dataset}/{relative}')
            print(f'{dataset}/{relative}: {sha}')


if __name__ == '__main__':
    fetch_training()
