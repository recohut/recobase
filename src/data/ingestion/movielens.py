from .base import AbstractDataset

from pathlib import Path
import dvc.api
import os


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return {'path':'ml1m/v0',
                'repo':'https://github.com/sparsh-ai/reco-data'}

    @classmethod
    def all_raw_file_names(cls):
        return ['movies.dat',
                'ratings.dat',
                'users.dat']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)
        if all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
    
        print("Raw file doesn't exist. Downloading...")
        for filename in self.all_raw_file_names():
            with open(os.path.join(folder_path,filename), "w") as f:
                with dvc.api.open(
                    path=self.url()['path']+'/'+filename,
                    repo=self.url()['repo'],
                    encoding='ISO-8859-1') as scan:
                    f.write(scan.read())
        print()