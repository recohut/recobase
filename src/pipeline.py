import dvc.api
import os
from pathlib import Path

from src.datasets import DATASETS
from src.config import *

stages = ['ingestion', 'preprocessing']

parent_path = Path(__file__).parent.parent

#### Data Ingestion into Bronzer Layer ####

def stage_ingestion(args):
    from src.datasets import dataset_factory
    dataset = dataset_factory(args)
    dataset.maybe_download_raw_dataset()

def stage_preprocessing():
    from src.preprocess import preprocess as prep
    prep()


import argparse

parser = argparse.ArgumentParser(description='Arguments to customize the pipeline.')

if __name__== '__main__':
    parser.add_argument('--stage', type=str, choices=stages)
    parser.add_argument('--dataset_code', type=str, default='ml-1m', choices=DATASETS.keys())
    parser.add_argument('--dvc', type=bool)
    args = parser.parse_args()

    # Stage - ingestion
    if args.stage == 'ingestion':
        if args.dvc==True:
            data_path = Path(os.path.join(parent_path, RAW_DATASET_ROOT_FOLDER, 'ml-1m'))
            if not data_path.is_dir():
                data_path.mkdir(parents=True)
            for file in ['ratings.dat', 'movies.dat', 'users.dat']:
                with open(os.path.join(str(data_path), file), 'w') as f:
                    with dvc.api.open('ml1m/v0/{}'.format(file), repo='https://github.com/sparsh-ai/reco-data') as scan:
                        f.write(scan.read())
        else:
            stage_ingestion(args)

    # Stage - preprocess
    if args.stage == 'preprocessing':
        stage_preprocessing()