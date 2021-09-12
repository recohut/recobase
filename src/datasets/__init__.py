from .movielens import ML1MDataset
from .ml_1m import ML1MDataset
from .gmcf import GMCFDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
}

def data_ingestion_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)