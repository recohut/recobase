from .movielens import ML1MDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
}

def data_ingestion_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)