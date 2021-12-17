from .datasets import ML1MDataset
from .sampling import RandomNegativeSampler
from .dataloader import BertDataloader


DATASETS = {
    ML1MDataset.code(): ML1MDataset
}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)


NEGATIVE_SAMPLERS = {
    RandomNegativeSampler.code(): RandomNegativeSampler,
}

def negative_sampler_factory(code, train, val, test, user_count, item_count, sample_size, seed, save_folder):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, val, test, user_count, item_count, sample_size, seed, save_folder)


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
}

def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test