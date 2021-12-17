from abc import *
from pathlib import Path
import pickle
import os
from tqdm import trange
import numpy as np
from collections import Counter


class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, flag, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.flag = flag
        self.save_path = save_path

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        print("Negative samples don't exist. Generating.")
        seen_samples, negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump([seen_samples, negative_samples], f)
        return seen_samples, negative_samples

    def _get_save_path(self):
        folder = Path(self.save_path)
        if not folder.is_dir():
            folder.mkdir(parents=True)
        # filename = '{}-sample_size{}-seed{}-{}.pkl'.format(
        #     self.code(), self.sample_size, self.seed, self.flag)
        filename = 'negative_samples_{}.pkl'.format(self.flag)
        return folder.joinpath(filename)


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(self.item_count, num_samples) + 1

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items randomly...')
        j = 0
        for i in trange(self.user_count):
            user = i + 1
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            seen_samples[user] = seen

            samples = []
            while len(samples) < self.sample_size:
                item = all_samples[j % num_samples]
                j += 1
                if item in seen or item in samples:
                    continue
                samples.append(item)
            negative_samples[user] = samples

        return seen_samples, negative_samples


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        popularity = self.items_by_popularity()
        items = list(popularity.keys())
        total = 0
        for i in range(len(items)):
            total += popularity[items[i]]
        for i in range(len(items)):
            popularity[items[i]] /= total
        probs = list(popularity.values())
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(items, num_samples, p=probs)

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items by popularity...')
        j = 0
        for i in trange(self.user_count):
            user = i + 1
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            seen_samples[user] = seen

            samples = []
            while len(samples) < self.sample_size:
                item = all_samples[j % num_samples]
                j += 1
                if item in seen or item in samples:
                    continue
                samples.append(item)
            negative_samples[user] = samples

        return seen_samples, negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        self.users = sorted(self.train.keys())
        for user in self.users:
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])

        popularity = dict(popularity)
        popularity = {k: v for k, v in sorted(popularity.items(), key=lambda item: item[1], reverse=True)}
        return popularity


NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
}

def negative_sampler_factory(code, train, val, test, 
                             user_count, item_count,
                             sample_size, seed, flag,
                             save_path):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, val, test, user_count,
                            item_count, sample_size, seed, 
                            flag, save_path)


if __name__ == '__main__':
    PREP_DATASET_ROOT_FOLDER = 'data/silver'
    FEATURES_ROOT_FOLDER = 'data/gold'
    source_filepath = Path(os.path.join(PREP_DATASET_ROOT_FOLDER, 'ml-1m/dataset.pkl'))
    dataset = pickle.load(source_filepath.open('rb'))
    code = 'random'
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']
    umap = dataset['umap']
    smap = dataset['smap']
    user_count = len(umap)
    item_count = len(smap)
    sample_size = 100
    seed = 0
    flag = 'val'
    save_path = os.path.join(FEATURES_ROOT_FOLDER, 'ml-1m', 'negative_samples')
    negative_sampler = negative_sampler_factory(code, train, val, test, 
                             user_count, item_count,
                             sample_size, seed, flag,
                             save_path)
    _, _ = negative_sampler.get_negative_samples()