import pickle
import shutil
import tempfile
import os
from datetime import date
from pathlib import Path
import gzip
import argparse
from abc import *

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(
                lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
            train, val, test = {}, {}, {}
            for i in range(user_count):
                user = i + 1
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = Path(PREP_DATASET_ROOT_FOLDER)
        return root.joinpath(self.raw_code())

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        # folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
        #     .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        # return preprocessed_root.joinpath(folder_name)
        return preprocessed_root

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        df = self.load_ratings_df()
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None, engine='python')
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


if __name__ == '__main__':
    RAW_DATASET_ROOT_FOLDER = 'data/bronze'
    PREP_DATASET_ROOT_FOLDER = 'data/silver'
    class Args:
        min_rating = 0
        min_uc = 5
        min_sc = 5
        split = 'leave_one_out'
    args = Args()
    dataset = ML1MDataset(args)
    dataset.preprocess()