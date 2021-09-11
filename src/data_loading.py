from abc import *
import random
import torch
import pickle
import random
import os
from pathlib import Path
import torch.utils.data as data_utils


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.rng = random.Random()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.smap = dataset['smap']
        self.item_count = len(self.smap)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass


class BERTDataloader():
    def __init__(self, args, dataset, seen_samples, val_negative_samples):
        self.args = args
        self.rng = random.Random()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.smap = dataset['smap']
        self.item_count = len(self.smap)

        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        self.seen_samples = seen_samples
        self.val_negative_samples = val_negative_samples
        self.test_negative_samples = val_negative_samples

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BERTTrainDataset(
            self.train, self.max_len, self.mask_prob, self.max_predictions, self.sliding_size, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = BERTValidDataset(self.train, self.val, self.max_len, self.CLOZE_MASK_TOKEN, self.val_negative_samples)
        elif mode == 'test':
            dataset = BERTTestDataset(self.train, self.val, self.test, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BERTTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, max_predictions, sliding_size, mask_token, num_items, rng):
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.max_predictions = max_predictions
        self.sliding_step = int(sliding_size * max_len)
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        
        assert self.sliding_step > 0
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]

    def __len__(self):
        return len(self.all_seqs)
        # return len(self.users)

    def __getitem__(self, index):
        # user = self.users[index]
        # seq = self._getseq(user)
        seq = self.all_seqs[index]

        tokens = []
        labels = []
        covered_items = set()
        for i in range(len(seq)):
            s = seq[i]
            if (len(covered_items) >= self.max_predictions) or (s in covered_items):
                tokens.append(s)
                labels.append(0)
                continue
            
            temp_mask_prob = self.mask_prob
            if i == (len(seq) - 1):
                temp_mask_prob += 0.1 * (1 - self.mask_prob)

            prob = self.rng.random()
            if prob < temp_mask_prob:
                covered_items.add(s)
                prob /= temp_mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BERTValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, valid_users=None):
        self.u2seq = u2seq  # train
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class BERTTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, mask_token, negative_samples, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]  # append validation item after train seq
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


def dataloader_factory(args, dataset, seen_samples, val_negative_samples):
    if args.model_code == 'bert':
        dataloader = BERTDataloader(args, dataset, seen_samples, val_negative_samples)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test


if __name__ == '__main__':
    PREP_DATASET_ROOT_FOLDER = 'data/silver'
    FEATURES_ROOT_FOLDER = 'data/gold'
    prep_filepath = Path(os.path.join(PREP_DATASET_ROOT_FOLDER, 'ml-1m/dataset.pkl'))
    dataset = pickle.load(prep_filepath.open('rb'))
    ns_val_filepath = Path(os.path.join(FEATURES_ROOT_FOLDER, 'ml-1m/negative_samples/negative_samples_val.pkl'))
    seen_samples, val_negative_samples = pickle.load(ns_val_filepath.open('rb'))
    class Args:
        model_code = 'bert'
        sliding_window_size = 0.5
        bert_hidden_units = 64
        bert_dropout = 0.1
        bert_attn_dropout = 0.1
        bert_max_len = 200
        bert_mask_prob = 0.2
        bert_max_predictions = 40
        batch = 128
        train_batch_size = batch
        val_batch_size = batch
        test_batch_size = batch
        device = 'cpu'
        optimizer = 'AdamW'
        lr = 0.001
        weight_decay = 0.01
        enable_lr_schedule = True
        decay_step = 10000
        gamma = 1.
        enable_lr_warmup = False
        warmup_steps = 100
        num_epochs = 1
        metric_ks = [1, 5, 10]
        best_metric = 'NDCG@10'
        model_init_seed = 98765
        bert_num_blocks = 2
        bert_num_heads = 2
        bert_head_size = None
    args = Args()
    train_loader, val_loader, test_loader = dataloader_factory(args, dataset,
                                 seen_samples,
                                 val_negative_samples)
    dataloader_savepath = Path(os.path.join(FEATURES_ROOT_FOLDER, 'ml-1m/dataloaders'))
    if not dataloader_savepath.is_dir():
        dataloader_savepath.mkdir(parents=True)
    torch.save(train_loader, dataloader_savepath.joinpath('train_loader.pt'))
    torch.save(val_loader, dataloader_savepath.joinpath('val_loader.pt'))
    torch.save(test_loader, dataloader_savepath.joinpath('test_loader.pt'))