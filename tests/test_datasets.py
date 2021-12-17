# import torch

# from src import dataset_factory


# class Args:
#     RAW_DATASET_ROOT_FOLDER = '/content/data/raw'
#     dataset_code = ''
#     min_rating = 1
#     min_uc = 5
#     min_sc = 5
#     split = 'leave_one_out'

# args = Args()


# def testML1MDataset():
#     args.dataset_code = 'ml-1m'
#     dataset = dataset_factory(args)
#     dataset.load_dataset()


# if __name__ == "__main__":
#     testML1MDataset()